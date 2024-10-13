import re
import pandas as pd
import multiprocessing as mp
import submitit
import logging
import luigi
import string
import fasttext
import math
import unicodedata
import numpy as np
import pipeline.minhash as mh
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Set
from abc import ABC, abstractmethod
from collections import Counter
from pipeline.markdown_processing import ProcessMarkdown
from pipeline.utils import GLOBAL_CONF
from pipeline.s3_utils import S3Target, get_s3fs_client


TRANSLATION_TABLE_PUNCTUATION = str.maketrans("", "", string.punctuation)


@dataclass
class Document:
    url: str
    text: str
    images: List[Tuple[int, str]]  # index, url


def normalize(
    text: str,
    remove_punct: bool = True,
    lowercase: bool = True,
    nfd_unicode: bool = True,
    white_space: bool = True
) -> str:
    """ Normalize the text by lowercasing and removing punctuation. """
    # remove punctuation
    if remove_punct:
        text = text.translate(TRANSLATION_TABLE_PUNCTUATION)
    # lowercase
    if lowercase:
        text = text.lower()
    if white_space:
        text = text.strip()
        text = re.sub(r"\s+", " ", text)
    # NFD unicode normalization
    if nfd_unicode:
        text = unicodedata.normalize("NFD", text)
    return text


class QualityFeature(ABC):
    """
    The abstract filter class
    """
    @abstractmethod
    def __call__(self, document: Document):
        """
        Takes a document and returns a quality value
        """
        pass



class ContentLength(QualityFeature):
    """
    The character length of the document 
    """
    def __call__(self, document: Document) -> int:
        return {"content_length": len(document.text)}


class IsDomainBlacklisted(QualityFeature):
    domain_pattern = r"https?://(?:www\.)?([^\/:?]+)"

    def __init__(self, blacklisted_domains: Set[str] = set(), domain_parse_error_strategy: str = "keep"):
        self.blacklisted_domains = blacklisted_domains
        assert domain_parse_error_strategy in ("keep", "remove")
        self.domain_parse_error_strategy = domain_parse_error_strategy

    def __call__(self, document: Document) -> bool:
        match = re.search(self.domain_pattern, document.url)
        if match:
            domain = match.group(1)
            val = domain in self.blacklisted_domains
        else:
            val = self.domain_parse_error_strategy == "keep"
        return {"is_domain_blacklisted": val}
        

class RatioAlphanumericChars(QualityFeature):
    def __call__(self, document: Document) -> float:
        if len(document.text) == 0:
            val = 0.0
        else:
            val = len([char for char in document.text if char.isalnum()]) / len(document.text)
        return {"ratio_alphanumeric_chars": val}


class RatioHeadingsPerNonHeadingWord(QualityFeature):
    def __call__(self, document: Document) -> float:
        num_headings = 0
        num_non_heading_words = 0
        for line in document.text.split("\n"):
            if line.startswith("#"):
                num_headings += 1
            else:
                num_non_heading_words += len(normalize(line).split())
        return {"ratio_headings_per_non_heading_word": num_headings / num_non_heading_words if num_non_heading_words > 0 else 0.0}


class UnigramEntropy(QualityFeature):
    def __call__(self, document: Document):
        # count the number of times each word appears in the content
        counter = Counter(normalize(document.text).split())

        # calculate the entropy of the unigram distribution
        total = sum(counter.values())
        entropy = sum(map(
            lambda x: -x / total * math.log(x / total) if x > 0 else 0.0,
            counter.values()
        ))
        return {"unigram_entropy": entropy}


class Language(QualityFeature):
    def __init__(self, model_path: Path):
        self.model_path = model_path

    def __call__(self, document: Document):
        # Load the model here as it cannot be pickled
        if not hasattr(self, "model"):
            self.model = fasttext.load_model(str(self.model_path))

        label, score = self.model.predict(document.text.replace("\n", ""))
        label = label[0].replace("__label__", "")
        return {"language": label}


class MinHashLSH(QualityFeature):
    def __init__(self, buckets=14, bucketsize=8, shinglesize=4, seed=0x5eed):
        self.buckets = buckets
        self.bucketsize = bucketsize
        self.shinglesize = shinglesize
        self.seed = seed

        permutations = self.buckets * self.bucketsize

        rng = np.random.default_rng(seed=self.seed)

        self.a = rng.integers(low=1, high=mh.Pm, size=permutations, dtype=np.uint64)
        self.b = rng.integers(low=0, high=mh.Pm, size=permutations, dtype=np.uint64)
        self.c = rng.integers(low=1, high=mh.Pm, size=permutations, dtype=np.uint64)
        self.z = rng.integers(low=1, high=mh.Pm, size=1, dtype=np.uint64)


    def __call__(self, document: Document):
        # arr: L-shaped array of uint64
        arr = np.array([ord(c) for c in mh.normalize_for_minhash(document.text)], dtype=np.uint64)
        
        # hashes: buckets x bucketsize-shaped array of uint64 minhashes 
        hashes = mh.affine_61(self.c, mh._0, mh.hashit(self.a, self.b, self.z, self.shinglesize, arr))
        buckets = mh.sum_61(hashes.reshape(self.buckets, self.bucketsize))

        return {f'minhash_bucket_{ix}': bhash for ix, bhash in enumerate(buckets.tolist())}


class FilterDocuments(luigi.Task):
    dump = luigi.Parameter()
    # Values for these parameters are set in luigi.cfg
    blacklisted_domains = luigi.ListParameter(visibility=luigi.parameter.ParameterVisibility.HIDDEN)
    lid_model_path = luigi.PathParameter(
        significant=False, 
        visibility=luigi.parameter.ParameterVisibility.HIDDEN
    )

    content_length_filter_range = luigi.TupleParameter(visibility=luigi.parameter.ParameterVisibility.HIDDEN)
    ratio_alphanumeric_chars_filter_range = luigi.TupleParameter(visibility=luigi.parameter.ParameterVisibility.HIDDEN)
    ratio_headings_per_non_heading_word_filter_range = luigi.TupleParameter(visibility=luigi.parameter.ParameterVisibility.HIDDEN)
    unigram_entropy_filter_range = luigi.TupleParameter(visibility=luigi.parameter.ParameterVisibility.HIDDEN)

    @dataclass
    class Shard:
        convert_to_markdown_output_s3_path: str
        process_markdown_output_s3_path: str
        output_s3_path: str

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.features = [
            ContentLength(),
            IsDomainBlacklisted(set(self.blacklisted_domains)),
            RatioAlphanumericChars(),
            RatioHeadingsPerNonHeadingWord(),
            UnigramEntropy(),
            Language(self.lid_model_path),
            MinHashLSH()
        ]

    def requires(self):
        return ProcessMarkdown(dump=self.dump)

    def output(self):
        return [
            S3Target(shard.output_s3_path) for shard in self.get_shards()
        ]

    def get_shards(self) -> List[Shard]:
        """
        Returns list of shards like (input_shard_path, output_shard_path)
        """
        shards = []
        for convert_to_markdown_shard, process_markdown_shard in zip(self.requires().requires().get_shards(), self.requires().get_shards()):
            assert convert_to_markdown_shard.output_s3_path.split('/')[-1] == process_markdown_shard.output_s3_path.split('/')[-1], \
                "The shards must match"
            shards.append(
                FilterDocuments.Shard(
                    convert_to_markdown_output_s3_path=convert_to_markdown_shard.output_s3_path,
                    process_markdown_output_s3_path=process_markdown_shard.output_s3_path,
                    output_s3_path=f"s3://{GLOBAL_CONF.s3_bucket_name}/{GLOBAL_CONF.s3_prefix}/FilterDocuments/{self.dump}/{process_markdown_shard.output_s3_path.split('/')[-1]}"
                )
            )
        return shards
    
    def run(self):
        assert self.lid_model_path.exists(), f"{self.lid_model_path} does not exist!"
        s3 = get_s3fs_client()
        missing_shards = [
            shard for shard in self.get_shards() 
            if not s3.exists(shard.output_s3_path)
        ]

        log_dir = Path(GLOBAL_CONF.data_dir) / "logs" / "FilterDocuments" / self.dump
        log_dir.mkdir(parents=True, exist_ok=True)
        executor = submitit.AutoExecutor(folder=log_dir)
        executor.update_parameters(
            name=f"FilterDocuments_{self.dump}", 
            timeout_min=60*2,
            slurm_partition="main",
            mem_gb=400
        )
        job = executor.submit(self._run, missing_shards)
        logging.info(f"Submitting job {job.job_id} to filter documents for dump {self.dump}")
        job.result()

    def _run(self, shards=None):
        # This method is run from within the slurm job
        if shards is None:
            shards = self.get_shards()

        # Parallelize over shards
        with mp.Pool() as pool:
            n_done = 0
            for _ in pool.imap_unordered(self.process_shard, shards):
                n_done += 1
                self.set_progress_percentage(round(100 * n_done/len(shards), ndigits=1))

    def process_shard(self, shard: Shard):
        logging.info(f"Start processing for output: {shard.output_s3_path}")
        
        markdown_df = pd.read_parquet(shard.convert_to_markdown_output_s3_path, columns=["url"])
        processed_markdown_df = pd.read_parquet(shard.process_markdown_output_s3_path)

        # Verify indecies match, then concat columns
        assert (markdown_df.index == processed_markdown_df.index).all(), "Indices should match"
        df = pd.concat([markdown_df, processed_markdown_df], axis=1)

        # Compute features 
        feature_values = {}
        for row in df.itertuples():
            row_dict = {}
            for feature in self.features:
                val = feature(row)
                row_dict.update(val)
            feature_values[row.Index] = row_dict

        out_df = pd.DataFrame.from_dict(feature_values, orient="index")

        # Filter documents and save
        out_df["passes_all_quality_filters"] = out_df.apply(self.filter, axis=1)
        out_df.to_parquet(shard.output_s3_path)
        
    def filter(self, row) -> bool:
        for feature, value in row.items():
            match feature:
                case "content_length":
                    if not (self.content_length_filter_range[0] <= value <= self.content_length_filter_range[1]):
                        return False
                case "is_domain_blacklisted":
                    if value:
                        return False
                case "ratio_alphanumeric_chars":
                    if not (self.ratio_alphanumeric_chars_filter_range[0] <= value <= self.ratio_alphanumeric_chars_filter_range[1]):
                        return False
                case "ratio_headings_per_non_heading_word":
                    if not (self.ratio_headings_per_non_heading_word_filter_range[0] <= value <= self.ratio_headings_per_non_heading_word_filter_range[1]):
                        return False
                case "unigram_entropy_chars":
                    if not (self.ratio_alphanumeric_chars_filter_range[0] <= value <= self.unigram_entropy_filter_range[1]):
                        return False
        return True
