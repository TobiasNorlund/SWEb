import luigi
import os
import re
import os
import pandas as pd
import logging
import pyarrow as pa
import ftfy
from collections import OrderedDict
from typing import Optional
from dataclasses import dataclass
from collections import OrderedDict
from typing import Dict, List, Tuple, Union
from pathlib import Path
from dask_jobqueue import SLURMCluster
from dask.distributed import Client, as_completed
from pipeline.warc_processing import ConvertToMarkdown
from pipeline.utils import AssignGpuToWorker, FullNodeAdaptive, GLOBAL_CONF
from pipeline.s3_utils import get_s3fs_client, S3Target
from tqdm import tqdm


# Suppress annoying warning regarding padding
logging.getLogger("transformers.models.longformer.modeling_longformer").setLevel(logging.ERROR)


class ProcessMarkdown(luigi.Task):
    """
    This task does the following:
     1. For each line in the input markdown content, predict whether it should be kept or removed
     2. On the resulting content (the kept lines), normalize using `ftfy`
     3. Extract the markdown image references ![...](...) into a separate dataframe column
     4. Sanitize the image urls, i.e. add potentially missing protocol, fix relative domain-relative urls etc.

    This results in two columns that are saved in the output parquet file:
     · text (string): the extracted markdown content
     · images (map[int, string]): for each image, maps from its position in text (char index) to the santized image url 
    """
    dump = luigi.Parameter()
    line_filter_threshold = luigi.FloatParameter(0.5)
    model_path = luigi.PathParameter(
        visibility=luigi.parameter.ParameterVisibility.HIDDEN
    )

    @dataclass
    class Shard:
        input_s3_path: str
        output_s3_path: str

    def requires(self):
        return ConvertToMarkdown(dump=self.dump)
    
    def output(self):
        return [
            S3Target(shard.output_s3_path)
            for shard in self.get_shards()
        ]
    
    def get_shards(self) -> List[Shard]:
        """
        Returns list of shards like (input_shard_path, output_shard_path)
        """
        return [
            ProcessMarkdown.Shard(
                input_s3_path=input_shard.output_s3_path,
                output_s3_path=f"s3://{GLOBAL_CONF.s3_bucket_name}/{GLOBAL_CONF.s3_prefix}/ProcessMarkdown/{self.dump}/{input_shard.output_s3_path.split('/')[-1]}",
            )
            for input_shard in self.requires().get_shards()
        ]

    def run(self):
        assert self.model_path.exists(), f"{self.model_path} does not exist!"
        
        # Setup dask cluster
        log_dir = Path(GLOBAL_CONF.data_dir) / "logs" / "ProcessMarkdown" / self.dump
        log_dir.mkdir(parents=True, exist_ok=True)
        cluster = SLURMCluster(
            cores=8,
            memory="400GB",
            processes=8,
            account="pdc-bus-2024-1",
            walltime="1-00:00:00",
            queue="gpu",
            job_script_prologue=["rm -f /tmp/counter"],
            scheduler_options={
                "dashboard_address": ":0",
                "port": 0  # port=0 means autoselecting an available one
            },
            log_directory=log_dir
        )

        client = Client(cluster)
        client.register_plugin(AssignGpuToWorker())
        cluster.adapt(minimum_jobs=1, maximum_jobs=10, Adaptive=FullNodeAdaptive, worker_key=lambda ws: ws.host)
        
        s3 = get_s3fs_client()
        missing_shards = [
            shard for shard in self.get_shards() 
            if not s3.exists(shard.output_s3_path)
        ]
        logging.info(f"{len(missing_shards)} missing processed markdown shards for {self.dump}")

        futures = [
            client.submit(self.process_shard, shard)
            for shard in missing_shards
        ]

        # Wait until all finish
        for future in tqdm(as_completed(futures), total=len(futures)):
            # Deleting all future refs is necessary for the task result to clear from worker 
            # memory, and for the task not to be recomputed if/when the worker fails
            del futures[futures.index(future)]

            self.set_progress_percentage(round(100 * (len(missing_shards) - len(futures))/len(missing_shards), ndigits=2))
            
    @staticmethod
    def get_num_tokens(content: str, tokenizer):
        return len(tokenizer.encode(content))

    def process_shard(self, shard: Shard):
        # This runs in a dask worker and has access to a gpu
        
        import torch
        import pytorch_lightning as pl
        from pytorch_lightning.plugins.environments import SLURMEnvironment
        from torch.utils.data import DataLoader
        from transformers import AutoTokenizer, AutoConfig
        from pipeline.line_classification.data_loading import IterableLineClassificationDataset, collator
        from pipeline.line_classification.model import LineClassificationModel
        from pipeline.line_classification.predict import PredictCallback

        torch.set_float32_matmul_precision("medium")

        logging.basicConfig(level=logging.INFO)
        if torch.cuda.is_available():
            logging.info(f"Running inference on GPU {os.environ['CUDA_VISIBLE_DEVICES']} for {shard.input_s3_path}")
        else:
            logging.info("Running inference on CPU")

        # We want to sort all documents by token length for batching with minimal padding
        tokenizer = AutoTokenizer.from_pretrained("severinsimmler/xlm-roberta-longformer-base-16384")
        df = pd.read_parquet(shard.input_s3_path, columns=["url", "content"])
        original_index = df.index.copy()
        df["token_length"] = [len(encoding) for encoding in tokenizer(df["content"].tolist()).encodings]
        df = df.sort_values("token_length")

        config = AutoConfig.from_pretrained("severinsimmler/xlm-roberta-longformer-base-16384")
        config.num_labels = 1
        ds = IterableLineClassificationDataset(df["content"].tolist(), tokenizer)
        dl = DataLoader(ds, batch_size=16, collate_fn=collator, num_workers=1)
        logging.info(f"Loaded {len(df)} examples in this shard")

        logging.info(f"Loading model")
        model = LineClassificationModel.load_from_checkpoint(
            self.model_path,
            config=config,
            map_location=torch.device("cpu")
        )

        callback = PredictCallback()
        trainer = pl.Trainer(
            accelerator="gpu" if torch.cuda.is_available() else "cpu",
            callbacks=[callback],
            precision="bf16-mixed",
            enable_progress_bar=False,
            logger=False,
            plugins=[SLURMEnvironment(auto_requeue=False)]  # Disables auto re-queuing
        )

        # Run inference
        trainer.predict(model, dataloaders=dl)
        assert len(df) == len(callback.predictions), \
            "The number of documents and predictions didn't match"

        # Filter markdown lines based on predictions, split text/images and sanitize image urls
        logging.info(f"Filtering lines using threshold={self.line_filter_threshold}")
        text_col = {}
        for i in range(len(df)):
            id = df.index[i]
            content = df["content"][i]
            url = df["url"][i]

            # Make sure we have one prediction for each newline
            line_logits = callback.predictions[i]
            num_newlines = content.count('\n')
            assert line_logits.shape[0] == num_newlines, \
                f"Number of lines and and predictions didn't match for ex {i} ({line_logits.shape[0]} != {num_newlines})"
            
            line_predictions = torch.nn.functional.sigmoid(line_logits).tolist()
            filtered_content = "".join([line for line, pred in zip(content.splitlines(keepends=True), line_predictions)
                                        if pred >= self.line_filter_threshold])
            
            # Normalize text
            text = ftfy.fix_text(filtered_content)
            
            # Split text and images
            text, _ = split_text_and_images(text)

            text_col[id] = text

        # Load only the index (in the original order)
        output_df = pd.DataFrame(index=original_index)
        output_df["text"] = pd.Series(text_col)
        
        schema = pa.schema([
            ('id', pa.string()),
            ('text', pa.string()),
        ])
        output_df.to_parquet(shard.output_s3_path, schema=schema)
        logging.info(f"Wrote {shard.output_s3_path}")


def split_text_and_images(
    markdown: str,
    image_match_regex: str = "!\[(.*?)\]\((.*?)\)",
) -> Tuple[str, OrderedDict]:
    """Splits a document in markdown format to a text & image component.

    Image links are removed from the text and their character positions are calculated
    as they relate to their insert positions in the final cleaned text.

    The images are stored in a dict in the format {char_pos_int: "url as string"}

    Args:
        markdown (str): The markdown string to do the splitting on.
        image_match_regex (str, optional): Image matching regex. Defaults to
            "!\[(.*?)\]\((.*?)\)".

    Returns:
        Tuple[str, OrderedDict]: text, images
    """

    images = OrderedDict()
    # Initialize offset to track cumulative length of removed segments
    offset = 0

    def repl(match):
        # Use nonlocal to modify offset variable from the enclosing scope
        nonlocal offset
        # Adjust start position by offset
        image_start = match.start() - offset
        image_url = match.group(2)
        images[image_start] = image_url
        # Update offset by the length of the matched segment
        offset += match.end() - match.start()
        return ""

    text = re.sub(
        pattern=image_match_regex,
        repl=repl,
        string=markdown,
    )

    return text, images


def insert_images(
    text: str,
    images: Union[Dict[str, str], OrderedDict[int, str]],
) -> str:
    """Inserts images back into the text at their original positions.

    Args:
        text (str): The cleaned text without images.
        images (Union[Dict[str, str], OrderedDict[int, str]]): A dictionary
            containing image positions and URLs.

    Returns:
        str: The text with images inserted.
    """
    if isinstance(images, OrderedDict):
        # Confirm if all keys are integers
        if not all(isinstance(key, int) for key in images.keys()):
            raise ValueError(
                "Keys in `images` must be integers when passing in an OrderedDict."
            )
    elif isinstance(images, dict):
        images = OrderedDict((int(key), value) for key, value in images.items())
    else:
        raise ValueError("`images` must be an OrderedDict or Dict.")

    # Reconstruct the original text by inserting images at their original positions
    reconstructed_text = ""
    current_position = 0
    for position, url in images.items():
        # Insert the text from the current position to the image position
        reconstructed_text += text[current_position:position]
        # Insert the image markdown syntax
        reconstructed_text += f"![]({url})"
        # Update the current position
        current_position = position
    # Insert the remaining text after the last image
    reconstructed_text += text[current_position:]

    return reconstructed_text



if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    task = ProcessMarkdown(dump="2023-40")
    task.run()
    #task.filter_markdown(Path("/cfs/klemming/home/t/tnorlund/data/common-crawl/markdown-documents/2023-40/0000.json.gz"))