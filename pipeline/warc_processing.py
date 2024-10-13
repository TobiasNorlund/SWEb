import charset_normalizer
import luigi
import logging
import shutil
import requests
import gzip
import re
import io
import uuid
import json
import os
import subprocess
import multiprocessing as mp
import submitit
import pypandoc
import pandas as pd
from typing import Optional
from dataclasses import dataclass
from urllib.parse import urlparse
from io import BytesIO
from typing import List, Tuple
from bs4 import BeautifulSoup
from collections import defaultdict
from time import time, sleep
from datetime import timedelta
from itertools import chain, takewhile
from warcio import ArchiveIterator
from urllib3.util import Retry
from requests.adapters import HTTPAdapter
from requests.exceptions import ChunkedEncodingError
from urllib3.exceptions import ProtocolError
from pathlib import Path
from pipeline.s3_utils import S3Target, get_boto3_s3_client
from pipeline.utils import GLOBAL_CONF, group
from pipeline.cc_net import CCNet



class CCIndex(luigi.Task):
    dump = luigi.Parameter()

    resources = {
        "common_crawl_parallel_requests": 1,
        "network_bw": 1,
    }

    def run(self):
        # Download all index files to scratch storage
        logging.info(f"Starting to download index for {self.dump}")
        cc_index_dir = Path(GLOBAL_CONF.scratch_dir) /  "cc-index" / "collections" / ("CC-MAIN-" + self.dump)
        log_dir = Path(GLOBAL_CONF.data_dir) / "logs" / "CCIndex" / self.dump
        cc_index_dir.mkdir(parents=True, exist_ok=True)
        log_dir.mkdir(parents=True, exist_ok=True)
        subprocess.run([
            "sbatch", 
                "-p", "shared",
                "--output", str(log_dir / "%A.out"),
                "--error", str(log_dir / "%A.err"),
                "--cpus-per-task", "40",
                "--time", str(60*10),  # 10h
                "--wait",
                "pipeline/cc_download.sh", f"crawl-data/CC-MAIN-{self.dump}/cc-index.paths.gz"
        ], check=True)
        logging.info(f"Finished downloading index for {self.dump}")
    
    def complete(self):
        # Check whether all index files exist
        paths_file = Path(GLOBAL_CONF.scratch_dir) / "crawl-data" / f"CC-MAIN-{self.dump}" / "cc-index.paths.gz"
        if paths_file.exists():
            for file in gzip.open(paths_file, "rt").readlines():
                if not Path(GLOBAL_CONF.scratch_dir, file.strip()).exists():
                    return False
        else:
            return False
        return True

class WARCIndex(luigi.Task):
    dump = luigi.Parameter()

    def requires(self):
        return [CCIndex(dump=self.dump), CCNet(dump=self.dump)]
    
    def output(self):
        return S3Target(
            f"s3://{GLOBAL_CONF.s3_bucket_name}/{GLOBAL_CONF.s3_prefix}/WARCIndex/{self.dump}/index.json.gz"
        )
    
    def run(self):
        log_dir = Path(GLOBAL_CONF.data_dir) / "logs" / "WARCIndex" / self.dump
        log_dir.mkdir(parents=True, exist_ok=True)
        executor = submitit.AutoExecutor(folder=log_dir)
        executor.update_parameters(name=f"WARCIndex_{self.dump}", timeout_min=60, slurm_partition="main", mem_gb=400)
        job = executor.submit(self._run)
        logging.info(f"Submitting job {job.job_id}")
        job.result()
    
    @property
    def cc_index_dir(self) -> Path:
        # Output of CCIndex task
        return Path(GLOBAL_CONF.scratch_dir) /  "cc-index" / "collections" / ("CC-MAIN-" + self.dump)

    def _run(self):
        # This method is run from within the slurm job

        # Collect all URLs from CCNet output
        global urls  # Set to global for workers in _process_index_file to have access to it without pickling
        logging.info(f"Collecting all urls in dump {self.dump}")
        s3_client = get_boto3_s3_client()
        mined_file_prefixes = [
            obj["Key"]
            for obj in s3_client.list_objects(
                Bucket=GLOBAL_CONF.s3_bucket_name, Prefix=f"{GLOBAL_CONF.s3_prefix}/CCNet/mined/{self.dump}/"
            )["Contents"]
            if obj["Key"].endswith(".json.gz")
        ]
        with mp.Pool(5) as pool: # Limit number of workers for data to fit in memory
            urls = pool.map(self._get_urls, mined_file_prefixes)
        urls = set(chain.from_iterable(urls))
        logging.info(f"Found {len(urls)} urls")

        # Extract warc offset+length for each url from index, and group by warc file
        logging.info("Searching WARC index for these urls")
        index_files = self.cc_index_dir.glob("**/cdx-*.gz")
        
        with mp.Pool() as pool:
            res = pool.map(self._process_index_file, index_files)
            logging.info("Finished processing all index files")
            
            dicts, num_urls = zip(*res)
            num_urls = sum(num_urls)
            index_dict = defaultdict(list)
            for d in dicts:
                for key, value in d.items():
                    index_dict[key].extend(value)
            logging.info(f"Found {num_urls} urls in warc index")

            # Write to gzipped json to s3
            buffer = BytesIO()
            with gzip.GzipFile(fileobj=buffer, mode="w") as gz_file:
                gz_file.write(json.dumps(index_dict, ensure_ascii=False).encode("utf-8"))
            
            s3_client.put_object(
                Bucket=GLOBAL_CONF.s3_bucket_name, 
                Key=f"{GLOBAL_CONF.s3_prefix}/WARCIndex/{self.dump}/index.json.gz", 
                Body=buffer.getvalue(),
                ContentType="application/json", 
                ContentEncoding="gzip"
            )

            logging.info(f"Wrote s3://{GLOBAL_CONF.s3_bucket_name}/{GLOBAL_CONF.s3_prefix}/WARCIndex/{self.dump}/index.json.gz")

    @staticmethod
    def _get_urls(file_key):
        urls = set()
        s3_client = get_boto3_s3_client()
        response = s3_client.get_object(Bucket=GLOBAL_CONF.s3_bucket_name, Key=file_key)
        json_lines_file = io.StringIO(gzip.decompress(response["Body"].read()).decode("utf-8"))
        try:
            for line_no, line in enumerate(json_lines_file):
                parsed = json.loads(line)
                urls.add(parsed["url"])
            return urls
        except Exception as e:
            logging.error(f"Error in parsing {str(file_key)} line {line_no}", exc_info=e)
            raise e
    
    @staticmethod
    def _process_index_file(index_file_path):
        global urls
        def find_start(line):
            cnt = 0
            for i, c in enumerate(line):
                if c == " ":
                    if cnt == 1:
                        return i+1
                    else:
                        cnt += 1

        d = defaultdict(list)
        num_urls = 0
        with gzip.open(index_file_path) as f:
            logging.info(f"Opened {str(index_file_path)}")
            for line in f:
                line = line.decode()
                parsed = json.loads(line[find_start(line):])
                if parsed["url"] in urls:
                    d[parsed["filename"]].append((int(parsed["offset"]), int(parsed["length"])))
                    num_urls += 1
            logging.info(f"Finished {str(index_file_path)} with {num_urls} found urls")
        return d, num_urls


class LoggingRetry(Retry):
    def increment(self, *args, **kwargs):
        # Log the retry attempt
        consecutive_errors_len = len(
            list(
                takewhile(lambda x: x.redirect_location is None, reversed(self.history))
            )
        )
        if consecutive_errors_len > 5:
            logging.warning(f"Retrying attempt: {consecutive_errors_len}")
        return super().increment(*args, **kwargs)


class FetchHTMLDocuments(luigi.Task):
    dump = luigi.Parameter()

    priority = 10  # Set high priority as this is a bottleneck task that we always want running
    resources = {"common_crawl_parallel_requests": 1}

    def requires(self):
        return WARCIndex(dump=self.dump)
    
    def output(self):
        return luigi.LocalTarget(
            self.output_dir / "crawl-data" / ("CC-MAIN-" + self.dump) / ".warc_success"
        )
    
    @property
    def output_dir(self) -> Path:
        return Path(GLOBAL_CONF.scratch_dir)

    def run(self):
        log_dir = Path(GLOBAL_CONF.data_dir) / "logs" / "FetchHTMLDocuments" / self.dump
        log_dir.mkdir(parents=True, exist_ok=True)
        executor = submitit.AutoExecutor(folder=log_dir)
        executor.update_parameters(
            name=f"FetchHTMLDocuments_{self.dump}", 
            timeout_min=60*24, 
            slurm_partition="shared", 
            cpus_per_task=16,
            mem_gb=64
        )
        job = executor.submit(self._run)
        logging.info(f"Submitting job {job.job_id} to fetch HTML documents for dump {self.dump}")
        job.result()
    
    def _run(self):
        # This method is run from within the slurm job

        # Using the index, collect the HTML using multi-range http requests
        logging.info(f"Loading WARC index for {self.dump}")
        s3 = get_boto3_s3_client()
        key = urlparse(self.requires().output().path).path.lstrip('/')
        response = s3.get_object(Bucket=GLOBAL_CONF.s3_bucket_name, Key=key)
        warc_index = json.loads(gzip.decompress(response["Body"].read()).decode("utf-8"))
        logging.info(f"WARC index successfully loaded")
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        # Using 200 parallel downloads seems to not result in many "slow down" responses
        with mp.Pool(processes=200, initializer=self._worker_init) as pool:
            logging.info(f"Started parallel pool of {pool._processes} workers")
            t = time()
            speed = 0
            sleep(1)  # To avoid division by zero when computing records_per_second below
            for i, n_docs in enumerate(pool.imap_unordered(self._download_warc_multirange, warc_index.items())):
                if i % 100 == 0 and i > 0:
                    speed = 0.5 * speed + 0.5 * 100 / (time() - t)  # estimated records downloaded per second
                    eta = timedelta(seconds=(len(warc_index)-i) / speed)
                    logging.info(f"Finished {i}/{len(warc_index)} ({i*100/len(warc_index):.2f}%) warc records. Time remaining: {str(eta)}")
                    self.set_progress_percentage(round(100 * i / len(warc_index), ndigits=2))
                    t = time()

        # Write success file
        (Path(self.output_dir) / "crawl-data" / ("CC-MAIN-" + self.dump) / ".warc_success").touch()
    
    def _download_warc_multirange(self, args):
        global session
        warc_file, ranges = args

        # deduplicate and sort ranges by offset
        ranges = sorted(set(tuple(x) for x in ranges), key=lambda x: x[0])

        output_file = self.output_dir / warc_file
        tmp_output_file = output_file.with_suffix(".tmp")

        if output_file.exists():
            return 0

        def _try_decompress(body):
            try:
                return gzip.decompress(body)
            except gzip.BadGzipFile as e:
                logging.warning(f"Document not a gzipped file, in {warc_file}")
            except EOFError as e:
                logging.warning(f"Compressed file ended before the end-of-stream marker was reached, in {warc_file}")
            return None

        output_file.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(tmp_output_file, 'wb') as f:
            num_docs_downloaded = 0
            max_attempts = 10
            for range_group in group(ranges, group_size=100):
                attempts = 0
                while True:
                    try:
                        response = session.get(
                            "https://data.commoncrawl.org/" + warc_file, 
                            headers={
                                "Range": f"bytes={', '.join(f'{offset}-{offset+length-1}' for offset, length in range_group)}"
                            },
                            timeout=60.0*5,
                        )
                        break
                    except (ProtocolError, ChunkedEncodingError, requests.exceptions.RequestException) as e:
                        attempts += 1
                        if attempts < max_attempts:
                            logging.warning(f"Attempt {attempts} failed with connection error: {e}")
                        else:
                            logging.error(f"Failed request after {attempts} attempts for {warc_file}. Skipping these ranges...")
                            response = None
                            break
                
                if response is None:
                    continue
                elif response.status_code == 206:  # Partial content; multiple ranges might be included
                    content_type = response.headers['Content-Type']
                    # Check if it's a multipart response
                    if 'multipart/byteranges' in content_type:
                        # Extract boundary
                        boundary = content_type.split("boundary=")[-1]
                        parts = response.content.split(f'--{boundary}'.encode())
                        for part in parts:
                            if part and (b"Content-Range" in part):
                                # Split headers from the body
                                headers, body = part.split(b'\r\n\r\n', 1)
                                if uncompressed := _try_decompress(body.strip(b"\r\n")):
                                    f.write(uncompressed)
                                    num_docs_downloaded += 1
                    elif content_type == "application/octet-stream" and len(range_group) == 1:
                        if uncompressed := _try_decompress(response.content.strip(b"\r\n")):
                            f.write(uncompressed)
                            num_docs_downloaded += 1
                    else:
                        logging.warning(f"Unexpected content type '{content_type}' for ranges '{range_group}' in {warc_file}")
                else:
                    logging.warning(f"Got status code: {response.status_code}")
        
        if num_docs_downloaded > 0 and os.path.exists(tmp_output_file):
            os.rename(tmp_output_file, output_file)
            logging.debug(f"Finished downloading {num_docs_downloaded}/{len(ranges)} documents from {warc_file}")
        else:
            if os.path.exists(tmp_output_file):
                os.remove(tmp_output_file)
            logging.warning(f"Couldn't download any documents from {warc_file}")
        
        return num_docs_downloaded

    @staticmethod
    def _worker_init():
        # Each worker has a session to reuse underlying connection between requests
        global session
        retry_strategy = LoggingRetry(
            total=100,  # Maximum number of retries
            read=100,
            connect=100,
            backoff_factor=1.0,
            raise_on_status=False,
            status_forcelist=[500, 503, 403, 429]
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session = requests.Session()
        session.mount('http://', adapter)
        session.mount('https://', adapter)


class ConvertToMarkdown(luigi.Task):
    dump = luigi.Parameter()
    copy_pandoc_to_shm = luigi.BoolParameter(
        significant=False,
        visibility=luigi.parameter.ParameterVisibility.HIDDEN,
    )
    num_shards = luigi.IntParameter(
        significant=False, 
        visibility=luigi.parameter.ParameterVisibility.HIDDEN
    )
    parallel_nodes = luigi.IntParameter(
        significant=False, 
        visibility=luigi.parameter.ParameterVisibility.HIDDEN
    )
    parallel_processes = luigi.IntParameter(
        significant=False, 
        visibility=luigi.parameter.ParameterVisibility.HIDDEN
    )

    @dataclass
    class Shard:
        input_warc_file_paths: List[Path]
        output_s3_path: str

    def requires(self):
        return FetchHTMLDocuments(dump=self.dump)
    
    def output(self) -> List[S3Target]:
        return [
            S3Target(
                f"s3://{GLOBAL_CONF.s3_bucket_name}/{GLOBAL_CONF.s3_prefix}/ConvertToMarkdown/{self.dump}/{shard_idx:04d}.parquet",
            )
            for shard_idx in range(self.num_shards)
        ]
    
    def run(self):
        # Convert the HTML to markdown using Pandoc
        pypandoc.ensure_pandoc_installed("2.9.2.1")

        missing_shards = [
            shard for shard, target in zip(self.get_shards(), self.output())
            if not target.exists()
        ]
        logging.info(f"Missing {len(missing_shards)} markdown shards for {self.dump}")

        # We group shards together and process them in parallel on each node
        parallel_groups = [shard_group for shard_group in group(missing_shards, num_groups=self.parallel_nodes)]

        log_dir = Path(GLOBAL_CONF.data_dir) / "logs" / "ConvertToMarkdown" / self.dump
        log_dir.mkdir(parents=True, exist_ok=True)
        executor = submitit.AutoExecutor(folder=log_dir)
        executor.update_parameters(
            name=f"ConvertToMarkdown_{self.dump}", 
            timeout_min=60*24, 
            slurm_partition="main"
        )
        jobs = executor.map_array(self._run, parallel_groups)
        logging.info(f"Submitting job array to convert html to markdown for dump {self.dump}")
        
        # Wait until all done
        exceptions = []
        for job in jobs:
            try:
                _ = job.result()
            except (submitit.core.utils.UncompletedJobError, submitit.core.utils.FailedJobError) as e:
                logging.error(f"Job {job.job_id} failed:", exc_info=e)
                exceptions.append(e)
        if exceptions:
            raise Exception(exceptions)

    def _run(self, shards):
        # This method is run from within the slurm job
        
        # Hack: Copy pandoc binary to shm for great speedup on Dardel
        if self.copy_pandoc_to_shm:
            pandoc_path = shutil.which("pandoc")
            shutil.copy2(pandoc_path, "/dev/shm/pandoc")
        
        # Process each shard in sequence
        for shard in shards:
            self.process_shard(shard)

    def get_shards(self) -> List[Shard]:
        assert self.requires().complete(), "get_shards requires the dependency to be completed"
        input_warc_files = list(Path(self.requires().output_dir / "crawl-data" / ("CC-MAIN-" + self.dump)).glob("**/*.warc.gz"))
        shards = [
            ConvertToMarkdown.Shard(
                input_warc_file_paths=warc_files,
                output_s3_path=s3_target.path,
            )
            for warc_files, s3_target in zip(group(input_warc_files, num_groups=self.num_shards), self.output())
        ]
        return shards

    @property
    def pandoc_path(self):
        if self.copy_pandoc_to_shm:
            return "/dev/shm/pandoc"
        else:
            return "pandoc"
    
    def process_shard(self, shard: Shard):
        # Process shard in parallel
        logging.info(f"Starting markdown conversion for {shard.output_s3_path}")

        input_queue = mp.JoinableQueue(maxsize=10_000)
        output_queue = mp.Queue()

        def consumer(input_queue: mp.JoinableQueue, output_queue: mp.Queue):
            while True:
                record = input_queue.get()
                if record is None:
                    input_queue.task_done()
                    # Send sentinel to output queue to signal done
                    output_queue.put(None)
                    break

                try:
                    match = charset_normalizer.from_bytes(record["payload"]).best()
                    encoding = match.encoding if match else "utf-8"
                    html = record["payload"].decode(encoding, errors="ignore")
                    markdown = self.convert_html_to_markdown(html, pandoc_path=self.pandoc_path)
                    output_queue.put({
                       "id": record["id"],
                       "url": record["url"],
                       "warc_file": record["warc_file"],
                       "warc_date": record["warc_date"],
                       "warc_block_digest": record["warc_block_digest"],
                       "content": markdown
                    })                    
                except Exception as e:
                    logging.warning(f"Error converting record in {record['warc_file']}", exc_info=e)
                finally:
                    input_queue.task_done()

        consumers = [mp.Process(target=consumer, args=(input_queue, output_queue)) 
                     for _ in range(self.parallel_processes)]
        for c in consumers:
            c.start()

        for warc_file in shard.input_warc_file_paths:
            warc_file_suffix = re.search(r'(crawl-data/.*)', str(warc_file)).group(1) \
                if "crawl-data" in str(warc_file) else str(warc_file)
            with gzip.open(warc_file, "rb") as stream:
                for record in ArchiveIterator(stream):
                    url = record.rec_headers["WARC-Target-URI"]
                    warc_date = record.rec_headers["WARC-Date"]
                    warc_block_digest = record.rec_headers["WARC-Block-Digest"]
                    payload = record.content_stream().read()
                    input_queue.put({
                        "id": str(uuid.uuid4()),
                        "url": url,
                        "warc_file": warc_file_suffix,
                        "warc_date": warc_date,
                        "warc_block_digest": warc_block_digest,
                        "payload": payload
                    })

        # Put a sentinel None for each consumer into the queue to signal them to stop
        for _ in range(self.parallel_processes):
            input_queue.put(None)

        # Wait for all consumers to finish working off the queue
        input_queue.join()
        logging.info(f"Finished converting, writing {shard.output_s3_path}")
        
        # Read output_queue records into dataframe and save
        data = []
        sentinel_counts = 0
        while sentinel_counts < self.parallel_processes:
            record = output_queue.get()
            if record is None:
                sentinel_counts += 1
            else:
                data.append(record)
        df = pd.DataFrame.from_records(data)
        df.set_index("id", inplace=True)
        df.to_parquet(shard.output_s3_path, compression="gzip")
        logging.info(f"Done writing {shard.output_s3_path}")

        # Join consumers now when output_queue is consumed
        for c in consumers:
            c.join()

        logging.info(f"Consumers quit. Shard done.")

    @staticmethod
    def convert_html_to_markdown(html, pandoc_path="pandoc"):
        soup = BeautifulSoup(html, 'html.parser')

        # Remove script tags
        for script_tag in soup.find_all('script'):
            script_tag.decompose()  

        # Run pandoc
        res = subprocess.run([pandoc_path, "--from", "html", "--to", "gfm-raw_html", 
                              "--atx-headers", "--wrap=none", 
                              f"--lua-filter={str(Path(__file__).parent / 'pandoc-filters.lua')}"],
                              input=str(soup).encode(errors="ignore"),
                              stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE,
                              timeout=60
        )
        
        # check that pandoc returned successfully
        if res.returncode != 0:
            raise RuntimeError(
                'Pandoc died with exitcode "%s" during conversion: %s' % (res.returncode, res.stderr)
            )
        
        # if there is output on stderr, process it and send to logger
        if res.stderr:
            for level, msg in pypandoc._classify_pandoc_logging(res.stderr if type(res.stderr) is str else res.stderr.decode()):
                if level >= logging.ERROR:
                    logging.log(level, msg)
        text = res.stdout.decode()
        return text
