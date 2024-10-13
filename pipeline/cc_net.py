import luigi
import os
import shutil
import logging
import subprocess
from pipeline.utils import GLOBAL_CONF
from pipeline.s3_utils import S3Target, upload_directory_to_s3, get_boto3_s3_client
from pathlib import Path

from cc_net.mine import main as mine
from cc_net.process_wet_file import cc_segments


class DownloadWET(luigi.Task):
    dump = luigi.Parameter()
    resources = {
        "common_crawl_parallel_requests": 1,
        "network_bw": 1,
    }

    def complete(self):
        segments = cc_segments(self.dump, cache_dir=Path(GLOBAL_CONF.scratch_dir) / "crawl-data" / f"CC-MAIN-{self.dump}")
        for segment_file in segments:
            if not (Path(GLOBAL_CONF.scratch_dir) / segment_file).exists():
                logging.info(f"At least {Path(GLOBAL_CONF.scratch_dir) / segment_file} is currently missing")
                return False
        return True

    def run(self):
        logging.info(f"Start downloading wet for dump {self.dump}")
        log_dir = Path(GLOBAL_CONF.data_dir) / "logs" / "DownloadWET" / self.dump
        log_dir.mkdir(parents=True, exist_ok=True)
        subprocess.run([
            "sbatch",
            "-p", "shared",
            "--output", str(log_dir / "%A.out"),
            "--error", str(log_dir / "%A.err"),
            "--cpus-per-task", "40",
            "--time", str(60*10),  # 10h
            "--wait",
            "pipeline/cc_download.sh", f"crawl-data/CC-MAIN-{self.dump}/wet.paths.gz"
        ], check=True)


class CCNet(luigi.Task):
    dump = luigi.Parameter()

    def requires(self):
        return DownloadWET(dump=self.dump)

    def output(self):
        return S3Target(
            f"s3://{GLOBAL_CONF.s3_bucket_name}/{GLOBAL_CONF.s3_prefix}/CCNet/mined/{self.dump}/.success"
        )

    @property
    def local_output_dir(self):
        return Path(GLOBAL_CONF.data_dir) / "CCNet"

    @property
    def local_cache_dir(self):
        return Path(GLOBAL_CONF.scratch_dir)

    def run(self):
        logging.info(f"Start mining of Common Crawl dump: {self.dump}")
        os.environ["SBATCH_PARTITION"] = "main"
        mine(
           config="nordic_pile_v2_dardel",
           mined_dir="mined",
           dump=self.dump,
           output_dir=self.local_output_dir,
           cache_dir=self.local_cache_dir
        )
        del os.environ["SBATCH_PARTITION"] 

        # Upload to s3
        local_mined_dir = self.local_output_dir / "mined" / self.dump
        s3_prefix = Path(GLOBAL_CONF.s3_prefix) / "CCNet" / "mined" / self.dump
        s3_client = get_boto3_s3_client()
        upload_directory_to_s3(
            directory_path=local_mined_dir,
            s3_bucket_name=GLOBAL_CONF.s3_bucket_name,
            s3_prefix=s3_prefix,
            s3_client=s3_client
        )

        # Clear local copy of mined data and line hashes
        shutil.rmtree(local_mined_dir)
        if (local_hashes_dir := self.local_output_dir / "hashes" / self.dump).exists():
            shutil.rmtree(local_hashes_dir)

        # Write success file
        s3_client.put_object(Bucket=GLOBAL_CONF.s3_bucket_name, Key=str(s3_prefix / ".success"), Body=b'')
