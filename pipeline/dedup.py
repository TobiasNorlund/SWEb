import luigi
import os
import re
import pandas as pd
import dask.dataframe as dd
from dask import delayed
from dask_jobqueue import SLURMCluster
from pathlib import Path
from pipeline.utils import ALL_CC_DUMPS, GLOBAL_CONF
from pipeline.s3_utils import S3Target, get_boto3_s3_client
from pipeline.quality_filtering import FilterDocuments
from dask.distributed import Client, progress


class Deduplicate(luigi.Task):
    dumps = luigi.Parameter(default="all")

    def requires(self):
        return [
            FilterDocuments(dump=dump)
            for dump in self._dumps
        ]
    
    def output(self):
        return S3Target(f"{self._output_path_prefix}/.success")
    
    @property
    def _dumps(self):
        return [
            dump for dump in (ALL_CC_DUMPS if self.dumps == "all" else self.dumps.split(","))
        ]
    
    @property
    def _output_path_prefix(self):
        return f"s3://{GLOBAL_CONF.s3_bucket_name}/{GLOBAL_CONF.s3_prefix}/Deduplicate/dumps={self.dumps}"

    def run(self):
        # Setup dask cluster
        log_dir = Path(GLOBAL_CONF.data_dir) / "logs" / "Deduplicate" / f"dumps={self.dumps}"
        log_dir.mkdir(parents=True, exist_ok=True)

        # Note: Dardel specific settings
        cluster = SLURMCluster(
            cores=256,
            memory="200GB",
            worker_extra_args=["--memory-limit", "10GB"],
            processes=256,
            account=os.environ["SBATCH_ACCOUNT"],
            walltime="01:00:00",
            queue="main",
            scheduler_options={
                "dashboard_address": ":0",
                "port": 0  # port=0 means autoselecting an available one
            },
            log_directory=log_dir
        )
        
        client = Client(cluster)

        # Note: A single node might run out of memory when running on more than 4 or 5 dumps
        cluster.scale(1)
        
        def load_minhash_shard(shard_path):
            df = pd.read_parquet(shard_path, columns=[f"minhash_bucket_{i}" for i in range(14)])
            df["shard"] = shard_path
            return df

        # Load each shard and add "shard" column
        features_df = dd.from_delayed([
            delayed(load_minhash_shard)(shard.output_s3_path)
            for filter_documents in self.requires()
            for shard in filter_documents.get_shards()
        ])
        
        # Repartition to fewer shards for speedup
        # TODO: Increase number of partitions if we increase number of workers/nodes
        features_df = features_df.repartition(npartitions=256)
        
        # Deduplicate by dropping duplicates for each bucket column
        deduped_df = features_df
        for bix in range(14):
            deduped_df = deduped_df.drop_duplicates(f"minhash_bucket_{bix}", split_out=features_df.npartitions)

        # Drop bucket columns to save memory as they are not needed anymore
        features_df = features_df.drop([f"minhash_bucket_{bix}" for bix in range(14)], axis=1)

        # Add "dedup_keep" indicator column
        features_df["dedup_keep"] = features_df.merge(deduped_df, how='left', left_index=True, right_index=True, indicator='dedup_keep')["dedup_keep"] == "both"

        def write_output_shard(partition):
            orig_shard_path = partition.iloc[0].shard
            dump, shard_idx = re.search(r"([0-9]{4}-[0-9]{2})\/([^\/]+)\.parquet", orig_shard_path).groups()
            # Load original index
            out_df = pd.read_parquet(orig_shard_path, columns=[])
            # Join in the dedup_keep column, to keep the original index for this shard
            out_df = out_df.merge(partition[["dedup_keep"]], how="left", left_index=True, right_index=True)
            out_df.to_parquet(f"{self._output_path_prefix}/{dump}/{shard_idx}.parquet")

        # Group by shard and write output. Persist to not block, for progress bar to work
        write_output_op = features_df.groupby("shard").apply(write_output_shard, meta={}).persist()

        # Track progress to stdout
        progress(write_output_op)

        # We also run compute to raise potential errors
        write_output_op.compute()

        # Touch the output file
        s3_client = get_boto3_s3_client()
        s3_client.put_object(
           Bucket=GLOBAL_CONF.s3_bucket_name,
           Key=self.output().path.replace(f"s3://{GLOBAL_CONF.s3_bucket_name}/", ""),
        )

        client.close()


class DeduplicateIndividually(luigi.WrapperTask):
    dumps = luigi.Parameter(default="all")

    def requires(self):
        return [
            Deduplicate(dumps=dump)
            for dump in self._dumps
        ]

    @property
    def _dumps(self):
        return [
            dump for dump in (ALL_CC_DUMPS if self.dumps == "all" else self.dumps.split(","))
        ]
