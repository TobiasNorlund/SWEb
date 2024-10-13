from pathlib import Path
import pandas as pd
import multiprocessing as mp
from datatrove.pipeline.formatters import PIIFormatter
from pipeline.utils import ALL_CC_DUMPS
from tqdm import tqdm


def convert_shard(args):
    base_path, dump, shard = args
    # Loads:
    # - url, warc_file, warc_date from ConverToMarkdown stage
    # - text from ProcessMarkdown stage
    # - language, passes_all_quality_filters from FilterDocuments
    # - dedup_keep from Deduplicate step

    convert_to_markdown_df = pd.read_parquet(base_path / "ConvertToMarkdown" / dump / f"{shard:04d}.parquet", columns=["url", "warc_file", "warc_date"])
    process_markdown_df = pd.read_parquet(base_path / "ProcessMarkdown" / dump / f"{shard:04d}.parquet", columns=["text"])
    filter_documents_df = pd.read_parquet(base_path / "FilterDocuments" / dump / f"{shard:04d}.parquet", columns=["language", "passes_all_quality_filters"])
    deduplicate_df = pd.read_parquet(base_path / "Deduplicate" / f"dumps={dump}" / dump / f"{shard:04d}.parquet", columns=["dedup_keep"])

    assert all(convert_to_markdown_df.index == process_markdown_df.index)
    assert all(convert_to_markdown_df.index == filter_documents_df.index)
    assert all(convert_to_markdown_df.index == deduplicate_df.index)

    df = pd.concat([convert_to_markdown_df, process_markdown_df, filter_documents_df, deduplicate_df], axis=1)

    # Only keep filtered and deduped examples
    df = df[(df["passes_all_quality_filters"] == True) & (df["dedup_keep"] == True)]
    df.drop(columns=["dedup_keep", "passes_all_quality_filters"], inplace=True)

    # PII filter text
    pii_formatter = PIIFormatter(
        remove_emails=True,
        remove_ips=True,
        only_remove_public_ips=True
    )
    df["text"] = df["text"].apply(pii_formatter.format)

    # Save to base_path / SWEb
    (base_path / "SWEb" / dump).mkdir(parents=True, exist_ok=True)
    df.to_parquet(base_path / "SWEb" / dump / f"{shard:04d}.parquet", compression="gzip")


def main(args):

    with mp.Pool() as p:
        all_dumps_and_shards = [(args.base_path, dump, shard) for dump in ALL_CC_DUMPS for shard in range(1000)]
        progress_bar = tqdm(range(len(all_dumps_and_shards)))
        for _ in p.imap_unordered(convert_shard, all_dumps_and_shards):
            progress_bar.update()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-path", default="/mnt/NFS/nordic_pile_v2/common-crawl/", type=Path)
    args = parser.parse_args()

    main(args)