"""
Step 1: Inspect BeIR/nq and BeIR/nq-qrels
----------------------------------------
Loads BeIR/nq (corpus and queries) and BeIR/nq-qrels using chunked / streaming loading,
prints the exact schema and 3 sample rows for each, and calculates key statistics
(number of passages, unique article titles, and passages-per-title distribution).
"""

from collections import Counter
from datasets import load_dataset
import pandas as pd
import pyarrow.parquet as pq


def main():
    print("=" * 70)
    print("STEP 1: INSPECTING BEIR/NQ AND BEIR/NQ-QRELS")
    print("=" * 70)

    # -------------------------------------------------------------------------
    # 1. Corpus: Stream/batch load via Parquet chunks to maintain low memory
    # -------------------------------------------------------------------------
    print("\n[1] Loading Corpus (BeIR/nq - 'corpus')...")
    # Using streaming=True loads data on demand without filling RAM
    corpus_stream = load_dataset("BeIR/nq", "corpus", split="corpus", streaming=True)
    corpus_iter = iter(corpus_stream)
    
    # Grab first sample to inspect schema
    sample_corpus_1 = next(corpus_iter)
    sample_corpus_2 = next(corpus_iter)
    sample_corpus_3 = next(corpus_iter)
    corpus_samples = [sample_corpus_1, sample_corpus_2, sample_corpus_3]
    corpus_fields = list(sample_corpus_1.keys())

    print(f"Corpus Field Names: {corpus_fields}")
    print("\nCorpus 3 Sample Rows:")
    for idx, sample in enumerate(corpus_samples, 1):
        print(f"  Sample {idx}:")
        print(f"    _id  : {sample['_id']}")
        print(f"    title: {sample['title']}")
        print(f"    text : {sample['text'][:120]}... [length: {len(sample['text'])} chars]")

    # -------------------------------------------------------------------------
    # 2. Queries: Stream load via BeIR/nq - 'queries'
    # -------------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("[2] Loading Queries (BeIR/nq - 'queries')...")
    queries_stream = load_dataset("BeIR/nq", "queries", split="queries", streaming=True)
    queries_iter = iter(queries_stream)
    
    queries_samples = [next(queries_iter) for _ in range(3)]
    queries_fields = list(queries_samples[0].keys())

    print(f"Queries Field Names: {queries_fields}")
    print("\nQueries 3 Sample Rows:")
    for idx, sample in enumerate(queries_samples, 1):
        print(f"  Sample {idx}:")
        print(f"    _id  : {sample['_id']}")
        print(f"    title: {repr(sample['title'])}")
        print(f"    text : {sample['text']}")

    # -------------------------------------------------------------------------
    # 3. Qrels: Load BeIR/nq-qrels (test split)
    # -------------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("[3] Loading Qrels (BeIR/nq-qrels)...")
    qrels_stream = load_dataset("BeIR/nq-qrels", split="test", streaming=True)
    qrels_iter = iter(qrels_stream)
    
    qrels_samples = [next(qrels_iter) for _ in range(3)]
    qrels_fields = list(qrels_samples[0].keys())

    print(f"Qrels Field Names: {qrels_fields}")
    print("\nQrels 3 Sample Rows:")
    for idx, sample in enumerate(qrels_samples, 1):
        print(f"  Sample {idx}:")
        print(f"    query-id : {sample['query-id']}")
        print(f"    corpus-id: {sample['corpus-id']}")
        print(f"    score    : {sample['score']}")

    # -------------------------------------------------------------------------
    # 4. Corpus Statistics: Passage count, unique titles, and distribution
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("CORPUS STATISTICS (Chunked processing)")
    print("=" * 70)
    
    # We locate the cached parquet file to stream/chunk solely the 'title' column
    # This prevents loading passages or text into memory (< 200 MB RAM used).
    from huggingface_hub import hf_hub_download
    parquet_path = hf_hub_download(
        repo_id="BeIR/nq",
        filename="corpus/corpus-00000-of-00001.parquet",
        repo_type="dataset",
    )
    
    pf = pq.ParquetFile(parquet_path)
    total_passages = pf.metadata.num_rows
    
    title_counts = Counter()
    # Read in batches of 100k rows, extracting only the 'title' column
    for batch in pf.iter_batches(batch_size=100000, columns=["title"]):
        title_counts.update(batch["title"].to_pylist())

    unique_titles = len(title_counts)
    counts_series = pd.Series(list(title_counts.values()))

    print(f"Total Passages (rows)       : {total_passages:,}")
    print(f"Unique Titles (articles)    : {unique_titles:,}")
    print(f"Average Passages per Title  : {counts_series.mean():.2f}")
    print(f"Median Passages per Title   : {counts_series.median():.0f}")
    print(f"Min Passages per Title      : {counts_series.min()}")
    print(f"Max Passages per Title      : {counts_series.max()}")
    
    print("\nPercentile Distribution of Passages per Title:")
    percentiles = [0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
    perc_table = counts_series.quantile(percentiles)
    for p, val in zip(percentiles, perc_table):
        print(f"  {int(p*100):2d}th percentile: {int(val):4d} passages")

    # Group into intuitive bucket distribution
    bins = [1, 5, 10, 20, 50, 100, 200, 500, float("inf")]
    labels = ["1-4", "5-9", "10-19", "20-49", "50-99", "100-199", "200-499", "500+"]
    binned = pd.cut(counts_series, bins=bins, labels=labels, right=False).value_counts(sort=False)
    
    print("\nPassage Count Distribution across Articles (Titles):")
    for bucket, count in binned.items():
        pct = (count / unique_titles) * 100
        print(f"  {bucket:10s} passages: {count:6,d} titles ({pct:5.2f}%)")


if __name__ == "__main__":
    main()

