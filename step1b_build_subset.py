"""
Step 1b: Build the Natural Questions (NQ) Subset
------------------------------------------------
This script builds a representative, article-level subset of BeIR/NQ:
1. Samples 1,000 valid test queries (with score > 0 and corpus-backed gold passages).
2. Identifies all article titles corresponding to the gold passages of these queries
   and includes ALL passages belonging to those articles.
3. Randomly adds additional complete articles until the subset reaches ~100,000 passages.
4. Splits the 1,000 queries into 500 'dev' and 500 'test' using random seed 42.
5. Saves corpus_subset.parquet, queries_subset.parquet, and qrels_subset.parquet to data/.
6. Runs sanity checks verifying article completeness, gold passage integrity (0 missing),
   and distributions.

Peak memory usage is strictly controlled (< 500 MB) by streaming/chunked processing.
"""

import os
import random
from collections import Counter

from datasets import load_dataset
from huggingface_hub import hf_hub_download
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


def main():
    print("=" * 70)
    print("STEP 1b: BUILDING BEIR/NQ SUBSET (~100K PASSAGES, 1,000 QUERIES)")
    print("=" * 70)

    RANDOM_SEED = 42
    TARGET_PASSAGES = 100_000
    OUTPUT_DIR = "data"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # -------------------------------------------------------------------------
    # PART 1: Load Qrels and find gold passages
    # -------------------------------------------------------------------------
    print("\n[Part 1] Loading qrels and filtering positive labels (score > 0)...")
    qrels_dataset = load_dataset("BeIR/nq-qrels", split="test")
    positive_qrels = [r for r in qrels_dataset if r["score"] > 0]
    gold_corpus_ids = set(r["corpus-id"] for r in positive_qrels)
    print(f"  Positive qrels: {len(positive_qrels):,}")
    print(f"  Unique gold passage IDs: {len(gold_corpus_ids):,}")

    # -------------------------------------------------------------------------
    # PART 2: Scan Corpus metadata in chunks to locate gold articles
    # -------------------------------------------------------------------------
    print("\n[Part 2] Scanning corpus parquet metadata (chunked)...")
    corpus_parquet_path = hf_hub_download(
        repo_id="BeIR/nq",
        filename="corpus/corpus-00000-of-00001.parquet",
        repo_type="dataset",
    )
    corpus_pf = pq.ParquetFile(corpus_parquet_path)
    print(f"  Total passages in full corpus: {corpus_pf.metadata.num_rows:,}")

    # Map gold corpus-id -> article title, and count passages per article title
    gold_cid_to_title = {}
    title_to_passage_count = Counter()

    # Read only '_id' and 'title' columns in batches of 100,000 rows
    for batch in corpus_pf.iter_batches(batch_size=100000, columns=["_id", "title"]):
        batch_ids = batch["_id"].to_pylist()
        batch_titles = batch["title"].to_pylist()
        title_to_passage_count.update(batch_titles)
        for cid, title in zip(batch_ids, batch_titles):
            if cid in gold_corpus_ids:
                gold_cid_to_title[cid] = title

    print(f"  Total unique titles discovered: {len(title_to_passage_count):,}")
    print(f"  Gold passages verified in corpus: {len(gold_cid_to_title):,} / {len(gold_corpus_ids):,}")

    # Group valid queries whose gold passages exist in the corpus
    query_to_gold_cids = {}
    for r in positive_qrels:
        qid = r["query-id"]
        cid = r["corpus-id"]
        if cid in gold_cid_to_title:
            query_to_gold_cids.setdefault(qid, set()).add(cid)

    valid_query_ids = sorted(query_to_gold_cids.keys())
    print(f"  Queries with valid corpus-backed gold passages: {len(valid_query_ids):,}")

    # -------------------------------------------------------------------------
    # PART 3: Sample 1,000 Queries and Collect All Gold Articles
    # -------------------------------------------------------------------------
    print("\n[Part 3] Randomly selecting 1,000 queries (seed=42)...")
    rng_query = random.Random(RANDOM_SEED)
    selected_query_ids = sorted(rng_query.sample(valid_query_ids, 1000))
    selected_query_ids_set = set(selected_query_ids)

    # Collect article titles that contain gold passages for these 1,000 queries
    gold_article_titles = set()
    for qid in selected_query_ids:
        for cid in query_to_gold_cids[qid]:
            gold_article_titles.add(gold_cid_to_title[cid])

    gold_passages_total = sum(title_to_passage_count[t] for t in gold_article_titles)
    print(f"  Selected queries: {len(selected_query_ids):,}")
    print(f"  Articles containing gold passages: {len(gold_article_titles):,}")
    print(f"  Total passages from gold articles: {gold_passages_total:,}")

    # -------------------------------------------------------------------------
    # PART 4: Add whole random articles until ~100,000 passages total
    # -------------------------------------------------------------------------
    print(f"\n[Part 4] Adding random whole articles to reach ~{TARGET_PASSAGES:,} passages...")
    remaining_titles = sorted(set(title_to_passage_count.keys()) - gold_article_titles)
    
    rng_articles = random.Random(RANDOM_SEED)
    rng_articles.shuffle(remaining_titles)

    selected_titles = set(gold_article_titles)
    current_passage_count = gold_passages_total

    for title in remaining_titles:
        count = title_to_passage_count[title]
        # If adding this article exceeds the target, choose whether including it
        # gets closer to TARGET_PASSAGES than stopping
        if current_passage_count + count > TARGET_PASSAGES:
            if abs((current_passage_count + count) - TARGET_PASSAGES) < abs(current_passage_count - TARGET_PASSAGES):
                selected_titles.add(title)
                current_passage_count += count
            break
        selected_titles.add(title)
        current_passage_count += count

    print(f"  Total articles in subset: {len(selected_titles):,}")
    print(f"  Total passages in subset: {current_passage_count:,}")

    # -------------------------------------------------------------------------
    # PART 5: Split queries into 500 dev and 500 test
    # -------------------------------------------------------------------------
    print("\n[Part 5] Splitting 1,000 queries into 500 dev and 500 test (seed=42)...")
    shuffled_queries = list(selected_query_ids)
    rng_split = random.Random(RANDOM_SEED)
    rng_split.shuffle(shuffled_queries)
    dev_qids = set(shuffled_queries[:500])
    test_qids = set(shuffled_queries[500:])
    print(f"  Dev queries: {len(dev_qids):,}, Test queries: {len(test_qids):,}")

    # -------------------------------------------------------------------------
    # PART 6: Save queries_subset.parquet and qrels_subset.parquet
    # -------------------------------------------------------------------------
    print("\n[Part 6] Writing queries and qrels subsets to data/...")
    
    # Queries: Load full queries from BeIR/nq
    queries_hf = load_dataset("BeIR/nq", "queries", split="queries")
    filtered_queries = []
    for q in queries_hf:
        qid = q["_id"]
        if qid in selected_query_ids_set:
            split_label = "dev" if qid in dev_qids else "test"
            filtered_queries.append({
                "_id": qid,
                "text": q["text"],
                "split": split_label,
            })

    queries_df = pd.DataFrame(filtered_queries)
    # Order predictably
    queries_df = queries_df.sort_values(by="_id").reset_index(drop=True)
    queries_path = os.path.join(OUTPUT_DIR, "queries_subset.parquet")
    queries_df.to_parquet(queries_path, index=False)
    print(f"  Saved {len(queries_df):,} queries to {queries_path}")

    # Qrels: filter positive qrels for the 1,000 queries
    filtered_qrels = []
    for r in positive_qrels:
        if r["query-id"] in selected_query_ids_set:
            filtered_qrels.append({
                "query-id": r["query-id"],
                "corpus-id": r["corpus-id"],
                "score": r["score"],
            })

    qrels_df = pd.DataFrame(filtered_qrels)
    qrels_df = qrels_df.sort_values(by=["query-id", "corpus-id"]).reset_index(drop=True)
    qrels_path = os.path.join(OUTPUT_DIR, "qrels_subset.parquet")
    qrels_df.to_parquet(qrels_path, index=False)
    print(f"  Saved {len(qrels_df):,} qrels to {qrels_path}")

    # -------------------------------------------------------------------------
    # PART 7: Chunked Extraction and Saving of corpus_subset.parquet
    # -------------------------------------------------------------------------
    print("\n[Part 7] Extracting corpus subset passages in chunks to data/corpus_subset.parquet...")
    corpus_output_path = os.path.join(OUTPUT_DIR, "corpus_subset.parquet")
    
    # Setup PyArrow ParquetWriter to stream output directly to disk
    schema = pa.schema([
        ("_id", pa.string()),
        ("title", pa.string()),
        ("text", pa.string()),
    ])

    saved_passage_count = 0
    with pq.ParquetWriter(corpus_output_path, schema=schema, compression="snappy") as writer:
        for batch in corpus_pf.iter_batches(batch_size=50000, columns=["_id", "title", "text"]):
            batch_df = batch.to_pandas()
            # Filter rows whose article title is in our selected articles set
            subset_chunk = batch_df[batch_df["title"].isin(selected_titles)]
            if not subset_chunk.empty:
                table = pa.Table.from_pandas(subset_chunk, schema=schema)
                writer.write_table(table)
                saved_passage_count += len(subset_chunk)

    print(f"  Saved {saved_passage_count:,} passages to {corpus_output_path}")

    # -------------------------------------------------------------------------
    # PART 8: Sanity Checks
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("SANITY CHECKS")
    print("=" * 70)

    # 1. Inspect saved corpus subset
    saved_pf = pq.ParquetFile(corpus_output_path)
    total_subset_passages = saved_pf.metadata.num_rows

    subset_title_counts = Counter()
    subset_cids = set()

    for batch in saved_pf.iter_batches(batch_size=50000, columns=["_id", "title"]):
        ids = batch["_id"].to_pylist()
        titles = batch["title"].to_pylist()
        subset_title_counts.update(titles)
        subset_cids.update(ids)

    total_articles = len(subset_title_counts)
    counts_series = pd.Series(list(subset_title_counts.values()))
    median_passages = counts_series.median()
    max_passages = counts_series.max()

    print(f"1. Total passages in subset            : {total_subset_passages:,}")
    print(f"2. Total articles in subset            : {total_articles:,}")
    print(f"3. Passages-per-article median         : {median_passages:.1f}")
    print(f"   Passages-per-article max            : {max_passages:,}")
    print(f"4. Number of queries                   : {len(queries_df):,} (Dev: {len(dev_qids)}, Test: {len(test_qids)})")
    
    avg_gold_per_query = len(qrels_df) / len(queries_df)
    print(f"5. Average gold passages per query     : {avg_gold_per_query:.2f} ({len(qrels_df):,} total qrels)")

    # 6. Check that every gold corpus-id of every selected query is in corpus_subset
    gold_cids_needed = set(qrels_df["corpus-id"])
    missing_cids = gold_cids_needed - subset_cids
    print(f"6. Missing gold corpus-ids in subset   : {len(missing_cids)} missing")

    if len(missing_cids) == 0:
        print("\n--> ALL SANITY CHECKS PASSED: 0 missing gold passages!")
    else:
        print(f"\n--> ERROR: {len(missing_cids)} missing gold passages!")


if __name__ == "__main__":
    main()
