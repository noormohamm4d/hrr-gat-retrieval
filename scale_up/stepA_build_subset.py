"""
Scale-Up Step A: Build the Scale-Up Subset (~200,000 Passages, ~3,452 Queries, 5 Grouped Folds)
---------------------------------------------------------------------------------------------
1. Identifies all BeIR/nq test queries having at least one positive qrel (score > 0)
   whose gold passages exist in the corpus.
2. Identifies all Wikipedia article titles containing gold passages for these kept queries
   and includes ALL passages belonging to those articles.
3. Randomly samples whole articles without replacement (seed 42) until the corpus reaches
   approximately 200,000 passages.
4. Assigns each query a group ID = article title of its first gold passage (lowest corpus-id).
5. Partitions queries into 5 grouped folds using sklearn GroupKFold (so no group/article
   appears in multiple folds).
6. Streams and writes:
   - scale_up/data/corpus.parquet
   - scale_up/data/queries.parquet (with 'fold' column)
   - scale_up/data/qrels.parquet
7. Runs verification checks:
   - Total passages and unique articles.
   - Number of queries and average gold passages per query.
   - Number of groups and fold size distribution.
   - Confirms 0 missing gold passages and 0 group leakage across folds.
"""

from collections import Counter
import os
import random
import time

from datasets import load_dataset
from huggingface_hub import hf_hub_download
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.model_selection import GroupKFold

RANDOM_SEED = 42
TARGET_PASSAGES = 200_000
OUTPUT_DIR = os.path.join("scale_up", "data")


def parse_corpus_id(cid: str):
    """Sorts corpus-ids numerically if 'doc' prefix followed by digits, else lexicographically."""
    if cid.startswith("doc") and cid[3:].isdigit():
        return int(cid[3:])
    return cid


def main():
    print("=" * 80)
    print("SCALE-UP STEP A: BUILDING SUBSET (~200K PASSAGES, 5 GROUPED FOLDS)")
    print("=" * 80)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # -------------------------------------------------------------------------
    # PART 1: Load Qrels and find gold passages
    # -------------------------------------------------------------------------
    print("\n[Part 1] Loading BeIR/nq-qrels (test split) and filtering score > 0...")
    t0 = time.time()
    qrels_dataset = load_dataset("BeIR/nq-qrels", split="test")
    positive_qrels = [r for r in qrels_dataset if r["score"] > 0]
    gold_corpus_ids = set(r["corpus-id"] for r in positive_qrels)
    print(f"  Positive qrels: {len(positive_qrels):,}")
    print(f"  Unique gold passage IDs: {len(gold_corpus_ids):,}")

    # -------------------------------------------------------------------------
    # PART 2: Scan full corpus metadata (chunked)
    # -------------------------------------------------------------------------
    print("\n[Part 2] Scanning full BeIR/nq corpus metadata in chunks...")
    corpus_parquet_path = hf_hub_download(
        repo_id="BeIR/nq",
        filename="corpus/corpus-00000-of-00001.parquet",
        repo_type="dataset",
    )
    corpus_pf = pq.ParquetFile(corpus_parquet_path)
    total_raw_rows = corpus_pf.metadata.num_rows
    print(f"  Full corpus total passages: {total_raw_rows:,}")

    gold_cid_to_title = {}
    title_to_passage_count = Counter()

    # Small timing benchmark on first batch
    t_bench_start = time.time()
    for batch_idx, batch in enumerate(corpus_pf.iter_batches(batch_size=100000, columns=["_id", "title"])):
        batch_ids = batch["_id"].to_pylist()
        batch_titles = batch["title"].to_pylist()
        title_to_passage_count.update(batch_titles)
        for cid, title in zip(batch_ids, batch_titles):
            if cid in gold_corpus_ids:
                gold_cid_to_title[cid] = title

        if batch_idx == 0:
            batch_time = time.time() - t_bench_start
            est_total_scan = (total_raw_rows / 100000.0) * batch_time
            print(f"  Benchmark: 100,000 rows metadata scan took {batch_time:.2f}s.")
            print(f"  Estimated scan duration for all {total_raw_rows:,} rows: {est_total_scan:.1f}s ({est_total_scan/60:.2f} min).")

    print(f"  Metadata scan complete. Total unique articles discovered: {len(title_to_passage_count):,}")
    print(f"  Gold passages verified in corpus: {len(gold_cid_to_title):,} / {len(gold_corpus_ids):,}")

    # Group valid queries whose gold passages exist in the corpus
    query_to_gold_cids = {}
    for r in positive_qrels:
        qid = r["query-id"]
        cid = r["corpus-id"]
        if cid in gold_cid_to_title:
            query_to_gold_cids.setdefault(qid, []).append(cid)

    kept_query_ids = sorted(query_to_gold_cids.keys())
    print(f"\n[Step 1 Result] Kept test queries with valid corpus-backed gold passages: {len(kept_query_ids):,}")

    # -------------------------------------------------------------------------
    # PART 3: Collect All Gold Articles and Add Whole Random Articles
    # -------------------------------------------------------------------------
    print("\n[Part 3] Collecting all gold articles and expanding corpus to ~200,000 passages...")
    gold_article_titles = set()
    for qid in kept_query_ids:
        for cid in query_to_gold_cids[qid]:
            gold_article_titles.add(gold_cid_to_title[cid])

    gold_passages_total = sum(title_to_passage_count[t] for t in gold_article_titles)
    print(f"  Unique articles containing gold passages: {len(gold_article_titles):,}")
    print(f"  Total passages from gold articles: {gold_passages_total:,}")

    # Sample whole articles without replacement using seed 42
    remaining_titles = sorted(set(title_to_passage_count.keys()) - gold_article_titles)
    rng_articles = random.Random(RANDOM_SEED)
    rng_articles.shuffle(remaining_titles)

    selected_titles = set(gold_article_titles)
    current_passage_count = gold_passages_total

    for title in remaining_titles:
        count = title_to_passage_count[title]
        if current_passage_count + count > TARGET_PASSAGES:
            # Pick whichever option gets closest to 200,000
            if abs((current_passage_count + count) - TARGET_PASSAGES) < abs(current_passage_count - TARGET_PASSAGES):
                selected_titles.add(title)
                current_passage_count += count
            break
        selected_titles.add(title)
        current_passage_count += count

    print(f"  Total articles in scale-up subset: {len(selected_titles):,}")
    print(f"  Total passages in scale-up subset: {current_passage_count:,}")

    # -------------------------------------------------------------------------
    # PART 4: Group Assignment and 5-Fold GroupKFold
    # -------------------------------------------------------------------------
    print("\n[Part 4] Assigning query groups and generating 5 GroupKFold splits...")
    query_records = []
    queries_hf = load_dataset("BeIR/nq", "queries", split="queries")
    kept_query_ids_set = set(kept_query_ids)
    queries_text_map = {}
    for q in queries_hf:
        if q["_id"] in kept_query_ids_set:
            queries_text_map[q["_id"]] = q["text"]

    for qid in kept_query_ids:
        cids = query_to_gold_cids[qid]
        # First gold passage by lowest corpus-id
        first_gold_cid = min(cids, key=parse_corpus_id)
        group_id = gold_cid_to_title[first_gold_cid]
        query_records.append({
            "_id": qid,
            "text": queries_text_map[qid],
            "group": group_id,
        })

    queries_df = pd.DataFrame(query_records)
    groups = queries_df["group"].values
    num_unique_groups = len(set(groups))
    print(f"  Number of unique query groups: {num_unique_groups:,}")

    # 5-fold GroupKFold
    gkf = GroupKFold(n_splits=5)
    queries_df["fold"] = -1
    for fold_idx, (train_indices, val_indices) in enumerate(gkf.split(queries_df, groups=groups)):
        queries_df.loc[val_indices, "fold"] = fold_idx

    fold_sizes = queries_df["fold"].value_counts().sort_index().to_dict()
    print(f"  Fold sizes: {fold_sizes}")

    # -------------------------------------------------------------------------
    # PART 5: Save queries.parquet and qrels.parquet
    # -------------------------------------------------------------------------
    print("\n[Part 5] Saving queries.parquet and qrels.parquet to scale_up/data/...")
    queries_out_path = os.path.join(OUTPUT_DIR, "queries.parquet")
    queries_df.to_parquet(queries_out_path, index=False)
    print(f"  Saved queries to '{queries_out_path}'.")

    # Qrels
    kept_qrels = []
    for r in positive_qrels:
        if r["query-id"] in kept_query_ids_set:
            kept_qrels.append({
                "query-id": r["query-id"],
                "corpus-id": r["corpus-id"],
                "score": r["score"],
            })

    qrels_df = pd.DataFrame(kept_qrels).sort_values(by=["query-id", "corpus-id"]).reset_index(drop=True)
    qrels_out_path = os.path.join(OUTPUT_DIR, "qrels.parquet")
    qrels_df.to_parquet(qrels_out_path, index=False)
    print(f"  Saved qrels to '{qrels_out_path}'.")

    # -------------------------------------------------------------------------
    # PART 6: Extract and Save corpus.parquet (Streamed in Chunks)
    # -------------------------------------------------------------------------
    print("\n[Part 6] Extracting corpus passages in chunks to scale_up/data/corpus.parquet...")
    corpus_out_path = os.path.join(OUTPUT_DIR, "corpus.parquet")
    schema = pa.schema([
        ("_id", pa.string()),
        ("title", pa.string()),
        ("text", pa.string()),
    ])

    saved_passage_count = 0
    t_write_start = time.time()
    with pq.ParquetWriter(corpus_out_path, schema=schema, compression="snappy") as writer:
        for batch in corpus_pf.iter_batches(batch_size=50000, columns=["_id", "title", "text"]):
            batch_df = batch.to_pandas()
            subset_chunk = batch_df[batch_df["title"].isin(selected_titles)]
            if not subset_chunk.empty:
                table = pa.Table.from_pandas(subset_chunk, schema=schema)
                writer.write_table(table)
                saved_passage_count += len(subset_chunk)

    write_elapsed = time.time() - t_write_start
    print(f"  Finished saving {saved_passage_count:,} passages in {write_elapsed:.2f}s to '{corpus_out_path}'.")

    # -------------------------------------------------------------------------
    # PART 7: Final Verification Checks
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("FINAL SANITY CHECKS & VERIFICATION")
    print("=" * 80)

    saved_corpus_pf = pq.ParquetFile(corpus_out_path)
    total_passages = saved_corpus_pf.metadata.num_rows

    subset_title_counter = Counter()
    subset_cids = set()
    for batch in saved_corpus_pf.iter_batches(batch_size=50000, columns=["_id", "title"]):
        subset_title_counter.update(batch["title"].to_pylist())
        subset_cids.update(batch["_id"].to_pylist())

    total_articles = len(subset_title_counter)
    num_queries = len(queries_df)
    avg_gold_per_query = len(qrels_df) / num_queries

    # Verify no missing gold passages
    gold_cids_needed = set(qrels_df["corpus-id"])
    missing_gold = gold_cids_needed - subset_cids

    # Verify no group appears in two folds
    group_fold_map = {}
    leakage_groups = 0
    for _, row in queries_df.iterrows():
        grp = row["group"]
        f = row["fold"]
        if grp in group_fold_map and group_fold_map[grp] != f:
            leakage_groups += 1
        group_fold_map[grp] = f

    print(f"1. Total Passages in Corpus           : {total_passages:,}")
    print(f"2. Total Unique Articles in Corpus    : {total_articles:,}")
    print(f"3. Number of Queries                  : {num_queries:,}")
    print(f"4. Average Gold Passages per Query    : {avg_gold_per_query:.2f} ({len(qrels_df):,} total qrels)")
    print(f"5. Number of Groups                   : {num_unique_groups:,}")
    print(f"6. Fold Sizes                         : {fold_sizes}")
    print(f"7. Missing Gold Passages in Corpus    : {len(missing_gold)}")
    print(f"8. Groups Appearing in >1 Fold        : {leakage_groups}")

    assert len(missing_gold) == 0, f"Error: {len(missing_gold)} missing gold passages!"
    assert leakage_groups == 0, f"Error: {leakage_groups} groups leaked across folds!"
    assert total_passages == saved_passage_count, "Passage count mismatch!"

    print("\n--> ALL VERIFICATIONS PASSED: 0 missing gold passages, 0 group leakage across folds.")
    print("=" * 80)


if __name__ == "__main__":
    main()
