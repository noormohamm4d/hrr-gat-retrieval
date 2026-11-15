"""
Step 2 Part 2: BM25 Baseline Retrieval
--------------------------------------
Implements BM25 retrieval using rank_bm25 on the Natural Questions subset:
1. Loads corpus passages from data/corpus_subset.parquet.
2. Formats each passage text as: title + " " + text.
3. Tokenizes by lowercasing and splitting on non-alphanumeric characters.
4. Builds a BM25Okapi index.
5. Runs first on 50 dev queries to report latency and preview metrics.
6. Runs on all 500 dev queries and all 500 test queries, retrieving the top 100 passages.
7. Saves results to results/bm25_dev.json and results/bm25_test.json in the standard format:
   {query_id: [[corpus_id, score], ... top 100]}.
8. Evaluates and prints Recall@10, Recall@100, MRR@10, and nDCG@10 using eval.metrics.
"""

import json
import os
import re
import sys
import time

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from rank_bm25 import BM25Okapi

# Add parent/current directory to path for importing eval.metrics
sys.path.insert(0, ".")
from eval.metrics import evaluate


def tokenize(text: str) -> list[str]:
    """
    Tokenizes text by lowercasing and splitting on non-alphanumeric characters.
    """
    return [token for token in re.split(r"[^a-zA-Z0-9]+", text.lower()) if token]


def retrieve_top_k(bm25: BM25Okapi, corpus_ids: list[str], query_tokens: list[str], k: int = 100):
    """
    Scores all passages for a single query using BM25 and returns the top-k
    as [[corpus_id, score], ...] sorted in descending score order.
    """
    scores = bm25.get_scores(query_tokens)
    # Use argpartition for fast top-k selection
    if len(scores) <= k:
        top_k_indices = np.argsort(-scores)
    else:
        top_k_partition = np.argpartition(scores, -k)[-k:]
        top_k_indices = top_k_partition[np.argsort(-scores[top_k_partition])]
    
    return [[corpus_ids[idx], float(scores[idx])] for idx in top_k_indices]


def run_batch_retrieval(bm25: BM25Okapi, corpus_ids: list[str], queries_df: pd.DataFrame, k: int = 100):
    """
    Runs retrieval for all queries in a DataFrame.
    Returns:
        results: {qid: [[cid, score], ...]}
        elapsed_seconds: float
    """
    results = {}
    t0 = time.time()
    for _, row in queries_df.iterrows():
        qid = str(row["_id"])
        q_tokens = tokenize(row["text"])
        results[qid] = retrieve_top_k(bm25, corpus_ids, q_tokens, k=k)
    elapsed = time.time() - t0
    return results, elapsed


def print_metrics_table(split_name: str, metrics: dict, elapsed: float, num_queries: int):
    print(f"\n--- {split_name} Results ({num_queries} queries, {elapsed:.2f}s | {elapsed/num_queries*1000:.1f}ms/query) ---")
    print(f"  Recall@10  : {metrics['Recall@10']:.4f} ({metrics['Recall@10']*100:.2f}%)")
    print(f"  Recall@100 : {metrics['Recall@100']:.4f} ({metrics['Recall@100']*100:.2f}%)")
    print(f"  MRR@10     : {metrics['MRR@10']:.4f}")
    print(f"  nDCG@10    : {metrics['nDCG@10']:.4f}")


def main():
    print("=" * 70)
    print("STEP 2 PART 2: BM25 RETRIEVAL (rank_bm25)")
    print("=" * 70)

    os.makedirs("results", exist_ok=True)

    # 1. Load Data
    print("\n[1/5] Loading data from data/...")
    corpus_table = pq.read_table("data/corpus_subset.parquet")
    corpus_ids = corpus_table["_id"].to_pylist()
    titles = corpus_table["title"].to_pylist()
    texts = corpus_table["text"].to_pylist()
    print(f"  Loaded {len(corpus_ids):,} passages.")

    queries_df = pq.read_table("data/queries_subset.parquet").to_pandas()
    dev_queries = queries_df[queries_df["split"] == "dev"].reset_index(drop=True)
    test_queries = queries_df[queries_df["split"] == "test"].reset_index(drop=True)
    print(f"  Loaded {len(dev_queries):,} dev queries and {len(test_queries):,} test queries.")

    qrels_df = pq.read_table("data/qrels_subset.parquet").to_pandas()
    print(f"  Loaded {len(qrels_df):,} qrels.")

    # 2. Tokenize Corpus
    print("\n[2/5] Tokenizing corpus passages (title + ' ' + text)...")
    t_tok_start = time.time()
    tokenized_corpus = [tokenize(f"{t} {txt}") for t, txt in zip(titles, texts)]
    print(f"  Tokenization finished in {time.time() - t_tok_start:.2f}s.")

    # 3. Build BM25 Index
    print("\n[3/5] Building BM25Okapi index...")
    t_idx_start = time.time()
    bm25 = BM25Okapi(tokenized_corpus)
    print(f"  BM25 index built in {time.time() - t_idx_start:.2f}s.")

    # 4. Pilot run on 50 dev queries
    print("\n[4/5] Running pilot evaluation on first 50 dev queries...")
    pilot_queries = dev_queries.head(50)
    pilot_results, pilot_time = run_batch_retrieval(bm25, corpus_ids, pilot_queries, k=100)
    pilot_metrics = evaluate(pilot_results, qrels_df)
    
    print_metrics_table("50 Dev Queries (Pilot)", pilot_metrics, pilot_time, len(pilot_queries))
    est_500_time = (pilot_time / 50) * 500
    print(f"\n  --> Estimated time for 500 queries: ~{est_500_time:.1f}s ({est_500_time/60:.2f} min)")
    print(f"  --> Estimated time for all 1,000 queries: ~{est_500_time * 2:.1f}s ({(est_500_time * 2)/60:.2f} min)")

    # 5. Full Runs: Dev (500) and Test (500)
    print("\n[5/5] Running retrieval on all 500 dev queries and all 500 test queries...")
    
    # Dev run
    print("  Processing 500 dev queries...")
    dev_results, dev_time = run_batch_retrieval(bm25, corpus_ids, dev_queries, k=100)
    dev_metrics = evaluate(dev_results, qrels_df)
    print_metrics_table("Full Dev Split (500 queries)", dev_metrics, dev_time, len(dev_queries))
    
    dev_out_path = os.path.join("results", "bm25_dev.json")
    with open(dev_out_path, "w", encoding="utf-8") as f:
        json.dump(dev_results, f)
    print(f"  Saved dev results to {dev_out_path}")

    # Test run
    print("\n  Processing 500 test queries...")
    test_results, test_time = run_batch_retrieval(bm25, corpus_ids, test_queries, k=100)
    test_metrics = evaluate(test_results, qrels_df)
    print_metrics_table("Full Test Split (500 queries)", test_metrics, test_time, len(test_queries))

    test_out_path = os.path.join("results", "bm25_test.json")
    with open(test_out_path, "w", encoding="utf-8") as f:
        json.dump(test_results, f)
    print(f"  Saved test results to {test_out_path}")

    print("\n" + "=" * 70)
    print("BM25 EXPERIMENT COMPLETED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    main()
