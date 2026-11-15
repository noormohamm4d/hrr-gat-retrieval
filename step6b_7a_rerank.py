"""
Step 6b & 7a: Error Analysis on Top-10 Failures and Cross-Encoder Reranking
--------------------------------------------------------------------------
Part 1 (TEST, Flat Dense Error Diagnosis):
- Analyzes the 75 test queries that fail to retrieve any gold passage in top 10.
- Reports:
  (a) In how many the gold passage's article appears among top-10 and top-3 retrieved passages.
  (b) The median dense rank of the first gold passage in the top-100 (for those that have one).
  (c) Across all 500 test queries, the fraction where the gold article appears among top-10 retrieved passages.

Part 2 (Cross-Encoder Reranking):
- Uses sentence_transformers CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2') on GPU.
- For each query, reranks the top 30 flat-dense candidates using text = title + " " + text.
- Keeps candidates 31-100 in their original dense order.
- Runs Dev and Test, saving results/ce_dev.json and results/ce_test.json.
- Evaluates Recall@10, Recall@100, MRR@10, nDCG@10.
- Runs paired bootstrap on TEST (1,000 resamples, seed 42) vs Flat Dense for Recall@10, MRR@10, nDCG@10.
"""

import json
import math
import os
import sys
import time

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from sentence_transformers import CrossEncoder

# Current directory for eval.metrics
sys.path.insert(0, ".")
from eval.metrics import evaluate, _normalize_qrels, _normalize_ranked_list

CORPUS_PATH = "data/corpus_subset.parquet"
QUERIES_PATH = "data/queries_subset.parquet"
QRELS_PATH = "data/qrels_subset.parquet"

RESULTS_DIR = "results"
DENSE_DEV_PATH = os.path.join(RESULTS_DIR, "dense_dev.json")
DENSE_TEST_PATH = os.path.join(RESULTS_DIR, "dense_test.json")
CE_DEV_PATH = os.path.join(RESULTS_DIR, "ce_dev.json")
CE_TEST_PATH = os.path.join(RESULTS_DIR, "ce_test.json")


def compute_per_query_metrics(run: dict, qrels_norm: dict) -> dict[str, dict[str, float]]:
    """Computes exact per-query IR metrics matching eval/metrics.py."""
    per_query = {}
    for qid, raw_ranked in run.items():
        qid_str = str(qid)
        gold_set = qrels_norm.get(qid_str, set())
        num_gold = len(gold_set)

        if num_gold == 0:
            per_query[qid_str] = {"Recall@10": 0.0, "Recall@100": 0.0, "MRR@10": 0.0, "nDCG@10": 0.0}
            continue

        ranked_cids = _normalize_ranked_list(raw_ranked)
        top_10 = ranked_cids[:10]
        top_100 = ranked_cids[:100]

        # 1. Recall@10 and Recall@100
        gold_in_top_10 = sum(1 for cid in top_10 if cid in gold_set)
        gold_in_top_100 = sum(1 for cid in top_100 if cid in gold_set)
        recall_10 = gold_in_top_10 / num_gold
        recall_100 = gold_in_top_100 / num_gold

        # 2. MRR@10
        mrr_10 = 0.0
        for rank, cid in enumerate(top_10, start=1):
            if cid in gold_set:
                mrr_10 = 1.0 / rank
                break

        # 3. nDCG@10
        dcg_10 = 0.0
        for rank, cid in enumerate(top_10, start=1):
            if cid in gold_set:
                dcg_10 += 1.0 / math.log2(rank + 1)
        ideal_count = min(num_gold, 10)
        idcg_10 = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_count + 1))
        ndcg_10 = (dcg_10 / idcg_10) if idcg_10 > 0.0 else 0.0

        per_query[qid_str] = {
            "Recall@10": recall_10,
            "Recall@100": recall_100,
            "MRR@10": mrr_10,
            "nDCG@10": ndcg_10,
        }
    return per_query


def run_paired_bootstrap(
    metrics_A: list[float],
    metrics_B: list[float],
    n_resamples: int = 1000,
    seed: int = 42,
) -> tuple[float, float, float, float]:
    """Performs paired bootstrap test between systems A and B (A - B)."""
    diffs = np.array(metrics_A) - np.array(metrics_B)
    observed_mean = float(np.mean(diffs))

    rng = np.random.default_rng(seed)
    n = len(diffs)
    boot_diffs = np.empty(n_resamples, dtype=np.float64)

    for i in range(n_resamples):
        sample_indices = rng.integers(0, n, size=n)
        boot_diffs[i] = np.mean(diffs[sample_indices])

    ci_lower = float(np.percentile(boot_diffs, 2.5))
    ci_upper = float(np.percentile(boot_diffs, 97.5))
    p_above_zero = float(np.mean(boot_diffs > 0.0))

    return observed_mean, ci_lower, ci_upper, p_above_zero


def main():
    print("=" * 75)
    print("STEP 6b & 7a: ERROR ANALYSIS & CROSS-ENCODER RERANKING")
    print("=" * 75)

    # 1. Load Data
    print("\n[1/5] Loading data and previous Dense results...")
    corpus_table = pq.read_table(CORPUS_PATH)
    corpus_ids = corpus_table["_id"].to_pylist()
    corpus_titles = corpus_table["title"].to_pylist()
    corpus_texts = corpus_table["text"].to_pylist()
    corpus_lookup = {
        cid: {"title": t, "text": txt}
        for cid, t, txt in zip(corpus_ids, corpus_titles, corpus_texts)
    }

    queries_df = pq.read_table(QUERIES_PATH).to_pandas()
    dev_queries = queries_df[queries_df["split"] == "dev"].reset_index(drop=True)
    test_queries = queries_df[queries_df["split"] == "test"].reset_index(drop=True)
    qrels_df = pq.read_table(QRELS_PATH).to_pandas()
    qrels_norm = _normalize_qrels(qrels_df)

    with open(DENSE_DEV_PATH, "r", encoding="utf-8") as f:
        dense_dev = json.load(f)
    with open(DENSE_TEST_PATH, "r", encoding="utf-8") as f:
        dense_test = json.load(f)

    # =========================================================================
    # PART 1: TEST FLAT DENSE ERROR DIAGNOSIS
    # =========================================================================
    print("\n" + "=" * 75)
    print("PART 1: TEST FLAT DENSE ERROR DIAGNOSIS")
    print("=" * 75)

    test_qids = [str(qid) for qid in test_queries["_id"]]
    no_gold_in_top10_qids = []

    for qid in test_qids:
        gold = qrels_norm.get(qid, set())
        retrieved_10 = _normalize_ranked_list(dense_test.get(qid, []))[:10]
        if not bool(set(retrieved_10) & gold):
            no_gold_in_top10_qids.append(qid)

    print(f"Total test queries with no gold passage in top 10: {len(no_gold_in_top10_qids)} / {len(test_qids)} ({len(no_gold_in_top10_qids)/len(test_qids)*100:.1f}%)")

    # (a) Among the 75 queries: gold article in top 10 retrieved articles, and top 3 retrieved articles
    gold_art_in_top10_retrieved = 0
    gold_art_in_top3_retrieved = 0
    first_gold_ranks = []

    for qid in no_gold_in_top10_qids:
        gold_cids = qrels_norm.get(qid, set())
        gold_articles = {corpus_lookup[cid]["title"] for cid in gold_cids if cid in corpus_lookup}

        retrieved_100 = _normalize_ranked_list(dense_test.get(qid, []))[:100]
        retrieved_10_articles = {corpus_lookup[cid]["title"] for cid in retrieved_100[:10] if cid in corpus_lookup}
        retrieved_3_articles = {corpus_lookup[cid]["title"] for cid in retrieved_100[:3] if cid in corpus_lookup}

        if bool(gold_articles & retrieved_10_articles):
            gold_art_in_top10_retrieved += 1
        if bool(gold_articles & retrieved_3_articles):
            gold_art_in_top3_retrieved += 1

        # (b) First gold rank in top 100
        for rank, cid in enumerate(retrieved_100, start=1):
            if cid in gold_cids:
                first_gold_ranks.append(rank)
                break

    print(f"\nAmong the {len(no_gold_in_top10_qids)} queries with NO gold passage in top 10:")
    print(f"  (a) Gold article appears in top-10 retrieved articles : {gold_art_in_top10_retrieved} / {len(no_gold_in_top10_qids)} ({gold_art_in_top10_retrieved/len(no_gold_in_top10_qids)*100:.1f}%)")
    print(f"      Gold article appears in top-3 retrieved articles  : {gold_art_in_top3_retrieved} / {len(no_gold_in_top10_qids)} ({gold_art_in_top3_retrieved/len(no_gold_in_top10_qids)*100:.1f}%)")
    print(f"  (b) Queries having at least one gold passage in top-100: {len(first_gold_ranks)} / {len(no_gold_in_top10_qids)} ({len(first_gold_ranks)/len(no_gold_in_top10_qids)*100:.1f}%)")
    if first_gold_ranks:
        print(f"      Median dense rank of first gold passage in top-100: {np.median(first_gold_ranks):.1f} (min: {min(first_gold_ranks)}, max: {max(first_gold_ranks)})")

    # (c) Across ALL 500 test queries: fraction where gold article appears among top-10 retrieved articles
    all_500_gold_art_in_top10 = 0
    for qid in test_qids:
        gold_cids = qrels_norm.get(qid, set())
        gold_articles = {corpus_lookup[cid]["title"] for cid in gold_cids if cid in corpus_lookup}
        retrieved_10 = _normalize_ranked_list(dense_test.get(qid, []))[:10]
        retrieved_10_articles = {corpus_lookup[cid]["title"] for cid in retrieved_10 if cid in corpus_lookup}
        if bool(gold_articles & retrieved_10_articles):
            all_500_gold_art_in_top10 += 1

    pct_all_500 = (all_500_gold_art_in_top10 / len(test_qids)) * 100
    print(f"\nAcross ALL {len(test_qids)} test queries:")
    print(f"  Fraction where gold article appears in top-10 retrieved articles: {all_500_gold_art_in_top10} / {len(test_qids)} ({pct_all_500:.2f}%)")

    # =========================================================================
    # PART 2: CROSS-ENCODER RERANKING
    # =========================================================================
    print("\n" + "=" * 75)
    print("PART 2: CROSS-ENCODER RERANKING ('cross-encoder/ms-marco-MiniLM-L-6-v2')")
    print("=" * 75)

    print("\n[2/5] Loading CrossEncoder model onto GPU...")
    ce = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device="cuda")
    print(f"  CrossEncoder loaded on: {ce.model.device}")

    def rerank_split(queries_df_split: pd.DataFrame, dense_run: dict, split_name: str, k_rerank: int = 30):
        print(f"\n[3/5] Reranking top {k_rerank} candidates for {split_name} ({len(queries_df_split)} queries)...")
        t0 = time.time()
        reranked_run = {}

        # Prepare all pairs for efficient batch inference
        all_pairs = []
        query_slices = []
        current_offset = 0

        for _, row in queries_df_split.iterrows():
            qid = str(row["_id"])
            q_text = row["text"]
            dense_top100 = dense_run[qid]

            top_candidates = dense_top100[:k_rerank]
            num_cand = len(top_candidates)

            for item in top_candidates:
                cid = item[0]
                p_info = corpus_lookup[cid]
                p_text = f"{p_info['title']} {p_info['text']}"
                all_pairs.append((q_text, p_text))

            query_slices.append((qid, current_offset, current_offset + num_cand, dense_top100))
            current_offset += num_cand

        print(f"  Scoring {len(all_pairs):,} query-passage pairs with CrossEncoder (batch_size=128)...")
        all_scores = ce.predict(all_pairs, batch_size=128, show_progress_bar=True)

        for qid, start_idx, end_idx, original_top100 in query_slices:
            slice_scores = all_scores[start_idx:end_idx]
            cand_slice = original_top100[:k_rerank]

            # Sort top k_rerank candidates by cross-encoder score descending
            sort_order = np.argsort(-slice_scores)
            reranked_top = [
                [cand_slice[idx][0], float(slice_scores[idx])]
                for idx in sort_order
            ]

            # Append candidates 31-100 in their original dense order
            tail = original_top100[k_rerank:100]
            reranked_run[qid] = reranked_top + tail

        elapsed = time.time() - t0
        print(f"  Reranking for {split_name} finished in {elapsed:.2f}s ({elapsed/len(queries_df_split)*1000:.1f}ms/query).")
        return reranked_run

    # Rerank Dev
    ce_dev_results = rerank_split(dev_queries, dense_dev, "DEV Split", k_rerank=30)
    with open(CE_DEV_PATH, "w", encoding="utf-8") as f:
        json.dump(ce_dev_results, f)
    print(f"  Saved dev cross-encoder results to '{CE_DEV_PATH}'.")

    # Rerank Test
    ce_test_results = rerank_split(test_queries, dense_test, "TEST Split", k_rerank=30)
    with open(CE_TEST_PATH, "w", encoding="utf-8") as f:
        json.dump(ce_test_results, f)
    print(f"  Saved test cross-encoder results to '{CE_TEST_PATH}'.")

    # [4/5] Evaluate and Print Comparison Tables
    print("\n[4/5] Evaluating performance...")
    dense_dev_metrics = evaluate(dense_dev, qrels_df)
    dense_test_metrics = evaluate(dense_test, qrels_df)
    ce_dev_metrics = evaluate(ce_dev_results, qrels_df)
    ce_test_metrics = evaluate(ce_test_results, qrels_df)

    def print_eval_table(split_name: str, dense_m: dict, ce_m: dict):
        print(f"\n=========================================================================")
        print(f"          {split_name.upper()}: FLAT DENSE vs DENSE + CROSS-ENCODER")
        print(f"=========================================================================")
        print(f"{'Metric':<14} | {'Flat Dense':<12} | {'Dense + CE':<12} | {'Delta':<10}")
        print(f"{'-'*14}-+-{'-'*12}-+-{'-'*12}-+-{'-'*10}")
        for m in ["Recall@10", "Recall@100", "MRR@10", "nDCG@10"]:
            d_val = dense_m[m]
            c_val = ce_m[m]
            delta = c_val - d_val
            print(f"{m:<14} | {d_val:<12.4f} | {c_val:<12.4f} | {delta:<+10.4f}")
        print("=========================================================================")

    print_eval_table("Dev Split (500 queries)", dense_dev_metrics, ce_dev_metrics)
    print_eval_table("Test Split (500 queries)", dense_test_metrics, ce_test_metrics)

    # [5/5] Paired Bootstrap on TEST (1,000 resamples, seed 42)
    print("\n[5/5] Running Paired Bootstrap on TEST (Dense + CE vs Flat Dense)...")
    pq_dense_test = compute_per_query_metrics(dense_test, qrels_norm)
    pq_ce_test = compute_per_query_metrics(ce_test_results, qrels_norm)

    print("\n" + "=" * 90)
    print("PAIRED BOOTSTRAP SIGNIFICANCE TESTING: DENSE + CE vs FLAT DENSE (TEST SPLIT)")
    print("=" * 90)
    print(f"{'Metric':<14} | {'Dense Mean':<12} | {'CE Mean':<12} | {'Mean Diff':<10} | {'95% CI':<20} | {'P(Diff > 0)':<12} | {'Verdict'}")
    print("-" * 90)

    for m in ["Recall@10", "MRR@10", "nDCG@10"]:
        vals_ce = [pq_ce_test[qid][m] for qid in test_qids]
        vals_dense = [pq_dense_test[qid][m] for qid in test_qids]

        mean_d, ci_l, ci_u, p_gt_0 = run_paired_bootstrap(vals_ce, vals_dense, n_resamples=1000, seed=42)

        if ci_l > 0:
            verdict = "Clear Improvement (p < 0.05)"
        elif ci_u < 0:
            verdict = "Clear Decline (p < 0.05)"
        else:
            verdict = "Within Noise (Not Significant)"

        ci_str = f"[{ci_l:+.4f}, {ci_u:+.4f}]"
        print(f"{m:<14} | {dense_test_metrics[m]:<12.4f} | {ce_test_metrics[m]:<12.4f} | {mean_d:<+10.4f} | {ci_str:<20} | {p_gt_0*100:<11.1f}% | {verdict}")

    print("=" * 90)
    print("\nSTEP 6b & 7a COMPLETED SUCCESSFULLY!")


if __name__ == "__main__":
    main()
