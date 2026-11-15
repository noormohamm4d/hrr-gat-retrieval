"""
Step 8: Article-Evidence Aggregation
------------------------------------
Implements article-evidence aggregation on top of cross-encoder reranking:
1. Candidate Scoring:
   For each query, takes the top 30 flat-dense candidates and their cross-encoder scores CE(p).
   For each candidate p:
       final(p) = CE(p) + beta * max_{p' in top30, article(p') == article(p)} CE(p')
   Candidates 31-100 are kept after them in their original dense order.
2. Tuning on DEV:
   Tuning beta in {0, 0.1, 0.25, 0.5, 1.0} on DEV queries.
   Confirms beta = 0 reproduces Dense + Cross-Encoder dev results exactly.
   Picks best beta by dev MRR@10.
3. Test Evaluation & Saving:
   Evaluates ONCE on TEST split with the best beta.
   Saves results/agg_dev.json and results/agg_test.json.
4. Reporting:
   - Dev beta sweep table.
   - Dev and Test performance tables comparing Flat Dense, Dense + CE, and Dense + CE + Aggregation.
   - Paired bootstrap on TEST (1,000 resamples, seed 42) comparing Aggregation vs Dense + CE.
"""

import json
import math
import os
import sys

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

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
AGG_DEV_PATH = os.path.join(RESULTS_DIR, "agg_dev.json")
AGG_TEST_PATH = os.path.join(RESULTS_DIR, "agg_test.json")


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


def apply_article_aggregation(
    ce_run: dict,
    dense_run: dict,
    cid_to_title: dict[str, str],
    beta: float,
    k_rerank: int = 30,
    k_total: int = 100,
) -> dict:
    """
    Applies article-evidence aggregation:
    final(p) = CE(p) + beta * max_{p' in top30, article(p') == article(p)} CE(p')
    Keeps candidates (k_rerank to k_total) in their original dense order.
    """
    agg_run = {}
    for qid, ce_list in ce_run.items():
        top30 = ce_list[:k_rerank]

        # 1. Compute max CE score per article among the top 30 candidates
        art_max_ce = {}
        for cid, ce_score in top30:
            art = cid_to_title[cid]
            if art not in art_max_ce or ce_score > art_max_ce[art]:
                art_max_ce[art] = ce_score

        # 2. Compute final aggregated score for each candidate in top 30
        scored_top30 = []
        for cid, ce_score in top30:
            art = cid_to_title[cid]
            final_score = ce_score + beta * art_max_ce[art]
            scored_top30.append([cid, final_score])

        # 3. Sort top 30 candidates in descending order of aggregated score
        scored_top30.sort(key=lambda item: item[1], reverse=True)

        # 4. Append candidates 31-100 in original dense order
        tail = dense_run[qid][k_rerank:k_total]
        agg_run[qid] = scored_top30 + tail

    return agg_run


def main():
    print("=" * 80)
    print("STEP 8: ARTICLE-EVIDENCE AGGREGATION")
    print("=" * 80)

    # 1. Load Data
    print("\n[1/5] Loading corpus mapping, qrels, and saved results...")
    corpus_table = pq.read_table(CORPUS_PATH)
    corpus_ids = corpus_table["_id"].to_pylist()
    corpus_titles = corpus_table["title"].to_pylist()
    cid_to_title = dict(zip(corpus_ids, corpus_titles))

    queries_df = pq.read_table(QUERIES_PATH).to_pandas()
    test_queries = queries_df[queries_df["split"] == "test"].reset_index(drop=True)
    test_qids = [str(qid) for qid in test_queries["_id"]]

    qrels_df = pq.read_table(QRELS_PATH).to_pandas()
    qrels_norm = _normalize_qrels(qrels_df)

    with open(DENSE_DEV_PATH, "r", encoding="utf-8") as f: dense_dev = json.load(f)
    with open(DENSE_TEST_PATH, "r", encoding="utf-8") as f: dense_test = json.load(f)
    with open(CE_DEV_PATH, "r", encoding="utf-8") as f: ce_dev = json.load(f)
    with open(CE_TEST_PATH, "r", encoding="utf-8") as f: ce_test = json.load(f)

    # 2. Parameter Sweep: beta in {0, 0.1, 0.25, 0.5, 1.0} on DEV
    print("\n[2/5] Tuning beta in {0, 0.1, 0.25, 0.5, 1.0} on DEV queries (500 queries)...")
    beta_candidates = [0.0, 0.1, 0.25, 0.5, 1.0]
    dev_sweep_results = {}
    best_beta = None
    best_dev_mrr = -1.0

    print(f"\n{'beta':<8} | {'Recall@10':<10} | {'Recall@100':<10} | {'MRR@10':<8} | {'nDCG@10':<8}")
    print("-" * 54)

    for beta in beta_candidates:
        agg_dev = apply_article_aggregation(ce_dev, dense_dev, cid_to_title, beta)
        m = evaluate(agg_dev, qrels_df)
        dev_sweep_results[beta] = {"results": agg_dev, "metrics": m}

        mrr = m["MRR@10"]
        if mrr > best_dev_mrr:
            best_dev_mrr = mrr
            best_beta = beta

        print(
            f"{beta:<8.2f} | "
            f"{m['Recall@10']:<10.4f} | "
            f"{m['Recall@100']:<10.4f} | "
            f"{m['MRR@10']:<8.4f} | "
            f"{m['nDCG@10']:<8.4f}"
        )

    print("-" * 54)

    # Sanity Check: beta = 0 must reproduce dense + cross-encoder results exactly
    print("\n--- Verifying beta = 0 reproduces Dense + Cross-Encoder Dev results ---")
    ce_dev_metrics = evaluate(ce_dev, qrels_df)
    beta0_metrics = dev_sweep_results[0.0]["metrics"]
    print(f"  beta = 0.00 metrics: {beta0_metrics}")
    print(f"  Dense + CE metrics : {ce_dev_metrics}")

    exact_match = (
        round(beta0_metrics["Recall@10"], 6) == round(ce_dev_metrics["Recall@10"], 6)
        and round(beta0_metrics["Recall@100"], 6) == round(ce_dev_metrics["Recall@100"], 6)
        and round(beta0_metrics["MRR@10"], 6) == round(ce_dev_metrics["MRR@10"], 6)
        and round(beta0_metrics["nDCG@10"], 6) == round(ce_dev_metrics["nDCG@10"], 6)
    )
    print(f"  CHECK: beta = 0 matches Dense + Cross-Encoder exactly -> {exact_match}")
    assert exact_match, "Error: beta=0 does not reproduce Dense + Cross-Encoder results!"

    print(f"\n--> Best beta selected based on Dev MRR@10: beta = {best_beta:.2f} (MRR@10 = {best_dev_mrr:.4f})")

    # 3. Evaluate ONCE on TEST with best beta
    print(f"\n[3/5] Evaluating ONCE on TEST split with best beta = {best_beta:.2f}...")
    agg_test = apply_article_aggregation(ce_test, dense_test, cid_to_title, best_beta)
    agg_test_metrics = evaluate(agg_test, qrels_df)

    # 4. Save results/agg_dev.json and results/agg_test.json
    best_agg_dev = dev_sweep_results[best_beta]["results"]
    best_agg_dev_metrics = dev_sweep_results[best_beta]["metrics"]

    with open(AGG_DEV_PATH, "w", encoding="utf-8") as f:
        json.dump(best_agg_dev, f)
    print(f"  Saved best dev aggregation results to '{AGG_DEV_PATH}'.")

    with open(AGG_TEST_PATH, "w", encoding="utf-8") as f:
        json.dump(agg_test, f)
    print(f"  Saved best test aggregation results to '{AGG_TEST_PATH}'.")

    # 5. Print Comparison Tables (Dev and Test)
    dense_dev_metrics = evaluate(dense_dev, qrels_df)
    dense_test_metrics = evaluate(dense_test, qrels_df)
    ce_test_metrics = evaluate(ce_test, qrels_df)

    def print_benchmark_table(split_name: str, d_m: dict, ce_m: dict, agg_m: dict, chosen_b: float):
        print(f"\n=========================================================================================")
        print(f"                        {split_name.upper()} RETRIEVAL BENCHMARK")
        print(f"=========================================================================================")
        header = f"{'System':<38} | {'Recall@10':<10} | {'Recall@100':<10} | {'MRR@10':<8} | {'nDCG@10':<8}"
        print(header)
        print("-" * len(header))
        print(
            f"{'Flat Dense':<38} | "
            f"{d_m['Recall@10']:<10.4f} | "
            f"{d_m['Recall@100']:<10.4f} | "
            f"{d_m['MRR@10']:<8.4f} | "
            f"{d_m['nDCG@10']:<8.4f}"
        )
        print(
            f"{'Dense + Cross-Encoder':<38} | "
            f"{ce_m['Recall@10']:<10.4f} | "
            f"{ce_m['Recall@100']:<10.4f} | "
            f"{ce_m['MRR@10']:<8.4f} | "
            f"{ce_m['nDCG@10']:<8.4f}"
        )
        agg_label = f"Dense + CE + Aggregation (beta={chosen_b:.2f})"
        print(
            f"{agg_label:<38} | "
            f"{agg_m['Recall@10']:<10.4f} | "
            f"{agg_m['Recall@100']:<10.4f} | "
            f"{agg_m['MRR@10']:<8.4f} | "
            f"{agg_m['nDCG@10']:<8.4f}"
        )
        print("=" * len(header))

    print_benchmark_table("Dev Split (500 queries)", dense_dev_metrics, ce_dev_metrics, best_agg_dev_metrics, best_beta)
    print_benchmark_table("Test Split (500 queries)", dense_test_metrics, ce_test_metrics, agg_test_metrics, best_beta)

    # 6. Paired Bootstrap on TEST (1,000 resamples, seed 42): Aggregation vs Dense + Cross-Encoder
    print("\n[5/5] Running Paired Bootstrap on TEST (Aggregation vs Dense + Cross-Encoder)...")
    pq_ce = compute_per_query_metrics(ce_test, qrels_norm)
    pq_agg = compute_per_query_metrics(agg_test, qrels_norm)

    print("\n" + "=" * 90)
    print("PAIRED BOOTSTRAP SIGNIFICANCE TESTING: AGGREGATION vs DENSE + CE (TEST SPLIT)")
    print("=" * 90)
    print(f"{'Metric':<14} | {'CE Mean':<12} | {'Agg Mean':<12} | {'Mean Diff':<10} | {'95% CI':<20} | {'P(Diff > 0)':<12} | {'Verdict'}")
    print("-" * 90)

    for m in ["Recall@10", "MRR@10", "nDCG@10"]:
        vals_agg = [pq_agg[qid][m] for qid in test_qids]
        vals_ce = [pq_ce[qid][m] for qid in test_qids]

        mean_d, ci_l, ci_u, p_gt_0 = run_paired_bootstrap(vals_agg, vals_ce, n_resamples=1000, seed=42)

        if ci_l > 0:
            verdict = "Clear Improvement (p < 0.05)"
        elif ci_u < 0:
            verdict = "Clear Decline (p < 0.05)"
        else:
            verdict = "Within Noise (Not Significant)"

        ci_str = f"[{ci_l:+.4f}, {ci_u:+.4f}]"
        print(f"{m:<14} | {ce_test_metrics[m]:<12.4f} | {agg_test_metrics[m]:<12.4f} | {mean_d:<+10.4f} | {ci_str:<20} | {p_gt_0*100:<11.1f}% | {verdict}")

    print("=" * 90)
    print("\nSTEP 8 COMPLETED SUCCESSFULLY!")


if __name__ == "__main__":
    main()
