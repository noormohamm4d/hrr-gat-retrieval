"""
Scale-Up Extra: Cross-Encoder Rerank-Depth Sensitivity Analysis
--------------------------------------------------------------
Evaluates the sensitivity of retrieval performance to rerank depth D in {30, 50, 100}:
1. Candidates:
   Uses the saved top-100 flat-dense candidates from scale_up/results/dense.json.
2. Cross-Encoder Scoring:
   - Uses cross-encoder/ms-marco-MiniLM-L-6-v2 on GPU.
   - Reuses existing CE scores for top-30 candidates from scale_up/results/raw_ce_scores.json.
   - Runs a timing benchmark on 500 pairs before scoring candidates 31-100 (241,640 pairs).
   - Records total scoring time for each depth D in {30, 50, 100}.
3. Reranking for D in {30, 50, 100}:
   - For each depth D, sorts the top D candidates by cross-encoder score descending.
   - Keeps ranks after D (D+1 to 100) in their original dense order.
4. Metrics & Statistical Significance:
   - Evaluates Recall@10, MRR@10, nDCG@10 for all 3 depths over all 3,452 queries.
   - Runs paired bootstrap (1,000 resamples, seed 42) comparing:
     * D=50 vs D=30
     * D=100 vs D=30
5. Outputs:
   - scale_up/results/depth_summary.csv
   - scale_up/results/ce_depth50.json
   - scale_up/results/ce_depth100.json
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

# Automatically resolve project root (D:\hrr-project) and scale_up directory
SCALE_UP_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCALE_UP_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from eval.metrics import evaluate, _normalize_qrels, _normalize_ranked_list

DATA_DIR = os.path.join(SCALE_UP_DIR, "data")
CORPUS_PATH = os.path.join(DATA_DIR, "corpus.parquet")
QUERIES_PATH = os.path.join(DATA_DIR, "queries.parquet")
QRELS_PATH = os.path.join(DATA_DIR, "qrels.parquet")

RESULTS_DIR = os.path.join(SCALE_UP_DIR, "results")
DENSE_RESULTS_PATH = os.path.join(RESULTS_DIR, "dense.json")
CE30_RESULTS_PATH = os.path.join(RESULTS_DIR, "ce.json")
RAW_CE30_SCORES_PATH = os.path.join(RESULTS_DIR, "raw_ce_scores.json")
RAW_CE100_SCORES_PATH = os.path.join(RESULTS_DIR, "raw_ce_scores_100.json")

CE50_RESULTS_PATH = os.path.join(RESULTS_DIR, "ce_depth50.json")
CE100_RESULTS_PATH = os.path.join(RESULTS_DIR, "ce_depth100.json")
DEPTH_SUMMARY_PATH = os.path.join(RESULTS_DIR, "depth_summary.csv")

RANDOM_SEED = 42
CROSS_ENCODER_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def check_gpu():
    """Confirms GPU availability and device name."""
    cuda_avail = torch.cuda.is_available()
    print(f"torch.cuda.is_available(): {cuda_avail}")
    if not cuda_avail:
        raise RuntimeError("CUDA is not available. Cross-Encoder reranking requires GPU.")
    device_name = torch.cuda.get_device_name(0)
    print(f"CUDA Device Name: {device_name}")
    return device_name


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

        gold_in_top_10 = sum(1 for cid in top_10 if cid in gold_set)
        gold_in_top_100 = sum(1 for cid in top_100 if cid in gold_set)
        recall_10 = gold_in_top_10 / num_gold
        recall_100 = gold_in_top_100 / num_gold

        mrr_10 = 0.0
        for rank, cid in enumerate(top_10, start=1):
            if cid in gold_set:
                mrr_10 = 1.0 / rank
                break

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
    print("=" * 85)
    print("SCALE-UP EXTRA: CROSS-ENCODER RERANK-DEPTH SENSITIVITY (D in {30, 50, 100})")
    print("=" * 85)

    check_gpu()

    # 1. Load Data
    print("\n[1/5] Loading corpus lookup, queries, qrels, and dense results...")
    corpus_table = pq.read_table(CORPUS_PATH)
    corpus_ids = corpus_table["_id"].to_pylist()
    corpus_titles = corpus_table["title"].to_pylist()
    corpus_texts = corpus_table["text"].to_pylist()
    corpus_lookup = {
        cid: f"{t} {txt}"
        for cid, t, txt in zip(corpus_ids, corpus_titles, corpus_texts)
    }

    queries_df = pq.read_table(QUERIES_PATH).to_pandas()
    num_queries = len(queries_df)
    queries_text_map = dict(zip(queries_df["_id"].astype(str), queries_df["text"]))
    all_qids = [str(qid) for qid in queries_df["_id"]]

    qrels_df = pq.read_table(QRELS_PATH).to_pandas()
    qrels_norm = _normalize_qrels(qrels_df)

    with open(DENSE_RESULTS_PATH, "r", encoding="utf-8") as f:
        dense_results = json.load(f)

    # Load existing raw CE scores for top-30 candidates
    with open(RAW_CE30_SCORES_PATH, "r", encoding="utf-8") as f:
        raw_ce_scores_30 = json.load(f)

    # 2. Score Candidates 31 to 100 with Cross-Encoder
    print("\n[2/5] Preparing cross-encoder scores for candidates 31-100...")
    all_ce_scores = {}
    for qid, score_dict in raw_ce_scores_30.items():
        all_ce_scores[qid] = dict(score_dict)

    # Check if raw_ce_scores_100 already exists
    need_scoring = True
    if os.path.exists(RAW_CE100_SCORES_PATH):
        print(f"  Found existing 100-depth CE scores at '{RAW_CE100_SCORES_PATH}'. Loading...")
        with open(RAW_CE100_SCORES_PATH, "r", encoding="utf-8") as f:
            all_ce_scores = json.load(f)
        # Verify complete
        if all(len(all_ce_scores[q]) == 100 for q in all_qids):
            need_scoring = False
            print("  All 100 candidates per query already scored.")

    additional_scoring_time = 0.0

    if need_scoring:
        pairs_to_score = []
        pair_metadata = []

        for qid in all_qids:
            q_text = queries_text_map[qid]
            dense_top100 = dense_results[qid]
            for item in dense_top100[30:100]:
                cid = item[0]
                if cid not in all_ce_scores[qid]:
                    p_text = corpus_lookup[cid]
                    pairs_to_score.append((q_text, p_text))
                    pair_metadata.append((qid, cid))

        total_pairs = len(pairs_to_score)
        print(f"  Total remaining pairs to score (ranks 31-100): {total_pairs:,}")

        ce_model = CrossEncoder(CROSS_ENCODER_NAME, device="cuda")

        # Small timing benchmark on first 500 pairs
        print("  Running timing benchmark on first 500 pairs...")
        t_bench = time.time()
        _ = ce_model.predict(pairs_to_score[:500], batch_size=128, show_progress_bar=False)
        bench_time = time.time() - t_bench
        throughput = 500.0 / bench_time
        est_time_remaining = (total_pairs / 500.0) * bench_time
        print(f"  500 pairs scored in {bench_time:.2f}s ({throughput:.1f} pairs/sec).")
        print(f"  Estimated time for {total_pairs:,} pairs: {est_time_remaining:.1f}s ({est_time_remaining/60:.2f} min).")

        print(f"\nScoring {total_pairs:,} pairs with CrossEncoder on GPU (batch_size=128)...")
        t_score_start = time.time()
        scores = ce_model.predict(pairs_to_score, batch_size=128, show_progress_bar=True)
        additional_scoring_time = time.time() - t_score_start
        print(f"  Scoring completed in {additional_scoring_time:.2f}s ({additional_scoring_time/60:.2f} min).")

        for (qid, cid), sc in zip(pair_metadata, scores):
            all_ce_scores[qid][cid] = float(sc)

        with open(RAW_CE100_SCORES_PATH, "w", encoding="utf-8") as f:
            json.dump(all_ce_scores, f)
        print(f"  Saved complete 100-candidate raw CE scores to '{RAW_CE100_SCORES_PATH}'.")

        del ce_model
        torch.cuda.empty_cache()

    # Total Cross-Encoder inference times:
    # Top 30 took 127.40s in Step B.
    # Candidates 31-100 took additional_scoring_time (or proportional at ~812 pairs/sec).
    time_ce_30 = 127.40  # actual time from Step B
    # Per-pair rate from Step B: 127.40s / 103,560 pairs = 1.2302 ms / pair
    # Total time for D=50 (172,600 pairs): 172,600 * 0.0012302 = ~212.3s
    # Total time for D=100 (345,200 pairs): 345,200 * 0.0012302 = ~424.7s
    if additional_scoring_time > 0:
        rate = additional_scoring_time / 241640.0
        time_ce_50 = time_ce_30 + (3452 * 20) * rate
        time_ce_100 = time_ce_30 + additional_scoring_time
    else:
        time_ce_50 = time_ce_30 * (50.0 / 30.0)
        time_ce_100 = time_ce_30 * (100.0 / 30.0)

    # 3. Build Reranked Outputs for D in {30, 50, 100}
    print("\n[3/5] Generating reranked candidate lists for depths D in {30, 50, 100}...")

    def build_reranked_run(depth: int):
        reranked_run = {}
        for qid in all_qids:
            dense_top100 = dense_results[qid]
            top_D = dense_top100[:depth]

            # Rerank top D by CE score
            scored_top_D = [
                [item[0], all_ce_scores[qid][item[0]]]
                for item in top_D
            ]
            scored_top_D.sort(key=lambda x: x[1], reverse=True)

            # Keep ranks after D in dense order
            tail = dense_top100[depth:100]
            reranked_run[qid] = scored_top_D + tail
        return reranked_run

    with open(CE30_RESULTS_PATH, "r", encoding="utf-8") as f:
        run_d30 = json.load(f)

    run_d50 = build_reranked_run(50)
    with open(CE50_RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(run_d50, f)
    print(f"  Saved D=50 results to '{CE50_RESULTS_PATH}'.")

    run_d100 = build_reranked_run(100)
    with open(CE100_RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(run_d100, f)
    print(f"  Saved D=100 results to '{CE100_RESULTS_PATH}'.")

    # 4. Evaluate Metrics
    print("\n[4/5] Evaluating metrics for each depth...")
    m_d30 = evaluate(run_d30, qrels_df)
    m_d50 = evaluate(run_d50, qrels_df)
    m_d100 = evaluate(run_d100, qrels_df)

    depth_times = {30: time_ce_30, 50: time_ce_50, 100: time_ce_100}
    depth_metrics = {30: m_d30, 50: m_d50, 100: m_d100}

    print("\n" + "=" * 95)
    print("      CROSS-ENCODER RERANK-DEPTH SENSITIVITY BENCHMARK (ALL 3,452 QUERIES)")
    print("=" * 95)
    header = f"{'Depth (D)':<12} | {'Pairs Scored':<14} | {'CE Time (s)':<14} | {'Recall@10':<10} | {'MRR@10':<8} | {'nDCG@10':<8}"
    print(header)
    print("-" * len(header))
    for d in [30, 50, 100]:
        pairs_cnt = 3452 * d
        t_sec = depth_times[d]
        m = depth_metrics[d]
        print(f"D = {d:<8} | {pairs_cnt:<14,} | {t_sec:<6.1f}s ({t_sec/60:.2f}m) | {m['Recall@10']:<10.4f} | {m['MRR@10']:<8.4f} | {m['nDCG@10']:<8.4f}")
    print("=" * len(header))

    # 5. Paired Bootstrap: D=50 vs D=30 and D=100 vs D=30
    print("\n[5/5] Running Paired Bootstrap (1,000 resamples, seed 42)...")
    pq_d30 = compute_per_query_metrics(run_d30, qrels_norm)
    pq_d50 = compute_per_query_metrics(run_d50, qrels_norm)
    pq_d100 = compute_per_query_metrics(run_d100, qrels_norm)

    bootstrap_comparisons = [
        ("D=50 vs D=30", pq_d50, pq_d30),
        ("D=100 vs D=30", pq_d100, pq_d30),
    ]

    print("\n" + "=" * 95)
    print("PAIRED BOOTSTRAP SIGNIFICANCE TESTING (1,000 RESAMPLES, SEED 42)")
    print("=" * 95)
    print(f"{'Comparison':<22} | {'Metric':<10} | {'Mean Diff':<10} | {'95% CI':<20} | {'P(Diff > 0)':<12} | {'Verdict'}")
    print("-" * 95)

    bootstrap_rows = []
    for comp_name, run_A, run_B in bootstrap_comparisons:
        for m in ["Recall@10", "MRR@10", "nDCG@10"]:
            vals_A = [run_A[qid][m] for qid in all_qids]
            vals_B = [run_B[qid][m] for qid in all_qids]

            mean_d, ci_l, ci_u, p_gt_0 = run_paired_bootstrap(vals_A, vals_B, n_resamples=1000, seed=RANDOM_SEED)

            if ci_l > 0:
                verdict = "Clear Improvement (p < 0.05)"
            elif ci_u < 0:
                verdict = "Clear Decline (p < 0.05)"
            else:
                verdict = "Within Noise (Not Significant)"

            ci_str = f"[{ci_l:+.4f}, {ci_u:+.4f}]"
            print(f"{comp_name:<22} | {m:<10} | {mean_d:<+10.4f} | {ci_str:<20} | {p_gt_0*100:<11.1f}% | {verdict}")

            bootstrap_rows.append({
                "comparison": comp_name,
                "metric": m,
                "mean_diff": mean_d,
                "ci_lower": ci_l,
                "ci_upper": ci_u,
                "p_above_zero": p_gt_0,
                "verdict": verdict,
            })
    print("=" * 95)

    # 6. Save depth_summary.csv
    summary_data = []
    for d in [30, 50, 100]:
        m = depth_metrics[d]
        summary_data.append({
            "depth": d,
            "pairs_scored": 3452 * d,
            "ce_time_sec": round(depth_times[d], 2),
            "ce_time_min": round(depth_times[d] / 60.0, 2),
            "Recall@10": m["Recall@10"],
            "Recall@100": m["Recall@100"],
            "MRR@10": m["MRR@10"],
            "nDCG@10": m["nDCG@10"],
        })

    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(DEPTH_SUMMARY_PATH, index=False)
    print(f"\nSaved depth sensitivity summary to '{DEPTH_SUMMARY_PATH}'.")

    print("\nEXTRA TASK COMPLETED SUCCESSFULLY!")


if __name__ == "__main__":
    main()
