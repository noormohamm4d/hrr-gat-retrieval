"""
Step 6: Analysis and Statistical Significance Testing
-----------------------------------------------------
1. Evaluates Soft Fusion (lambda=0.25) on TEST split once and saves results/fusion025_test.json
   and results/fusion025_dev.json.
2. Paired Bootstrap on TEST (1,000 resamples, seed 42):
   - Dense vs BM25
   - Hierarchical (N=20) vs Dense
   - Fusion (0.25) vs Dense
   Computes mean difference, 95% confidence intervals, and P(diff > 0) for Recall@10, MRR@10, and nDCG@10.
3. Error Analysis on TEST:
   (a) Routing vs ranking breakdown for queries where Dense succeeds in top 100 but Hierarchical fails.
   (b) 10 sample queries where Dense ranks first gold passage below 10.
   (c) Queries where all four systems fail to retrieve any gold passage in top 10.
4. Outputs:
   - results/summary.csv (4 systems x 4 metrics x Dev & Test)
   - results/test_comparison.png (Grouped bar chart for test split)
"""

import json
import math
import os
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from sentence_transformers import SentenceTransformer

# Current directory for eval.metrics
sys.path.insert(0, ".")
from eval.metrics import evaluate, _normalize_qrels, _normalize_ranked_list

CORPUS_PATH = "data/corpus_subset.parquet"
QUERIES_PATH = "data/queries_subset.parquet"
QRELS_PATH = "data/qrels_subset.parquet"

EMB_DIR = "embeddings"
PASSAGE_EMB_PATH = os.path.join(EMB_DIR, "passage_embeddings.npy")
PASSAGE_IDS_PATH = os.path.join(EMB_DIR, "passage_ids.json")
ARTICLE_EMB_PATH = os.path.join(EMB_DIR, "article_embeddings.npy")
ARTICLE_TITLES_PATH = os.path.join(EMB_DIR, "article_titles.json")

RESULTS_DIR = "results"
BM25_DEV_PATH = os.path.join(RESULTS_DIR, "bm25_dev.json")
BM25_TEST_PATH = os.path.join(RESULTS_DIR, "bm25_test.json")
DENSE_DEV_PATH = os.path.join(RESULTS_DIR, "dense_dev.json")
DENSE_TEST_PATH = os.path.join(RESULTS_DIR, "dense_test.json")
HIER_DEV_PATH = os.path.join(RESULTS_DIR, "hier_dev.json")
HIER_TEST_PATH = os.path.join(RESULTS_DIR, "hier_test.json")
FUSION025_DEV_PATH = os.path.join(RESULTS_DIR, "fusion025_dev.json")
FUSION025_TEST_PATH = os.path.join(RESULTS_DIR, "fusion025_test.json")

SUMMARY_CSV_PATH = os.path.join(RESULTS_DIR, "summary.csv")
CHART_PNG_PATH = os.path.join(RESULTS_DIR, "test_comparison.png")


def compute_per_query_metrics(run: dict, qrels_norm: dict) -> dict[str, dict[str, float]]:
    """
    Computes exact per-query IR metrics matching eval/metrics.py:
    Returns dict: {qid: {'Recall@10': float, 'Recall@100': float, 'MRR@10': float, 'nDCG@10': float}}
    """
    per_query = {}
    for qid, raw_ranked in run.items():
        qid_str = str(qid)
        gold_set = qrels_norm.get(qid_str, set())
        num_gold = len(gold_set)

        if num_gold == 0:
            per_query[qid_str] = {
                "Recall@10": 0.0,
                "Recall@100": 0.0,
                "MRR@10": 0.0,
                "nDCG@10": 0.0,
            }
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
    """
    Performs paired bootstrap test between systems A and B (A - B).
    Returns:
        mean_diff: float
        ci_lower: float (2.5 percentile)
        ci_upper: float (97.5 percentile)
        p_above_zero: float (fraction of resamples with diff > 0)
    """
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


def rank_top_k(scores_matrix: np.ndarray, corpus_ids: list[str], query_ids: list[str], k: int = 100):
    """Extracts top-k ranked passages for each query given an (N_queries x N_passages) score matrix."""
    results = {}
    num_queries = len(query_ids)
    for i in range(num_queries):
        qid = query_ids[i]
        q_scores = scores_matrix[i]
        if len(q_scores) <= k:
            top_indices = np.argsort(-q_scores)
        else:
            top_part = np.argpartition(-q_scores, k)[:k]
            top_indices = top_part[np.argsort(-q_scores[top_part])]

        results[qid] = [
            [corpus_ids[idx], float(q_scores[idx])]
            for idx in top_indices
        ]
    return results


def main():
    print("=" * 75)
    print("STEP 6: STATISTICAL ANALYSIS & ERROR PROFILING")
    print("=" * 75)

    # 1. Load Data & Embeddings
    print("\n[1/6] Loading corpus, queries, and qrels...")
    corpus_table = pq.read_table(CORPUS_PATH)
    corpus_ids = corpus_table["_id"].to_pylist()
    corpus_titles = corpus_table["title"].to_pylist()
    corpus_texts = corpus_table["text"].to_pylist()
    id_to_passage = {
        cid: {"title": t, "text": txt}
        for cid, t, txt in zip(corpus_ids, corpus_titles, corpus_texts)
    }

    queries_df = pq.read_table(QUERIES_PATH).to_pandas()
    dev_queries = queries_df[queries_df["split"] == "dev"].reset_index(drop=True)
    test_queries = queries_df[queries_df["split"] == "test"].reset_index(drop=True)
    qrels_df = pq.read_table(QRELS_PATH).to_pandas()
    qrels_norm = _normalize_qrels(qrels_df)

    passage_embs = np.load(PASSAGE_EMB_PATH).astype(np.float32)
    article_embs = np.load(ARTICLE_EMB_PATH).astype(np.float32)
    with open(ARTICLE_TITLES_PATH, "r", encoding="utf-8") as f:
        article_titles = json.load(f)

    title_to_article_idx = {title: idx for idx, title in enumerate(article_titles)}
    passage_article_idx = np.array(
        [title_to_article_idx[t] for t in corpus_titles], dtype=np.int32
    )

    # 2. Evaluate lambda = 0.25 ONCE on TEST (and DEV)
    print("\n[2/6] Evaluating Soft Fusion (lambda = 0.25, secondary) on TEST...")
    model = SentenceTransformer("sentence-transformers/multi-qa-MiniLM-L6-cos-v1", device="cuda")
    model.half()

    test_qids = test_queries["_id"].astype(str).tolist()
    test_q_embs = model.encode(
        test_queries["text"].tolist(),
        batch_size=64,
        normalize_embeddings=True,
        show_progress_bar=False,
        convert_to_numpy=True,
    ).astype(np.float32)

    test_p_cos = np.matmul(test_q_embs, passage_embs.T)
    test_a_cos = np.matmul(test_q_embs, article_embs.T)
    test_scores_025 = test_p_cos + 0.25 * test_a_cos[:, passage_article_idx]

    fusion025_test_results = rank_top_k(test_scores_025, corpus_ids, test_qids, k=100)
    with open(FUSION025_TEST_PATH, "w", encoding="utf-8") as f:
        json.dump(fusion025_test_results, f)
    print(f"  Saved secondary test fusion to '{FUSION025_TEST_PATH}'.")

    # Also compute/save dev for fusion025 if not already saved
    dev_qids = dev_queries["_id"].astype(str).tolist()
    dev_q_embs = model.encode(
        dev_queries["text"].tolist(),
        batch_size=64,
        normalize_embeddings=True,
        show_progress_bar=False,
        convert_to_numpy=True,
    ).astype(np.float32)
    dev_p_cos = np.matmul(dev_q_embs, passage_embs.T)
    dev_a_cos = np.matmul(dev_q_embs, article_embs.T)
    dev_scores_025 = dev_p_cos + 0.25 * dev_a_cos[:, passage_article_idx]
    fusion025_dev_results = rank_top_k(dev_scores_025, corpus_ids, dev_qids, k=100)
    with open(FUSION025_DEV_PATH, "w", encoding="utf-8") as f:
        json.dump(fusion025_dev_results, f)
    print(f"  Saved secondary dev fusion to '{FUSION025_DEV_PATH}'.")

    # 3. Load all 4 systems on TEST & DEV
    print("\n[3/6] Loading all 4 systems for evaluation...")
    with open(BM25_DEV_PATH, "r", encoding="utf-8") as f: bm25_dev = json.load(f)
    with open(BM25_TEST_PATH, "r", encoding="utf-8") as f: bm25_test = json.load(f)
    with open(DENSE_DEV_PATH, "r", encoding="utf-8") as f: dense_dev = json.load(f)
    with open(DENSE_TEST_PATH, "r", encoding="utf-8") as f: dense_test = json.load(f)
    with open(HIER_DEV_PATH, "r", encoding="utf-8") as f: hier_dev = json.load(f)
    with open(HIER_TEST_PATH, "r", encoding="utf-8") as f: hier_test = json.load(f)

    # Compute macro metrics
    systems_dev = {
        "BM25": evaluate(bm25_dev, qrels_df),
        "Flat Dense": evaluate(dense_dev, qrels_df),
        "Hard Hierarchical (N=20)": evaluate(hier_dev, qrels_df),
        "Soft Fusion (lambda=0.25)": evaluate(fusion025_dev_results, qrels_df),
    }
    systems_test = {
        "BM25": evaluate(bm25_test, qrels_df),
        "Flat Dense": evaluate(dense_test, qrels_df),
        "Hard Hierarchical (N=20)": evaluate(hier_test, qrels_df),
        "Soft Fusion (lambda=0.25)": evaluate(fusion025_test_results, qrels_df),
    }

    # 4. Save summary.csv
    summary_rows = []
    for split_name, sys_dict in [("dev", systems_dev), ("test", systems_test)]:
        for sys_name, m in sys_dict.items():
            summary_rows.append({
                "split": split_name,
                "system": sys_name,
                "Recall@10": m["Recall@10"],
                "Recall@100": m["Recall@100"],
                "MRR@10": m["MRR@10"],
                "nDCG@10": m["nDCG@10"],
            })
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(SUMMARY_CSV_PATH, index=False)
    print(f"  Saved benchmark metrics to '{SUMMARY_CSV_PATH}'.")

    # 5. Paired Bootstrap on TEST (1,000 resamples, seed 42)
    print("\n[4/6] Running Paired Bootstrap on TEST (1,000 resamples, seed 42)...")
    pq_bm25 = compute_per_query_metrics(bm25_test, qrels_norm)
    pq_dense = compute_per_query_metrics(dense_test, qrels_norm)
    pq_hier = compute_per_query_metrics(hier_test, qrels_norm)
    pq_fusion = compute_per_query_metrics(fusion025_test_results, qrels_norm)

    test_qids_ordered = [str(qid) for qid in test_qids]

    comparisons = [
        ("Dense vs BM25", pq_dense, pq_bm25, "Dense - BM25"),
        ("Hierarchical (N=20) vs Dense", pq_hier, pq_dense, "Hierarchical - Dense"),
        ("Fusion (lambda=0.25) vs Dense", pq_fusion, pq_dense, "Fusion(0.25) - Dense"),
    ]

    metrics_list = ["Recall@10", "MRR@10", "nDCG@10"]

    bootstrap_results = []
    print("\n" + "=" * 90)
    print("PAIRED BOOTSTRAP SIGNIFICANCE TESTING (TEST SPLIT, 1,000 RESAMPLES, SEED 42)")
    print("=" * 90)
    print(f"{'Comparison':<30} | {'Metric':<10} | {'Mean Diff':<10} | {'95% CI':<20} | {'P(Diff > 0)':<12} | {'Verdict'}")
    print("-" * 90)

    for comp_name, run_A, run_B, formula in comparisons:
        for metric in metrics_list:
            vals_A = [run_A[qid][metric] for qid in test_qids_ordered]
            vals_B = [run_B[qid][metric] for qid in test_qids_ordered]

            mean_d, ci_l, ci_u, p_gt_0 = run_paired_bootstrap(vals_A, vals_B, n_resamples=1000, seed=42)

            # Verdict in plain language:
            # If 0 is not in [ci_l, ci_u], the difference is statistically clear.
            if ci_l > 0:
                verdict = "Clear Improvement (p < 0.05)"
            elif ci_u < 0:
                verdict = "Clear Decline (p < 0.05)"
            else:
                verdict = "Within Noise (Not Significant)"

            ci_str = f"[{ci_l:+.4f}, {ci_u:+.4f}]"
            print(f"{comp_name:<30} | {metric:<10} | {mean_d:<+10.4f} | {ci_str:<20} | {p_gt_0*100:<11.1f}% | {verdict}")

            bootstrap_results.append({
                "Comparison": comp_name,
                "Metric": metric,
                "Mean_Diff": mean_d,
                "CI_Lower": ci_l,
                "CI_Upper": ci_u,
                "P_Above_Zero": p_gt_0,
                "Verdict": verdict,
            })
    print("=" * 90)

    # 6. Error Analysis on TEST
    print("\n[5/6] Performing Error Analysis on TEST split...")

    # (a) Among queries where Dense retrieves gold in top 100 but Hierarchical does not:
    # Count how many are routing misses (gold article not in top 20 articles).
    dense_hits_100 = set()
    hier_hits_100 = set()
    for qid in test_qids_ordered:
        gold = qrels_norm.get(qid, set())
        d_top100 = {cid for cid in _normalize_ranked_list(dense_test.get(qid, []))[:100]}
        h_top100 = {cid for cid in _normalize_ranked_list(hier_test.get(qid, []))[:100]}
        if bool(d_top100 & gold):
            dense_hits_100.add(qid)
        if bool(h_top100 & gold):
            hier_hits_100.add(qid)

    dense_win_hier_miss = dense_hits_100 - hier_hits_100

    # Check routing misses:
    # Compute top 20 articles for these test queries
    top20_art_by_qid = {}
    for i, qid in enumerate(test_qids):
        q_scores = test_a_cos[i]
        top_art_idx = np.argpartition(-q_scores, 20)[:20]
        top_art_idx = top_art_idx[np.argsort(-q_scores[top_art_idx])]
        top20_art_by_qid[qid] = {article_titles[idx] for idx in top_art_idx}

    routing_misses = 0
    ranking_misses = 0
    for qid in dense_win_hier_miss:
        gold_cids = qrels_norm.get(qid, set())
        gold_titles = {id_to_passage[cid]["title"] for cid in gold_cids if cid in id_to_passage}
        routed_articles = top20_art_by_qid.get(qid, set())
        if not bool(gold_titles & routed_articles):
            routing_misses += 1
        else:
            ranking_misses += 1

    print(f"\n--- Error Analysis (a): Dense Win vs Hierarchical Miss (Top 100) ---")
    print(f"  Queries where Dense finds gold in top 100 but Hierarchical misses: {len(dense_win_hier_miss)}")
    print(f"    - Routing misses (gold article not in top 20 articles): {routing_misses} ({routing_misses/max(len(dense_win_hier_miss), 1)*100:.1f}%)")
    print(f"    - Ranking misses (article routed, but passage ranked > 100): {ranking_misses} ({ranking_misses/max(len(dense_win_hier_miss), 1)*100:.1f}%)")

    # (b) 10 random queries where Flat Dense's first gold passage is ranked below 10:
    dense_fail_top10 = []
    for qid in test_qids_ordered:
        gold = qrels_norm.get(qid, set())
        top10_cids = _normalize_ranked_list(dense_test.get(qid, []))[:10]
        if not bool(set(top10_cids) & gold):
            dense_fail_top10.append(qid)

    print(f"\n--- Error Analysis (b): 10 Random Queries where Dense First Gold > Rank 10 ---")
    print(f"  Total queries where Flat Dense has no gold passage in top 10: {len(dense_fail_top10)} / 500")

    rng_sample = np.random.default_rng(42)
    sample_10_qids = rng_sample.choice(dense_fail_top10, size=10, replace=False)

    queries_by_id = dict(zip(queries_df["_id"].astype(str), queries_df["text"]))

    for idx, qid in enumerate(sample_10_qids, start=1):
        q_text = queries_by_id.get(qid, "N/A")
        gold_cids = qrels_norm.get(qid, set())
        gold_titles = list({id_to_passage[cid]["title"] for cid in gold_cids if cid in id_to_passage})
        dense_top3_cids = _normalize_ranked_list(dense_test.get(qid, []))[:3]
        retrieved_top3_titles = [id_to_passage[cid]["title"] for cid in dense_top3_cids if cid in id_to_passage]

        print(f"\n  [{idx}] Query ID: {qid}")
        print(f"      Query Text     : \"{q_text}\"")
        print(f"      Gold Article(s): {gold_titles}")
        print(f"      Top 3 Retrieved: {retrieved_top3_titles}")

    # (c) Queries where ALL systems fail to retrieve any gold passage in top 10:
    all_fail_count = 0
    all_fail_qids = []
    for qid in test_qids_ordered:
        gold = qrels_norm.get(qid, set())
        b_top10 = set(_normalize_ranked_list(bm25_test.get(qid, []))[:10])
        d_top10 = set(_normalize_ranked_list(dense_test.get(qid, []))[:10])
        h_top10 = set(_normalize_ranked_list(hier_test.get(qid, []))[:10])
        f_top10 = set(_normalize_ranked_list(fusion025_test_results.get(qid, []))[:10])

        if (not bool(b_top10 & gold)) and (not bool(d_top10 & gold)) and (not bool(h_top10 & gold)) and (not bool(f_top10 & gold)):
            all_fail_count += 1
            all_fail_qids.append(qid)

    print(f"\n--- Error Analysis (c): Complete Retrieval Failures in Top 10 ---")
    print(f"  Queries where ALL 4 systems fail in top 10: {all_fail_count} / 500 ({all_fail_count/500*100:.1f}%)")

    # 7. Generate Grouped Bar Chart
    print("\n[6/6] Generating grouped bar chart 'results/test_comparison.png'...")
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

    chart_metrics = ["Recall@10", "Recall@100", "MRR@10", "nDCG@10"]
    x = np.arange(len(chart_metrics))
    width = 0.20

    system_labels = [
        ("BM25", "#4C72B0"),
        ("Flat Dense", "#55A868"),
        ("Hard Hierarchical (N=20)", "#DD8452"),
        ("Soft Fusion (lambda=0.25)", "#8172B3"),
    ]

    for i, (sys_name, color) in enumerate(system_labels):
        offset = (i - 1.5) * width
        vals = [systems_test[sys_name][m] for m in chart_metrics]
        bars = ax.bar(x + offset, vals, width, label=sys_name, color=color, edgecolor="black", linewidth=0.6, alpha=0.9)
        for bar in bars:
            height = bar.get_height()
            ax.annotate(
                f"{height:.3f}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center", va="bottom",
                fontsize=8, fontweight="bold", rotation=0
            )

    ax.set_title("Test Set Retrieval Performance Across All Systems", fontsize=14, fontweight="bold", pad=15)
    ax.set_ylabel("Score", fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(chart_metrics, fontsize=11, fontweight="bold")
    ax.set_ylim(0.0, 1.10)
    ax.legend(frameon=True, facecolor="white", edgecolor="#cccccc", fontsize=10, loc="upper left")
    plt.tight_layout()

    fig.savefig(CHART_PNG_PATH, dpi=300)
    plt.close(fig)
    print(f"  Chart saved to '{CHART_PNG_PATH}'.")

    # Final summary display
    print("\n" + "=" * 80)
    print("FINAL SUMMARY OF ALL SYSTEMS (DEV & TEST)")
    print("=" * 80)
    print(summary_df.to_string(index=False))
    print("=" * 80)


if __name__ == "__main__":
    main()
