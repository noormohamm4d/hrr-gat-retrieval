"""
Step 5: Soft Hierarchical Fusion
--------------------------------
Implements soft hierarchical fusion combining passage and article representations:
1. Representation Combination:
   For each query:
   - passage_cosine: cosine similarity between query and passage embedding.
   - article_cosine: cosine similarity between query and the embedding of the article
     to which the passage belongs.
   - final_score(p) = passage_cosine(p) + lambda * article_cosine(article(p))
   - Returns top 100 passages per query.
2. Tuning on DEV:
   - Tests lambda in {0, 0.1, 0.25, 0.5, 1.0, 2.0} exclusively on the 500 dev queries.
   - Confirms that lambda = 0 reproduces Flat Dense dev results exactly.
   - Selects the best lambda based on dev MRR@10.
3. Test Split Evaluation:
   - Evaluates once on the 500 test queries using the best lambda chosen from dev.
   - Saves results/fusion_dev.json and results/fusion_test.json.
4. Benchmark Comparison:
   - Displays lambda sweep results on dev.
   - Displays full comparison table across BM25, Flat Dense, Hard Hierarchical (N=20),
     and Soft Fusion on both Dev and Test splits.
"""

import json
import os
import sys
import time

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from sentence_transformers import SentenceTransformer

# Current directory for eval.metrics
sys.path.insert(0, ".")
from eval.metrics import evaluate

MODEL_NAME = "sentence-transformers/multi-qa-MiniLM-L6-cos-v1"
CORPUS_PATH = "data/corpus_subset.parquet"
QUERIES_PATH = "data/queries_subset.parquet"
QRELS_PATH = "data/qrels_subset.parquet"

EMB_DIR = "embeddings"
PASSAGE_EMB_PATH = os.path.join(EMB_DIR, "passage_embeddings.npy")
PASSAGE_IDS_PATH = os.path.join(EMB_DIR, "passage_ids.json")
ARTICLE_EMB_PATH = os.path.join(EMB_DIR, "article_embeddings.npy")
ARTICLE_TITLES_PATH = os.path.join(EMB_DIR, "article_titles.json")

RESULTS_DIR = "results"
FUSION_DEV_PATH = os.path.join(RESULTS_DIR, "fusion_dev.json")
FUSION_TEST_PATH = os.path.join(RESULTS_DIR, "fusion_test.json")
BM25_DEV_PATH = os.path.join(RESULTS_DIR, "bm25_dev.json")
BM25_TEST_PATH = os.path.join(RESULTS_DIR, "bm25_test.json")
DENSE_DEV_PATH = os.path.join(RESULTS_DIR, "dense_dev.json")
DENSE_TEST_PATH = os.path.join(RESULTS_DIR, "dense_test.json")
HIER_DEV_PATH = os.path.join(RESULTS_DIR, "hier_dev.json")
HIER_TEST_PATH = os.path.join(RESULTS_DIR, "hier_test.json")


def check_gpu():
    """Confirms GPU availability and prints hardware details."""
    cuda_avail = torch.cuda.is_available()
    print(f"torch.cuda.is_available(): {cuda_avail}")
    if not cuda_avail:
        raise RuntimeError("CUDA is not available. Query encoding requires GPU.")
    device_name = torch.cuda.get_device_name(0)
    print(f"CUDA Device Name: {device_name}")
    return device_name


def load_model():
    """Loads SentenceTransformer model on GPU for query encoding."""
    print(f"\n[1/6] Loading encoder model '{MODEL_NAME}' onto GPU...")
    model = SentenceTransformer(MODEL_NAME, device="cuda")
    model.half()
    print("  Model loaded successfully.")
    return model


def rank_top_k(scores_matrix: np.ndarray, corpus_ids: list[str], query_ids: list[str], k: int = 100):
    """
    Extracts top-k ranked passages for each query given an (N_queries x N_passages) score matrix.
    Returns: {query_id: [[corpus_id, score], ... top-k]}
    """
    results = {}
    num_queries = len(query_ids)
    for i in range(num_queries):
        qid = query_ids[i]
        q_scores = scores_matrix[i]
        if len(q_scores) <= k:
            sort_order = np.argsort(-q_scores)
            top_indices = sort_order
        else:
            top_part = np.argpartition(-q_scores, k)[:k]
            top_indices = top_part[np.argsort(-q_scores[top_part])]

        results[qid] = [
            [corpus_ids[idx], float(q_scores[idx])]
            for idx in top_indices
        ]
    return results


def print_comparison_table(
    split_name: str,
    bm25_metrics: dict,
    dense_metrics: dict,
    hier_metrics: dict,
    fusion_metrics: dict,
    best_lambda: float,
):
    """Prints a comparison table for BM25, Flat Dense, Hard Hierarchical, and Soft Fusion."""
    print(f"\n=========================================================================================")
    print(f"                         {split_name.upper()} RETRIEVAL BENCHMARK")
    print(f"=========================================================================================")
    header = f"{'Model':<28} | {'Recall@10':<10} | {'Recall@100':<10} | {'MRR@10':<8} | {'nDCG@10':<8}"
    print(header)
    print("-" * len(header))

    # BM25
    print(
        f"{'BM25 (Baseline)':<28} | "
        f"{bm25_metrics['Recall@10']:<10.4f} | "
        f"{bm25_metrics['Recall@100']:<10.4f} | "
        f"{bm25_metrics['MRR@10']:<8.4f} | "
        f"{bm25_metrics['nDCG@10']:<8.4f}"
    )

    # Flat Dense
    print(
        f"{'Flat Dense (MiniLM)':<28} | "
        f"{dense_metrics['Recall@10']:<10.4f} | "
        f"{dense_metrics['Recall@100']:<10.4f} | "
        f"{dense_metrics['MRR@10']:<8.4f} | "
        f"{dense_metrics['nDCG@10']:<8.4f}"
    )

    # Hard Hierarchical (N=20)
    print(
        f"{'Hard Hierarchical (N=20)':<28} | "
        f"{hier_metrics['Recall@10']:<10.4f} | "
        f"{hier_metrics['Recall@100']:<10.4f} | "
        f"{hier_metrics['MRR@10']:<8.4f} | "
        f"{hier_metrics['nDCG@10']:<8.4f}"
    )

    # Soft Fusion
    fusion_label = f"Soft Fusion (lambda={best_lambda})"
    print(
        f"{fusion_label:<28} | "
        f"{fusion_metrics['Recall@10']:<10.4f} | "
        f"{fusion_metrics['Recall@100']:<10.4f} | "
        f"{fusion_metrics['MRR@10']:<8.4f} | "
        f"{fusion_metrics['nDCG@10']:<8.4f}"
    )
    print("=" * len(header))


def main():
    print("=" * 70)
    print("STEP 5: SOFT HIERARCHICAL FUSION (Passage + Lambda * Article)")
    print("=" * 70)

    # 1. Hardware verification & model loading
    check_gpu()
    model = load_model()

    # 2. Load corpus data and saved embeddings
    print("\n[2/6] Loading corpus and precomputed embeddings...")
    corpus_table = pq.read_table(CORPUS_PATH)
    corpus_ids = corpus_table["_id"].to_pylist()
    corpus_titles = corpus_table["title"].to_pylist()

    passage_embeddings = np.load(PASSAGE_EMB_PATH).astype(np.float32)
    with open(PASSAGE_IDS_PATH, "r", encoding="utf-8") as f:
        saved_passage_ids = json.load(f)
    assert saved_passage_ids == corpus_ids, "Passage IDs mismatch with corpus table!"

    article_embeddings = np.load(ARTICLE_EMB_PATH).astype(np.float32)
    with open(ARTICLE_TITLES_PATH, "r", encoding="utf-8") as f:
        article_titles = json.load(f)

    print(f"  Passage embeddings: {passage_embeddings.shape}")
    print(f"  Article embeddings: {article_embeddings.shape}")

    # Build mapping from each passage row to its article row index
    title_to_article_idx = {title: idx for idx, title in enumerate(article_titles)}
    passage_article_idx = np.array(
        [title_to_article_idx[t] for t in corpus_titles], dtype=np.int32
    )

    # 3. Load queries and qrels
    queries_df = pq.read_table(QUERIES_PATH).to_pandas()
    dev_queries = queries_df[queries_df["split"] == "dev"].reset_index(drop=True)
    test_queries = queries_df[queries_df["split"] == "test"].reset_index(drop=True)
    qrels_df = pq.read_table(QRELS_PATH).to_pandas()

    dev_query_ids = dev_queries["_id"].astype(str).tolist()
    test_query_ids = test_queries["_id"].astype(str).tolist()

    # 4. Encode Queries (normalized, float32)
    print("\n[3/6] Encoding Dev and Test queries...")
    dev_query_embs = model.encode(
        dev_queries["text"].tolist(),
        batch_size=64,
        normalize_embeddings=True,
        show_progress_bar=False,
        convert_to_numpy=True,
    ).astype(np.float32)

    test_query_embs = model.encode(
        test_queries["text"].tolist(),
        batch_size=64,
        normalize_embeddings=True,
        show_progress_bar=False,
        convert_to_numpy=True,
    ).astype(np.float32)
    print("  Queries encoded successfully.")

    # 5. Precompute Cosine Similarities on Dev
    print("\n[4/6] Precomputing Dev similarities (Passage Cosine & Article Cosine)...")
    t0 = time.time()
    dev_passage_cos = np.matmul(dev_query_embs, passage_embeddings.T)
    dev_article_cos = np.matmul(dev_query_embs, article_embeddings.T)
    # Broadcast article score to corresponding passages
    dev_p_art_cos = dev_article_cos[:, passage_article_idx]
    print(f"  Dev matrices computed in {time.time() - t0:.2f}s.")

    # 6. Parameter Sweep: lambda in {0, 0.1, 0.25, 0.5, 1.0, 2.0} on Dev
    print("\n[5/6] Tuning lambda in {0, 0.1, 0.25, 0.5, 1.0, 2.0} on DEV queries (500 queries)...")
    lambda_candidates = [0.0, 0.1, 0.25, 0.5, 1.0, 2.0]
    dev_sweep_results = {}
    best_lambda = None
    best_dev_mrr = -1.0

    print(f"\n{'lambda':<8} | {'Recall@10':<10} | {'Recall@100':<10} | {'MRR@10':<8} | {'nDCG@10':<8} | {'Ranking Time':<12}")
    print("-" * 66)

    for lam in lambda_candidates:
        t_lam = time.time()
        # Final score: passage_cosine + lambda * article_cosine
        if lam == 0.0:
            dev_final_scores = dev_passage_cos
        else:
            dev_final_scores = dev_passage_cos + lam * dev_p_art_cos

        lam_results = rank_top_k(dev_final_scores, corpus_ids, dev_query_ids, k=100)
        lam_metrics = evaluate(lam_results, qrels_df)
        elapsed_lam = time.time() - t_lam

        dev_sweep_results[lam] = {
            "results": lam_results,
            "metrics": lam_metrics,
        }

        mrr = lam_metrics["MRR@10"]
        if mrr > best_dev_mrr:
            best_dev_mrr = mrr
            best_lambda = lam

        print(
            f"{lam:<8.2f} | "
            f"{lam_metrics['Recall@10']:<10.4f} | "
            f"{lam_metrics['Recall@100']:<10.4f} | "
            f"{lam_metrics['MRR@10']:<8.4f} | "
            f"{lam_metrics['nDCG@10']:<8.4f} | "
            f"{elapsed_lam:<12.2f}s"
        )

    print("-" * 66)

    # Verification: lambda = 0 must reproduce flat dense dev results exactly
    print("\n--- Verifying lambda = 0 against Flat Dense Dev ---")
    with open(DENSE_DEV_PATH, "r", encoding="utf-8") as f:
        dense_dev_results = json.load(f)
    dense_dev_metrics = evaluate(dense_dev_results, qrels_df)
    lambda0_metrics = dev_sweep_results[0.0]["metrics"]

    print(f"  lambda = 0.00 metrics: {lambda0_metrics}")
    print(f"  Flat Dense metrics   : {dense_dev_metrics}")
    exact_match = (
        round(lambda0_metrics["Recall@10"], 6) == round(dense_dev_metrics["Recall@10"], 6)
        and round(lambda0_metrics["Recall@100"], 6) == round(dense_dev_metrics["Recall@100"], 6)
        and round(lambda0_metrics["MRR@10"], 6) == round(dense_dev_metrics["MRR@10"], 6)
        and round(lambda0_metrics["nDCG@10"], 6) == round(dense_dev_metrics["nDCG@10"], 6)
    )
    print(f"  CHECK: lambda = 0 reproduces flat dense dev results exactly -> {exact_match}")
    assert exact_match, "Error: lambda=0 does not match flat dense dev results!"

    print(f"\n--> Best lambda selected based on Dev MRR@10: lambda = {best_lambda} (MRR@10 = {best_dev_mrr:.4f})")

    # 7. Evaluate on TEST split ONCE with best lambda
    print(f"\n[6/6] Evaluating on TEST split (500 queries) with best lambda = {best_lambda}...")
    t_test_0 = time.time()
    test_passage_cos = np.matmul(test_query_embs, passage_embeddings.T)
    test_article_cos = np.matmul(test_query_embs, article_embeddings.T)
    test_p_art_cos = test_article_cos[:, passage_article_idx]

    if best_lambda == 0.0:
        test_final_scores = test_passage_cos
    else:
        test_final_scores = test_passage_cos + best_lambda * test_p_art_cos

    test_results = rank_top_k(test_final_scores, corpus_ids, test_query_ids, k=100)
    test_metrics = evaluate(test_results, qrels_df)
    test_elapsed = time.time() - t_test_0
    print(f"  Test evaluation completed in {test_elapsed:.2f}s ({test_elapsed/len(test_queries)*1000:.1f}ms/query).")

    # 8. Save results/fusion_dev.json and results/fusion_test.json
    os.makedirs(RESULTS_DIR, exist_ok=True)
    best_dev_results = dev_sweep_results[best_lambda]["results"]
    best_dev_metrics = dev_sweep_results[best_lambda]["metrics"]

    with open(FUSION_DEV_PATH, "w", encoding="utf-8") as f:
        json.dump(best_dev_results, f)
    print(f"  Saved dev fusion results to '{FUSION_DEV_PATH}'.")

    with open(FUSION_TEST_PATH, "w", encoding="utf-8") as f:
        json.dump(test_results, f)
    print(f"  Saved test fusion results to '{FUSION_TEST_PATH}'.")

    # 9. Load baselines for full comparison table
    with open(BM25_DEV_PATH, "r", encoding="utf-8") as f:
        bm25_dev_metrics = evaluate(json.load(f), qrels_df)
    with open(BM25_TEST_PATH, "r", encoding="utf-8") as f:
        bm25_test_metrics = evaluate(json.load(f), qrels_df)

    with open(DENSE_TEST_PATH, "r", encoding="utf-8") as f:
        dense_test_metrics = evaluate(json.load(f), qrels_df)

    with open(HIER_DEV_PATH, "r", encoding="utf-8") as f:
        hier_dev_metrics = evaluate(json.load(f), qrels_df)
    with open(HIER_TEST_PATH, "r", encoding="utf-8") as f:
        hier_test_metrics = evaluate(json.load(f), qrels_df)

    # Print comparison tables
    print_comparison_table(
        "Dev Split (500 queries)",
        bm25_dev_metrics,
        dense_dev_metrics,
        hier_dev_metrics,
        best_dev_metrics,
        best_lambda,
    )

    print_comparison_table(
        "Test Split (500 queries)",
        bm25_test_metrics,
        dense_test_metrics,
        hier_test_metrics,
        test_metrics,
        best_lambda,
    )

    print("=" * 70)
    print("STEP 5: SOFT HIERARCHICAL FUSION COMPLETED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    main()
