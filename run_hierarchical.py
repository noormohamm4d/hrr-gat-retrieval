"""
Step 4: Hierarchical Retrieval
------------------------------
Implements two-stage hierarchical retrieval on the BEIR/NQ subset:
1. Article Representation:
   Builds one L2-normalized embedding per article (article = corpus title)
   by averaging the embeddings of all passages belonging to that article.
   Saves to embeddings/article_embeddings.npy and embeddings/article_titles.json.
2. Two-Stage Routing & Ranking:
   - Stage 1 (Article Routing): Scores all 3,758 articles by cosine similarity
     to the query and retrieves the top N articles.
   - Stage 2 (Passage Ranking): Ranks ONLY the passages belonging to those top N
     articles by cosine similarity to the query, returning the top 100 passages.
3. Tuning on Dev Split:
   Tests N in {3, 5, 10, 20, 50} on all 500 DEV queries.
   Reports Recall@10, Recall@100, MRR@10, nDCG@10, Routing Accuracy, and Avg Pool Size.
   Picks the best N based on dev MRR@10.
4. Test Evaluation & Saving:
   Evaluates the test split once using the best N selected on dev.
   Saves results/hier_dev.json and results/hier_test.json.
5. Benchmark Comparison:
   Prints comprehensive side-by-side comparison tables for Dev and Test splits
   comparing BM25, Flat Dense, and Hierarchical Retrieval.
"""

from collections import defaultdict
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
HIER_DEV_PATH = os.path.join(RESULTS_DIR, "hier_dev.json")
HIER_TEST_PATH = os.path.join(RESULTS_DIR, "hier_test.json")
BM25_DEV_PATH = os.path.join(RESULTS_DIR, "bm25_dev.json")
BM25_TEST_PATH = os.path.join(RESULTS_DIR, "bm25_test.json")
DENSE_DEV_PATH = os.path.join(RESULTS_DIR, "dense_dev.json")
DENSE_TEST_PATH = os.path.join(RESULTS_DIR, "dense_test.json")


def check_gpu():
    """Confirms GPU availability and prints hardware details."""
    cuda_avail = torch.cuda.is_available()
    print(f"torch.cuda.is_available(): {cuda_avail}")
    if not cuda_avail:
        raise RuntimeError("CUDA is not available. Dense query encoding requires GPU.")
    device_name = torch.cuda.get_device_name(0)
    print(f"CUDA Device Name: {device_name}")
    return device_name


def load_model():
    """Loads SentenceTransformer model on GPU for encoding queries."""
    print(f"\n[1/6] Loading encoder model '{MODEL_NAME}' onto GPU...")
    model = SentenceTransformer(MODEL_NAME, device="cuda")
    model.half()
    print("  Model loaded successfully.")
    return model


def get_or_create_article_embeddings(
    passage_embeddings: np.ndarray,
    corpus_titles: list[str]
):
    """
    Computes article embeddings as the L2-normalized mean of passage embeddings for each article title.
    Saves to embeddings/article_embeddings.npy and embeddings/article_titles.json.
    Loads existing files if already present.
    """
    os.makedirs(EMB_DIR, exist_ok=True)

    if os.path.exists(ARTICLE_EMB_PATH) and os.path.exists(ARTICLE_TITLES_PATH):
        print(f"\n[2/6] Loading existing article embeddings from '{ARTICLE_EMB_PATH}'...")
        article_embeddings = np.load(ARTICLE_EMB_PATH)
        with open(ARTICLE_TITLES_PATH, "r", encoding="utf-8") as f:
            article_titles = json.load(f)
        print(f"  Loaded {len(article_titles):,} article embeddings, shape: {article_embeddings.shape}")
        return article_embeddings, article_titles

    print(f"\n[2/6] Computing article embeddings from {len(corpus_titles):,} passages...")
    t0 = time.time()

    # Map each title to passage row indices
    title_to_indices = defaultdict(list)
    for idx, title in enumerate(corpus_titles):
        title_to_indices[title].append(idx)

    article_titles = list(title_to_indices.keys())
    num_articles = len(article_titles)
    dim = passage_embeddings.shape[1]
    article_embeddings = np.zeros((num_articles, dim), dtype=np.float32)

    for i, title in enumerate(article_titles):
        indices = title_to_indices[title]
        # Mean of passage embeddings for this article
        mean_vec = np.mean(passage_embeddings[indices], axis=0)
        # L2 normalize
        norm = np.linalg.norm(mean_vec)
        article_embeddings[i] = mean_vec / max(norm, 1e-12)

    elapsed = time.time() - t0
    print(f"  Computed {num_articles:,} article embeddings in {elapsed:.2f}s, shape: {article_embeddings.shape}")

    print(f"  Saving article embeddings to '{ARTICLE_EMB_PATH}'...")
    np.save(ARTICLE_EMB_PATH, article_embeddings)

    print(f"  Saving article titles to '{ARTICLE_TITLES_PATH}'...")
    with open(ARTICLE_TITLES_PATH, "w", encoding="utf-8") as f:
        json.dump(article_titles, f)

    return article_embeddings, article_titles


def build_mappings(corpus_ids: list[str], corpus_titles: list[str], qrels_df: pd.DataFrame):
    """
    Builds lookup structures:
    - title_to_passage_indices: dict mapping article title to list of passage indices
    - corpus_id_to_title: dict mapping corpus passage _id to its article title
    - query_gold_titles: dict mapping query_id to the set of gold passage article titles
    """
    title_to_passage_indices = defaultdict(list)
    corpus_id_to_title = {}
    for idx, (cid, title) in enumerate(zip(corpus_ids, corpus_titles)):
        title_to_passage_indices[title].append(idx)
        corpus_id_to_title[cid] = title

    # Build gold titles per query for routing accuracy computation
    positive_qrels = qrels_df[qrels_df["score"] > 0]
    query_gold_titles = defaultdict(set)
    for _, row in positive_qrels.iterrows():
        qid = str(row["query-id"])
        cid = str(row["corpus-id"])
        if cid in corpus_id_to_title:
            query_gold_titles[qid].add(corpus_id_to_title[cid])

    return title_to_passage_indices, query_gold_titles


def run_hierarchical_retrieval(
    query_embeddings: np.ndarray,
    query_ids: list[str],
    article_embeddings: np.ndarray,
    article_titles: list[str],
    passage_embeddings: np.ndarray,
    corpus_ids: list[str],
    title_to_passage_indices: dict[str, list[int]],
    query_gold_titles: dict[str, set[str]],
    N: int,
    k: int = 100,
):
    """
    Performs two-stage hierarchical retrieval for all given queries:
    1. Score all articles with cosine similarity: (num_queries x num_articles)
    2. Select top N articles per query.
    3. Rank ONLY the candidate passages from those N articles by cosine similarity.
    4. Compute routing accuracy and average candidate pool size.
    Returns:
        results: {qid: [[corpus_id, score], ... top 100]}
        routing_accuracy: float
        avg_pool_size: float
        elapsed: float
    """
    t0 = time.time()
    num_queries = len(query_ids)

    # 1. Cosine similarity against all articles (both are L2-normalized)
    article_sim_matrix = np.matmul(query_embeddings, article_embeddings.T)

    results = {}
    routing_hits = 0
    total_candidate_passages = 0

    for i in range(num_queries):
        qid = query_ids[i]
        q_emb = query_embeddings[i]
        q_art_scores = article_sim_matrix[i]

        # Top N article indices
        if len(q_art_scores) <= N:
            top_art_indices = np.argsort(-q_art_scores)
        else:
            top_part = np.argpartition(-q_art_scores, N)[:N]
            top_art_indices = top_part[np.argsort(-q_art_scores[top_part])]

        top_art_titles = [article_titles[idx] for idx in top_art_indices]

        # Routing accuracy check: is at least one gold passage's article in the top N articles?
        gold_titles = query_gold_titles.get(qid, set())
        if any(t in gold_titles for t in top_art_titles):
            routing_hits += 1

        # Gather passage indices belonging to the top N articles
        candidate_indices = []
        for t in top_art_titles:
            candidate_indices.extend(title_to_passage_indices[t])

        num_candidates = len(candidate_indices)
        total_candidate_passages += num_candidates

        if num_candidates == 0:
            results[qid] = []
            continue

        # Score candidate passages by cosine similarity to query
        cand_passage_embs = passage_embeddings[candidate_indices]
        passage_scores = np.dot(cand_passage_embs, q_emb)

        # Retrieve top k (or all if fewer than k)
        if num_candidates <= k:
            sort_order = np.argsort(-passage_scores)
            top_indices = sort_order
        else:
            part = np.argpartition(-passage_scores, k)[:k]
            top_indices = part[np.argsort(-passage_scores[part])]

        results[qid] = [
            [corpus_ids[candidate_indices[idx]], float(passage_scores[idx])]
            for idx in top_indices
        ]

    elapsed = time.time() - t0
    routing_accuracy = routing_hits / max(num_queries, 1)
    avg_pool_size = total_candidate_passages / max(num_queries, 1)

    return results, routing_accuracy, avg_pool_size, elapsed


def print_comparison_table(
    split_name: str,
    bm25_metrics: dict,
    dense_metrics: dict,
    hier_metrics: dict,
    hier_routing: float,
    hier_pool_size: float,
    best_N: int,
):
    """Prints a side-by-side comparison table of BM25, Flat Dense, and Hierarchical."""
    print(f"\n=========================================================================================================")
    print(f"                                   {split_name.upper()} RETRIEVAL BENCHMARK")
    print(f"=========================================================================================================")
    header = f"{'Model':<22} | {'Recall@10':<10} | {'Recall@100':<10} | {'MRR@10':<8} | {'nDCG@10':<8} | {'Routing Acc':<12} | {'Avg Pool Size':<14}"
    print(header)
    print("-" * len(header))

    # BM25 row
    print(
        f"{'BM25 (Baseline)':<22} | "
        f"{bm25_metrics['Recall@10']:<10.4f} | "
        f"{bm25_metrics['Recall@100']:<10.4f} | "
        f"{bm25_metrics['MRR@10']:<8.4f} | "
        f"{bm25_metrics['nDCG@10']:<8.4f} | "
        f"{'-':<12} | "
        f"{'99,991 (100%)':<14}"
    )

    # Flat Dense row
    print(
        f"{'Flat Dense (MiniLM)':<22} | "
        f"{dense_metrics['Recall@10']:<10.4f} | "
        f"{dense_metrics['Recall@100']:<10.4f} | "
        f"{dense_metrics['MRR@10']:<8.4f} | "
        f"{dense_metrics['nDCG@10']:<8.4f} | "
        f"{'-':<12} | "
        f"{'99,991 (100%)':<14}"
    )

    # Hierarchical row
    hier_label = f"Hierarchical (N={best_N})"
    print(
        f"{hier_label:<22} | "
        f"{hier_metrics['Recall@10']:<10.4f} | "
        f"{hier_metrics['Recall@100']:<10.4f} | "
        f"{hier_metrics['MRR@10']:<8.4f} | "
        f"{hier_metrics['nDCG@10']:<8.4f} | "
        f"{hier_routing*100:<11.2f}% | "
        f"{hier_pool_size:<14.1f}"
    )
    print("=" * len(header))


def main():
    print("=" * 70)
    print("STEP 4: HIERARCHICAL RETRIEVAL (Two-Stage Article -> Passage)")
    print("=" * 70)

    # 1. Hardware verification
    check_gpu()
    model = load_model()

    # 2. Load corpus data and saved passage embeddings
    print("\nLoading corpus and saved passage embeddings...")
    corpus_table = pq.read_table(CORPUS_PATH)
    corpus_ids = corpus_table["_id"].to_pylist()
    corpus_titles = corpus_table["title"].to_pylist()

    passage_embeddings = np.load(PASSAGE_EMB_PATH).astype(np.float32)
    with open(PASSAGE_IDS_PATH, "r", encoding="utf-8") as f:
        saved_ids = json.load(f)
    assert saved_ids == corpus_ids, "Passage IDs mismatch with corpus table!"
    print(f"  Loaded passage embeddings shape: {passage_embeddings.shape}")

    # 3. Load queries and qrels
    queries_df = pq.read_table(QUERIES_PATH).to_pandas()
    dev_queries = queries_df[queries_df["split"] == "dev"].reset_index(drop=True)
    test_queries = queries_df[queries_df["split"] == "test"].reset_index(drop=True)
    print(f"  Loaded {len(dev_queries):,} dev queries and {len(test_queries):,} test queries.")

    qrels_df = pq.read_table(QRELS_PATH).to_pandas()
    print(f"  Loaded {len(qrels_df):,} qrels.")

    # 4. Article Embeddings & Mappings
    article_embeddings, article_titles = get_or_create_article_embeddings(passage_embeddings, corpus_titles)
    title_to_passage_indices, query_gold_titles = build_mappings(corpus_ids, corpus_titles, qrels_df)

    # 5. Encode Queries (normalized)
    print("\n[3/6] Encoding Dev and Test queries...")
    dev_query_texts = dev_queries["text"].tolist()
    dev_query_ids = dev_queries["_id"].astype(str).tolist()
    dev_query_embs = model.encode(
        dev_query_texts,
        batch_size=64,
        normalize_embeddings=True,
        show_progress_bar=False,
        convert_to_numpy=True
    ).astype(np.float32)

    test_query_texts = test_queries["text"].tolist()
    test_query_ids = test_queries["_id"].astype(str).tolist()
    test_query_embs = model.encode(
        test_query_texts,
        batch_size=64,
        normalize_embeddings=True,
        show_progress_bar=False,
        convert_to_numpy=True
    ).astype(np.float32)
    print("  Queries encoded successfully.")

    # 6. Tuning N in {3, 5, 10, 20, 50} on DEV split
    print("\n[4/6] Tuning parameter N in {3, 5, 10, 20, 50} on DEV queries (500 queries)...")
    N_candidates = [3, 5, 10, 20, 50]
    dev_tuning_results = {}
    best_N = None
    best_dev_mrr = -1.0

    print(f"\n{'N':<4} | {'Recall@10':<10} | {'Recall@100':<10} | {'MRR@10':<8} | {'nDCG@10':<8} | {'Routing Acc':<12} | {'Avg Pool Size':<14} | {'Time (s)':<8}")
    print("-" * 88)

    for N in N_candidates:
        dev_res, dev_routing, dev_pool_size, elapsed = run_hierarchical_retrieval(
            dev_query_embs,
            dev_query_ids,
            article_embeddings,
            article_titles,
            passage_embeddings,
            corpus_ids,
            title_to_passage_indices,
            query_gold_titles,
            N=N,
            k=100,
        )
        metrics = evaluate(dev_res, qrels_df)
        dev_tuning_results[N] = {
            "results": dev_res,
            "metrics": metrics,
            "routing_acc": dev_routing,
            "avg_pool_size": dev_pool_size,
        }

        mrr = metrics["MRR@10"]
        if mrr > best_dev_mrr:
            best_dev_mrr = mrr
            best_N = N

        print(
            f"{N:<4} | "
            f"{metrics['Recall@10']:<10.4f} | "
            f"{metrics['Recall@100']:<10.4f} | "
            f"{metrics['MRR@10']:<8.4f} | "
            f"{metrics['nDCG@10']:<8.4f} | "
            f"{dev_routing*100:<11.2f}% | "
            f"{dev_pool_size:<14.1f} | "
            f"{elapsed:<8.2f}"
        )

    print("-" * 88)
    print(f"--> Best N selected based on Dev MRR@10: N = {best_N} (MRR@10 = {best_dev_mrr:.4f})")

    # 7. Evaluate on TEST split with best N
    print(f"\n[5/6] Evaluating on TEST split (500 queries) with best N = {best_N}...")
    test_res, test_routing, test_pool_size, test_elapsed = run_hierarchical_retrieval(
        test_query_embs,
        test_query_ids,
        article_embeddings,
        article_titles,
        passage_embeddings,
        corpus_ids,
        title_to_passage_indices,
        query_gold_titles,
        N=best_N,
        k=100,
    )
    test_metrics = evaluate(test_res, qrels_df)
    print(f"  Test run finished in {test_elapsed:.2f}s ({test_elapsed/len(test_queries)*1000:.1f}ms/query).")
    print(f"  Test Routing Accuracy: {test_routing*100:.2f}%")
    print(f"  Test Avg Pool Size   : {test_pool_size:.1f} passages")

    # 8. Save results for best N
    os.makedirs(RESULTS_DIR, exist_ok=True)
    best_dev_res = dev_tuning_results[best_N]["results"]
    best_dev_metrics = dev_tuning_results[best_N]["metrics"]
    best_dev_routing = dev_tuning_results[best_N]["routing_acc"]
    best_dev_pool_size = dev_tuning_results[best_N]["avg_pool_size"]

    with open(HIER_DEV_PATH, "w", encoding="utf-8") as f:
        json.dump(best_dev_res, f)
    print(f"\n  Saved best dev hierarchical results to '{HIER_DEV_PATH}'.")

    with open(HIER_TEST_PATH, "w", encoding="utf-8") as f:
        json.dump(test_res, f)
    print(f"  Saved best test hierarchical results to '{HIER_TEST_PATH}'.")

    # 9. Load BM25 and Flat Dense for benchmark comparison table
    print("\n[6/6] Generating benchmark comparison tables...")
    with open(BM25_DEV_PATH, "r", encoding="utf-8") as f:
        bm25_dev_metrics = evaluate(json.load(f), qrels_df)
    with open(BM25_TEST_PATH, "r", encoding="utf-8") as f:
        bm25_test_metrics = evaluate(json.load(f), qrels_df)

    with open(DENSE_DEV_PATH, "r", encoding="utf-8") as f:
        dense_dev_metrics = evaluate(json.load(f), qrels_df)
    with open(DENSE_TEST_PATH, "r", encoding="utf-8") as f:
        dense_test_metrics = evaluate(json.load(f), qrels_df)

    print_comparison_table(
        "Dev Split (500 queries)",
        bm25_dev_metrics,
        dense_dev_metrics,
        best_dev_metrics,
        best_dev_routing,
        best_dev_pool_size,
        best_N,
    )

    print_comparison_table(
        "Test Split (500 queries)",
        bm25_test_metrics,
        dense_test_metrics,
        test_metrics,
        test_routing,
        test_pool_size,
        best_N,
    )

    print("=" * 70)
    print("STEP 4: HIERARCHICAL RETRIEVAL COMPLETED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    main()
