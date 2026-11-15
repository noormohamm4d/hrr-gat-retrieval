"""
Scale-Up Step B: Dense Retrieval & Cross-Encoder Reranking
---------------------------------------------------------
1. Passage & Article Embeddings:
   - Encodes all 199,990 passages using multi-qa-MiniLM-L6-cos-v1 (fp16, normalized) on GPU.
   - Runs 1,000-passage timing benchmark and checks safety thresholds (< 30 min, GPU active).
   - Saves scale_up/embeddings/passage_embeddings.npy and scale_up/embeddings/passage_ids.json.
   - Computes L2-normalized mean article embeddings and saves
     scale_up/embeddings/article_embeddings.npy and scale_up/embeddings/article_titles.json.
2. Flat Dense Retrieval:
   - Builds FAISS flat inner-product index on CPU over normalized passage embeddings.
   - Encodes all 3,452 queries (fp16, normalized) and retrieves top-100 per query.
   - Saves scale_up/results/dense.json.
3. Cross-Encoder Reranking:
   - Loads cross-encoder/ms-marco-MiniLM-L-6-v2 on GPU.
   - Benchmarks throughput on a small sample of query-passage pairs.
   - Scores the top 30 flat-dense candidates per query (3,452 x 30 = 103,560 pairs).
   - Reranks top 30 by cross-encoder score descending; keeps ranks 31-100 in original dense order.
   - Saves:
     - scale_up/results/ce.json (top 100 with reranked top 30)
     - scale_up/results/raw_ce_scores.json (raw cross-encoder scores for all top 30 candidates)
4. Evaluation:
   - Evaluates Recall@10, Recall@100, MRR@10, and nDCG@10 on all 3,452 queries.
   - Displays side-by-side benchmark table comparing Scale-Up results with Study 1 Test numbers.
"""

from collections import defaultdict
import json
import os
import sys
import time

import faiss
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from sentence_transformers import CrossEncoder, SentenceTransformer

# Automatically resolve project root (D:\hrr-project) and scale_up directory
SCALE_UP_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCALE_UP_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from eval.metrics import evaluate

DATA_DIR = os.path.join(SCALE_UP_DIR, "data")
CORPUS_PATH = os.path.join(DATA_DIR, "corpus.parquet")
QUERIES_PATH = os.path.join(DATA_DIR, "queries.parquet")
QRELS_PATH = os.path.join(DATA_DIR, "qrels.parquet")

EMB_DIR = os.path.join(SCALE_UP_DIR, "embeddings")
PASSAGE_EMB_PATH = os.path.join(EMB_DIR, "passage_embeddings.npy")
PASSAGE_IDS_PATH = os.path.join(EMB_DIR, "passage_ids.json")
ARTICLE_EMB_PATH = os.path.join(EMB_DIR, "article_embeddings.npy")
ARTICLE_TITLES_PATH = os.path.join(EMB_DIR, "article_titles.json")

RESULTS_DIR = os.path.join(SCALE_UP_DIR, "results")
DENSE_RESULTS_PATH = os.path.join(RESULTS_DIR, "dense.json")
CE_RESULTS_PATH = os.path.join(RESULTS_DIR, "ce.json")
RAW_CE_SCORES_PATH = os.path.join(RESULTS_DIR, "raw_ce_scores.json")

BI_ENCODER_NAME = "sentence-transformers/multi-qa-MiniLM-L6-cos-v1"
CROSS_ENCODER_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def check_gpu():
    """Confirms GPU availability and device name."""
    cuda_avail = torch.cuda.is_available()
    print(f"torch.cuda.is_available(): {cuda_avail}")
    if not cuda_avail:
        raise RuntimeError("CUDA is not available. Embedding requires GPU.")
    device_name = torch.cuda.get_device_name(0)
    print(f"CUDA Device Name: {device_name}")
    return device_name


def get_or_compute_passage_embeddings(model, passages: list[str], corpus_ids: list[str]):
    """Embeds all passages or loads from disk if already present."""
    os.makedirs(EMB_DIR, exist_ok=True)

    if os.path.exists(PASSAGE_EMB_PATH) and os.path.exists(PASSAGE_IDS_PATH):
        print(f"\n[Embeddings] Found existing passage embeddings at '{PASSAGE_EMB_PATH}'.")
        embeddings = np.load(PASSAGE_EMB_PATH)
        with open(PASSAGE_IDS_PATH, "r", encoding="utf-8") as f:
            saved_ids = json.load(f)
        print(f"  Loaded {len(embeddings):,} embeddings from disk, shape: {embeddings.shape}")
        return embeddings, saved_ids

    total_passages = len(passages)
    print(f"\n[Timing Benchmark] Embedding first 1,000 passages out of {total_passages:,}...")
    sample_passages = passages[:1000]

    batch_size = 64
    try:
        t0 = time.time()
        sample_embs = model.encode(
            sample_passages,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        sample_time = time.time() - t0
    except torch.cuda.OutOfMemoryError:
        print("  WARNING: OutOfMemory with batch_size=64. Falling back to batch_size=32...")
        torch.cuda.empty_cache()
        batch_size = 32
        t0 = time.time()
        sample_embs = model.encode(
            sample_passages,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        sample_time = time.time() - t0

    est_total_seconds = (sample_time / 1000.0) * total_passages
    est_total_minutes = est_total_seconds / 60.0
    throughput = 1000.0 / sample_time
    print(f"  1,000 passages encoded in: {sample_time:.2f}s ({throughput:.1f} passages/sec, batch_size={batch_size}).")
    print(f"  Estimated time for all {total_passages:,} passages: {est_total_seconds:.1f}s ({est_total_minutes:.2f} min).")

    # Safety checks
    if not torch.cuda.is_available():
        raise RuntimeError("GPU is not being used! Halting execution.")
    if est_total_minutes > 30.0:
        raise RuntimeError(f"Estimated encoding time ({est_total_minutes:.2f} min) exceeds 30-minute threshold! Halting.")

    print(f"\nEmbedding all {total_passages:,} passages with progress bar (batch_size={batch_size})...")
    t_start = time.time()
    embeddings = model.encode(
        passages,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
        convert_to_numpy=True,
    )
    total_time = time.time() - t_start
    print(f"  Finished embedding {total_passages:,} passages in {total_time:.2f}s ({total_time / 60:.2f} min).")

    embeddings_f32 = embeddings.astype(np.float32)

    print(f"  Saving embeddings to '{PASSAGE_EMB_PATH}'...")
    np.save(PASSAGE_EMB_PATH, embeddings_f32)

    print(f"  Saving passage IDs to '{PASSAGE_IDS_PATH}'...")
    with open(PASSAGE_IDS_PATH, "w", encoding="utf-8") as f:
        json.dump(corpus_ids, f)

    return embeddings_f32, corpus_ids


def get_or_compute_article_embeddings(passage_embeddings: np.ndarray, corpus_titles: list[str]):
    """Computes article embeddings as L2-normalized mean of constituent passage embeddings."""
    if os.path.exists(ARTICLE_EMB_PATH) and os.path.exists(ARTICLE_TITLES_PATH):
        print(f"\n[Article Embeddings] Loading existing article embeddings from '{ARTICLE_EMB_PATH}'...")
        article_embeddings = np.load(ARTICLE_EMB_PATH)
        with open(ARTICLE_TITLES_PATH, "r", encoding="utf-8") as f:
            article_titles = json.load(f)
        print(f"  Loaded {len(article_titles):,} article embeddings, shape: {article_embeddings.shape}")
        return article_embeddings, article_titles

    print(f"\n[Article Embeddings] Computing article embeddings from {len(corpus_titles):,} passages...")
    t0 = time.time()
    title_to_indices = defaultdict(list)
    for idx, title in enumerate(corpus_titles):
        title_to_indices[title].append(idx)

    article_titles = list(title_to_indices.keys())
    num_articles = len(article_titles)
    dim = passage_embeddings.shape[1]
    article_embeddings = np.zeros((num_articles, dim), dtype=np.float32)

    for i, title in enumerate(article_titles):
        indices = title_to_indices[title]
        mean_vec = np.mean(passage_embeddings[indices], axis=0)
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


def run_flat_dense_retrieval(model, passage_embeddings: np.ndarray, corpus_ids: list[str], queries_df: pd.DataFrame, k: int = 100):
    """Encodes queries and searches FAISS flat inner-product index for top 100 passages."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    if os.path.exists(DENSE_RESULTS_PATH):
        print(f"\n[Flat Dense] Loading existing dense results from '{DENSE_RESULTS_PATH}'...")
        with open(DENSE_RESULTS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)

    print("\n[Flat Dense] Building FAISS IndexFlatIP (CPU)...")
    t0 = time.time()
    dim = passage_embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(np.ascontiguousarray(passage_embeddings, dtype=np.float32))
    print(f"  FAISS index built with {index.ntotal:,} vectors in {time.time() - t0:.2f}s.")

    query_texts = queries_df["text"].tolist()
    query_ids = queries_df["_id"].astype(str).tolist()

    print(f"  Encoding {len(query_texts):,} queries on GPU (batch_size=64)...")
    t_q0 = time.time()
    q_embeddings = model.encode(
        query_texts,
        batch_size=64,
        normalize_embeddings=True,
        show_progress_bar=True,
        convert_to_numpy=True,
    ).astype(np.float32)
    print(f"  Queries encoded in {time.time() - t_q0:.2f}s.")

    print(f"  Searching top {k} passages for all {len(query_ids):,} queries...")
    t_search = time.time()
    scores, indices = index.search(np.ascontiguousarray(q_embeddings), k)
    print(f"  Search completed in {time.time() - t_search:.2f}s.")

    results = {}
    for i, qid in enumerate(query_ids):
        q_scores = scores[i]
        q_indices = indices[i]
        results[qid] = [
            [corpus_ids[idx], float(q_scores[j])]
            for j, idx in enumerate(q_indices)
        ]

    with open(DENSE_RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f)
    print(f"  Saved dense results to '{DENSE_RESULTS_PATH}'.")

    return results


def run_cross_encoder_reranking(
    ce_model: CrossEncoder,
    queries_df: pd.DataFrame,
    dense_results: dict,
    corpus_lookup: dict,
    k_rerank: int = 30,
    k_total: int = 100,
):
    """
    Reranks top 30 flat dense candidates with CrossEncoder and preserves ranks 31-100.
    Saves ce.json and raw_ce_scores.json.
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)
    if os.path.exists(CE_RESULTS_PATH) and os.path.exists(RAW_CE_SCORES_PATH):
        print(f"\n[Cross-Encoder] Loading existing results from '{CE_RESULTS_PATH}'...")
        with open(CE_RESULTS_PATH, "r", encoding="utf-8") as f:
            ce_results = json.load(f)
        with open(RAW_CE_SCORES_PATH, "r", encoding="utf-8") as f:
            raw_scores = json.load(f)
        return ce_results, raw_scores

    print(f"\n[Cross-Encoder] Preparing query-passage pairs for top {k_rerank} candidates...")
    all_pairs = []
    query_slices = []
    current_offset = 0

    for _, row in queries_df.iterrows():
        qid = str(row["_id"])
        q_text = row["text"]
        dense_top = dense_results[qid]
        top_candidates = dense_top[:k_rerank]
        num_cand = len(top_candidates)

        for item in top_candidates:
            cid = item[0]
            p_info = corpus_lookup[cid]
            p_text = f"{p_info['title']} {p_info['text']}"
            all_pairs.append((q_text, p_text))

        query_slices.append((qid, current_offset, current_offset + num_cand, dense_top))
        current_offset += num_cand

    total_pairs = len(all_pairs)
    print(f"  Total pairs to score: {total_pairs:,} ({len(queries_df):,} queries x {k_rerank} candidates).")

    # Small timing benchmark on first 500 pairs
    print("  Benchmarking CrossEncoder on first 500 pairs...")
    t_bench = time.time()
    _ = ce_model.predict(all_pairs[:500], batch_size=128, show_progress_bar=False)
    bench_time = time.time() - t_bench
    est_total_ce = (total_pairs / 500.0) * bench_time
    print(f"  500 pairs scored in {bench_time:.2f}s ({500.0 / bench_time:.1f} pairs/sec).")
    print(f"  Estimated time for all {total_pairs:,} pairs: {est_total_ce:.1f}s ({est_total_ce / 60:.2f} min).")

    print(f"\nScoring all {total_pairs:,} pairs with CrossEncoder on GPU (batch_size=128)...")
    t_score_start = time.time()
    all_scores = ce_model.predict(all_pairs, batch_size=128, show_progress_bar=True)
    total_score_time = time.time() - t_score_start
    print(f"  Scoring completed in {total_score_time:.2f}s ({total_score_time / 60:.2f} min).")

    ce_results = {}
    raw_ce_scores = {}

    for qid, start_idx, end_idx, original_top100 in query_slices:
        slice_scores = all_scores[start_idx:end_idx]
        cand_slice = original_top100[:k_rerank]

        # Save raw scores mapping for the top 30
        raw_ce_scores[qid] = {
            cand_slice[idx][0]: float(slice_scores[idx])
            for idx in range(len(cand_slice))
        }

        # Sort top 30 candidates by cross-encoder score descending
        sort_order = np.argsort(-slice_scores)
        reranked_top = [
            [cand_slice[idx][0], float(slice_scores[idx])]
            for idx in sort_order
        ]

        # Append candidates 31-100 in their original dense order
        tail = original_top100[k_rerank:k_total]
        ce_results[qid] = reranked_top + tail

    print(f"  Saving reranked results to '{CE_RESULTS_PATH}'...")
    with open(CE_RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(ce_results, f)

    print(f"  Saving raw cross-encoder scores to '{RAW_CE_SCORES_PATH}'...")
    with open(RAW_CE_SCORES_PATH, "w", encoding="utf-8") as f:
        json.dump(raw_ce_scores, f)

    return ce_results, raw_ce_scores


def main():
    print("=" * 80)
    print("SCALE-UP STEP B: DENSE RETRIEVAL & CROSS-ENCODER RERANKING")
    print("=" * 80)

    # 1. Hardware Check
    check_gpu()

    # 2. Load Corpus and Queries
    print("\n[1/4] Loading scale-up corpus, queries, and qrels...")
    corpus_table = pq.read_table(CORPUS_PATH)
    corpus_ids = corpus_table["_id"].to_pylist()
    corpus_titles = corpus_table["title"].to_pylist()
    corpus_texts = corpus_table["text"].to_pylist()
    corpus_lookup = {
        cid: {"title": t, "text": txt}
        for cid, t, txt in zip(corpus_ids, corpus_titles, corpus_texts)
    }
    passages = [f"{t} {txt}" for t, txt in zip(corpus_titles, corpus_texts)]
    print(f"  Loaded {len(passages):,} passages from '{CORPUS_PATH}'.")

    queries_df = pq.read_table(QUERIES_PATH).to_pandas()
    print(f"  Loaded {len(queries_df):,} queries from '{QUERIES_PATH}'.")

    qrels_df = pq.read_table(QRELS_PATH).to_pandas()
    print(f"  Loaded {len(qrels_df):,} positive qrels from '{QRELS_PATH}'.")

    # 3. Embed Passages & Articles
    print("\n[2/4] Loading bi-encoder model on GPU...")
    bi_encoder = SentenceTransformer(BI_ENCODER_NAME, device="cuda")
    bi_encoder.half()
    print(f"  Bi-encoder '{BI_ENCODER_NAME}' loaded in fp16.")

    passage_embs, passage_ids = get_or_compute_passage_embeddings(bi_encoder, passages, corpus_ids)
    assert passage_ids == corpus_ids, "Passage IDs mismatch!"
    assert passage_embs.shape == (199990, 384), f"Unexpected passage embeddings shape: {passage_embs.shape}"

    article_embs, article_titles = get_or_compute_article_embeddings(passage_embs, corpus_titles)
    assert article_embs.shape == (6850, 384), f"Unexpected article embeddings shape: {article_embs.shape}"

    # 4. Flat Dense Retrieval
    print("\n[3/4] Running flat dense retrieval on all queries...")
    dense_results = run_flat_dense_retrieval(bi_encoder, passage_embs, corpus_ids, queries_df, k=100)

    # Free bi-encoder VRAM before loading cross-encoder
    del bi_encoder
    torch.cuda.empty_cache()

    # 5. Cross-Encoder Reranking
    print("\n[4/4] Loading cross-encoder model on GPU...")
    ce_model = CrossEncoder(CROSS_ENCODER_NAME, device="cuda")
    print(f"  Cross-encoder '{CROSS_ENCODER_NAME}' loaded on {ce_model.model.device}.")

    ce_results, raw_ce_scores = run_cross_encoder_reranking(
        ce_model, queries_df, dense_results, corpus_lookup, k_rerank=30, k_total=100
    )

    # 6. Evaluation and Benchmark Comparison Table
    print("\nEvaluating retrieval metrics on all 3,452 scale-up queries...")
    scale_up_dense_metrics = evaluate(dense_results, qrels_df)
    scale_up_ce_metrics = evaluate(ce_results, qrels_df)

    # Study 1 Test numbers from results/summary.csv for comparison
    study1_dense = {"Recall@10": 0.8240, "Recall@100": 0.9847, "MRR@10": 0.5361, "nDCG@10": 0.5958}
    study1_ce = {"Recall@10": 0.9033, "Recall@100": 0.9847, "MRR@10": 0.6663, "nDCG@10": 0.7164}

    print("\n" + "=" * 105)
    print("                     SCALE-UP (200K CORPUS, 3,452 QUERIES) vs STUDY 1 TEST BENCHMARK")
    print("=" * 105)
    header = f"{'System':<30} | {'Corpus':<10} | {'Queries':<8} | {'Recall@10':<10} | {'Recall@100':<10} | {'MRR@10':<8} | {'nDCG@10':<8}"
    print(header)
    print("-" * len(header))

    # Dense Rows
    print(
        f"{'Flat Dense (Scale-Up)':<30} | {'199,990':<10} | {'3,452':<8} | "
        f"{scale_up_dense_metrics['Recall@10']:<10.4f} | {scale_up_dense_metrics['Recall@100']:<10.4f} | "
        f"{scale_up_dense_metrics['MRR@10']:<8.4f} | {scale_up_dense_metrics['nDCG@10']:<8.4f}"
    )
    print(
        f"{'Flat Dense (Study 1 Test)':<30} | {'99,991':<10} | {'500':<8} | "
        f"{study1_dense['Recall@10']:<10.4f} | {study1_dense['Recall@100']:<10.4f} | "
        f"{study1_dense['MRR@10']:<8.4f} | {study1_dense['nDCG@10']:<8.4f}"
    )
    print("-" * len(header))

    # Cross-Encoder Rows
    print(
        f"{'Dense + Cross-Encoder (Scale-Up)':<30} | {'199,990':<10} | {'3,452':<8} | "
        f"{scale_up_ce_metrics['Recall@10']:<10.4f} | {scale_up_ce_metrics['Recall@100']:<10.4f} | "
        f"{scale_up_ce_metrics['MRR@10']:<8.4f} | {scale_up_ce_metrics['nDCG@10']:<8.4f}"
    )
    print(
        f"{'Dense + Cross-Encoder (Study 1 Test)':<30} | {'99,991':<10} | {'500':<8} | "
        f"{study1_ce['Recall@10']:<10.4f} | {study1_ce['Recall@100']:<10.4f} | "
        f"{study1_ce['MRR@10']:<8.4f} | {study1_ce['nDCG@10']:<8.4f}"
    )
    print("=" * len(header))

    print("\nSTEP B COMPLETED SUCCESSFULLY!")


if __name__ == "__main__":
    main()
