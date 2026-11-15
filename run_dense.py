"""
Step 3: Dense Retrieval with sentence-transformers and FAISS
-----------------------------------------------------------
Implements dense retrieval using sentence-transformers/multi-qa-MiniLM-L6-cos-v1:
1. Loads model on GPU in fp16. Verifies CUDA availability and device name.
2. Times embedding on 1,000 passages and estimates time for all 99,991 passages.
   Safety check: stops if GPU is not available or estimated time exceeds 30 minutes.
3. Embeds all passages with batch size 64 (fallback to 32 if OOM) and progress bar.
   Saves embeddings/passage_embeddings.npy and embeddings/passage_ids.json.
   Loads existing embeddings on subsequent runs to prevent recomputation.
4. Confirms embeddings shape (99,991 x 384) and matches ID count to row count.
5. Builds a FAISS flat inner-product index (IndexFlatIP) on CPU.
6. Encodes queries and retrieves top 100 passages:
   - First runs a 50-query pilot on dev and prints metrics.
   - Then runs all 500 dev queries and saves results/dense_dev.json.
   - Runs all 500 test queries and saves results/dense_test.json.
7. Evaluates Recall@10, Recall@100, MRR@10, and nDCG@10 using eval.metrics.
8. Displays a comparison table side-by-side with BM25 baseline results.
"""

import json
import os
import sys
import time

import faiss
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

RESULTS_DIR = "results"
DENSE_DEV_PATH = os.path.join(RESULTS_DIR, "dense_dev.json")
DENSE_TEST_PATH = os.path.join(RESULTS_DIR, "dense_test.json")
BM25_DEV_PATH = os.path.join(RESULTS_DIR, "bm25_dev.json")
BM25_TEST_PATH = os.path.join(RESULTS_DIR, "bm25_test.json")


def check_gpu():
    """Confirms GPU availability and prints hardware details."""
    cuda_avail = torch.cuda.is_available()
    print(f"torch.cuda.is_available(): {cuda_avail}")
    if not cuda_avail:
        raise RuntimeError("CUDA is not available. Dense retrieval requires GPU.")
    device_name = torch.cuda.get_device_name(0)
    print(f"CUDA Device Name: {device_name}")
    return device_name


def load_model():
    """Loads SentenceTransformer model to GPU in fp16 mode."""
    print(f"\n[1/7] Loading encoder model '{MODEL_NAME}' onto GPU...")
    model = SentenceTransformer(MODEL_NAME, device="cuda")
    model.half()  # fp16 for faster inference and lower VRAM usage
    print("  Model loaded successfully and converted to fp16.")
    return model


def get_or_create_passage_embeddings(model, corpus_ids: list[str], passages: list[str]):
    """
    Computes passage embeddings or loads them if already saved.
    Includes 1,000-passage timing test and safety checks.
    """
    os.makedirs(EMB_DIR, exist_ok=True)

    if os.path.exists(PASSAGE_EMB_PATH) and os.path.exists(PASSAGE_IDS_PATH):
        print(f"\n[2/7] Found existing embeddings at '{PASSAGE_EMB_PATH}'.")
        print("  Loading saved embeddings and corpus IDs...")
        embeddings = np.load(PASSAGE_EMB_PATH)
        with open(PASSAGE_IDS_PATH, "r", encoding="utf-8") as f:
            saved_ids = json.load(f)
        print(f"  Loaded {len(embeddings):,} embeddings from disk.")
        return embeddings, saved_ids

    total_passages = len(passages)
    print(f"\n[2/7] Timing benchmark: embedding first 1,000 passages out of {total_passages:,}...")
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
    print(f"  1,000 passages encoded in: {sample_time:.2f} seconds.")
    print(f"  Throughput: {1000.0 / sample_time:.1f} passages/sec (batch_size={batch_size}).")
    print(f"  Estimated time for all {total_passages:,} passages: {est_total_seconds:.1f}s ({est_total_minutes:.2f} min).")

    # Safety check rule: stop if estimated time > 30 minutes or GPU not used
    if not torch.cuda.is_available():
        print("\nERROR: GPU is not being used. Stopping as per safety instructions.")
        sys.exit(1)
    if est_total_minutes > 30.0:
        print(f"\nERROR: Estimated time ({est_total_minutes:.2f} min) exceeds 30 minutes. Stopping as instructed.")
        sys.exit(1)

    print(f"\n[3/7] Embedding all {total_passages:,} passages with progress bar (batch_size={batch_size})...")
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

    # Ensure float32 for standard downstream compatibility and FAISS IndexFlatIP
    embeddings_f32 = embeddings.astype(np.float32)

    print(f"  Saving embeddings to '{PASSAGE_EMB_PATH}'...")
    np.save(PASSAGE_EMB_PATH, embeddings_f32)

    print(f"  Saving passage IDs to '{PASSAGE_IDS_PATH}'...")
    with open(PASSAGE_IDS_PATH, "w", encoding="utf-8") as f:
        json.dump(corpus_ids, f)

    print("  Embeddings and IDs successfully saved.")
    return embeddings_f32, corpus_ids


def build_faiss_index(embeddings: np.ndarray):
    """
    Builds a FAISS Flat Inner-Product index (faiss-cpu) over normalized embeddings.
    Since vectors are L2-normalized, inner product equals cosine similarity.
    """
    print("\n[4/7] Building FAISS IndexFlatIP (CPU)...")
    t0 = time.time()
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    # Ensure float32 and C-contiguous
    embeddings_f32 = np.ascontiguousarray(embeddings, dtype=np.float32)
    index.add(embeddings_f32)
    build_time = time.time() - t0
    print(f"  FAISS Index built with {index.ntotal:,} vectors (dim={dim}) in {build_time:.2f}s.")
    return index


def search_queries(
    model: SentenceTransformer,
    index: faiss.IndexFlatIP,
    corpus_ids: list[str],
    queries_df: pd.DataFrame,
    k: int = 100,
):
    """
    Encodes queries, retrieves top-k passages using FAISS, and returns:
    {query_id: [[corpus_id, score], ... top-k]}
    """
    t0 = time.time()
    query_texts = queries_df["text"].tolist()
    query_ids = queries_df["_id"].astype(str).tolist()

    # Encode queries (normalized)
    q_embeddings = model.encode(
        query_texts,
        batch_size=64,
        normalize_embeddings=True,
        show_progress_bar=False,
        convert_to_numpy=True,
    ).astype(np.float32)

    # Search FAISS index
    scores, indices = index.search(np.ascontiguousarray(q_embeddings), k)

    results = {}
    for i, qid in enumerate(query_ids):
        q_scores = scores[i]
        q_indices = indices[i]
        results[qid] = [
            [corpus_ids[idx], float(q_scores[j])]
            for j, idx in enumerate(q_indices)
        ]

    elapsed = time.time() - t0
    return results, elapsed


def print_comparison_table(bm25_metrics: dict, dense_metrics: dict, split_name: str):
    """Formats and prints side-by-side metric comparison table."""
    print(f"\n=======================================================")
    print(f"        {split_name.upper()} RETRIEVAL EVALUATION")
    print(f"=======================================================")
    print(f"{'Metric':<14} | {'BM25':<12} | {'Dense (MiniLM)':<14} | {'Delta':<10}")
    print(f"{'-'*14}-+-{'-'*12}-+-{'-'*14}-+-{'-'*10}")
    for metric in ["Recall@10", "Recall@100", "MRR@10", "nDCG@10"]:
        bm25_val = bm25_metrics.get(metric, 0.0)
        dense_val = dense_metrics.get(metric, 0.0)
        delta = dense_val - bm25_val
        delta_str = f"{delta:+.4f}"
        print(f"{metric:<14} | {bm25_val:<12.4f} | {dense_val:<14.4f} | {delta_str:<10}")
    print(f"=======================================================\n")


def main():
    print("=" * 70)
    print("STEP 3: DENSE RETRIEVAL (sentence-transformers + FAISS)")
    print("=" * 70)

    # 1. Hardware verification
    check_gpu()

    # 2. Load Model
    model = load_model()

    # 3. Load Data
    print("\nLoading corpus passages from 'data/corpus_subset.parquet'...")
    corpus_table = pq.read_table(CORPUS_PATH)
    corpus_ids = corpus_table["_id"].to_pylist()
    titles = corpus_table["title"].to_pylist()
    texts = corpus_table["text"].to_pylist()
    passages = [f"{t} {txt}" for t, txt in zip(titles, texts)]
    print(f"  Loaded {len(passages):,} passages.")

    queries_df = pq.read_table(QUERIES_PATH).to_pandas()
    dev_queries = queries_df[queries_df["split"] == "dev"].reset_index(drop=True)
    test_queries = queries_df[queries_df["split"] == "test"].reset_index(drop=True)
    print(f"  Loaded {len(dev_queries):,} dev queries and {len(test_queries):,} test queries.")

    qrels_df = pq.read_table(QRELS_PATH).to_pandas()
    print(f"  Loaded {len(qrels_df):,} qrels.")

    # 4. Embeddings (Compute or Load)
    embeddings, corpus_ids = get_or_create_passage_embeddings(model, corpus_ids, passages)

    # 5. Confirm array shape and ID count
    print("\n[Check Embeddings Array]")
    print(f"  Embeddings array shape: {embeddings.shape}")
    print(f"  Corpus IDs count: {len(corpus_ids):,}")
    assert embeddings.shape == (99991, 384), f"Expected shape (99991, 384), got {embeddings.shape}"
    assert len(corpus_ids) == embeddings.shape[0], f"ID count {len(corpus_ids)} != row count {embeddings.shape[0]}"
    print("  CONFIRMED: Shape is exactly 99,991 x 384 and number of IDs equals number of rows.")

    # 6. Build FAISS Index
    index = build_faiss_index(embeddings)

    # 7. 50-Query Pilot on Dev
    print("\n[5/7] Running 50-query pilot on dev...")
    pilot_queries = dev_queries.head(50)
    pilot_results, pilot_time = search_queries(model, index, corpus_ids, pilot_queries, k=100)
    pilot_metrics = evaluate(pilot_results, qrels_df)
    print(f"  50 Pilot queries processed in {pilot_time:.2f}s ({pilot_time/50*1000:.1f}ms/query).")
    print(f"  Pilot Recall@10 : {pilot_metrics['Recall@10']:.4f}")
    print(f"  Pilot Recall@100: {pilot_metrics['Recall@100']:.4f}")
    print(f"  Pilot MRR@10    : {pilot_metrics['MRR@10']:.4f}")
    print(f"  Pilot nDCG@10   : {pilot_metrics['nDCG@10']:.4f}")

    # 8. Full Dev Run (500 queries)
    print("\n[6/7] Running dense retrieval on all 500 dev queries...")
    dev_results, dev_time = search_queries(model, index, corpus_ids, dev_queries, k=100)
    dev_metrics = evaluate(dev_results, qrels_df)
    print(f"  Dev run finished in {dev_time:.2f}s ({dev_time/len(dev_queries)*1000:.1f}ms/query).")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(DENSE_DEV_PATH, "w", encoding="utf-8") as f:
        json.dump(dev_results, f)
    print(f"  Saved dev results to '{DENSE_DEV_PATH}'.")

    # 9. Full Test Run (500 queries)
    print("\n[7/7] Running dense retrieval on all 500 test queries...")
    test_results, test_time = search_queries(model, index, corpus_ids, test_queries, k=100)
    test_metrics = evaluate(test_results, qrels_df)
    print(f"  Test run finished in {test_time:.2f}s ({test_time/len(test_queries)*1000:.1f}ms/query).")

    with open(DENSE_TEST_PATH, "w", encoding="utf-8") as f:
        json.dump(test_results, f)
    print(f"  Saved test results to '{DENSE_TEST_PATH}'.")

    # 10. Load BM25 metrics for side-by-side comparison
    print("\nLoading BM25 results for side-by-side comparison...")
    with open(BM25_DEV_PATH, "r", encoding="utf-8") as f:
        bm25_dev_results = json.load(f)
    bm25_dev_metrics = evaluate(bm25_dev_results, qrels_df)

    with open(BM25_TEST_PATH, "r", encoding="utf-8") as f:
        bm25_test_results = json.load(f)
    bm25_test_metrics = evaluate(bm25_test_results, qrels_df)

    # Print comparison tables
    print_comparison_table(bm25_dev_metrics, dev_metrics, "Dev Split (500 queries)")
    print_comparison_table(bm25_test_metrics, test_metrics, "Test Split (500 queries)")

    print("=" * 70)
    print("STEP 3: DENSE RETRIEVAL COMPLETED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    main()
