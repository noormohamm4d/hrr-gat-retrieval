"""
Scale-Up Step C: 5-Fold Cross-Validated GAT Reranker & Article Aggregation
--------------------------------------------------------------------------
1. Recall@30 Ceiling:
   Computes and reports Recall@30 of flat-dense candidates across all 3,452 queries.
2. Cross-Validation (5 Folds):
   For each fold k in {0, 1, 2, 3, 4}:
   - Test set: fold k (from queries.parquet).
   - Training & Validation: remaining 4 folds split 85% train / 15% val by group (GroupShuffleSplit, seed 42).
   - Baseline Aggregation: tunes beta in {0, 0.1, 0.25, 0.5, 1.0} on the validation split by MRR@10.
   - GAT & Control Training:
     * Architecture: 2 layers, 4 heads, hidden_dim=256, dropout=0.1.
     * Features (771-d): q*p (384), |q-p| (384), dense_score (1), ce_score (1), article_cosine (1).
     * Edges: same-article + top-5 KNN + self-loops.
     * Control: self-loops only.
     * Final score = CE score + correction (last layer zero-initialized).
     * Loss: listwise softmax cross-entropy over 30 candidates.
     * Optimizer: Adam lr=1e-3, batch=32, up to 30 epochs, early stopping on val MRR@10.
   - Timing Safety Check: times Fold 0 and confirms 5-fold estimated duration is under 90 minutes.
   - Predicts test fold k ONCE for each model.
3. Pooled Evaluation & Bootstrap:
   - Pools test predictions across all 5 folds so every query is evaluated out-of-fold.
   - Reports Recall@10, MRR@10, nDCG@10 for all systems.
   - Runs paired bootstrap on pooled queries (1,000 resamples, seed 42) for Recall@10, MRR@10, nDCG@10.
4. Outputs:
   - scale_up/results/cv_summary.csv
   - scale_up/results/pooled_dense.json, pooled_ce.json, pooled_agg.json, pooled_gat.json, pooled_gat_noedges.json
"""

from collections import defaultdict
import json
import math
import os
import sys
import time

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.model_selection import GroupShuffleSplit
import torch
import torch.nn as nn
import torch.nn.functional as F

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

EMB_DIR = os.path.join(SCALE_UP_DIR, "embeddings")
PASSAGE_EMB_PATH = os.path.join(EMB_DIR, "passage_embeddings.npy")
ARTICLE_EMB_PATH = os.path.join(EMB_DIR, "article_embeddings.npy")
ARTICLE_TITLES_PATH = os.path.join(EMB_DIR, "article_titles.json")

RESULTS_DIR = os.path.join(SCALE_UP_DIR, "results")
DENSE_RESULTS_PATH = os.path.join(RESULTS_DIR, "dense.json")
CE_RESULTS_PATH = os.path.join(RESULTS_DIR, "ce.json")
RAW_CE_SCORES_PATH = os.path.join(RESULTS_DIR, "raw_ce_scores.json")

CV_SUMMARY_PATH = os.path.join(RESULTS_DIR, "cv_summary.csv")
POOLED_DENSE_PATH = os.path.join(RESULTS_DIR, "pooled_dense.json")
POOLED_CE_PATH = os.path.join(RESULTS_DIR, "pooled_ce.json")
POOLED_AGG_PATH = os.path.join(RESULTS_DIR, "pooled_agg.json")
POOLED_GAT_PATH = os.path.join(RESULTS_DIR, "pooled_gat.json")
POOLED_NOEDGES_PATH = os.path.join(RESULTS_DIR, "pooled_gat_noedges.json")

RANDOM_SEED = 42


def check_gpu():
    """Confirms GPU availability and prints hardware details."""
    cuda_avail = torch.cuda.is_available()
    print(f"torch.cuda.is_available(): {cuda_avail}")
    if not cuda_avail:
        raise RuntimeError("CUDA is not available. GAT training requires GPU.")
    device_name = torch.cuda.get_device_name(0)
    print(f"CUDA Device Name: {device_name}")
    return "cuda" if cuda_avail else "cpu"


# =========================================================================
# GAT ARCHITECTURE IN PLAIN PYTORCH (DENSE ADJACENCY MASK)
# =========================================================================

class DenseGraphAttentionLayer(nn.Module):
    def __init__(self, in_features: int, out_features: int, num_heads: int = 4, dropout: float = 0.1, alpha: float = 0.2):
        super().__init__()
        self.num_heads = num_heads
        self.out_features = out_features
        self.head_dim = out_features // num_heads
        assert out_features % num_heads == 0, "out_features must be divisible by num_heads"

        self.W = nn.Linear(in_features, out_features, bias=False)
        self.a_src = nn.Parameter(torch.empty(1, num_heads, 1, self.head_dim))
        self.a_dst = nn.Parameter(torch.empty(1, num_heads, 1, self.head_dim))
        self.leaky_relu = nn.LeakyReLU(alpha)
        self.dropout = nn.Dropout(dropout)

        nn.init.xavier_uniform_(self.W.weight)
        nn.init.xavier_uniform_(self.a_src)
        nn.init.xavier_uniform_(self.a_dst)

    def forward(self, h: torch.Tensor, adj_mask: torch.Tensor) -> torch.Tensor:
        B, N, _ = h.shape
        Wh = self.W(h).view(B, N, self.num_heads, self.head_dim)
        Wh = Wh.transpose(1, 2)

        attn_src = (Wh * self.a_src).sum(dim=-1, keepdim=True)
        attn_dst = (Wh * self.a_dst).sum(dim=-1, keepdim=True)

        scores = attn_src + attn_dst.transpose(-2, -1)
        scores = self.leaky_relu(scores)

        mask = (adj_mask.unsqueeze(1) == 0)
        scores = scores.masked_fill(mask, -1e9)

        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, Wh)
        out = out.transpose(1, 2).contiguous().view(B, N, self.out_features)
        return out


class GATReranker(nn.Module):
    def __init__(self, in_features: int = 771, hidden_dim: int = 256, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.gat1 = DenseGraphAttentionLayer(in_features, hidden_dim, num_heads=num_heads, dropout=dropout)
        self.gat2 = DenseGraphAttentionLayer(hidden_dim, hidden_dim, num_heads=num_heads, dropout=dropout)
        self.dropout = nn.Dropout(dropout)

        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, x: torch.Tensor, adj: torch.Tensor, ce_scores: torch.Tensor) -> torch.Tensor:
        h1 = F.elu(self.gat1(x, adj))
        h1 = self.dropout(h1)
        h2 = F.elu(self.gat2(h1, adj))
        h2 = self.dropout(h2)

        correction = self.mlp(h2).squeeze(-1)
        final_scores = ce_scores + correction
        return final_scores, correction


def listwise_cross_entropy_loss(final_scores: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    log_probs = F.log_softmax(final_scores, dim=-1)
    num_positives = targets.sum(dim=-1, keepdim=True)
    valid_mask = (num_positives.squeeze(-1) > 0)

    if not valid_mask.any():
        return torch.tensor(0.0, device=final_scores.device, requires_grad=True)

    target_dist = targets / num_positives.clamp(min=1e-12)
    per_query_loss = -(target_dist * log_probs).sum(dim=-1)
    return per_query_loss[valid_mask].mean()


# =========================================================================
# DATA PREPARATION & GRAPH BUILDER
# =========================================================================

def build_all_graph_tensors(
    queries_df: pd.DataFrame,
    dense_run: dict,
    raw_ce_scores: dict,
    query_embeddings: np.ndarray,
    passage_embeddings: np.ndarray,
    article_embeddings: np.ndarray,
    cid_to_row_idx: dict[str, int],
    cid_to_title: dict[str, str],
    title_to_article_idx: dict[str, int],
    qrels_norm: dict,
    k_rerank: int = 30,
    device: str = "cuda"
):
    """Builds node features, adjacency matrices, CE scores, and targets for all 3,452 queries."""
    num_queries = len(queries_df)
    dim = query_embeddings.shape[1]
    feat_dim = dim * 2 + 3

    print(f"  Building graph tensors for {num_queries:,} queries (feat_dim={feat_dim})...")
    X = np.zeros((num_queries, k_rerank, feat_dim), dtype=np.float32)
    A_full = np.zeros((num_queries, k_rerank, k_rerank), dtype=np.float32)
    A_noedges = np.zeros((num_queries, k_rerank, k_rerank), dtype=np.float32)
    CE_matrix = np.zeros((num_queries, k_rerank), dtype=np.float32)
    Y = np.zeros((num_queries, k_rerank), dtype=np.float32)
    cand_cids_list = []

    for i, (_, row) in enumerate(queries_df.iterrows()):
        qid = str(row["_id"])
        q_emb = query_embeddings[i]
        gold_cids = qrels_norm.get(qid, set())

        dense_top = dense_run[qid][:k_rerank]
        cids = [item[0] for item in dense_top]
        dense_scores = [float(item[1]) for item in dense_top]
        cand_cids_list.append(cids)

        ce_dict = raw_ce_scores.get(qid, {})
        ce_scores = [ce_dict.get(cid, 0.0) for cid in cids]

        p_row_indices = [cid_to_row_idx[cid] for cid in cids]
        P_embs = passage_embeddings[p_row_indices]

        p_titles = [cid_to_title[cid] for cid in cids]
        p_art_indices = [title_to_article_idx[t] for t in p_titles]
        A_embs = article_embeddings[p_art_indices]
        art_cos_scores = np.dot(A_embs, q_emb)

        # Node features
        X[i, :, 0:dim] = P_embs * q_emb
        X[i, :, dim:dim*2] = np.abs(P_embs - q_emb)
        X[i, :, dim*2] = dense_scores
        X[i, :, dim*2 + 1] = ce_scores
        X[i, :, dim*2 + 2] = art_cos_scores

        CE_matrix[i, :] = ce_scores

        for j, cid in enumerate(cids):
            if cid in gold_cids:
                Y[i, j] = 1.0

        # Adjacency
        np.fill_diagonal(A_full[i], 1.0)
        np.fill_diagonal(A_noedges[i], 1.0)

        for j1 in range(k_rerank):
            for j2 in range(j1 + 1, k_rerank):
                if p_titles[j1] == p_titles[j2]:
                    A_full[i, j1, j2] = 1.0
                    A_full[i, j2, j1] = 1.0

        pairwise_sim = np.matmul(P_embs, P_embs.T)
        np.fill_diagonal(pairwise_sim, -np.inf)
        for j1 in range(k_rerank):
            top5_neighbors = np.argpartition(-pairwise_sim[j1], 5)[:5]
            for nb in top5_neighbors:
                A_full[i, j1, nb] = 1.0
                A_full[i, nb, j1] = 1.0

    return (
        torch.tensor(X, dtype=torch.float32, device=device),
        torch.tensor(A_full, dtype=torch.float32, device=device),
        torch.tensor(A_noedges, dtype=torch.float32, device=device),
        torch.tensor(CE_matrix, dtype=torch.float32, device=device),
        torch.tensor(Y, dtype=torch.float32, device=device),
        cand_cids_list
    )


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


# =========================================================================
# MAIN CV PIPELINE
# =========================================================================

def main():
    print("=" * 85)
    print("SCALE-UP STEP C: 5-FOLD CROSS-VALIDATED GAT RERANKER & ARTICLE AGGREGATION")
    print("=" * 85)

    device = check_gpu()

    # 1. Load Data
    print("\n[1/6] Loading corpus, queries, and qrels...")
    corpus_table = pq.read_table(CORPUS_PATH)
    corpus_ids = corpus_table["_id"].to_pylist()
    corpus_titles = corpus_table["title"].to_pylist()
    cid_to_row_idx = {cid: idx for idx, cid in enumerate(corpus_ids)}
    cid_to_title = dict(zip(corpus_ids, corpus_titles))

    queries_df = pq.read_table(QUERIES_PATH).to_pandas()
    num_queries = len(queries_df)
    print(f"  Loaded {num_queries:,} queries.")

    qrels_df = pq.read_table(QRELS_PATH).to_pandas()
    qrels_norm = _normalize_qrels(qrels_df)

    with open(DENSE_RESULTS_PATH, "r", encoding="utf-8") as f:
        dense_results = json.load(f)
    with open(CE_RESULTS_PATH, "r", encoding="utf-8") as f:
        ce_results = json.load(f)
    with open(RAW_CE_SCORES_PATH, "r", encoding="utf-8") as f:
        raw_ce_scores = json.load(f)

    # 2. Check Recall@30 Ceiling
    print("\n[2/6] Calculating Recall@30 Ceiling of Flat Dense candidates...")
    rec_30_list = []
    for qid, ranked in dense_results.items():
        gold = qrels_norm.get(str(qid), set())
        if not gold:
            continue
        top30 = set(_normalize_ranked_list(ranked)[:30])
        rec_30_list.append(len(top30 & gold) / len(gold))

    recall_30_ceiling = float(np.mean(rec_30_list))
    print(f"  --> Flat Dense Recall@30 Ceiling over all {num_queries:,} queries: {recall_30_ceiling:.4f} ({recall_30_ceiling*100:.2f}%)")
    print("      (This is the theoretical maximum Recall@10 reachable by reranking the top 30 candidates).")

    # 3. Load Embeddings & Encode Queries
    print("\n[3/6] Loading precomputed embeddings & encoding queries on GPU...")
    passage_embs = np.load(PASSAGE_EMB_PATH).astype(np.float32)
    article_embs = np.load(ARTICLE_EMB_PATH).astype(np.float32)
    with open(ARTICLE_TITLES_PATH, "r", encoding="utf-8") as f:
        article_titles = json.load(f)
    title_to_article_idx = {title: idx for idx, title in enumerate(article_titles)}

    from sentence_transformers import SentenceTransformer
    bi_enc = SentenceTransformer("sentence-transformers/multi-qa-MiniLM-L6-cos-v1", device="cuda")
    bi_enc.half()
    query_embs = bi_enc.encode(
        queries_df["text"].tolist(),
        batch_size=64,
        normalize_embeddings=True,
        show_progress_bar=False,
        convert_to_numpy=True
    ).astype(np.float32)
    del bi_enc
    torch.cuda.empty_cache()

    # 4. Construct Graph Tensors for all 3,452 Queries
    print("\n[4/6] Precomputing graph representations for all queries...")
    X_all, A_full_all, A_noedges_all, CE_all, Y_all, cand_cids_all = build_all_graph_tensors(
        queries_df, dense_results, raw_ce_scores, query_embs, passage_embs, article_embs,
        cid_to_row_idx, cid_to_title, title_to_article_idx, qrels_norm, device=device
    )

    # 5. Cross-Validation Execution (5 Folds)
    print("\n[5/6] Starting 5-Fold Cross-Validation...")
    folds = sorted(queries_df["fold"].unique())
    assert len(folds) == 5, f"Expected 5 folds, found {len(folds)}"

    fold_metrics_records = []
    pooled_test_dense = {}
    pooled_test_ce = {}
    pooled_test_agg = {}
    pooled_test_gat = {}
    pooled_test_noedges = {}

    def train_cv_model(model_name: str, x_tr, adj_tr, ce_tr, y_tr, x_vl, adj_vl, ce_vl, y_vl, val_qids, cand_vl, seed=42):
        torch.manual_seed(seed)
        model = GATReranker(in_features=771, hidden_dim=256, num_heads=4, dropout=0.1).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

        batch_size = 32
        n_train = x_tr.shape[0]
        best_val_mrr = -1.0
        best_weights = None
        best_epoch = -1

        for epoch in range(1, 31):
            model.train()
            perm = torch.randperm(n_train, device=device)
            for start in range(0, n_train, batch_size):
                b_idx = perm[start:start+batch_size]
                optimizer.zero_grad()
                pred_scores, _ = model(x_tr[b_idx], adj_tr[b_idx], ce_tr[b_idx])
                loss = listwise_cross_entropy_loss(pred_scores, y_tr[b_idx])
                if torch.isnan(loss) or loss.item() == 0.0:
                    continue
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            # Validation evaluation
            model.eval()
            with torch.no_grad():
                val_scores, _ = model(x_vl, adj_vl, ce_vl)
                val_scores_np = val_scores.cpu().numpy()

            val_run = {}
            for i, qid in enumerate(val_qids):
                sort_idx = np.argsort(-val_scores_np[i])
                top_reranked = [[cand_vl[i][idx], float(val_scores_np[i][idx])] for idx in sort_idx]
                val_run[qid] = top_reranked + dense_results[qid][30:100]

            val_m = evaluate(val_run, qrels_df)
            v_mrr = val_m["MRR@10"]

            if v_mrr > best_val_mrr:
                best_val_mrr = v_mrr
                best_epoch = epoch
                best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        model.load_state_dict(best_weights)
        return model, best_val_mrr, best_epoch

    def predict_test_fold(model, x_te, adj_te, ce_te, test_qids, cand_te):
        model.eval()
        with torch.no_grad():
            pred_scores, _ = model(x_te, adj_te, ce_te)
            pred_scores_np = pred_scores.cpu().numpy()

        test_run = {}
        for i, qid in enumerate(test_qids):
            sort_idx = np.argsort(-pred_scores_np[i])
            top_reranked = [[cand_te[i][idx], float(pred_scores_np[i][idx])] for idx in sort_idx]
            test_run[qid] = top_reranked + dense_results[qid][30:100]
        return test_run

    total_cv_start_time = time.time()

    for fold_k in folds:
        print(f"\n" + "-" * 75)
        print(f"FOLD {fold_k} / 4")
        print("-" * 75)
        t_fold_start = time.time()

        test_mask = (queries_df["fold"] == fold_k).values
        train_val_mask = ~test_mask

        test_df = queries_df[test_mask].reset_index(drop=True)
        test_indices = np.where(test_mask)[0]
        test_qids = [str(qid) for qid in test_df["_id"]]

        # Split remaining 4 folds by group into 85% train / 15% val
        other_df = queries_df[train_val_mask].reset_index(drop=True)
        other_indices = np.where(train_val_mask)[0]
        other_groups = other_df["group"].values

        gss = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=RANDOM_SEED)
        tr_rel_idx, vl_rel_idx = next(gss.split(other_df, groups=other_groups))

        train_indices = other_indices[tr_rel_idx]
        val_indices = other_indices[vl_rel_idx]

        val_df = queries_df.iloc[val_indices].reset_index(drop=True)
        val_qids = [str(qid) for qid in val_df["_id"]]

        print(f"  Split sizes: Train={len(train_indices):,} queries, Val={len(val_indices):,} queries, Test={len(test_indices):,} queries.")

        # Data slices
        x_tr = X_all[train_indices]
        y_tr = Y_all[train_indices]
        ce_tr = CE_all[train_indices]

        x_vl = X_all[val_indices]
        y_vl = Y_all[val_indices]
        ce_vl = CE_all[val_indices]
        cand_vl = [cand_cids_all[idx] for idx in val_indices]

        x_te = X_all[test_indices]
        ce_te = CE_all[test_indices]
        cand_te = [cand_cids_all[idx] for idx in test_indices]

        # -------------------------------------------------------------
        # 1. Tune Beta for Aggregation on Validation split
        # -------------------------------------------------------------
        best_beta = None
        best_beta_val_mrr = -1.0
        beta_candidates = [0.0, 0.1, 0.25, 0.5, 1.0]

        for beta in beta_candidates:
            val_agg_run = {}
            for i, qid in enumerate(val_qids):
                cids_30 = cand_vl[i]
                ce_scores_30 = ce_vl[i].cpu().numpy()
                art_max = {}
                for cid, score in zip(cids_30, ce_scores_30):
                    art = cid_to_title[cid]
                    if art not in art_max or score > art_max[art]:
                        art_max[art] = score
                scored_30 = [
                    [cid, float(score + beta * art_max[cid_to_title[cid]])]
                    for cid, score in zip(cids_30, ce_scores_30)
                ]
                scored_30.sort(key=lambda item: item[1], reverse=True)
                val_agg_run[qid] = scored_30 + dense_results[qid][30:100]

            val_m = evaluate(val_agg_run, qrels_df)
            if val_m["MRR@10"] > best_beta_val_mrr:
                best_beta_val_mrr = val_m["MRR@10"]
                best_beta = beta

        print(f"  Aggregation validation tuning: best beta = {best_beta:.2f} (Val MRR@10 = {best_beta_val_mrr:.4f})")

        # -------------------------------------------------------------
        # 2. Train GAT (with edges)
        # -------------------------------------------------------------
        adj_tr_full = A_full_all[train_indices]
        adj_vl_full = A_full_all[val_indices]
        adj_te_full = A_full_all[test_indices]

        gat_model, best_gat_val_mrr, best_gat_ep = train_cv_model(
            "GAT (with edges)", x_tr, adj_tr_full, ce_tr, y_tr, x_vl, adj_vl_full, ce_vl, y_vl, val_qids, cand_vl
        )

        # -------------------------------------------------------------
        # 3. Train GAT-noedges (Control)
        # -------------------------------------------------------------
        adj_tr_noedges = A_noedges_all[train_indices]
        adj_vl_noedges = A_noedges_all[val_indices]
        adj_te_noedges = A_noedges_all[test_indices]

        noedges_model, best_noedges_val_mrr, best_noedges_ep = train_cv_model(
            "GAT-noedges (control)", x_tr, adj_tr_noedges, ce_tr, y_tr, x_vl, adj_vl_noedges, ce_vl, y_vl, val_qids, cand_vl
        )

        # -------------------------------------------------------------
        # 4. Predict on Test Fold k ONCE with each model
        # -------------------------------------------------------------
        test_dense_run = {qid: dense_results[qid] for qid in test_qids}
        test_ce_run = {qid: ce_results[qid] for qid in test_qids}

        # Test Aggregation
        test_agg_run = {}
        for i, qid in enumerate(test_qids):
            cids_30 = cand_te[i]
            ce_scores_30 = ce_te[i].cpu().numpy()
            art_max = {}
            for cid, score in zip(cids_30, ce_scores_30):
                art = cid_to_title[cid]
                if art not in art_max or score > art_max[art]:
                    art_max[art] = score
            scored_30 = [
                [cid, float(score + best_beta * art_max[cid_to_title[cid]])]
                for cid, score in zip(cids_30, ce_scores_30)
            ]
            scored_30.sort(key=lambda item: item[1], reverse=True)
            test_agg_run[qid] = scored_30 + dense_results[qid][30:100]

        # Test GAT & Noedges
        test_gat_run = predict_test_fold(gat_model, x_te, adj_te_full, ce_te, test_qids, cand_te)
        test_noedges_run = predict_test_fold(noedges_model, x_te, adj_te_noedges, ce_te, test_qids, cand_te)

        # Accumulate pooled predictions
        pooled_test_dense.update(test_dense_run)
        pooled_test_ce.update(test_ce_run)
        pooled_test_agg.update(test_agg_run)
        pooled_test_gat.update(test_gat_run)
        pooled_test_noedges.update(test_noedges_run)

        # Evaluate fold test metrics
        m_dense = evaluate(test_dense_run, qrels_df)
        m_ce = evaluate(test_ce_run, qrels_df)
        m_agg = evaluate(test_agg_run, qrels_df)
        m_gat = evaluate(test_gat_run, qrels_df)
        m_noedges = evaluate(test_noedges_run, qrels_df)

        print(f"\n  Fold {fold_k} Test MRR@10 Results ({len(test_qids)} queries):")
        print(f"    Dense Baseline         : {m_dense['MRR@10']:.4f}")
        print(f"    Dense + Cross-Encoder  : {m_ce['MRR@10']:.4f}")
        print(f"    Aggregation (beta={best_beta:.2f}) : {m_agg['MRR@10']:.4f}")
        print(f"    GAT (with edges)       : {m_gat['MRR@10']:.4f} (best ep: {best_gat_ep})")
        print(f"    GAT-noedges (Control)  : {m_noedges['MRR@10']:.4f} (best ep: {best_noedges_ep})")

        fold_metrics_records.append({
            "fold": fold_k,
            "test_size": len(test_qids),
            "chosen_beta": best_beta,
            "dense_mrr10": m_dense["MRR@10"],
            "ce_mrr10": m_ce["MRR@10"],
            "agg_mrr10": m_agg["MRR@10"],
            "gat_mrr10": m_gat["MRR@10"],
            "noedges_mrr10": m_noedges["MRR@10"],
        })

        t_fold_elapsed = time.time() - t_fold_start

        # Fold 0 Timing Benchmark and Safety Check
        if fold_k == 0:
            est_total_cv_time = t_fold_elapsed * 5.0
            print(f"\n  [Timing Benchmark] Fold 0 completed in {t_fold_elapsed:.2f}s.")
            print(f"  Estimated total time for all 5 folds: {est_total_cv_time:.1f}s ({est_total_cv_time/60:.2f} min).")
            if est_total_cv_time > 5400.0:  # 90 minutes
                raise RuntimeError(f"Estimated CV duration ({est_total_cv_time/60:.2f} min) exceeds 90 minutes! Stopping.")

    total_cv_time = time.time() - total_cv_start_time
    print(f"\nAll 5 Folds completed in {total_cv_time:.2f}s ({total_cv_time/60:.2f} min).")

    # -------------------------------------------------------------
    # 6. Save Pooled Test Rankings
    # -------------------------------------------------------------
    print("\n[6/6] Saving pooled rankings and computing pooled cross-validated metrics...")
    assert len(pooled_test_gat) == num_queries, f"Expected {num_queries} pooled queries, got {len(pooled_test_gat)}"

    with open(POOLED_DENSE_PATH, "w", encoding="utf-8") as f: json.dump(pooled_test_dense, f)
    with open(POOLED_CE_PATH, "w", encoding="utf-8") as f: json.dump(pooled_test_ce, f)
    with open(POOLED_AGG_PATH, "w", encoding="utf-8") as f: json.dump(pooled_test_agg, f)
    with open(POOLED_GAT_PATH, "w", encoding="utf-8") as f: json.dump(pooled_test_gat, f)
    with open(POOLED_NOEDGES_PATH, "w", encoding="utf-8") as f: json.dump(pooled_test_noedges, f)

    # Evaluate Pooled Out-of-Fold Performance
    pooled_m_dense = evaluate(pooled_test_dense, qrels_df)
    pooled_m_ce = evaluate(pooled_test_ce, qrels_df)
    pooled_m_agg = evaluate(pooled_test_agg, qrels_df)
    pooled_m_gat = evaluate(pooled_test_gat, qrels_df)
    pooled_m_noedges = evaluate(pooled_test_noedges, qrels_df)

    print("\n" + "=" * 90)
    print("           POOLED OUT-OF-FOLD TEST BENCHMARK (ALL 3,452 QUERIES)")
    print("=" * 90)
    header = f"{'System':<36} | {'Recall@10':<10} | {'MRR@10':<8} | {'nDCG@10':<8}"
    print(header)
    print("-" * len(header))
    print(f"{'Flat Dense':<36} | {pooled_m_dense['Recall@10']:<10.4f} | {pooled_m_dense['MRR@10']:<8.4f} | {pooled_m_dense['nDCG@10']:<8.4f}")
    print(f"{'Dense + Cross-Encoder':<36} | {pooled_m_ce['Recall@10']:<10.4f} | {pooled_m_ce['MRR@10']:<8.4f} | {pooled_m_ce['nDCG@10']:<8.4f}")
    print(f"{'Dense + CE + Aggregation':<36} | {pooled_m_agg['Recall@10']:<10.4f} | {pooled_m_agg['MRR@10']:<8.4f} | {pooled_m_agg['nDCG@10']:<8.4f}")
    print(f"{'GAT (with edges)':<36} | {pooled_m_gat['Recall@10']:<10.4f} | {pooled_m_gat['MRR@10']:<8.4f} | {pooled_m_gat['nDCG@10']:<8.4f}")
    print(f"{'GAT-no-edges (Control)':<36} | {pooled_m_noedges['Recall@10']:<10.4f} | {pooled_m_noedges['MRR@10']:<8.4f} | {pooled_m_noedges['nDCG@10']:<8.4f}")
    print("=" * len(header))

    # -------------------------------------------------------------
    # 7. Paired Bootstrap over Pooled Queries (1,000 Resamples, Seed 42)
    # -------------------------------------------------------------
    print("\n" + "=" * 95)
    print("PAIRED BOOTSTRAP SIGNIFICANCE TESTING (POOLED 3,452 QUERIES, 1,000 RESAMPLES, SEED 42)")
    print("=" * 95)
    print(f"{'Comparison':<34} | {'Metric':<10} | {'Mean Diff':<10} | {'95% CI':<20} | {'P(Diff > 0)':<12} | {'Verdict'}")
    print("-" * 95)

    all_qids = [str(qid) for qid in queries_df["_id"]]
    pq_dense = compute_per_query_metrics(pooled_test_dense, qrels_norm)
    pq_ce = compute_per_query_metrics(pooled_test_ce, qrels_norm)
    pq_agg = compute_per_query_metrics(pooled_test_agg, qrels_norm)
    pq_gat = compute_per_query_metrics(pooled_test_gat, qrels_norm)
    pq_noedges = compute_per_query_metrics(pooled_test_noedges, qrels_norm)

    bootstrap_comparisons = [
        ("Cross-Encoder vs Dense", pq_ce, pq_dense),
        ("Aggregation vs Cross-Encoder", pq_agg, pq_ce),
        ("GAT vs Cross-Encoder", pq_gat, pq_ce),
        ("GAT-no-edges vs Cross-Encoder", pq_noedges, pq_ce),
        ("GAT vs GAT-no-edges", pq_gat, pq_noedges),
    ]

    bootstrap_records = []
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
            print(f"{comp_name:<34} | {m:<10} | {mean_d:<+10.4f} | {ci_str:<20} | {p_gt_0*100:<11.1f}% | {verdict}")

            bootstrap_records.append({
                "comparison": comp_name,
                "metric": m,
                "mean_diff": mean_d,
                "ci_lower": ci_l,
                "ci_upper": ci_u,
                "p_above_zero": p_gt_0,
                "verdict": verdict,
            })
    print("=" * 95)

    # -------------------------------------------------------------
    # 8. Save cv_summary.csv
    # -------------------------------------------------------------
    # Per-fold summary rows
    summary_rows = []
    for r in fold_metrics_records:
        summary_rows.append({
            "section": "per_fold",
            "fold": r["fold"],
            "test_size": r["test_size"],
            "chosen_beta": r["chosen_beta"],
            "dense_mrr10": r["dense_mrr10"],
            "ce_mrr10": r["ce_mrr10"],
            "agg_mrr10": r["agg_mrr10"],
            "gat_mrr10": r["gat_mrr10"],
            "noedges_mrr10": r["noedges_mrr10"],
        })

    # Pooled summary rows
    for sys_name, p_m in [
        ("Flat Dense", pooled_m_dense),
        ("Dense + Cross-Encoder", pooled_m_ce),
        ("Dense + CE + Aggregation", pooled_m_agg),
        ("GAT (with edges)", pooled_m_gat),
        ("GAT-no-edges (Control)", pooled_m_noedges),
    ]:
        summary_rows.append({
            "section": "pooled_overall",
            "system": sys_name,
            "Recall@10": p_m["Recall@10"],
            "Recall@100": p_m["Recall@100"],
            "MRR@10": p_m["MRR@10"],
            "nDCG@10": p_m["nDCG@10"],
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(CV_SUMMARY_PATH, index=False)
    print(f"\nSaved cross-validation summary to '{CV_SUMMARY_PATH}'.")

    print("\nSTEP C COMPLETED SUCCESSFULLY!")


if __name__ == "__main__":
    main()
