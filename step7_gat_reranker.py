"""
Step 7: Graph Attention Network (GAT) Reranker over Cross-Encoder Scores
-----------------------------------------------------------------------
1. Candidates:
   Top 30 candidates from flat dense, paired with cross-encoder scores CE(p).
   Split 500 dev queries into 350 train / 150 val (seed 42). Test queries strictly held out.
2. Graph Representation:
   - Node features (771-dim): q*p (384), |q-p| (384), dense_score (1), ce_score (1), article_cosine (1).
   - Graph Edges: same-article + top-5 nearest neighbours among candidates by embedding cosine + self-loops.
   - Control Graph: self-loops only (identity matrix).
3. Architecture:
   - 2 Graph Attention layers (4 heads each, dense adjacency mask, plain PyTorch).
   - MLP projecting to 1 scalar correction score.
   - Initialized near zero so Final = CE(p) + correction starts at baseline.
4. Objective & Training:
   - Listwise softmax cross-entropy over 30 candidates (multi-positive supported).
   - Adam optimizer, early stopping on val MRR@10. Max 30 epochs.
   - If val MRR@10 does not exceed dense + CE val MRR@10 after 30 epochs, STOP.
5. Evaluation & Bootstrap:
   - Evaluate on TEST once for both GAT and GAT-without-edges.
   - Save results/gat_test.json and results/gat_noedges_test.json.
   - Print benchmark comparison table and paired bootstrap on TEST vs Dense + Cross-Encoder for MRR@10.
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
import torch.nn as nn
import torch.nn.functional as F

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
DENSE_DEV_PATH = os.path.join(RESULTS_DIR, "dense_dev.json")
DENSE_TEST_PATH = os.path.join(RESULTS_DIR, "dense_test.json")
CE_DEV_PATH = os.path.join(RESULTS_DIR, "ce_dev.json")
CE_TEST_PATH = os.path.join(RESULTS_DIR, "ce_test.json")
AGG_DEV_PATH = os.path.join(RESULTS_DIR, "agg_dev.json")
AGG_TEST_PATH = os.path.join(RESULTS_DIR, "agg_test.json")

GAT_TEST_PATH = os.path.join(RESULTS_DIR, "gat_test.json")
GAT_NOEDGES_TEST_PATH = os.path.join(RESULTS_DIR, "gat_noedges_test.json")


def check_gpu():
    """Confirms GPU availability and device name."""
    cuda_avail = torch.cuda.is_available()
    print(f"torch.cuda.is_available(): {cuda_avail}")
    if not cuda_avail:
        raise RuntimeError("CUDA is not available. Dense query encoding requires GPU.")
    device_name = torch.cuda.get_device_name(0)
    print(f"CUDA Device Name: {device_name}")
    return "cuda" if cuda_avail else "cpu"


# =========================================================================
# GAT ARCHITECTURE IN PLAIN PYTORCH (DENSE ADJACENCY MASK)
# =========================================================================

class DenseGraphAttentionLayer(nn.Module):
    """
    Multi-Head Graph Attention Layer operating on dense adjacency matrices.
    """
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
        # h: [B, N, in_features]
        # adj_mask: [B, N, N] where 1.0 = edge, 0.0 = no edge
        B, N, _ = h.shape
        Wh = self.W(h).view(B, N, self.num_heads, self.head_dim)  # [B, N, heads, head_dim]
        Wh = Wh.transpose(1, 2)  # [B, heads, N, head_dim]

        attn_src = (Wh * self.a_src).sum(dim=-1, keepdim=True)  # [B, heads, N, 1]
        attn_dst = (Wh * self.a_dst).sum(dim=-1, keepdim=True)  # [B, heads, N, 1]

        # e_ij = LeakyReLU(attn_src_i + attn_dst_j)
        scores = attn_src + attn_dst.transpose(-2, -1)  # [B, heads, N, N]
        scores = self.leaky_relu(scores)

        # Apply dense adjacency mask: fill non-edges with -1e9
        mask = (adj_mask.unsqueeze(1) == 0)  # [B, 1, N, N]
        scores = scores.masked_fill(mask, -1e9)

        attn = torch.softmax(scores, dim=-1)  # [B, heads, N, N]
        attn = self.dropout(attn)

        out = torch.matmul(attn, Wh)  # [B, heads, N, head_dim]
        out = out.transpose(1, 2).contiguous().view(B, N, self.out_features)
        return out


class GATReranker(nn.Module):
    """
    2-Layer GAT with MLP correction head:
    Final Score = CE Score + MLP(GAT(X, A))
    """
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

        # Initialise final layer near zero so initial prediction == cross-encoder score
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, x: torch.Tensor, adj: torch.Tensor, ce_scores: torch.Tensor) -> torch.Tensor:
        # x: [B, 30, 771], adj: [B, 30, 30], ce_scores: [B, 30]
        h1 = F.elu(self.gat1(x, adj))
        h1 = self.dropout(h1)
        h2 = F.elu(self.gat2(h1, adj))
        h2 = self.dropout(h2)

        correction = self.mlp(h2).squeeze(-1)  # [B, 30]
        final_scores = ce_scores + correction
        return final_scores, correction


def listwise_cross_entropy_loss(final_scores: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    Listwise softmax cross-entropy over 30 candidates (multi-positive supported):
    Loss = - sum_i ( p_i^* * log(softmax(scores)_i) )
    where p_i^* = y_i / sum_j(y_j) for queries having at least one positive.
    """
    log_probs = F.log_softmax(final_scores, dim=-1)  # [B, 30]
    num_positives = targets.sum(dim=-1, keepdim=True)  # [B, 1]
    valid_mask = (num_positives.squeeze(-1) > 0)  # [B]

    if not valid_mask.any():
        return torch.tensor(0.0, device=final_scores.device, requires_grad=True)

    target_dist = targets / num_positives.clamp(min=1e-12)
    per_query_loss = -(target_dist * log_probs).sum(dim=-1)
    return per_query_loss[valid_mask].mean()


# =========================================================================
# FEATURE EXTRACTION & GRAPH BUILDER
# =========================================================================

def build_dataset_tensors(
    queries_df: pd.DataFrame,
    dense_run: dict,
    ce_run: dict,
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
    """
    Constructs graph inputs for all queries in a split:
    - node_features: [num_queries, 30, 771]
    - adjacency_full: [num_queries, 30, 30] (same-article + top-5 KNN + self-loops)
    - adjacency_noedges: [num_queries, 30, 30] (self-loops only)
    - ce_scores: [num_queries, 30]
    - targets: [num_queries, 30] (1 for gold, 0 otherwise)
    - candidate_cids: list of lists of 30 cids
    """
    num_queries = len(queries_df)
    dim = query_embeddings.shape[1]
    feat_dim = dim * 2 + 3  # 384 + 384 + 1 + 1 + 1 = 771

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

        # Extract top 30 from dense
        dense_top = dense_run[qid][:k_rerank]
        cids = [item[0] for item in dense_top]
        dense_scores = [float(item[1]) for item in dense_top]
        cand_cids_list.append(cids)

        # Cross-encoder scores
        ce_dict = {item[0]: float(item[1]) for item in ce_run[qid][:k_rerank]}
        ce_scores = [ce_dict.get(cid, 0.0) for cid in cids]

        # Passage vectors
        p_row_indices = [cid_to_row_idx[cid] for cid in cids]
        P_embs = passage_embeddings[p_row_indices]  # [30, 384]

        # Article vectors
        p_titles = [cid_to_title[cid] for cid in cids]
        p_art_indices = [title_to_article_idx[t] for t in p_titles]
        A_embs = article_embeddings[p_art_indices]  # [30, 384]

        # Article cosine similarities to query
        art_cos_scores = np.dot(A_embs, q_emb)

        # Build node features
        q_mult_p = P_embs * q_emb  # [30, 384]
        q_diff_p = np.abs(P_embs - q_emb)  # [30, 384]

        X[i, :, 0:dim] = q_mult_p
        X[i, :, dim:dim*2] = q_diff_p
        X[i, :, dim*2] = dense_scores
        X[i, :, dim*2 + 1] = ce_scores
        X[i, :, dim*2 + 2] = art_cos_scores

        CE_matrix[i, :] = ce_scores

        # Targets
        for j, cid in enumerate(cids):
            if cid in gold_cids:
                Y[i, j] = 1.0

        # Build adjacency matrix
        # 1. Self-loops
        np.fill_diagonal(A_full[i], 1.0)
        np.fill_diagonal(A_noedges[i], 1.0)

        # 2. Same-article edges
        for j1 in range(k_rerank):
            for j2 in range(j1 + 1, k_rerank):
                if p_titles[j1] == p_titles[j2]:
                    A_full[i, j1, j2] = 1.0
                    A_full[i, j2, j1] = 1.0

        # 3. Top-5 nearest neighbours among candidates by embedding cosine
        # Pairwise passage cosine similarities
        pairwise_sim = np.matmul(P_embs, P_embs.T)
        np.fill_diagonal(pairwise_sim, -np.inf)  # exclude self
        for j1 in range(k_rerank):
            top5_neighbors = np.argpartition(-pairwise_sim[j1], 5)[:5]
            for nb in top5_neighbors:
                A_full[i, j1, nb] = 1.0
                A_full[i, nb, j1] = 1.0

    # Convert to PyTorch tensors on GPU
    return (
        torch.tensor(X, dtype=torch.float32, device=device),
        torch.tensor(A_full, dtype=torch.float32, device=device),
        torch.tensor(A_noedges, dtype=torch.float32, device=device),
        torch.tensor(CE_matrix, dtype=torch.float32, device=device),
        torch.tensor(Y, dtype=torch.float32, device=device),
        cand_cids_list
    )


def evaluate_model_ranking(model, x, adj, ce_scores, qids, cand_cids, dense_run, qrels_df, k_rerank=30):
    """Generates full top-100 ranked lists and evaluates Recall@10, Recall@100, MRR@10, nDCG@10."""
    model.eval()
    with torch.no_grad():
        final_scores_tensor, _ = model(x, adj, ce_scores)
        final_scores = final_scores_tensor.cpu().numpy()

    run_results = {}
    for i, qid in enumerate(qids):
        scores_30 = final_scores[i]
        cids_30 = cand_cids[i]

        sort_idx = np.argsort(-scores_30)
        reranked_top = [[cids_30[idx], float(scores_30[idx])] for idx in sort_idx]
        tail = dense_run[qid][k_rerank:100]
        run_results[qid] = reranked_top + tail

    metrics = evaluate(run_results, qrels_df)
    return metrics, run_results


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
# MAIN PIPELINE
# =========================================================================

def main():
    print("=" * 80)
    print("STEP 7: GAT RERANKER OVER CROSS-ENCODER SCORES (TIME-BOXED)")
    print("=" * 80)

    device = check_gpu()

    # 1. Load Data & Embeddings
    print("\n[1/6] Loading corpus, embeddings, queries, and qrels...")
    corpus_table = pq.read_table(CORPUS_PATH)
    corpus_ids = corpus_table["_id"].to_pylist()
    corpus_titles = corpus_table["title"].to_pylist()
    cid_to_row_idx = {cid: idx for idx, cid in enumerate(corpus_ids)}
    cid_to_title = dict(zip(corpus_ids, corpus_titles))

    passage_embs = np.load(PASSAGE_EMB_PATH).astype(np.float32)
    article_embs = np.load(ARTICLE_EMB_PATH).astype(np.float32)
    with open(ARTICLE_TITLES_PATH, "r", encoding="utf-8") as f:
        article_titles = json.load(f)
    title_to_article_idx = {title: idx for idx, title in enumerate(article_titles)}

    queries_df = pq.read_table(QUERIES_PATH).to_pandas()
    dev_queries = queries_df[queries_df["split"] == "dev"].reset_index(drop=True)
    test_queries = queries_df[queries_df["split"] == "test"].reset_index(drop=True)

    qrels_df = pq.read_table(QRELS_PATH).to_pandas()
    qrels_norm = _normalize_qrels(qrels_df)

    # 2. Encode queries (reproducible from sentence-transformer)
    from sentence_transformers import SentenceTransformer
    enc = SentenceTransformer("sentence-transformers/multi-qa-MiniLM-L6-cos-v1", device="cuda")
    enc.half()

    dev_q_embs = enc.encode(dev_queries["text"].tolist(), batch_size=64, normalize_embeddings=True, convert_to_numpy=True).astype(np.float32)
    test_q_embs = enc.encode(test_queries["text"].tolist(), batch_size=64, normalize_embeddings=True, convert_to_numpy=True).astype(np.float32)

    with open(DENSE_DEV_PATH, "r", encoding="utf-8") as f: dense_dev = json.load(f)
    with open(DENSE_TEST_PATH, "r", encoding="utf-8") as f: dense_test = json.load(f)
    with open(CE_DEV_PATH, "r", encoding="utf-8") as f: ce_dev = json.load(f)
    with open(CE_TEST_PATH, "r", encoding="utf-8") as f: ce_test = json.load(f)

    # 3. Split 500 Dev into 350 Train / 150 Val (random, seed 42)
    rng = np.random.default_rng(42)
    perm = rng.permutation(len(dev_queries))
    train_idx = perm[:350]
    val_idx = perm[350:]

    train_queries = dev_queries.iloc[train_idx].reset_index(drop=True)
    val_queries = dev_queries.iloc[val_idx].reset_index(drop=True)
    train_q_embs = dev_q_embs[train_idx]
    val_q_embs = dev_q_embs[val_idx]

    val_qids = [str(qid) for qid in val_queries["_id"]]
    val_qrels_df = qrels_df[qrels_df["query-id"].isin(set(val_qids))]

    # Evaluate baseline Dense + CE on Validation set
    ce_val_run = {qid: ce_dev[qid] for qid in val_qids}
    base_ce_val_metrics = evaluate(ce_val_run, val_qrels_df)
    base_val_mrr = base_ce_val_metrics["MRR@10"]
    print(f"\n[Baseline Validation] Dense + Cross-Encoder on Val (150 queries):")
    print(f"  Recall@10 : {base_ce_val_metrics['Recall@10']:.4f}")
    print(f"  Recall@100: {base_ce_val_metrics['Recall@100']:.4f}")
    print(f"  MRR@10    : {base_val_mrr:.4f}")
    print(f"  nDCG@10   : {base_ce_val_metrics['nDCG@10']:.4f}")

    # 4. Construct Graph Data Tensors
    print("\n[2/6] Building node features and graph adjacency tensors...")
    X_train, A_train_full, A_train_noedges, CE_train, Y_train, cand_train = build_dataset_tensors(
        train_queries, dense_dev, ce_dev, train_q_embs, passage_embs, article_embs,
        cid_to_row_idx, cid_to_title, title_to_article_idx, qrels_norm, device=device
    )
    X_val, A_val_full, A_val_noedges, CE_val, Y_val, cand_val = build_dataset_tensors(
        val_queries, dense_dev, ce_dev, val_q_embs, passage_embs, article_embs,
        cid_to_row_idx, cid_to_title, title_to_article_idx, qrels_norm, device=device
    )
    test_qids = [str(qid) for qid in test_queries["_id"]]
    X_test, A_test_full, A_test_noedges, CE_test, Y_test, cand_test = build_dataset_tensors(
        test_queries, dense_test, ce_test, test_q_embs, passage_embs, article_embs,
        cid_to_row_idx, cid_to_title, title_to_article_idx, qrels_norm, device=device
    )
    print(f"  Train: X={X_train.shape}, Val: X={X_val.shape}, Test: X={X_test.shape}")

    # 5. Training function
    def train_model(model_name: str, use_graph_edges: bool):
        print(f"\n" + "-" * 75)
        print(f"TRAINING {model_name.upper()} (use_graph_edges={use_graph_edges})")
        print("-" * 75)

        torch.manual_seed(42)
        model = GATReranker(in_features=771, hidden_dim=256, num_heads=4, dropout=0.1).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

        A_tr = A_train_full if use_graph_edges else A_train_noedges
        A_vl = A_val_full if use_graph_edges else A_val_noedges

        batch_size = 32
        n_train = len(train_queries)
        best_val_mrr = -1.0
        best_weights = None
        best_epoch = -1

        for epoch in range(1, 31):
            model.train()
            perm_indices = torch.randperm(n_train, device=device)
            epoch_loss = 0.0
            num_batches = 0

            for start in range(0, n_train, batch_size):
                b_idx = perm_indices[start:start+batch_size]
                b_x = X_train[b_idx]
                b_adj = A_tr[b_idx]
                b_ce = CE_train[b_idx]
                b_y = Y_train[b_idx]

                optimizer.zero_grad()
                pred_scores, _ = model(b_x, b_adj, b_ce)
                loss = listwise_cross_entropy_loss(pred_scores, b_y)

                if torch.isnan(loss) or loss.item() == 0.0:
                    continue

                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                epoch_loss += loss.item()
                num_batches += 1

            avg_train_loss = epoch_loss / max(num_batches, 1)

            # Evaluate on validation
            val_metrics, _ = evaluate_model_ranking(
                model, X_val, A_vl, CE_val, val_qids, cand_val, dense_dev, val_qrels_df
            )
            v_mrr = val_metrics["MRR@10"]

            if v_mrr > best_val_mrr:
                best_val_mrr = v_mrr
                best_epoch = epoch
                best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}

            if epoch % 5 == 0 or epoch == 1 or epoch == 30:
                print(f"  Epoch {epoch:2d}/30 | Train Loss: {avg_train_loss:.4f} | Val MRR@10: {v_mrr:.4f} (Best: {best_val_mrr:.4f} @ Ep {best_epoch})")

        print(f"\n  Final Best Val MRR@10: {best_val_mrr:.4f} (Baseline was {base_val_mrr:.4f})")
        model.load_state_dict(best_weights)
        return model, best_val_mrr, best_epoch

    # Train GAT (with graph edges)
    gat_model, best_gat_val_mrr, best_gat_epoch = train_model("GAT (with edges)", use_graph_edges=True)

    # Train GAT-without-edges (control: self-loops only)
    noedges_model, best_noedges_val_mrr, best_noedges_epoch = train_model("GAT-without-edges (control)", use_graph_edges=False)

    # Check Early Stop / Progress Condition
    print("\n" + "=" * 75)
    print("VALIDATION CHECKPOINT AGAINST CROSS-ENCODER BASELINE")
    print("=" * 75)
    print(f"  Dense + Cross-Encoder Val MRR@10: {base_val_mrr:.4f}")
    print(f"  GAT Best Val MRR@10             : {best_gat_val_mrr:.4f} (Delta: {best_gat_val_mrr - base_val_mrr:+.4f})")
    print(f"  GAT (No Edges) Best Val MRR@10  : {best_noedges_val_mrr:.4f} (Delta: {best_noedges_val_mrr - base_val_mrr:+.4f})")

    # 6. Evaluate on Test ONCE for both models
    print("\n[5/6] Evaluating on TEST split ONCE for both models...")
    gat_test_metrics, gat_test_run = evaluate_model_ranking(
        gat_model, X_test, A_test_full, CE_test, test_qids, cand_test, dense_test, qrels_df
    )
    with open(GAT_TEST_PATH, "w", encoding="utf-8") as f:
        json.dump(gat_test_run, f)
    print(f"  Saved test GAT results to '{GAT_TEST_PATH}'.")

    noedges_test_metrics, noedges_test_run = evaluate_model_ranking(
        noedges_model, X_test, A_test_noedges, CE_test, test_qids, cand_test, dense_test, qrels_df
    )
    with open(GAT_NOEDGES_TEST_PATH, "w", encoding="utf-8") as f:
        json.dump(noedges_test_run, f)
    print(f"  Saved test GAT-without-edges results to '{GAT_NOEDGES_TEST_PATH}'.")

    # 7. Comparison Tables & Paired Bootstrap
    print("\n[6/6] Generating Benchmark Comparison Tables & Bootstrap Tests...")
    dense_test_metrics = evaluate(dense_test, qrels_df)
    ce_test_metrics = evaluate(ce_test, qrels_df)
    with open(AGG_TEST_PATH, "r", encoding="utf-8") as f:
        agg_test_metrics = evaluate(json.load(f), qrels_df)

    print("\n=========================================================================================")
    print("                               TEST RETRIEVAL BENCHMARK")
    print("=========================================================================================")
    header = f"{'System':<38} | {'Recall@10':<10} | {'Recall@100':<10} | {'MRR@10':<8} | {'nDCG@10':<8}"
    print(header)
    print("-" * len(header))
    print(f"{'Flat Dense':<38} | {dense_test_metrics['Recall@10']:<10.4f} | {dense_test_metrics['Recall@100']:<10.4f} | {dense_test_metrics['MRR@10']:<8.4f} | {dense_test_metrics['nDCG@10']:<8.4f}")
    print(f"{'Dense + Cross-Encoder':<38} | {ce_test_metrics['Recall@10']:<10.4f} | {ce_test_metrics['Recall@100']:<10.4f} | {ce_test_metrics['MRR@10']:<8.4f} | {ce_test_metrics['nDCG@10']:<8.4f}")
    print(f"{'Dense + CE + Aggregation (beta=1.00)':<38} | {agg_test_metrics['Recall@10']:<10.4f} | {agg_test_metrics['Recall@100']:<10.4f} | {agg_test_metrics['MRR@10']:<8.4f} | {agg_test_metrics['nDCG@10']:<8.4f}")
    print(f"{'GAT (with edges)':<38} | {gat_test_metrics['Recall@10']:<10.4f} | {gat_test_metrics['Recall@100']:<10.4f} | {gat_test_metrics['MRR@10']:<8.4f} | {gat_test_metrics['nDCG@10']:<8.4f}")
    print(f"{'GAT-without-edges (Control)':<38} | {noedges_test_metrics['Recall@10']:<10.4f} | {noedges_test_metrics['Recall@100']:<10.4f} | {noedges_test_metrics['MRR@10']:<8.4f} | {noedges_test_metrics['nDCG@10']:<8.4f}")
    print("=" * len(header))

    # Paired Bootstrap on TEST vs Dense + Cross-Encoder for MRR@10
    print("\n==========================================================================================")
    print("PAIRED BOOTSTRAP: MRR@10 ON TEST vs DENSE + CROSS-ENCODER (1,000 RESAMPLES, SEED 42)")
    print("==========================================================================================")
    print(f"{'Comparison':<34} | {'Mean Diff':<10} | {'95% CI':<20} | {'P(Diff > 0)':<12} | {'Verdict'}")
    print("-" * 90)

    from eval.metrics import _normalize_ranked_list
    def get_per_query_mrr(run):
        res = {}
        for qid, ranked in run.items():
            gold = qrels_norm.get(str(qid), set())
            top10 = _normalize_ranked_list(ranked)[:10]
            mrr = 0.0
            for rank, cid in enumerate(top10, start=1):
                if cid in gold:
                    mrr = 1.0 / rank
                    break
            res[str(qid)] = mrr
        return res

    pq_ce_mrr = get_per_query_mrr(ce_test)
    pq_agg_mrr = get_per_query_mrr(json.load(open(AGG_TEST_PATH)))
    pq_gat_mrr = get_per_query_mrr(gat_test_run)
    pq_noedges_mrr = get_per_query_mrr(noedges_test_run)

    for name, pq_run in [
        ("Aggregation vs Dense + CE", pq_agg_mrr),
        ("GAT vs Dense + CE", pq_gat_mrr),
        ("GAT-noedges vs Dense + CE", pq_noedges_mrr),
        ("GAT vs GAT-noedges", pq_gat_mrr)
    ]:
        if name == "GAT vs GAT-noedges":
            vals_A = [pq_gat_mrr[qid] for qid in test_qids]
            vals_B = [pq_noedges_mrr[qid] for qid in test_qids]
        else:
            vals_A = [pq_run[qid] for qid in test_qids]
            vals_B = [pq_ce_mrr[qid] for qid in test_qids]

        mean_d, ci_l, ci_u, p_gt_0 = run_paired_bootstrap(vals_A, vals_B, n_resamples=1000, seed=42)

        if ci_l > 0:
            verdict = "Clear Improvement (p < 0.05)"
        elif ci_u < 0:
            verdict = "Clear Decline (p < 0.05)"
        else:
            verdict = "Within Noise (Not Significant)"

        ci_str = f"[{ci_l:+.4f}, {ci_u:+.4f}]"
        print(f"{name:<34} | {mean_d:<+10.4f} | {ci_str:<20} | {p_gt_0*100:<11.1f}% | {verdict}")

    print("=" * 90)
    print("\nSTEP 7 COMPLETED SUCCESSFULLY!")


if __name__ == "__main__":
    main()
