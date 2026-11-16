# Experimental Report: Hierarchical Retrieval, Cross-Encoder Reranking, and Graph Neural Rerankers

## Executive Summary

This report documents our empirical investigations into hierarchical retrieval, passage reranking, article-level aggregation, and graph-attention-based reranking on the BeIR Natural Questions benchmark. The investigation is structured into two sequential parts:

* **Study 1 (1,000 queries, 99,991 passages)**: Diagnostic benchmark evaluating lexical retrieval (BM25), dense passage retrieval, hard two-stage hierarchical retrieval, soft score fusion, cross-encoder reranking, and article-level aggregation. It diagnoses the exact failure modes of dense passage retrieval.
* **Study 2 (3,452 queries, 199,990 passages)**: A scale-up on the same dataset evaluating whether cross-encoder reranking, evidence aggregation, and graph neural rerankers generalize to larger query volumes and expanded corpora under 5-fold cross-validation.

> **Methodological Note on Study 2**: Study 2 is an evaluation of scale on the **same underlying dataset**, not an independent replication. All 3,452 test queries from BeIR Natural Questions were evaluated (subsuming the 1,000 queries analyzed in Study 1) against an expanded corpus of 199,990 passages across 6,850 Wikipedia articles.

---

# Part I: Study 1 — System Comparison, Failure Diagnosis, and Reranking (1,000 Queries)

## 1. Experimental Setup & System Overview

* **Corpus**: 99,991 passages across 3,256 whole Wikipedia articles ([step1b_build_subset.py](file:///d:/hrr-project/step1b_build_subset.py)).
* **Evaluation Splits**: 500 Dev queries and 500 Test queries from BeIR Natural Questions.
* **Retrieval Systems**:
  1. **BM25**: Lucene / Rank-BM25 over passage text ($k_1=1.5, b=0.75$).
  2. **Flat Dense**: FAISS Flat Inner Product index over `sentence-transformers/multi-qa-MiniLM-L6-cos-v1` (384-d, normalized).
  3. **Hard Hierarchical ($N=20$)**: Two-stage routing. Stage 1 routes each query to the top $N=20$ articles using mean passage embeddings ($L_2$-normalized). Stage 2 conducts dense search strictly among passages belonging to those 20 articles.
  4. **Soft Fusion**: Convex linear combination of passage score and its parent article score: $S(q, p) = (1 - \lambda) \cdot \cos(q, p) + \lambda \cdot \cos(q, A(p))$. The primary Dev selection rule (best MRR@10) selected $\lambda = 0.0$ (reproducing Flat Dense). $\lambda = 0.25$ was evaluated as a secondary exploratory point having the best Dev Recall@10.
  5. **Dense + Cross-Encoder**: Top 30 dense candidates reranked using `cross-encoder/ms-marco-MiniLM-L-6-v2` with input `title + " " + text`. Candidates 31–100 retain original dense rank order.
  6. **Dense + CE + Aggregation ($\beta=1.0$)**: Adds article-level context to cross-encoder scores: $\text{final}(p) = \text{CE}(p) + \beta \cdot \max_{p' \in A(p)} \text{CE}(p')$. Parameter $\beta$ tuned on Dev MRR@10 ($\beta=1.0$ selected).
  7. **GAT & GAT-no-edges**: Graph attention reranker and its ablation control trained on a 350-train / 150-val Dev split.

---

## 2. Study 1 Headline Results

### Table 1: Retrieval Performance on Study 1 Splits (500 Dev / 500 Test Queries)

| System | Dev Recall@10 | Dev MRR@10 | Dev nDCG@10 | Test Recall@10 | Test MRR@10 | Test nDCG@10 | Primary Source |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **BM25** | 0.6952 | 0.4427 | 0.4913 | 0.6583 | 0.4144 | 0.4611 | [results/bm25_test.json](file:///d:/hrr-project/results/bm25_test.json) |
| **Flat Dense** | 0.8397 | 0.5464 | 0.6066 | 0.8240 | 0.5361 | 0.5958 | [results/dense_test.json](file:///d:/hrr-project/results/dense_test.json) |
| **Hard Hierarchical ($N=20$)** | 0.8297 | 0.5395 | 0.5989 | 0.8040 | 0.5237 | 0.5819 | [results/hier_test.json](file:///d:/hrr-project/results/hier_test.json) |
| **Soft Fusion ($\lambda=0.0$)** | 0.8397 | 0.5464 | 0.6066 | 0.8240 | 0.5361 | 0.5958 | [results/fusion_test.json](file:///d:/hrr-project/results/fusion_test.json) |
| **Soft Fusion ($\lambda=0.25$, Secondary)** | 0.8507 | 0.5419 | 0.6058 | 0.8270 | 0.5356 | 0.5964 | [results/fusion025_test.json](file:///d:/hrr-project/results/fusion025_test.json) |
| **Dense + Cross-Encoder (Top 30)** | 0.9168 | 0.7027 | 0.7446 | 0.9033 | 0.6663 | 0.7164 | [results/ce_test.json](file:///d:/hrr-project/results/ce_test.json) |
| **Dense + CE + Aggregation ($\beta=1.0$)** | 0.9108 | 0.7048 | 0.7455 | 0.9093 | 0.6683 | 0.7197 | [results/agg_test.json](file:///d:/hrr-project/results/agg_test.json) |
| **GAT (with Graph Edges)** | `[MISSING]` | `[MISSING]` | `[MISSING]` | 0.9033 | 0.6683 | 0.7181 | [results/gat_test.json](file:///d:/hrr-project/results/gat_test.json) |
| **GAT-no-edges (Ablation Control)** | `[MISSING]` | `[MISSING]` | `[MISSING]` | 0.9063 | 0.6645 | 0.7162 | [results/gat_noedges_test.json](file:///d:/hrr-project/results/gat_noedges_test.json) |

*Dev metrics for GAT models are marked `[MISSING]` because the 500 dev queries were used for train/val splitting rather than full out-of-sample evaluation.*

---

## 3. Study 1 Paired Bootstrap Significance (Test Split, 1,000 Resamples, Seed 42)

* **Flat Dense vs. BM25**: Clear improvement across all metrics; the 95% interval excludes zero (Recall@10 $\Delta = +0.1657$ $[+0.1260, +0.2080]$; MRR@10 $\Delta = +0.1217$ $[+0.0859, +0.1594]$; nDCG@10 $\Delta = +0.1346$ $[+0.0991, +0.1677]$).
* **Hard Hierarchical ($N=20$) vs. Flat Dense**: Clear decline; the 95% interval excludes zero (Recall@10 $\Delta = -0.0200$ $[-0.0340, -0.0080]$; MRR@10 $\Delta = -0.0125$ $[-0.0229, -0.0035]$; nDCG@10 $\Delta = -0.0139$ $[-0.0253, -0.0047]$).
* **Soft Fusion ($\lambda=0.25$) vs. Flat Dense**: Within noise; the 95% interval includes zero (Recall@10 $\Delta = +0.0030$ $[-0.0070, +0.0130]$; MRR@10 $\Delta = -0.0005$ $[-0.0066, +0.0043]$).
* **Dense + Cross-Encoder vs. Flat Dense**: Clear improvement across all metrics; the 95% interval excludes zero (Recall@10 $\Delta = +0.0793$ $[+0.0527, +0.1047]$; MRR@10 $\Delta = +0.1302$ $[+0.1000, +0.1617]$; nDCG@10 $\Delta = +0.1206$ $[+0.0956, +0.1469]$).
* **Aggregation ($\beta=1.0$) vs. Dense + CE**: Borderline / slight improvement on Test (Recall@10 $\Delta = +0.0060$ $[+0.0000, +0.0140]$; MRR@10 $\Delta = +0.0020$ $[+0.0000, +0.0041]$; nDCG@10 $\Delta = +0.0033$ $[+0.0009, +0.0065]$).
* **GAT vs. Dense + CE & GAT-no-edges**: Within noise; the 95% intervals include zero (GAT vs. CE MRR@10 $\Delta = +0.0020$ $[-0.0040, +0.0082]$; GAT vs. No-Edges MRR@10 $\Delta = +0.0038$ $[-0.0041, +0.0117]$).

---

## 4. Failure Diagnosis & Route Analysis

1. **Why Dense Top-10 Retrieval Fails**:
   On the Test split, 75 out of 500 queries (15.0%) fail to retrieve a gold passage in the top 10. Crucially, **for 64 of those 75 queries (85.33%), the gold article was already present in the top-10 retrieved passages** (and for 61 of them / 81.33%, in the top 3).
2. **Dense Reranking Headroom**:
   Among the 75 top-10 misses, 68 queries (90.67%) had a gold passage in the top 100, with a **median dense rank of 17.0**. Only 7 queries (9.33%) had no gold passage anywhere in the top 100.
3. **Corpus-Wide Presence**:
   Across all 500 Test queries, **489 queries (97.80%) had the correct gold article in the top-10 retrieved passages**.
4. **Hierarchical Routing Bottleneck**:
   While Hard Hierarchical reduced the candidate pool from 99,991 to an average of 581.8 passages per query (99.4% compute reduction), routing misses bottlenecked recall: 17 queries had gold in the flat dense top 100 but were missed by hierarchical search, and **100% of these (17/17) were caused by routing misses** during Stage 1 article selection.

---

# Part II: Study 2 — Scale-Up Evaluation for Reranking and Graph Attention Questions

## 5. Motivation & Experimental Setup

Study 2 tests whether the cross-encoder gains, aggregation benefits, and lack of graph-edge advantages persist when scaling to the complete BeIR Natural Questions test query set and an expanded corpus.

* **Query Set**: All **3,452 BeIR Natural Questions test queries** (the 1,000 Study 1 queries are among them).
* **Corpus**: **199,990 passages** across **6,850 Wikipedia articles** (3,452 gold articles + 3,398 sampled distractor articles).
* **Cross-Validation**: 5-fold GroupKFold by article group ID (no article leakage across folds).
  * In each fold, the training portion was split 85/15 into train and validation by group.
  * Aggregation parameter $\beta \in \{0, 0.1, 0.25, 0.5, 1.0\}$ was tuned per fold strictly on validation MRR@10.
  * GAT and GAT-no-edges models were trained on fold train splits with early stopping on validation MRR@10 (settings unchanged from Study 1).
* **Systems Compared**: Flat Dense, Dense + Cross-Encoder (top 30), Dense + CE + Aggregation, GAT, and GAT-no-edges.
* **Recall@30 Ceiling**: Flat Dense Recall@30 across all 3,452 queries is **0.9147 (91.47%)**, defining the maximum possible Recall@10 reachable by any top-30 reranker.

---

## 6. Study 2 Pooled Out-of-Fold Results (3,452 Queries)

### Table 2: Study 2 Pooled Out-of-Fold Performance ([scale_up/results/cv_summary.csv](file:///d:/hrr-project/scale_up/results/cv_summary.csv))

| System | Recall@10 | Recall@100 | MRR@10 | nDCG@10 |
| :--- | :---: | :---: | :---: | :---: |
| **Flat Dense** | 0.8021 | 0.9663 | 0.5147 | 0.5732 |
| **Dense + Cross-Encoder (Top 30)** | 0.8708 | 0.9663 | 0.6574 | 0.6988 |
| **Dense + CE + Aggregation** | 0.8716 | 0.9663 | 0.6581 | 0.7004 |
| **GAT (with Graph Edges)** | 0.8715 | 0.9663 | 0.6560 | 0.6982 |
| **GAT-no-edges (Ablation Control)** | 0.8719 | 0.9663 | 0.6593 | 0.7009 |

### Per-Fold Validation Tuning & Test Results:
* **Fold 0** (691 queries, $\beta=0.5$): Dense MRR 0.5104 | CE 0.6474 | Agg 0.6483 | GAT 0.6475 | No-Edges 0.6515
* **Fold 1** (691 queries, $\beta=0.5$): Dense MRR 0.5150 | CE 0.6634 | Agg 0.6636 | GAT 0.6632 | No-Edges 0.6650
* **Fold 2** (690 queries, $\beta=1.0$): Dense MRR 0.4923 | CE 0.6500 | Agg 0.6510 | GAT 0.6435 | No-Edges 0.6532
* **Fold 3** (690 queries, $\beta=1.0$): Dense MRR 0.5297 | CE 0.6638 | Agg 0.6672 | GAT 0.6620 | No-Edges 0.6656
* **Fold 4** (690 queries, $\beta=1.0$): Dense MRR 0.5262 | CE 0.6623 | Agg 0.6605 | GAT 0.6639 | No-Edges 0.6614

---

## 7. Study 2 Paired Bootstrap Significance (3,452 Queries, 1,000 Resamples, Seed 42)

### Table 3: Bootstrap Comparisons on Scaled 5-Fold Pooled Test Set

| Comparison | Metric | Mean Diff ($\Delta$) | 95% Bootstrap CI | Statistical Assessment |
| :--- | :--- | :---: | :---: | :--- |
| **Cross-Encoder vs. Dense** | Recall@10 | +0.0687 | [+0.0594, +0.0787] | **Clear improvement** (the 95% interval excludes zero) |
| | MRR@10 | +0.1427 | [+0.1297, +0.1546] | **Clear improvement** (the 95% interval excludes zero) |
| | nDCG@10 | +0.1256 | [+0.1158, +0.1354] | **Clear improvement** (the 95% interval excludes zero) |
| **Aggregation vs. Cross-Encoder** | Recall@10 | +0.0008 | [-0.0022, +0.0040] | **Within noise** (the 95% interval includes zero) |
| | MRR@10 | +0.0008 | [-0.0002, +0.0018] | **Within noise** (the 95% interval includes zero) |
| | nDCG@10 | +0.0016 | [+0.0003, +0.0029] | **Negligible and inconsistent** (see narrative) |
| **GAT vs. Cross-Encoder** | MRR@10 | -0.0014 | [-0.0040, +0.0011] | **Within noise** (the 95% interval includes zero) |
| **GAT-no-edges vs. Cross-Encoder** | MRR@10 | +0.0020 | [-0.0013, +0.0050] | **Within noise** (the 95% interval includes zero) |
| **GAT vs. GAT-no-edges** | Recall@10 | -0.0004 | [-0.0032, +0.0025] | **Within noise** (the 95% interval includes zero) |
| | MRR@10 | -0.0033 | [-0.0072, +0.0005] | **No evidence that edges help** (see narrative) |
| | nDCG@10 | -0.0027 | [-0.0057, +0.0001] | **No evidence that edges help** (the 95% interval includes zero) |

### Evidence & Mechanism Analysis

1. **Aggregation Gain is Negligible and Inconsistent**:
   While the aggregation nDCG@10 difference (+0.0016) has a 95% bootstrap interval that excludes zero ($[+0.0003, +0.0029]$), this gain is **negligible in magnitude and inconsistent**:
   - Both the MRR@10 interval ($[-0.0002, +0.0018]$) and Recall@10 interval ($[-0.0022, +0.0040]$) include zero.
   - When evaluating 15 uncorrected comparisons across metrics and folds, a +0.0016 nDCG fluctuation cannot be considered substantive evidence of real-world utility.
2. **No Evidence that Graph Edges Help**:
   The GAT model with graph edges underperforms the no-edges control by **$-0.0033$ MRR@10**, with a 95% bootstrap interval of **$[-0.0072, +0.0005]$**. Because the upper bound of this interval is $+0.0005$, **edge gains larger than about 0.0005 are unlikely in this setup**. Graph attention message passing does not improve over treating candidates independently.
3. **Training Dynamics & Hypotheses**:
   - Early stopping on validation MRR@10 consistently stopped training at early epochs (epochs 1 to 3), indicating that subsequent epochs yielded over-fitting rather than useful structural representation learning.
   - Hypotheses such as graph over-smoothing or candidate-neighbour noise must be treated strictly as **untested hypotheses**, as neither mechanism was isolated or confirmed through ablation.

---

# Part III: Cross-Encoder Rerank-Depth Sensitivity Analysis

Because all rerankers in Study 1 and Study 2 were restricted to candidate pools of depth $D=30$ (bounded by Recall@30 = 0.9147), an additional experiment evaluated cross-encoder reranking across depths $D \in \{30, 50, 100\}$ on all 3,452 queries.

* **Reporting Principle**: Depths 30, 50, and 100 are all reported; **none was selected** as a single best system.
* **Scope**: GAT and aggregation were evaluated strictly at $D=30$.

### Table 4: Depth Sensitivity Performance & Compute Scaling ([scale_up/results/depth_summary.csv](file:///d:/hrr-project/scale_up/results/depth_summary.csv))

| Depth ($D$) | Pairs Scored | Scoring Time | Recall@10 | Recall@100 | MRR@10 | nDCG@10 |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **$D = 30$** | 103,560 | 127.4 s *(Measured)* | 0.8708 | 0.9663 | 0.6574 | 0.6988 |
| **$D = 50$** | 172,600 | 212.5 s *(Estimated)* | 0.8803 | 0.9663 | 0.6623 | 0.7048 |
| **$D = 100$** | 345,200 | 425.1 s *(Estimated)* | 0.8849 | 0.9663 | 0.6647 | 0.7076 |

*Scoring Time Note*: The time for $D=30$ was directly measured (127.4 s), whereas the times for $D=50$ and $D=100$ are **estimated from the measured per-pair throughput** (~812 pairs/sec) as recorded in [scale_up/step_extra_depth_sensitivity.py](file:///d:/hrr-project/scale_up/step_extra_depth_sensitivity.py#L245-L255).

### Table 5: Paired Bootstrap Significance of Rerank Depth vs. $D=30$ (1,000 Resamples, Seed 42)

| Comparison | Metric | Mean Diff ($\Delta$) | 95% Bootstrap CI | Statistical Assessment |
| :--- | :--- | :---: | :---: | :--- |
| **$D=50$ vs. $D=30$** | Recall@10 | +0.0095 | [+0.0047, +0.0146] | **Small but consistent** (the 95% interval excludes zero) |
| | MRR@10 | +0.0049 | [+0.0020, +0.0079] | **Small but consistent** (the 95% interval excludes zero) |
| | nDCG@10 | +0.0059 | [+0.0030, +0.0092] | **Small but consistent** (the 95% interval excludes zero) |
| **$D=100$ vs. $D=30$** | Recall@10 | +0.0141 | [+0.0075, +0.0205] | **Small but consistent** (the 95% interval excludes zero) |
| | MRR@10 | +0.0073 | [+0.0041, +0.0110] | **Small but consistent** (the 95% interval excludes zero) |
| | nDCG@10 | +0.0088 | [+0.0050, +0.0127] | **Small but consistent** (the 95% interval excludes zero) |

### Key Depth Takeaways & Future Work

1. **Small but Consistent Gains**:
   Expanding rerank depth beyond 30 yields small but statistically consistent improvements, as the 95% bootstrap intervals for both $D=50$ and $D=100$ vs. $D=30$ exclude zero across all metrics.
2. **Efficiency Profile of $D=50$**:
   **$D=50$ delivers roughly two-thirds of the improvement seen at $D=100$** (+0.0049 vs. +0.0073 MRR@10; +0.0095 vs. +0.0141 Recall@10) while incurring only **half the additional scoring cost** (+69,040 pairs for $D=50$ vs. +241,640 pairs for $D=100$).
3. **Headroom and Future Direction**:
   Even at $D=100$, Recall@10 reaches **0.8849**, which remains substantially below the Flat Dense Recall@100 ceiling of **0.9663**. Because 96.63% of queries already contain a gold passage within the top-100 pool, the bottleneck is not candidate retrieval but reranker discriminative power. Future research should prioritize **stronger or task-fine-tuned cross-encoders** rather than complex candidate-graph message passing.
