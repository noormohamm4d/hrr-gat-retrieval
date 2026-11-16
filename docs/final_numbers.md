# Evaluation Report & Master Benchmark Sheet: Study 1 and Study 2

This report provides the unified evaluation results and analysis across **Study 1** (1,000-query diagnostic study on ~100k passages) and **Study 2** (3,452-query scale-up study on ~200k passages). 

> **Important Study Framing**: Study 2 is a **scale-up on the same dataset**, not an independent replication. It scales the corpus from 99,991 to 199,990 passages (6,850 articles) and evaluates all 3,452 BEIR Natural Questions test queries (which include the 1,000 queries from Study 1) under a rigorous 5-fold cross-validation scheme.

---

# Part I: Study 1 — System Comparison, Failure Diagnosis, and Reranking (1,000 Queries)

Study 1 benchmarks initial retrieval systems, diagnoses failure modes of flat dense retrieval, and tests whether article-level evidence or cross-encoder reranking recovers missing relevant passages.

* **Corpus**: 99,991 passages across 3,256 whole Wikipedia articles (Source: [step1b_build_subset.py](file:///d:/hrr-project/step1b_build_subset.py))
* **Queries**: 500 Dev queries, 500 Test queries from BeIR Natural Questions
* **Encoders**:
  * Dense Retriever: `sentence-transformers/multi-qa-MiniLM-L6-cos-v1` (384 dimensions, cosine / normalized inner product, FAISS flat index)
  * Cross-Encoder: `cross-encoder/ms-marco-MiniLM-L-6-v2` (reranking top 30 candidates)

---

## 1.1 Benchmark Performance Tables

### Dev Set Performance (500 Queries)

| System | Recall@10 | Recall@100 | MRR@10 | nDCG@10 | Source File |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **BM25 (Lucene / Rank-BM25)** | 0.6952 | 0.9030 | 0.4427 | 0.4913 | [results/bm25_dev.json](file:///d:/hrr-project/results/bm25_dev.json), [results/summary.csv](file:///d:/hrr-project/results/summary.csv) |
| **Flat Dense** | 0.8397 | 0.9840 | 0.5464 | 0.6066 | [results/dense_dev.json](file:///d:/hrr-project/results/dense_dev.json), [results/summary.csv](file:///d:/hrr-project/results/summary.csv) |
| **Hard Hierarchical ($N=20$)** | 0.8297 | 0.9710 | 0.5395 | 0.5989 | [results/hier_dev.json](file:///d:/hrr-project/results/hier_dev.json), [results/summary.csv](file:///d:/hrr-project/results/summary.csv) |
| **Soft Fusion ($\lambda=0.0$)** | 0.8397 | 0.9840 | 0.5464 | 0.6066 | [results/fusion_dev.json](file:///d:/hrr-project/results/fusion_dev.json) *(Reproduces Flat Dense exactly)* |
| **Soft Fusion ($\lambda=0.25$, Secondary)** | 0.8507 | 0.9880 | 0.5419 | 0.6058 | [results/fusion025_dev.json](file:///d:/hrr-project/results/fusion025_dev.json), [results/summary.csv](file:///d:/hrr-project/results/summary.csv) |
| **Dense + Cross-Encoder (Top 30)** | 0.9168 | 0.9840 | 0.7027 | 0.7446 | [results/ce_dev.json](file:///d:/hrr-project/results/ce_dev.json) |
| **Dense + CE + Aggregation ($\beta=1.0$)** | 0.9108 | 0.9840 | 0.7048 | 0.7455 | [results/agg_dev.json](file:///d:/hrr-project/results/agg_dev.json) |
| **GAT (with Graph Edges)** | `[MISSING]` | `[MISSING]` | `[MISSING]` | `[MISSING]` | *Split into 350 train / 150 val; full 500-dev out-of-sample run not generated* |
| **GAT-no-edges (Ablation Control)** | `[MISSING]` | `[MISSING]` | `[MISSING]` | `[MISSING]` | *Split into 350 train / 150 val; full 500-dev out-of-sample run not generated* |

* **Soft Fusion Selection**: The primary selection rule on Dev (best MRR@10) selected $\lambda = 0.0$ (identical to Flat Dense). $\lambda = 0.25$ was evaluated strictly as a secondary point because it achieved the highest Dev Recall@10.
* **Aggregation Tuning on Dev MRR@10**: Sweeping $\beta \in \{0, 0.1, 0.25, 0.5, 1.0\}$ yielded MRR@10 of $0.7027$ ($\beta=0$, reproducing CE), $0.7037$ ($\beta=0.1$), $0.7040$ ($\beta=0.25$), $0.7046$ ($\beta=0.5$), and $0.7048$ ($\beta=1.0$). Thus $\beta = 1.0$ was selected for Test evaluation.

---

### Test Set Performance (500 Queries)

| System | Recall@10 | Recall@100 | MRR@10 | nDCG@10 | Source File |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **BM25 (Lucene / Rank-BM25)** | 0.6583 | 0.8957 | 0.4144 | 0.4611 | [results/bm25_test.json](file:///d:/hrr-project/results/bm25_test.json), [results/summary.csv](file:///d:/hrr-project/results/summary.csv) |
| **Flat Dense** | 0.8240 | 0.9847 | 0.5361 | 0.5958 | [results/dense_test.json](file:///d:/hrr-project/results/dense_test.json), [results/summary.csv](file:///d:/hrr-project/results/summary.csv) |
| **Hard Hierarchical ($N=20$)** | 0.8040 | 0.9553 | 0.5237 | 0.5819 | [results/hier_test.json](file:///d:/hrr-project/results/hier_test.json), [results/summary.csv](file:///d:/hrr-project/results/summary.csv) |
| **Soft Fusion ($\lambda=0.0$)** | 0.8240 | 0.9847 | 0.5361 | 0.5958 | [results/fusion_test.json](file:///d:/hrr-project/results/fusion_test.json) *(Reproduces Flat Dense exactly)* |
| **Soft Fusion ($\lambda=0.25$, Secondary)** | 0.8270 | 0.9833 | 0.5356 | 0.5964 | [results/fusion025_test.json](file:///d:/hrr-project/results/fusion025_test.json), [results/summary.csv](file:///d:/hrr-project/results/summary.csv) |
| **Dense + Cross-Encoder (Top 30)** | 0.9033 | 0.9847 | 0.6663 | 0.7164 | [results/ce_test.json](file:///d:/hrr-project/results/ce_test.json) |
| **Dense + CE + Aggregation ($\beta=1.0$)** | 0.9093 | 0.9847 | 0.6683 | 0.7197 | [results/agg_test.json](file:///d:/hrr-project/results/agg_test.json) |
| **GAT (with Graph Edges)** | 0.9033 | 0.9847 | 0.6683 | 0.7181 | [results/gat_test.json](file:///d:/hrr-project/results/gat_test.json) |
| **GAT-no-edges (Ablation Control)** | 0.9063 | 0.9847 | 0.6645 | 0.7162 | [results/gat_noedges_test.json](file:///d:/hrr-project/results/gat_noedges_test.json) |

---

## 1.2 Study 1 Paired Bootstrap Significance Testing

* **Setup**: Paired query-level bootstrap on the TEST split (500 queries), 1,000 resamples, Random Seed 42.
* **Sources**: [run_analysis.py](file:///d:/hrr-project/run_analysis.py), [step6b_7a_rerank.py](file:///d:/hrr-project/step6b_7a_rerank.py), [run_step8_aggregation.py](file:///d:/hrr-project/run_step8_aggregation.py), [step7_gat_reranker.py](file:///d:/hrr-project/step7_gat_reranker.py).

| Comparison | Metric | Mean Diff ($\Delta$) | 95% Bootstrap CI | $P(\Delta > 0)$ | Plain-Language Statistical Assessment | Source File |
| :--- | :--- | :---: | :---: | :---: | :--- | :--- |
| **Flat Dense vs. BM25** | Recall@10 | +0.1657 | [+0.1260, +0.2080] | 100.0% | **Clear improvement** (the 95% interval excludes zero) | `run_analysis.py` |
| | MRR@10 | +0.1217 | [+0.0859, +0.1594] | 100.0% | **Clear improvement** (the 95% interval excludes zero) | `run_analysis.py` |
| | nDCG@10 | +0.1346 | [+0.0991, +0.1677] | 100.0% | **Clear improvement** (the 95% interval excludes zero) | `run_analysis.py` |
| **Hard Hierarchical vs. Flat Dense** | Recall@10 | -0.0200 | [-0.0340, -0.0080] | 0.0% | **Clear decline** (the 95% interval excludes zero) | `run_analysis.py` |
| | MRR@10 | -0.0125 | [-0.0229, -0.0035] | 0.4% | **Clear decline** (the 95% interval excludes zero) | `run_analysis.py` |
| | nDCG@10 | -0.0139 | [-0.0253, -0.0047] | 0.3% | **Clear decline** (the 95% interval excludes zero) | `run_analysis.py` |
| **Soft Fusion (0.25) vs. Flat Dense** | Recall@10 | +0.0030 | [-0.0070, +0.0130] | 67.7% | **Within noise** (the 95% interval includes zero) | `run_analysis.py` |
| | MRR@10 | -0.0005 | [-0.0066, +0.0043] | 47.9% | **Within noise** (the 95% interval includes zero) | `run_analysis.py` |
| | nDCG@10 | +0.0007 | [-0.0049, +0.0055] | 63.3% | **Within noise** (the 95% interval includes zero) | `run_analysis.py` |
| **Dense + Cross-Encoder vs. Flat Dense** | Recall@10 | +0.0793 | [+0.0527, +0.1047] | 100.0% | **Clear improvement** (the 95% interval excludes zero) | `step6b_7a_rerank.py` |
| | MRR@10 | +0.1302 | [+0.1000, +0.1617] | 100.0% | **Clear improvement** (the 95% interval excludes zero) | `step6b_7a_rerank.py` |
| | nDCG@10 | +0.1206 | [+0.0956, +0.1469] | 100.0% | **Clear improvement** (the 95% interval excludes zero) | `step6b_7a_rerank.py` |
| **Aggregation ($\beta=1.0$) vs. Dense + CE** | Recall@10 | +0.0060 | [+0.0000, +0.0140] | 94.6% | **Within noise / Borderline** (lower bound touches 0.0) | `run_step8_aggregation.py` |
| | MRR@10 | +0.0020 | [+0.0000, +0.0041] | 97.5% | **Slight improvement** (lower bound touches 0.0) | `run_step8_aggregation.py` |
| | nDCG@10 | +0.0033 | [+0.0009, +0.0065] | 99.7% | **Slight improvement** (the 95% interval excludes zero) | `run_step8_aggregation.py` |
| **GAT vs. Dense + CE** | MRR@10 | +0.0020 | [-0.0040, +0.0082] | 74.3% | **Within noise** (the 95% interval includes zero) | `step7_gat_reranker.py` |
| **GAT-no-edges vs. Dense + CE** | MRR@10 | -0.0018 | [-0.0064, +0.0021] | 17.0% | **Within noise** (the 95% interval includes zero) | `step7_gat_reranker.py` |
| **GAT vs. GAT-no-edges** | MRR@10 | +0.0038 | [-0.0041, +0.0117] | 85.0% | **Within noise** (the 95% interval includes zero) | `step7_gat_reranker.py` |

---

## 1.3 Failure Diagnosis & Routing Analysis

* **Sources**: [step6b_7a_rerank.py](file:///d:/hrr-project/step6b_7a_rerank.py), [run_analysis.py](file:///d:/hrr-project/run_analysis.py), [run_hierarchical.py](file:///d:/hrr-project/run_hierarchical.py).

### Diagnosis of Flat Dense Top-10 Failures (Test Split)
* **Queries missing gold passage in top 10**: **75 / 500 (15.00%)**
* **Gold article presence among top-10 retrieved articles for these 75 queries**: **64 / 75 (85.33%)**
* **Gold article presence among top-3 retrieved articles for these 75 queries**: **61 / 75 (81.33%)**
* **First gold passage present in top 100 for these 75 queries**: **68 / 75 (90.67%)**
* **Median dense rank of first gold passage in top 100 (for the 68 queries)**: **17.0** (min: 11, max: 87)
* **First gold passage missing from top 100 entirely**: **7 / 75 (9.33%)**

### Corpus-Wide Presence & Hard Hierarchical Routing Behavior
* **Gold article presence in top-10 retrieved passages across ALL 500 Test queries**: **489 / 500 (97.80%)**
* **Hard Hierarchical top-20 article routing accuracy**:
  * Dev: **98.20%** (491 / 500 queries successfully routed to the gold article)
  * Test: **95.80%** (479 / 500 queries successfully routed to the gold article)
* **Candidate search space reduction**: Average of **581.8 passages** searched per test query under Hard Hierarchical (a 99.4% reduction compared to searching all 99,991 passages).
* **Cause of Hierarchical Misses**: For all 17 queries where Flat Dense found a gold passage in its top 100 but Hard Hierarchical missed it, **17 / 17 (100.0%) were routing misses** (the gold article was ranked outside the top 20 article candidates during Stage 1).
* **Hard Query Failure (Entire Top-10 failure across BM25, Dense, Hierarchical, and Fusion)**: **43 / 500 queries (8.60%)**.

---

# Part II: Study 2 — Scale-Up Evaluation for Reranking and Graph Attention Questions

Study 2 investigates whether the reranking and structural graph patterns observed in Study 1 hold when scaling up the query set and corpus size on the same dataset distribution.

### Study 2 Experimental Protocol
* **Dataset Scope**: All **3,452 test queries** from BeIR Natural Questions (the 1,000 Study 1 queries are among them).
* **Scaled Corpus**: **199,990 passages** across **6,850 Wikipedia articles** (3,452 gold articles + 3,398 sampled distractor articles).
* **Evaluation Protocol**: **5-fold GroupKFold cross-validation** (every query is evaluated exactly once, out-of-fold; zero gold-article leakage across folds).
  * Within each fold, the training split was divided by group into **85% Train / 15% Validation**.
  * Aggregation baseline parameter $\beta \in \{0, 0.1, 0.25, 0.5, 1.0\}$ was selected strictly on each fold's validation MRR@10.
  * Graph Attention Network (GAT) and GAT-no-edges models were trained on the train split with early stopping on validation MRR@10 (settings unchanged from Study 1: 2 layers, 4 heads, listwise softmax cross-entropy, lr 1e-3).
* **Candidate Ceiling**: Flat Dense top-30 Recall ceiling across all 3,452 queries is **0.9147 (91.47%)**, defining the upper bound for any top-30 reranker.

---

## 2.1 Study 2 Corpus & Query Statistics

* **Sources**: [scale_up/data/corpus.parquet](file:///d:/hrr-project/scale_up/data/corpus.parquet), [scale_up/data/queries.parquet](file:///d:/hrr-project/scale_up/data/queries.parquet), [scale_up/data/qrels.parquet](file:///d:/hrr-project/scale_up/data/qrels.parquet), [scale_up/stepA_build_subset.py](file:///d:/hrr-project/scale_up/stepA_build_subset.py).

| Statistic | Value | Source File |
| :--- | :---: | :--- |
| **Total Passages in Scaled Corpus** | **199,990** | `scale_up/data/corpus.parquet` |
| **Total Articles (Unique Titles)** | **6,850** | `scale_up/data/corpus.parquet` |
| — *Articles with Gold Passages* | 3,452 | `scale_up/stepA_build_subset.py` |
| — *Random Distractor Articles* | 3,398 | `scale_up/stepA_build_subset.py` |
| **Total Evaluated Queries** | **3,452** | `scale_up/data/queries.parquet` |
| **Total Positive Qrels** | **4,201** | `scale_up/data/qrels.parquet` |
| **Average Positive Passages per Query** | **1.22** | `scale_up/stepA_build_subset.py` |
| **Total Group IDs (GroupKFold)** | **3,452** | `scale_up/stepA_build_subset.py` |
| **Fold 0 Query Count** | 691 | `scale_up/stepC_cross_validated_gat.py` |
| **Fold 1 Query Count** | 691 | `scale_up/stepC_cross_validated_gat.py` |
| **Fold 2 Query Count** | 690 | `scale_up/stepC_cross_validated_gat.py` |
| **Fold 3 Query Count** | 690 | `scale_up/stepC_cross_validated_gat.py` |
| **Fold 4 Query Count** | 690 | `scale_up/stepC_cross_validated_gat.py` |
| **Missing Gold Passages / Fold Leakage** | **0 / 0** | `scale_up/stepA_build_subset.py` |

---

## 2.2 Study 2 Pooled Benchmark Performance (3,452 Out-of-Fold Queries)

* **Source**: [scale_up/results/cv_summary.csv](file:///d:/hrr-project/scale_up/results/cv_summary.csv).

| System | Recall@10 | Recall@100 | MRR@10 | nDCG@10 | Source File |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Flat Dense** | 0.8021 | 0.9663 | 0.5147 | 0.5732 | [scale_up/results/cv_summary.csv](file:///d:/hrr-project/scale_up/results/cv_summary.csv), [scale_up/results/pooled_dense.json](file:///d:/hrr-project/scale_up/results/pooled_dense.json) |
| **Dense + Cross-Encoder (Top 30)** | 0.8708 | 0.9663 | 0.6574 | 0.6988 | [scale_up/results/cv_summary.csv](file:///d:/hrr-project/scale_up/results/cv_summary.csv), [scale_up/results/pooled_ce.json](file:///d:/hrr-project/scale_up/results/pooled_ce.json) |
| **Dense + CE + Aggregation** | 0.8716 | 0.9663 | 0.6581 | 0.7004 | [scale_up/results/cv_summary.csv](file:///d:/hrr-project/scale_up/results/cv_summary.csv), [scale_up/results/pooled_agg.json](file:///d:/hrr-project/scale_up/results/pooled_agg.json) |
| **GAT (with Graph Edges)** | 0.8715 | 0.9663 | 0.6560 | 0.6982 | [scale_up/results/cv_summary.csv](file:///d:/hrr-project/scale_up/results/cv_summary.csv), [scale_up/results/pooled_gat.json](file:///d:/hrr-project/scale_up/results/pooled_gat.json) |
| **GAT-no-edges (Ablation Control)** | 0.8719 | 0.9663 | 0.6593 | 0.7009 | [scale_up/results/cv_summary.csv](file:///d:/hrr-project/scale_up/results/cv_summary.csv), [scale_up/results/pooled_gat_noedges.json](file:///d:/hrr-project/scale_up/results/pooled_gat_noedges.json) |

### Fold-by-Fold Test MRR@10 Breakdown
* **Fold 0** (691 queries, $\beta = 0.5$): Dense: 0.5104 | CE: 0.6474 | Agg: 0.6483 | GAT: 0.6475 | No-Edges: 0.6515
* **Fold 1** (691 queries, $\beta = 0.5$): Dense: 0.5150 | CE: 0.6634 | Agg: 0.6636 | GAT: 0.6632 | No-Edges: 0.6649
* **Fold 2** (690 queries, $\beta = 1.0$): Dense: 0.4923 | CE: 0.6500 | Agg: 0.6510 | GAT: 0.6435 | No-Edges: 0.6532
* **Fold 3** (690 queries, $\beta = 1.0$): Dense: 0.5297 | CE: 0.6638 | Agg: 0.6672 | GAT: 0.6620 | No-Edges: 0.6656
* **Fold 4** (690 queries, $\beta = 1.0$): Dense: 0.5262 | CE: 0.6623 | Agg: 0.6605 | GAT: 0.6639 | No-Edges: 0.6614

---

## 2.3 Study 2 Paired Bootstrap Significance & Evidence Analysis

* **Setup**: Paired bootstrap over all 3,452 pooled out-of-fold test queries, 1,000 resamples, Random Seed 42.
* **Source**: [scale_up/results/cv_summary.csv](file:///d:/hrr-project/scale_up/results/cv_summary.csv).

| Comparison | Metric | Mean Diff ($\Delta$) | 95% Bootstrap CI | $P(\Delta > 0)$ | Plain-Language Statistical Assessment | Source File |
| :--- | :--- | :---: | :---: | :---: | :--- | :--- |
| **Cross-Encoder vs. Dense** | Recall@10 | +0.0687 | [+0.0594, +0.0787] | 100.0% | **Clear improvement** (the 95% interval excludes zero) | `scale_up/results/cv_summary.csv` |
| | MRR@10 | +0.1427 | [+0.1297, +0.1546] | 100.0% | **Clear improvement** (the 95% interval excludes zero) | `scale_up/results/cv_summary.csv` |
| | nDCG@10 | +0.1256 | [+0.1158, +0.1354] | 100.0% | **Clear improvement** (the 95% interval excludes zero) | `scale_up/results/cv_summary.csv` |
| **Aggregation vs. Cross-Encoder** | Recall@10 | +0.0008 | [-0.0022, +0.0040] | 72.0% | **Within noise** (the 95% interval includes zero) | `scale_up/results/cv_summary.csv` |
| | MRR@10 | +0.0008 | [-0.0002, +0.0018] | 94.1% | **Within noise** (the 95% interval includes zero) | `scale_up/results/cv_summary.csv` |
| | nDCG@10 | +0.0016 | [+0.0003, +0.0029] | 98.9% | **Negligible and inconsistent** (see narrative below) | `scale_up/results/cv_summary.csv` |
| **GAT vs. Cross-Encoder** | Recall@10 | +0.0007 | [-0.0013, +0.0026] | 76.2% | **Within noise** (the 95% interval includes zero) | `scale_up/results/cv_summary.csv` |
| | MRR@10 | -0.0014 | [-0.0040, +0.0011] | 13.3% | **Within noise** (the 95% interval includes zero) | `scale_up/results/cv_summary.csv` |
| | nDCG@10 | -0.0007 | [-0.0027, +0.0012] | 26.6% | **Within noise** (the 95% interval includes zero) | `scale_up/results/cv_summary.csv` |
| **GAT-no-edges vs. Cross-Encoder** | Recall@10 | +0.0011 | [-0.0014, +0.0036] | 81.9% | **Within noise** (the 95% interval includes zero) | `scale_up/results/cv_summary.csv` |
| | MRR@10 | +0.0020 | [-0.0013, +0.0050] | 88.8% | **Within noise** (the 95% interval includes zero) | `scale_up/results/cv_summary.csv` |
| | nDCG@10 | +0.0020 | [-0.0004, +0.0044] | 95.7% | **Within noise** (the 95% interval includes zero) | `scale_up/results/cv_summary.csv` |
| **GAT vs. GAT-no-edges** | Recall@10 | -0.0004 | [-0.0032, +0.0025] | 37.8% | **Within noise** (the 95% interval includes zero) | `scale_up/results/cv_summary.csv` |
| | MRR@10 | -0.0033 | [-0.0072, +0.0005] | 3.7% | **No evidence that edges help** (see narrative below) | `scale_up/results/cv_summary.csv` |
| | nDCG@10 | -0.0027 | [-0.0057, +0.0001] | 2.6% | **No evidence that edges help** (the 95% interval includes zero) | `scale_up/results/cv_summary.csv` |

### Critical Statistical Narrative & Findings

1. **Aggregation Gain is Negligible and Inconsistent**:
   While the pooled nDCG@10 bootstrap difference for Aggregation vs. Cross-Encoder (+0.0016) has a 95% interval excluding zero ($[+0.0003, +0.0029]$), this gain is **negligible in magnitude and inconsistent**:
   - The corresponding MRR@10 interval ($[-0.0002, +0.0018]$) and Recall@10 interval ($[-0.0022, +0.0040]$) both clearly include zero.
   - Across the 15 uncorrected comparisons evaluated across 3 metrics and 5 folds, an isolated +0.0016 movement in nDCG cannot be treated as robust evidence of a real-world effect.
2. **No Evidence that Graph Edges Help**:
   Comparing the full GAT model against the GAT-no-edges control yields an MRR@10 difference of **$-0.0033$**, with a 95% bootstrap confidence interval of **$[-0.0072, +0.0005]$**. Because the upper bound of this interval is $+0.0005$, **edge gains larger than about 0.0005 are unlikely in this setup**. 
3. **Training Dynamics & Hypotheses**:
   - Early stopping on validation MRR@10 consistently halted model training during early epochs (epochs 1 to 3), confirming that additional iterations caused over-fitting rather than structural learning.
   - Explanations such as graph over-smoothing or candidate-neighbour noise must be treated strictly as **untested hypotheses**, as neither was explicitly measured or isolated in these experiments.
4. **Candidate Pool Ceiling**:
   Because all rerankers evaluated in this setup operated on the top 30 candidates retrieved by Flat Dense, their Recall@10 was bounded by the **Recall@30 ceiling of 0.9147 (91.47%)**.

---

# Part III: Cross-Encoder Rerank-Depth Sensitivity Analysis

To test whether the top-30 candidate pool restricted the cross-encoder's reach, reranking sensitivity was systematically evaluated across depths $D \in \{30, 50, 100\}$ over all 3,452 queries.

* **Protocol**: Candidates up to depth $D$ were reranked with `cross-encoder/ms-marco-MiniLM-L-6-v2`. Candidates from $D+1$ to 100 retained their original dense rank order.
* **Reporting Principle**: All three depths ($D=30, 50, 100$) are reported concurrently; **none was selected** as a chosen configuration.
* **Sources**: [scale_up/results/depth_summary.csv](file:///d:/hrr-project/scale_up/results/depth_summary.csv), [scale_up/step_extra_depth_sensitivity.py](file:///d:/hrr-project/scale_up/step_extra_depth_sensitivity.py).

---

## 3.1 Depth Performance and Scoring Cost

| Rerank Depth ($D$) | Total Evaluated Pairs | Scoring Time *(Measurement Status)* | Recall@10 | Recall@100 | MRR@10 | nDCG@10 | Source File |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **$D = 30$** | 103,560 pairs | 127.4 s (2.12 min) *(Measured)* | 0.8708 | 0.9663 | 0.6574 | 0.6988 | `scale_up/results/depth_summary.csv` |
| **$D = 50$** | 172,600 pairs | 212.5 s (3.54 min) *(Estimated)* | 0.8803 | 0.9663 | 0.6623 | 0.7048 | `scale_up/results/depth_summary.csv` |
| **$D = 100$** | 345,200 pairs | 425.1 s (7.09 min) *(Estimated)* | 0.8849 | 0.9663 | 0.6647 | 0.7076 | `scale_up/results/depth_summary.csv` |

*Timing Note*: As documented in [scale_up/step_extra_depth_sensitivity.py](file:///d:/hrr-project/scale_up/step_extra_depth_sensitivity.py#L245-L255), the inference time for $D=30$ was directly measured (127.4 s), whereas the times for $D=50$ and $D=100$ are **estimated from the measured per-pair throughput** (~812 pairs/second).

---

## 3.2 Paired Bootstrap Significance of Rerank Depth (1,000 Resamples, Seed 42)

| Comparison | Metric | Mean Diff ($\Delta$) | 95% Bootstrap CI | $P(\Delta > 0)$ | Plain-Language Assessment | Source File |
| :--- | :--- | :---: | :---: | :---: | :--- | :--- |
| **$D=50$ vs. $D=30$** | Recall@10 | +0.0095 | [+0.0047, +0.0146] | 100.0% | **Small but consistent** (the 95% interval excludes zero) | `scale_up/results/depth_summary.csv` |
| | MRR@10 | +0.0049 | [+0.0020, +0.0079] | 100.0% | **Small but consistent** (the 95% interval excludes zero) | `scale_up/results/depth_summary.csv` |
| | nDCG@10 | +0.0059 | [+0.0030, +0.0092] | 100.0% | **Small but consistent** (the 95% interval excludes zero) | `scale_up/results/depth_summary.csv` |
| **$D=100$ vs. $D=30$** | Recall@10 | +0.0141 | [+0.0075, +0.0205] | 100.0% | **Small but consistent** (the 95% interval excludes zero) | `scale_up/results/depth_summary.csv` |
| | MRR@10 | +0.0073 | [+0.0041, +0.0110] | 100.0% | **Small but consistent** (the 95% interval excludes zero) | `scale_up/results/depth_summary.csv` |
| | nDCG@10 | +0.0088 | [+0.0050, +0.0127] | 100.0% | **Small but consistent** (the 95% interval excludes zero) | `scale_up/results/depth_summary.csv` |

---

## 3.3 Depth Insights & Future Work

1. **Gains are Small but Consistent**:
   Increasing reranking depth beyond $D=30$ produces gains that are small in absolute magnitude (+0.0049 MRR@10 for $D=50$, +0.0073 for $D=100$) but statistically consistent, as all 95% bootstrap confidence intervals vs. $D=30$ exclude zero.
2. **Cost-Benefit Trade-off**:
   **$D=50$ achieves roughly two-thirds of the total $D=100$ improvement** (+0.0049 vs. +0.0073 MRR@10; +0.0095 vs. +0.0141 Recall@10) while requiring only **half the scoring cost** (+69,040 additional pairs for $D=50$ compared to +241,640 additional pairs for $D=100$).
3. **Scope Limitation**:
   Both the GAT models and the aggregation heuristic were evaluated **strictly at $D=30$**; they were not tested at $D=50$ or $D=100$.
4. **Future Work — Stronger or Fine-Tuned Reranker**:
   Even at $D=100$, the achieved Recall@10 is **0.8849**, which remains far below the Flat Dense Recall@100 ceiling of **0.9663**. Because 96.63% of queries already have a gold passage inside the top-100 pool, the primary headroom lies in reranking accuracy rather than candidate retrieval. Exploring a stronger or task-fine-tuned cross-encoder is the most promising path for closing this gap.
