# Hierarchical Retrieval and Reranking (HRR) on Natural Questions

## Project Summary

This project investigates passage retrieval, hierarchical article routing, cross-encoder reranking, article-evidence aggregation, and graph attention networks (GAT) on the BeIR Natural Questions benchmark. Through a diagnostic study (**Study 1**, 1,000 queries, ~100k passages) and a scale-up study (**Study 2**, 3,452 queries, ~200k passages, 5-fold cross-validation), we evaluate whether hierarchical structural priors or cross-candidate attention improve passage retrieval over flat dense retrieval and cross-encoder reranking. Our findings demonstrate that while cross-encoder reranking provides substantial and statistically robust gains over flat dense retrieval, hard hierarchical routing is bottlenecked by Stage 1 article routing misses, and graph attention networks over candidate passages provide no statistically distinguishable benefit over an ablation control that ignores graph edges.

---

## Environment & Hardware

The experiments were developed, benchmarked, and validated on:
* **Operating System**: Windows 11
* **Python Version**: Python 3.11.x (64-bit)
* **GPU**: NVIDIA GeForce RTX 4050 Laptop GPU (6 GB GDDR6 VRAM, CUDA capability 8.9)
* **CUDA Version**: CUDA 12.6
* **Host Memory (RAM)**: 16 GB DDR5 (peak RAM usage designed and capped under 8 GB)

---

## Installation

### 1. Create and Activate Virtual Environment

```bash
# Windows (PowerShell)
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Linux / macOS (bash)
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install PyTorch with CUDA Support

Install PyTorch compiled for CUDA 12.6 (matches our GPU configuration):
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```

> **Note**: A CPU-only installation (`pip install torch torchvision`) is supported and functions correctly, but embedding generation, cross-encoder reranking, and GAT training will run significantly slower.

### 3. Install Required Dependencies

```bash
pip install -r requirements.txt
```

---

## Dataset Instructions

* **Data Source**: The corpus, queries, and relevance judgements (qrels) are streamed and downloaded automatically from Hugging Face Hub using the `datasets` library:
  * Corpus & Queries: [`BeIR/nq`](https://huggingface.co/datasets/BeIR/nq)
  * Relevance Judgments: [`BeIR/nq-qrels`](https://huggingface.co/datasets/BeIR/nq-qrels)
* **Hugging Face Cache Location (`HF_HOME`)**:
  By default, downloads are cached in `~/.cache/huggingface/hub`. To redirect the cache to a dedicated drive:
  ```bash
  # Windows PowerShell
  $env:HF_HOME = "D:\hf_cache"

  # Linux / macOS
  export HF_HOME="/path/to/hf_cache"
  ```
* **Expected Download Size**: ~1.2 GB compressed data downloaded from Hugging Face during subset construction.
* **Excluded Data & Generated Artifacts**:
  To keep the submission repository compact, raw parquet datasets and generated dense embedding arrays are **excluded from the zip archive**:
  * `data/` (~35 MB parquet files)
  * `embeddings/` (~150 MB numpy arrays)
  * `scale_up/data/` (~70 MB parquet files)
  * `scale_up/embeddings/` (~310 MB numpy arrays)
  
  All of these files are generated automatically from scratch by running the build scripts.

---

## Folder Structure

```text
hrr-project/
├── main.py                          # Unified command-line launcher (Study 1 & Study 2)
├── requirements.txt                 # Frozen Python dependencies (torch install instructions at top)
├── README.md                        # Project documentation and reproduction guide
├── PROJECT_BRIEF.md                 # Original project specifications and constraints
├── docs/
│   ├── final_numbers.md             # Master reference sheet with every headline number & source file
│   └── report.md                    # Formal experimental report with full methodology & analysis
├── eval/
│   └── metrics.py                   # IR evaluation metric implementations (Recall@K, MRR@K, nDCG@K)
├── tests/
│   └── test_metrics.py              # Unit tests for evaluation metrics
├── results/                         # Study 1 experimental outputs & evaluation files
│   ├── bm25_dev.json / bm25_test.json
│   ├── dense_dev.json / dense_test.json
│   ├── hier_dev.json / hier_test.json
│   ├── fusion_dev.json / fusion_test.json
│   ├── fusion025_dev.json / fusion025_test.json
│   ├── ce_dev.json / ce_test.json
│   ├── agg_dev.json / agg_test.json
│   ├── gat_test.json / gat_noedges_test.json
│   ├── summary.csv
│   └── test_comparison.png
├── scale_up/                        # Study 2 scale-up pipeline (3,452 queries, 200k passages)
│   ├── stepA_build_subset.py        # Scale-up dataset construction (whole articles, 5 folds)
│   ├── stepB_dense_and_rerank.py    # Scaled passage embedding, FAISS search & top-30 CE scoring
│   ├── stepC_cross_validated_gat.py # 5-fold cross-validated GAT, GAT-no-edges & aggregation
│   ├── step_extra_depth_sensitivity.py # Cross-encoder rerank-depth sensitivity (D in {30, 50, 100})
│   └── results/                     # Study 2 experimental outputs & summary tables
│       ├── dense.json / ce.json
│       ├── raw_ce_scores.json / raw_ce_scores_100.json
│       ├── pooled_dense.json / pooled_ce.json / pooled_agg.json
│       ├── pooled_gat.json / pooled_gat_noedges.json
│       ├── ce_depth50.json / ce_depth100.json
│       ├── cv_summary.csv
│       └── depth_summary.csv
├── step1_inspect_data.py            # Initial dataset inspection utility
├── step1b_build_subset.py           # Study 1 subset builder (100k passages, 1,000 queries)
├── run_bm25.py                      # Study 1 BM25 baseline
├── run_dense.py                     # Study 1 dense retrieval & article pooling
├── run_hierarchical.py              # Study 1 hard two-stage hierarchical retrieval
├── run_fusion.py                    # Study 1 soft score fusion
├── step6b_7a_rerank.py              # Study 1 cross-encoder reranking & failure diagnosis
├── run_step8_aggregation.py         # Study 1 article-evidence aggregation
├── step7_gat_reranker.py            # Study 1 GAT reranker & no-edges control
└── run_analysis.py                  # Study 1 paired bootstrap & failure analysis
```

---

## Main File to Run (`main.py`)

The project provides a thin, non-intrusive launcher `main.py` that executes the existing standalone scripts in their exact required order using `sys.executable`.

```bash
# Show usage and help
python main.py --help

# Dry-run: print commands without executing them
python main.py --study 1 --dry-run
python main.py --study 2 --dry-run

# Run full pipelines
python main.py --study 1             # Executes all 9 Study 1 steps sequentially
python main.py --study 2             # Executes all 4 Study 2 steps sequentially

# Run individual steps
python main.py --study 1 --step bm25
python main.py --study 1 --step cross_encoder
python main.py --study 2 --step stepA
python main.py --study 2 --step depth
```

---

## Execution Order, Runtimes, and Outputs

### Study 1: Diagnostic Benchmark (1,000 Queries, ~100k Passages)

All commands should be executed from the project root (`D:\hrr-project`):

| Step | Script Command | Description | Expected Runtime | Key Output Files |
| :---: | :--- | :--- | :---: | :--- |
| **1** | `python step1b_build_subset.py` | Streams `BeIR/nq`, samples 3,256 articles (99,991 passages), 500 Dev / 500 Test queries | ~2–3 min | `data/corpus.parquet`<br>`data/queries.parquet`<br>`data/qrels.parquet` |
| **2** | `python run_bm25.py` | Builds Rank-BM25 index over 99,991 passages and evaluates Dev & Test | ~1.5–2 min | `results/bm25_dev.json`<br>`results/bm25_test.json` |
| **3** | `python run_dense.py` | Encodes passages with `multi-qa-MiniLM` (GPU), runs FAISS search, builds article embeddings | ~4–5 min | `embeddings/passage_embeddings.npy`<br>`embeddings/article_embeddings.npy`<br>`results/dense_dev.json`<br>`results/dense_test.json` |
| **4** | `python run_hierarchical.py` | Two-stage hard hierarchical search ($N=20$ articles routed, then passage search) | ~20 sec | `results/hier_dev.json`<br>`results/hier_test.json` |
| **5** | `python run_fusion.py` | Soft linear score fusion ($\lambda=0.0$ and $\lambda=0.25$) | ~10 sec | `results/fusion_dev.json`<br>`results/fusion_test.json`<br>`results/fusion025_dev.json`<br>`results/fusion025_test.json` |
| **6** | `python step6b_7a_rerank.py` | Cross-encoder reranking of top 30 dense candidates (`ms-marco-MiniLM-L-6-v2`) | ~45 sec | `results/ce_dev.json`<br>`results/ce_test.json` |
| **7** | `python run_step8_aggregation.py` | Article-evidence aggregation (tunes $\beta \in \{0, 0.1, 0.25, 0.5, 1.0\}$ on Dev, tests on Test) | ~5 sec | `results/agg_dev.json`<br>`results/agg_test.json` |
| **8** | `python step7_gat_reranker.py` | Trains 2-layer 4-head GAT and GAT-no-edges on 350 queries, validates on 150, tests on Test | ~1.5–2 min | `results/gat_test.json`<br>`results/gat_noedges_test.json` |
| **9** | `python run_analysis.py` | Runs 1,000 paired bootstrap resamples on Test, routing error analysis, outputs summary | ~15 sec | `results/summary.csv`<br>`results/test_comparison.png` |

*Total Study 1 Execution Time: ~11–15 minutes.*

---

### Study 2: Scale-Up Pipeline (3,452 Queries, ~200k Passages, 5-Fold CV)

Study 2 is a **scale-up on the same dataset**, not an independent replication.

| Step | Script Command | Description | Expected Runtime | Key Output Files |
| :---: | :--- | :--- | :---: | :--- |
| **A** | `python scale_up/stepA_build_subset.py` | Builds scaled corpus: 199,990 passages, 6,850 articles, 3,452 test queries, 5 grouped folds | ~4–5 min | `scale_up/data/corpus.parquet`<br>`scale_up/data/queries.parquet`<br>`scale_up/data/qrels.parquet` |
| **B** | `python scale_up/stepB_dense_and_rerank.py` | Encodes 199,990 passages (GPU), retrieves top 100 with flat dense, reranks top 30 with CE | ~10–12 min | `scale_up/embeddings/passage_embeddings.npy`<br>`scale_up/results/dense.json`<br>`scale_up/results/ce.json`<br>`scale_up/results/raw_ce_scores.json` |
| **C** | `python scale_up/stepC_cross_validated_gat.py` | 5-fold CV: trains GAT and GAT-no-edges per fold with early stopping, tunes $\beta$, bootstrap tests | ~8–10 min | `scale_up/results/cv_summary.csv`<br>`scale_up/results/pooled_dense.json`<br>`scale_up/results/pooled_ce.json`<br>`scale_up/results/pooled_agg.json`<br>`scale_up/results/pooled_gat.json`<br>`scale_up/results/pooled_gat_noedges.json` |
| **D** | `python scale_up/step_extra_depth_sensitivity.py` | Cross-encoder sensitivity for rerank depth $D \in \{30, 50, 100\}$ across all 3,452 queries | ~5–6 min | `scale_up/results/depth_summary.csv`<br>`scale_up/results/ce_depth50.json`<br>`scale_up/results/ce_depth100.json`<br>`scale_up/results/raw_ce_scores_100.json` |

*Total Study 2 Execution Time: ~27–33 minutes.*

---

## How to Reproduce the Headline Numbers

All headline numbers are recorded with their generating files in [docs/final_numbers.md](file:///d:/hrr-project/docs/final_numbers.md) and [docs/report.md](file:///d:/hrr-project/docs/report.md). You can inspect and verify them directly:

### 1. Study 1 Test Set Headline Table (500 Queries)

Sourced from `results/summary.csv`, `results/ce_test.json`, `results/agg_test.json`, `results/gat_test.json`, and `results/gat_noedges_test.json`:

```text
System                      Recall@10   Recall@100   MRR@10   nDCG@10   Source File
---------------------------------------------------------------------------------------------------------
BM25                        0.6583      0.8957       0.4144   0.4611    results/bm25_test.json
Flat Dense                  0.8240      0.9847       0.5361   0.5958    results/dense_test.json
Hard Hierarchical (N=20)    0.8040      0.9553       0.5237   0.5819    results/hier_test.json
Soft Fusion (lambda=0.0)    0.8240      0.9847       0.5361   0.5958    results/fusion_test.json (reproduces dense)
Soft Fusion (lambda=0.25)   0.8270      0.9833       0.5356   0.5964    results/fusion025_test.json
Dense + Cross-Encoder       0.9033      0.9847       0.6663   0.7164    results/ce_test.json
Dense + CE + Aggregation    0.9093      0.9847       0.6683   0.7197    results/agg_test.json
GAT (with Graph Edges)      0.9033      0.9847       0.6683   0.7181    results/gat_test.json
GAT-no-edges (Ablation)     0.9063      0.9847       0.6645   0.7162    results/gat_noedges_test.json
```

To display Study 1 summary table:
```bash
python -c "import pandas as pd; print(pd.read_csv('results/summary.csv').to_string())"
```

### 2. Study 1 Failure Diagnosis & Route Analysis Numbers

* To reproduce the dense top-10 failure analysis (75 misses, 64/75 gold article present in top 10, 61/75 in top 3, median dense rank 17.0):
  ```bash
  python step6b_7a_rerank.py
  ```
* To reproduce the routing miss verification (17/17 hierarchical misses caused by Stage 1 routing misses) and paired bootstrap tests:
  ```bash
  python run_analysis.py
  ```

### 3. Study 2 Pooled 5-Fold Cross-Validation Table (3,452 Queries)

Sourced from `scale_up/results/cv_summary.csv`:

```text
System                      Recall@10   Recall@100   MRR@10   nDCG@10   Source File
---------------------------------------------------------------------------------------------------------
Flat Dense                  0.8021      0.9663       0.5147   0.5732    scale_up/results/cv_summary.csv
Dense + Cross-Encoder       0.8708      0.9663       0.6574   0.6988    scale_up/results/cv_summary.csv
Dense + CE + Aggregation    0.8716      0.9663       0.6581   0.7004    scale_up/results/cv_summary.csv
GAT (with Graph Edges)      0.8715      0.9663       0.6560   0.6982    scale_up/results/cv_summary.csv
GAT-no-edges (Ablation)     0.8719      0.9663       0.6593   0.7009    scale_up/results/cv_summary.csv
```

To display Study 2 CV summary:
```bash
python -c "import pandas as pd; print(pd.read_csv('scale_up/results/cv_summary.csv').to_string())"
```

* **Candidate Pool Ceiling**: Flat Dense Recall@30 across all 3,452 queries is **0.9147 (91.47%)**, printed by `scale_up/stepC_cross_validated_gat.py`.
* **Statistical Finding**: Comparing GAT vs. GAT-no-edges shows MRR@10 $\Delta = -0.0033$, 95% bootstrap CI $[-0.0072, +0.0005]$. Because the upper bound is $+0.0005$, **edge gains larger than about 0.0005 are unlikely in this setup**, providing no evidence that graph attention edges improve over independent scoring.
* **Aggregation Finding**: The aggregation nDCG@10 difference of $+0.0016$ ($[+0.0003, +0.0029]$) is **negligible and inconsistent**, as both MRR@10 ($[-0.0002, +0.0018]$) and Recall@10 ($[-0.0022, +0.0040]$) intervals include zero across 15 uncorrected comparisons.

### 4. Cross-Encoder Rerank-Depth Sensitivity Table ($D \in \{30, 50, 100\}$)

Sourced from `scale_up/results/depth_summary.csv`:

```text
Depth (D)   Pairs Scored   Inference Time          Recall@10   Recall@100   MRR@10   nDCG@10
-----------------------------------------------------------------------------------------
D = 30      103,560        127.4 s (measured)      0.8708      0.9663       0.6574   0.6988
D = 50      172,600        212.5 s (estimated)     0.8803      0.9663       0.6623   0.7048
D = 100     345,200        425.1 s (estimated)     0.8849      0.9663       0.6647   0.7076
```

To display Depth Sensitivity summary:
```bash
python -c "import pandas as pd; print(pd.read_csv('scale_up/results/depth_summary.csv').to_string())"
```

* Depths 30, 50, and 100 were all reported (**none was selected**).
* Gains are small but consistent (all 95% bootstrap intervals vs. $D=30$ exclude zero).
* $D=50$ captures roughly two-thirds of the $D=100$ improvement for half the additional scoring cost.
* Scoring times for $D=50$ and $D=100$ are estimated from the measured per-pair throughput (~812 pairs/sec).

---

## Known Limitations

1. **Subsampled Corpora**: While Study 2 scales the corpus to ~200k passages, this represents a subset of the full 2.68M BeIR/nq collection. Findings reflect the dense clustering of Wikipedia articles within the sampled subsets.
2. **GPU Nondeterminism**: PyTorch GPU operations (CUDA atomic adds in listwise softmax backward pass and scatter operations) may introduce minor numerical differences ($< 0.001$ in MRR) across different GPU architectures or CUDA library versions despite fixing random seeds (`seed=42`).
3. **Candidate Pool Truncation**: All neural rerankers (CE, GAT, Aggregation) in Study 1 and Study 2 were restricted to the top 30 flat dense candidates. As evidenced by the depth sensitivity analysis, reranking deeper candidate pools ($D=50, 100$) yields small further gains.
