Goal: compare BM25, dense retrieval and hierarchical retrieval on a subset of BEIR Natural Questions.
Data: Hugging Face BeIR/nq (corpus, queries) + BeIR/nq-qrels. Subset: ~1,000 queries, ~100K passages.
Hierarchy: article (title) -> passages. Build the subset by sampling WHOLE ARTICLES, not random passages.
Encoder: sentence-transformers multi-qa-MiniLM-L6-cos-v1. Search: faiss-cpu.
Metrics: Recall@10, MRR@10, nDCG@10.
Hardware: RTX 4050 (6 GB VRAM), 16 GB RAM. Keep memory under ~8 GB.
Rules: one step at a time. Minimal code. Explain every part in plain language. Print real field names, never guess them. Test on 50 queries first.
