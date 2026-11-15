"""
Evaluation Metrics for Retrieval
--------------------------------
Implements exact standard Information Retrieval metrics for binary relevance:
- Recall@10
- Recall@100
- MRR@10 (Mean Reciprocal Rank at cutoff 10)
- nDCG@10 (Normalized Discounted Cumulative Gain at cutoff 10)

Definitions:
- Relevance: a passage is considered gold if its qrel score > 0.
- Recall@k: (number of gold passages retrieved in top-k) / (total gold passages for query).
- MRR@10: 1 / rank of the first gold passage (1-indexed), if rank <= 10; else 0.0.
- nDCG@10: DCG@10 / IDCG@10
    DCG@10 = sum_{i=1..10} rel_i / log2(i + 1)
    IDCG@10 = sum_{i=1..min(|Gold|, 10)} 1 / log2(i + 1)
All metrics are computed per query and averaged over all queries in the run.
"""

import math
from typing import Any, Dict, List, Set, Union


def _normalize_qrels(qrels: Any) -> Dict[str, Set[str]]:
    """
    Normalizes qrels into a dictionary mapping: query_id -> set of gold corpus_ids.
    Supports:
    - dict {qid: set([cid1, ...])}
    - dict {qid: {cid: score, ...}}
    - pandas DataFrame or list of dicts with ['query-id', 'corpus-id', 'score']
    """
    if isinstance(qrels, dict):
        normalized = {}
        for qid, val in qrels.items():
            qid_str = str(qid)
            if isinstance(val, (set, list, tuple)):
                normalized[qid_str] = {str(cid) for cid in val}
            elif isinstance(val, dict):
                normalized[qid_str] = {str(cid) for cid, score in val.items() if score > 0}
            else:
                raise ValueError(f"Unsupported qrels value type for query {qid}: {type(val)}")
        return normalized

    # If it's a pandas DataFrame or iterable of rows
    try:
        import pandas as pd
        if isinstance(qrels, pd.DataFrame):
            # Expects columns: 'query-id', 'corpus-id', 'score'
            filtered = qrels[qrels["score"] > 0]
            grouped = filtered.groupby("query-id")["corpus-id"].apply(lambda s: set(s.astype(str))).to_dict()
            return {str(k): v for k, v in grouped.items()}
    except ImportError:
        pass

    if isinstance(qrels, (list, tuple)):
        normalized = {}
        for row in qrels:
            if row.get("score", 0) > 0:
                qid = str(row["query-id"])
                cid = str(row["corpus-id"])
                normalized.setdefault(qid, set()).add(cid)
        return normalized

    raise TypeError(f"Unsupported qrels type: {type(qrels)}")


def _normalize_ranked_list(ranked: Any) -> List[str]:
    """
    Ensures ranked list is a list of corpus_id strings.
    Accepts:
    - [cid1, cid2, ...]
    - [[cid1, score1], [cid2, score2], ...]
    - [{'corpus_id': cid1, ...}, ...]
    """
    if not ranked:
        return []
    
    first = ranked[0]
    if isinstance(first, str):
        return [str(cid) for cid in ranked]
    if isinstance(first, (list, tuple)) and len(first) >= 1:
        return [str(item[0]) for item in ranked]
    if isinstance(first, dict):
        cid_key = "corpus-id" if "corpus-id" in first else "corpus_id"
        return [str(item[cid_key]) for item in ranked]
    
    return [str(cid) for cid in ranked]


def evaluate(
    run: Dict[str, Any],
    qrels: Any
) -> Dict[str, float]:
    """
    Computes Recall@10, Recall@100, MRR@10, and nDCG@10.

    Args:
        run: Dict mapping query_id to a ranked list of retrieved corpus_ids.
             Format can be {qid: [cid1, cid2, ...]} or {qid: [[cid1, score1], ...]}.
        qrels: Relevance ground truth. Can be dict {qid: set(gold_cids)},
               dict {qid: {cid: score}}, or qrels DataFrame.

    Returns:
        Dict with keys: 'Recall@10', 'Recall@100', 'MRR@10', 'nDCG@10'.
    """
    norm_qrels = _normalize_qrels(qrels)

    if not run:
        return {
            "Recall@10": 0.0,
            "Recall@100": 0.0,
            "MRR@10": 0.0,
            "nDCG@10": 0.0,
        }

    total_recall_10 = 0.0
    total_recall_100 = 0.0
    total_mrr_10 = 0.0
    total_ndcg_10 = 0.0

    num_queries = len(run)

    for qid, raw_ranked in run.items():
        qid_str = str(qid)
        gold_set = norm_qrels.get(qid_str, set())
        num_gold = len(gold_set)

        if num_gold == 0:
            # No gold passages annotated for this query
            continue

        ranked_cids = _normalize_ranked_list(raw_ranked)

        # ---------------------------------------------------------------------
        # 1. Recall@10 and Recall@100
        # ---------------------------------------------------------------------
        top_10 = ranked_cids[:10]
        top_100 = ranked_cids[:100]

        gold_in_top_10 = sum(1 for cid in top_10 if cid in gold_set)
        gold_in_top_100 = sum(1 for cid in top_100 if cid in gold_set)

        recall_10 = gold_in_top_10 / num_gold
        recall_100 = gold_in_top_100 / num_gold

        # ---------------------------------------------------------------------
        # 2. MRR@10
        # ---------------------------------------------------------------------
        # Rank of the first gold passage (1-indexed). 0 if not found in top 10.
        mrr_10 = 0.0
        for rank, cid in enumerate(top_10, start=1):
            if cid in gold_set:
                mrr_10 = 1.0 / rank
                break

        # ---------------------------------------------------------------------
        # 3. nDCG@10 (with binary relevance)
        # ---------------------------------------------------------------------
        # DCG@10 = sum_{i=1..10} rel_i / log2(i + 1)
        dcg_10 = 0.0
        for rank, cid in enumerate(top_10, start=1):
            if cid in gold_set:
                dcg_10 += 1.0 / math.log2(rank + 1)

        # Ideal DCG@10: assumes min(|Gold|, 10) gold documents are ranked at the top
        ideal_count = min(num_gold, 10)
        idcg_10 = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_count + 1))

        ndcg_10 = (dcg_10 / idcg_10) if idcg_10 > 0.0 else 0.0

        # Accumulate
        total_recall_10 += recall_10
        total_recall_100 += recall_100
        total_mrr_10 += mrr_10
        total_ndcg_10 += ndcg_10

    # Average over all queries evaluated in the run
    return {
        "Recall@10": total_recall_10 / num_queries,
        "Recall@100": total_recall_100 / num_queries,
        "MRR@10": total_mrr_10 / num_queries,
        "nDCG@10": total_ndcg_10 / num_queries,
    }
