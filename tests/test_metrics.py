"""
Tests for Evaluation Metrics (eval/metrics.py)
---------------------------------------------
Tests Recall@10, Recall@100, MRR@10, and nDCG@10 on a tiny, hand-verifiable example:
3 queries (q1, q2, q3) across 6 passages (p1, p2, p3, p4, p5, p6).
"""

import math
import sys
import unittest

# Ensure root directory is on sys.path
sys.path.insert(0, ".")
from eval.metrics import evaluate


class TestRetrievalMetrics(unittest.TestCase):

    def setUp(self):
        # 3 queries, 6 candidate passages
        # Gold relevance annotations (score > 0):
        # q1: 2 gold passages (p1, p2)
        # q2: 1 gold passage (p3)
        # q3: 3 gold passages (p4, p5, p6)
        self.qrels = {
            "q1": {"p1": 1, "p2": 1},
            "q2": {"p3": 1},
            "q3": {"p4": 1, "p5": 1, "p6": 1},
        }

        # Ranked outputs:
        # q1 retrieves p1 at rank 1, p2 at rank 3
        # q2 retrieves p3 at rank 6
        # q3 retrieves [p1, p2, p3] (none of its gold passages p4, p5, p6)
        self.run = {
            "q1": ["p1", "p3", "p2", "p4", "p5", "p6"],
            "q2": ["p1", "p2", "p4", "p5", "p6", "p3"],
            "q3": ["p1", "p2", "p3"],
        }

    def test_metrics_hand_example(self):
        metrics = evaluate(self.run, self.qrels)

        # -------------------------------------------------------------
        # Hand calculations:
        #
        # Query 1:
        #   Gold = {p1, p2} (total = 2)
        #   Retrieved top-10: [p1 (rank 1), p3 (rank 2), p2 (rank 3), p4, p5, p6]
        #   Gold in top-10 = 2 -> Recall@10 = 2/2 = 1.0, Recall@100 = 1.0
        #   First gold at rank 1 -> MRR@10 = 1/1 = 1.0
        #   DCG@10 = 1/log2(2) + 1/log2(4) = 1.0 + 0.5 = 1.5
        #   Ideal DCG@10 (top 2 gold): 1/log2(2) + 1/log2(3) = 1.0 + 1/log2(3)
        #   nDCG@10 = 1.5 / (1.0 + 1/log2(3)) ≈ 0.9197208
        #
        # Query 2:
        #   Gold = {p3} (total = 1)
        #   Retrieved top-10: [p1, p2, p4, p5, p6, p3 (rank 6)]
        #   Gold in top-10 = 1 -> Recall@10 = 1/1 = 1.0, Recall@100 = 1.0
        #   First gold at rank 6 -> MRR@10 = 1/6 ≈ 0.1666667
        #   DCG@10 = 1/log2(7) ≈ 0.3562072
        #   Ideal DCG@10 (top 1 gold): 1/log2(2) = 1.0
        #   nDCG@10 = (1/log2(7)) / 1.0 = 1/log2(7) ≈ 0.3562072
        #
        # Query 3:
        #   Gold = {p4, p5, p6} (total = 3)
        #   Retrieved top-10: [p1, p2, p3] -> none of p4, p5, p6
        #   Recall@10 = 0/3 = 0.0, Recall@100 = 0.0
        #   MRR@10 = 0.0
        #   DCG@10 = 0.0 -> nDCG@10 = 0.0
        #
        # Macro Averages over 3 queries:
        #   Recall@10  = (1.0 + 1.0 + 0.0) / 3 = 2/3 ≈ 0.6666667
        #   Recall@100 = (1.0 + 1.0 + 0.0) / 3 = 2/3 ≈ 0.6666667
        #   MRR@10     = (1.0 + 1/6 + 0.0) / 3 = 7/18 ≈ 0.3888889
        #   nDCG@10    = (0.9197208 + 0.3562072 + 0.0) / 3 ≈ 0.4253093
        # -------------------------------------------------------------
        expected_recall_10 = 2.0 / 3.0
        expected_recall_100 = 2.0 / 3.0
        expected_mrr_10 = (1.0 + (1.0 / 6.0) + 0.0) / 3.0
        
        ndcg_q1 = 1.5 / (1.0 + (1.0 / math.log2(3)))
        ndcg_q2 = 1.0 / math.log2(7)
        ndcg_q3 = 0.0
        expected_ndcg_10 = (ndcg_q1 + ndcg_q2 + ndcg_q3) / 3.0

        self.assertAlmostEqual(metrics["Recall@10"], expected_recall_10, places=6)
        self.assertAlmostEqual(metrics["Recall@100"], expected_recall_100, places=6)
        self.assertAlmostEqual(metrics["MRR@10"], expected_mrr_10, places=6)
        self.assertAlmostEqual(metrics["nDCG@10"], expected_ndcg_10, places=6)

    def test_supports_tuple_format(self):
        # Also supports [[cid, score], ...]
        run_with_scores = {
            "q1": [["p1", 5.2], ["p3", 3.1], ["p2", 2.0], ["p4", 1.0]],
            "q2": [["p1", 4.0], ["p2", 3.0], ["p4", 2.5], ["p5", 2.0], ["p6", 1.5], ["p3", 1.0]],
            "q3": [["p1", 3.0], ["p2", 2.0], ["p3", 1.0]],
        }
        metrics = evaluate(run_with_scores, self.qrels)
        self.assertAlmostEqual(metrics["Recall@10"], 2.0 / 3.0, places=6)


if __name__ == "__main__":
    unittest.main()
