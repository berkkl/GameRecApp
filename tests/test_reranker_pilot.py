import numpy as np
import pytest

from scripts.eval.reranker_pilot import chunk_trace, reorder, summarize
from src.gamerec.search.faiss_store import FAISSStore


def test_reranker_pilot_keeps_unscored_tail_and_rejects_broken_scores():
    games = [{"app_id": i, "name": f"Game {i}"} for i in (1, 2, 3)]
    result = reorder(games, np.array([-4, 5]), 2)
    assert [g["app_id"] for g in result] == [2, 1, 3]
    assert "cross_encoder_logit" not in result[-1]
    assert games == [{"app_id": i, "name": f"Game {i}"} for i in (1, 2, 3)]
    with pytest.raises(ValueError, match="correspondence"):
        reorder(games, np.array([float("nan"), 5]), 2)
    with pytest.raises(ValueError, match="correspondence"):
        reorder(games, np.array([5]), 2)


def test_chunk_measurement_restores_search_after_exception(monkeypatch):
    def search(self, vector, k=10):
        return [(1, 0.5)] * k

    monkeypatch.setattr(FAISSStore, "search", search)
    with pytest.raises(RuntimeError):
        with chunk_trace() as trace:
            assert len(FAISSStore.search(None, None, 400)) == 400
            assert trace == [{"requested": 400, "returned": 400}]
            raise RuntimeError("Interrupted evaluation")
    assert FAISSStore.search is search


def test_pilot_reports_missing_target_as_zero_without_claiming_human_relevance():
    rows = [
        {
            "group": "curated_target",
            "variant": "current",
            "target_rank": 2,
            "latency_ms": 10,
        },
        {
            "group": "curated_target",
            "variant": "current",
            "target_rank": None,
            "latency_ms": 20,
        },
    ]
    metrics = summarize(rows, "curated_target")["current"]
    assert metrics["target_hit_at_10"] == 0.5
    assert metrics["target_mrr_at_10"] == 0.25
    assert metrics["p95_ms"] == 20
    assert "not verifiable" in metrics["constraint_evaluation"]
