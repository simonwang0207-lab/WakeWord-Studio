import pytest

from wakeword_studio.backends.repcnn import RepCNNBackend


def test_repcnn_runtime_contract_defaults_and_observable_score_state():
    backend = RepCNNBackend()
    state = backend.score_state()
    assert backend.window_seconds == pytest.approx(2.0)
    assert backend.hop_seconds == pytest.approx(0.20)
    assert state == {
        "raw_score": 0.0,
        "decision_score": 0.0,
        "window_seconds": 2.0,
        "hop_seconds": 0.20,
        "smoothing": {
            "mode": "raw",
            "window_size": 3,
            "hybrid_max_weight": 0.5,
            "history": [],
        },
    }


def test_repcnn_runtime_rejects_non_deployment_window():
    with pytest.raises(ValueError, match="exactly 2.0"):
        RepCNNBackend(window_seconds=1.5)
