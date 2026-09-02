import pytest

from wakeword_studio.runtime.score_smoothing import RollingScoreSmoother


def test_raw_mode_is_identity_and_default_safe():
    smoother = RollingScoreSmoother()
    assert [smoother.update(value) for value in (0.1, 0.9, 0.2)] == [0.1, 0.9, 0.2]


def test_short_mean_is_causal_and_bounded():
    smoother = RollingScoreSmoother("mean", window_size=3)
    assert smoother.update(0.3) == pytest.approx(0.3)
    assert smoother.update(0.9) == pytest.approx(0.6)
    assert smoother.update(0.6) == pytest.approx(0.6)
    assert smoother.update(0.0) == pytest.approx(0.5)


def test_max_mean_hybrid_and_reset():
    smoother = RollingScoreSmoother(
        "max_mean_hybrid", window_size=2, hybrid_max_weight=0.5
    )
    smoother.update(0.2)
    assert smoother.update(0.8) == pytest.approx(0.65)
    smoother.reset()
    assert smoother.update(0.4) == pytest.approx(0.4)
