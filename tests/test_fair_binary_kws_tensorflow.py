from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml


tf = pytest.importorskip("tensorflow")

from wakeword_studio.training.binary_kws_models import (  # noqa: E402
    build_binary_kws_model,
    count_trainable_parameters,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    ("model_name", "config_name"),
    [
        ("bcresnet", "bcresnet_binary_fair.yaml"),
        ("convmixer", "convmixer_binary_fair.yaml"),
    ],
)
def test_model_forward_train_step_and_checkpoint(
    model_name: str, config_name: str, tmp_path: Path
) -> None:
    config = yaml.safe_load(
        (PROJECT_ROOT / "configs/models" / config_name).read_text(encoding="utf-8")
    )
    model = build_binary_kws_model(model_name, (99, 40), config["model"])
    features = tf.random.normal((2, 99, 40), seed=7)
    targets = tf.constant([[1.0], [0.0]])
    before = np.asarray(model(features, training=False))
    assert before.shape == (2, 1)
    assert np.all((before >= 0.0) & (before <= 1.0))
    assert model.count_params() > 0
    assert 0 < count_trainable_parameters(model) <= model.count_params()

    optimizer = tf.keras.optimizers.Adam(1e-3)
    with tf.GradientTape() as tape:
        predictions = model(features, training=True)
        loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(targets, predictions))
    gradients = tape.gradient(loss, model.trainable_variables)
    assert gradients and all(value is not None for value in gradients)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    assert np.isfinite(float(loss.numpy()))

    checkpoint = tf.train.Checkpoint(model=model, optimizer=optimizer)
    saved = checkpoint.save(str(tmp_path / "ckpt"))
    restored = build_binary_kws_model(model_name, (99, 40), config["model"])
    tf.train.Checkpoint(model=restored).restore(saved).expect_partial()
    np.testing.assert_allclose(
        np.asarray(model(features, training=False)),
        np.asarray(restored(features, training=False)),
        rtol=0,
        atol=1e-6,
    )

