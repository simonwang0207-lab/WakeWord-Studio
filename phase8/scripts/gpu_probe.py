"""Short TensorFlow device smoke; performs no training and writes no files."""

from __future__ import annotations

import json

import tensorflow as tf


gpus = tf.config.list_physical_devices("GPU")
gpu_names = [
    str(tf.config.experimental.get_device_details(device).get("device_name", device.name))
    for device in gpus
]
with tf.device("/GPU:0" if gpus else "/CPU:0"):
    result = tf.linalg.matmul(tf.ones((64, 64)), tf.ones((64, 64)))
total = float(tf.reduce_sum(result).numpy())
print(
    json.dumps(
        {
            "framework": "TensorFlow",
            "tensorflow_version": tf.__version__,
            "gpu_count": len(gpus),
            "gpus": gpu_names,
            "op_device": result.device,
            "op_sum": total,
            "gpu_op_executed": bool(gpus and "GPU:0" in result.device.upper()),
        },
        ensure_ascii=False,
    )
)
