# RepCNN ESP32-S3 build probe

This minimal ESP-IDF application embeds the trained Phase 0.5 RepCNN full-INT8
classifier, registers exactly its seven TFLite operators, allocates tensors, and
invokes one zero-feature inference when flashed.

It is a classifier/TFLM integration probe, not complete microphone firmware.
No hardware runtime result is claimed until a board is flashed and monitored.

```powershell
idf.py set-target esp32s3
idf.py build
```

The model is copied from
`phase1/artifacts/models/livekit_repcnn_xxlarge_smoke/livekit_repcnn_xxlarge_smoke_full_int8.tflite`.

