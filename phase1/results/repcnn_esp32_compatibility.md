# RepCNN full-INT8 operator compatibility

Model: `livekit_repcnn_xxlarge_smoke_full_int8.tflite`

Required built-ins and firmware resolver registrations:

| TFLite operator | TFLM resolver call |
|---|---|
| `RESHAPE` | `AddReshape()` |
| `CONV_2D` | `AddConv2D()` |
| `DEPTHWISE_CONV_2D` | `AddDepthwiseConv2D()` |
| `ADD` | `AddAdd()` |
| `MEAN` | `AddMean()` |
| `FULLY_CONNECTED` | `AddFullyConnected()` |
| `LOGISTIC` | `AddLogistic()` |

The resolver uses no Flex/custom operators. Compile verification is the decisive
compatibility test; a PC TFLite invocation alone is not an ESP32 compatibility
claim. The project pins Espressif `esp-tflite-micro` 1.4.0.

