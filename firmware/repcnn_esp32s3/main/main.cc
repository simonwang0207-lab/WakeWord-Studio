#include <algorithm>
#include <cstddef>
#include <cstdint>

#include "esp_heap_caps.h"
#include "esp_log.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"

extern const uint8_t model_tflite_start[] asm("_binary_model_tflite_start");
extern const uint8_t model_tflite_end[] asm("_binary_model_tflite_end");

namespace {
constexpr char kTag[] = "repcnn_phase05";
constexpr size_t kTensorArenaBytes = 1024 * 1024;

bool RegisterOps(tflite::MicroMutableOpResolver<7>& resolver) {
  return resolver.AddReshape() == kTfLiteOk &&
         resolver.AddConv2D() == kTfLiteOk &&
         resolver.AddDepthwiseConv2D() == kTfLiteOk &&
         resolver.AddAdd() == kTfLiteOk &&
         resolver.AddMean() == kTfLiteOk &&
         resolver.AddFullyConnected() == kTfLiteOk &&
         resolver.AddLogistic() == kTfLiteOk;
}
}  // namespace

extern "C" void app_main(void) {
  tflite::InitializeTarget();
  const size_t model_bytes = model_tflite_end - model_tflite_start;
  ESP_LOGI(kTag, "embedded RepCNN model: %u bytes", static_cast<unsigned>(model_bytes));

  const tflite::Model* model = tflite::GetModel(model_tflite_start);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    ESP_LOGE(kTag, "schema mismatch: model=%d runtime=%d", model->version(), TFLITE_SCHEMA_VERSION);
    return;
  }

  static tflite::MicroMutableOpResolver<7> resolver;
  if (!RegisterOps(resolver)) {
    ESP_LOGE(kTag, "operator registration failed");
    return;
  }

  uint8_t* tensor_arena = static_cast<uint8_t*>(
      heap_caps_malloc(kTensorArenaBytes, MALLOC_CAP_8BIT));
  if (tensor_arena == nullptr) {
    ESP_LOGE(kTag, "failed to allocate %u-byte tensor arena", static_cast<unsigned>(kTensorArenaBytes));
    return;
  }

  tflite::MicroInterpreter interpreter(model, resolver, tensor_arena, kTensorArenaBytes);
  if (interpreter.AllocateTensors() != kTfLiteOk) {
    ESP_LOGE(kTag, "AllocateTensors failed; hardware arena tuning is still required");
    heap_caps_free(tensor_arena);
    return;
  }
  TfLiteTensor* input = interpreter.input(0);
  TfLiteTensor* output = interpreter.output(0);
  if (input->type != kTfLiteInt8 || output->type != kTfLiteInt8) {
    ESP_LOGE(kTag, "expected full INT8 I/O, got input=%d output=%d", input->type, output->type);
    heap_caps_free(tensor_arena);
    return;
  }
  std::fill_n(input->data.int8, input->bytes, static_cast<int8_t>(input->params.zero_point));
  if (interpreter.Invoke() != kTfLiteOk) {
    ESP_LOGE(kTag, "zero-feature smoke Invoke failed");
    heap_caps_free(tensor_arena);
    return;
  }
  const float score = (output->data.int8[0] - output->params.zero_point) * output->params.scale;
  ESP_LOGI(kTag, "TFLM integration ready; zero-feature raw score=%.6f", static_cast<double>(score));
  heap_caps_free(tensor_arena);
}

