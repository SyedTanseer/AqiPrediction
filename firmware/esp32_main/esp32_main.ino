#include <Arduino.h>

#include "model_data.h"

#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"

// TensorFlow objects
const tflite::Model* model;
tflite::MicroInterpreter* interpreter;
TfLiteTensor* input;
TfLiteTensor* output;

// Tensor memory
constexpr int tensorArenaSize = 60 * 1024;
uint8_t tensorArena[tensorArenaSize];

void setup() {

  Serial.begin(115200);
  delay(2000);

  Serial.println("Initializing AI model...");

  model = tflite::GetModel(dense_model_tflite);

  // FIX (Flaw 7): Add model schema version check
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.print("Model schema version mismatch! Expected: ");
    Serial.print(TFLITE_SCHEMA_VERSION);
    Serial.print(", Got: ");
    Serial.println(model->version());
    Serial.println("Restarting in 5 seconds...");
    delay(5000);
    ESP.restart();
  }

  // FIX (Flaw 6): Only register ops actually used by Dense model
  // Removed: AddSoftmax(), AddLogistic() (classification ops, not needed for regression)
  static tflite::MicroMutableOpResolver<8> resolver;

  resolver.AddFullyConnected();
  resolver.AddRelu();
  resolver.AddReshape();
  resolver.AddShape();
  resolver.AddQuantize();
  resolver.AddDequantize();
  resolver.AddStridedSlice();
  resolver.AddPack();

  // Create interpreter
  static tflite::MicroInterpreter static_interpreter(
    model,
    resolver,
    tensorArena,
    tensorArenaSize
  );

  interpreter = &static_interpreter;

  // FIX (Flaw 8): Replace while(1) halt with recovery restart
  if (interpreter->AllocateTensors() != kTfLiteOk) {
    Serial.println("Tensor allocation failed! Restarting in 5 seconds...");
    delay(5000);
    ESP.restart();
  }

  input = interpreter->input(0);
  output = interpreter->output(0);

  Serial.println("Model loaded successfully!");
  Serial.print("Input size: ");
  Serial.print(input->bytes / sizeof(float));
  Serial.println(" floats");
}

void loop() {

  // FIX (Flaw 5): Use Normal(0,1) distribution to approximate z-score normalized data
  // Training data is z-score normalized with mean~0, std~1
  // random(0,100)/100.0 was wrong — it produced [0, 0.99] (all positive, wrong range)
  // Box-Muller approximation: sum of 6 uniform random values, centered and scaled
  for (int i = 0; i < input->bytes / sizeof(float); i++) {
    // Approximate Normal(0,1) using sum of uniform randoms (Central Limit Theorem)
    float sum = 0;
    for (int j = 0; j < 6; j++) {
      sum += random(0, 1000) / 1000.0;
    }
    input->data.f[i] = (sum - 3.0);  // center at 0, range roughly [-3, +3]
  }

  if (interpreter->Invoke() != kTfLiteOk) {
    Serial.println("Inference failed");
    return;
  }

  float prediction = output->data.f[0];

  Serial.print("Prediction (normalized CO_GT): ");
  Serial.println(prediction);

  delay(3000);
}