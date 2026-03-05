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

  // Register operators used by the model
  static tflite::MicroMutableOpResolver<10> resolver;

  resolver.AddFullyConnected();
  resolver.AddRelu();
  resolver.AddSoftmax();
  resolver.AddReshape();
  resolver.AddShape();
  resolver.AddQuantize();
  resolver.AddDequantize();
  resolver.AddLogistic();
  resolver.AddStridedSlice(); 
  resolver.AddPack();  // FIX for your latest error

  // Create interpreter
  static tflite::MicroInterpreter static_interpreter(
    model,
    resolver,
    tensorArena,
    tensorArenaSize
  );

  interpreter = &static_interpreter;

  if (interpreter->AllocateTensors() != kTfLiteOk) {
    Serial.println("Tensor allocation failed");
    while (1);
  }

  input = interpreter->input(0);
  output = interpreter->output(0);

  Serial.println("Model loaded successfully!");
}

void loop() {

  // Dummy input values (replace with sensor values later)
 for (int i = 0; i < input->bytes / sizeof(float); i++) {
    input->data.f[i] = random(0,100) / 100.0;
}

  if (interpreter->Invoke() != kTfLiteOk) {
    Serial.println("Inference failed");
    return;
  }

  float prediction = output->data.f[0];

  Serial.print("Prediction: ");
  Serial.println(prediction);

  delay(3000);
}