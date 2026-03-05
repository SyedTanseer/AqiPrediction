/*
 * Example usage of the generated model_data.h in ESP32 firmware
 * This demonstrates how to integrate the TensorFlow Lite model
 */

#include "model_data.h"
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"

// Example function showing model integration
void setup_tflite_model() {
    // Load the model from the embedded byte array
    const tflite::Model* model = tflite::GetModel(dense_model_tflite);
    
    // Verify model version
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        Serial.println("Model schema version mismatch!");
        return;
    }
    
    // Print model information
    Serial.print("Model loaded successfully! Size: ");
    Serial.print(dense_model_tflite_len);
    Serial.println(" bytes");
    
    // Model is now ready for inference setup...
}

/*
 * Key points for ESP32 integration:
 * 
 * 1. Include the generated header: #include "model_data.h"
 * 
 * 2. Access the model data: dense_model_tflite[]
 * 
 * 3. Get model size: dense_model_tflite_len
 * 
 * 4. Load with TFLite: tflite::GetModel(dense_model_tflite)
 * 
 * 5. Model expects input shape: (1, 24, 4)
 *    - 1 batch
 *    - 24 timesteps (hours)
 *    - 4 features (CO_GT, NO2_GT, T, RH)
 * 
 * 6. Model outputs shape: (1, 1)
 *    - 1 batch
 *    - 1 prediction (next hour CO_GT)
 */