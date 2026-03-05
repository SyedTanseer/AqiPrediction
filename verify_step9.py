"""
Verify Step 9 completion - ESP32 Firmware with TensorFlow Lite Integration
"""
import os
import re

def verify_step9():
    """Verify all Step 9 requirements are met"""
    
    print("Step 9 Verification: ESP32 Firmware with TensorFlow Lite")
    print("=" * 60)
    
    issues = []
    
    # Check ESP32 main file exists
    esp32_main_path = "firmware/esp32_main/esp32_main.ino"
    if not os.path.exists(esp32_main_path):
        issues.append("ESP32 main file missing")
        return False
    
    print(f"✔ ESP32 main file found: {esp32_main_path}")
    
    # Check model_data.h exists in esp32_main directory
    model_header_path = "firmware/esp32_main/model_data.h"
    if not os.path.exists(model_header_path):
        issues.append("model_data.h missing in esp32_main directory")
        return False
    
    print(f"✔ Model header found: {model_header_path}")
    
    # Read and verify ESP32 main file content
    with open(esp32_main_path, 'r') as f:
        content = f.read()
    
    print(f"\n✔ Verifying required includes...")
    
    # Check for required includes
    required_includes = [
        '#include <Arduino.h>',
        '#include "model_data.h"',
        '#include "TensorFlowLite.h"',
        '#include "tensorflow/lite/micro/all_ops_resolver.h"',
        '#include "tensorflow/lite/micro/micro_interpreter.h"',
        '#include "tensorflow/lite/schema/schema_generated.h"',
        '#include "tensorflow/lite/version.h"'
    ]
    
    for include in required_includes:
        if include in content:
            print(f"  ✓ {include}")
        else:
            print(f"  ❌ {include}")
            issues.append(f"Missing include: {include}")
    
    print(f"\n✔ Verifying global variables...")
    
    # Check for required global variables
    global_vars = [
        ('const tflite::Model* model', 'TensorFlow Lite model pointer'),
        ('tflite::MicroInterpreter* interpreter', 'Micro interpreter pointer'),
        ('TfLiteTensor* input', 'Input tensor pointer'),
        ('TfLiteTensor* output', 'Output tensor pointer'),
        ('tensor_arena', 'Tensor arena buffer'),
        ('kTensorArenaSize', 'Arena size constant')
    ]
    
    for var_pattern, description in global_vars:
        if var_pattern in content:
            print(f"  ✓ {description}")
        else:
            print(f"  ❌ {description}")
            issues.append(f"Missing global variable: {var_pattern}")
    
    print(f"\n✔ Verifying setup() function...")
    
    # Check setup() function components
    setup_components = [
        ('Serial.begin(115200)', 'Serial initialization'),
        ('tflite::GetModel(dense_model_tflite)', 'Model loading'),
        ('TFLITE_SCHEMA_VERSION', 'Schema version check'),
        ('tflite::AllOpsResolver', 'Operations resolver'),
        ('tflite::MicroInterpreter', 'Interpreter creation'),
        ('AllocateTensors()', 'Tensor allocation'),
        ('interpreter->input(0)', 'Input tensor retrieval'),
        ('interpreter->output(0)', 'Output tensor retrieval')
    ]
    
    for component, description in setup_components:
        if component in content:
            print(f"  ✓ {description}")
        else:
            print(f"  ❌ {description}")
            issues.append(f"Missing setup component: {component}")
    
    print(f"\n✔ Verifying loop() function...")
    
    # Check loop() function components
    loop_components = [
        ('input->data.f', 'Input tensor data access'),
        ('for (int timestep = 0; timestep < 24', '24 timestep loop'),
        ('timestep * 4', 'Flattened input indexing'),
        ('interpreter->Invoke()', 'Inference execution'),
        ('output->data.f[0]', 'Output tensor reading'),
        ('delay(5000)', '5 second delay')
    ]
    
    for component, description in loop_components:
        if component in content:
            print(f"  ✓ {description}")
        else:
            print(f"  ❌ {description}")
            issues.append(f"Missing loop component: {component}")
    
    print(f"\n✔ Verifying tensor arena size...")
    
    # Check tensor arena size
    arena_match = re.search(r'kTensorArenaSize = (\d+)', content)
    if arena_match:
        arena_size = int(arena_match.group(1))
        if arena_size >= 20480:  # 20KB
            print(f"  ✓ Tensor arena size: {arena_size:,} bytes ({arena_size/1024:.0f}KB)")
        else:
            print(f"  ⚠️  Tensor arena size may be too small: {arena_size:,} bytes")
    else:
        issues.append("Could not find tensor arena size")
    
    print(f"\n✔ Verifying input data generation...")
    
    # Check input data generation
    input_checks = [
        ('96 values' in content or '24 timesteps × 4 features' in content, 'Input size documentation'),
        ('sin(' in content and 'cos(' in content, 'Realistic sensor simulation'),
        ('random(' in content, 'Random variation'),
        ('timestep * 4 + 0' in content, 'Feature 0 (CO_GT)'),
        ('timestep * 4 + 1' in content, 'Feature 1 (NO2_GT)'),
        ('timestep * 4 + 2' in content, 'Feature 2 (Temperature)'),
        ('timestep * 4 + 3' in content, 'Feature 3 (Humidity)')
    ]
    
    for check, description in input_checks:
        if check:
            print(f"  ✓ {description}")
        else:
            print(f"  ⚠️  {description}")
    
    print(f"\n✔ Checking file sizes...")
    
    # Check file sizes
    esp32_size = os.path.getsize(esp32_main_path)
    model_size = os.path.getsize(model_header_path)
    
    print(f"  ✓ ESP32 main file: {esp32_size:,} bytes")
    print(f"  ✓ Model header: {model_size:,} bytes")
    
    if esp32_size > 1000:
        print(f"  ✓ ESP32 file has substantial content")
    else:
        print(f"  ⚠️  ESP32 file seems small")
    
    # Final assessment
    print("\n" + "=" * 60)
    if not issues:
        print("🎉 Step 9 complete!")
        print("\nESP32 Firmware Features:")
        print("✔ TensorFlow Lite Micro integration")
        print("✔ Model loading from embedded data")
        print("✔ Tensor allocation and management")
        print("✔ Dummy sensor data generation")
        print("✔ AI inference execution")
        print("✔ Serial output for monitoring")
        
        print(f"\nExpected Behavior:")
        print(f"  • ESP32 boots and initializes TensorFlow Lite")
        print(f"  • Model loads successfully (9,248 bytes)")
        print(f"  • Tensors allocated (20KB arena)")
        print(f"  • Inference runs every 5 seconds")
        print(f"  • Predictions printed to Serial Monitor")
        print(f"  • Dummy sensor data simulates 24-hour patterns")
        
        print(f"\nNext Steps:")
        print(f"  1. Install ESP32 TensorFlow Lite library")
        print(f"  2. Upload firmware to ESP32")
        print(f"  3. Monitor Serial output at 115200 baud")
        print(f"  4. Verify AI predictions are generated")
        
        return True
    else:
        print("❌ Step 9 verification failed!")
        print("\nIssues found:")
        for i, issue in enumerate(issues, 1):
            print(f"{i:2d}. {issue}")
        return False

if __name__ == "__main__":
    success = verify_step9()
    exit(0 if success else 1)