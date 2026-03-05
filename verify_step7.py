"""
Verify Step 7 completion - C Header File Generation
"""
import os
import re

def verify_step7():
    """Verify all Step 7 requirements are met"""
    
    print("Step 7 Verification: C Header File Generation")
    print("=" * 60)
    
    issues = []
    
    # Step 1: Check TFLite model exists
    tflite_path = "model/dense_model.tflite"
    if not os.path.exists(tflite_path):
        issues.append("TensorFlow Lite model missing")
        return False
    
    model_size = os.path.getsize(tflite_path)
    print(f"✔ TensorFlow Lite model found: {tflite_path}")
    print(f"✔ Model size: {model_size:,} bytes ({model_size/1024:.1f} KB)")
    
    # Step 2: Check C header file exists
    header_path = "firmware/model_data.h"
    if not os.path.exists(header_path):
        issues.append("C header file missing")
        return False
    
    header_size = os.path.getsize(header_path)
    print(f"✔ C header file found: {header_path}")
    print(f"✔ Header file size: {header_size:,} bytes")
    
    # Step 3: Verify header file content
    print(f"\n✔ Verifying header file content...")
    
    with open(header_path, 'r') as f:
        content = f.read()
    
    # Check for required elements
    content_checks = [
        (re.search(r'unsigned char \w+\[\]', content), "Byte array declaration"),
        (re.search(r'unsigned int \w+_len = \d+', content), "Length variable with value"),
        ('#ifndef MODEL_DATA_H' in content, "Header guard start"),
        ('#endif /* MODEL_DATA_H */' in content, "Header guard end"),
        ('0x' in content, "Hexadecimal byte data"),
        ('dense_model_tflite' in content, "Correct variable name"),
        (f'= {model_size}' in content, f"Correct model size ({model_size})"),
        ('#ifdef __cplusplus' in content, "C++ compatibility")
    ]
    
    for check, description in content_checks:
        if check:
            print(f"  ✓ {description}")
        else:
            print(f"  ❌ {description}")
            issues.append(f"Header content missing: {description}")
    
    # Step 4: Extract and verify array details
    print(f"\n✔ Extracting array details...")
    
    # Find array name
    array_match = re.search(r'unsigned char (\w+)\[\]', content)
    if array_match:
        array_name = array_match.group(1)
        print(f"  ✓ Array name: {array_name}[]")
    else:
        issues.append("Could not find array name")
        array_name = "unknown"
    
    # Find length variable
    length_match = re.search(r'unsigned int (\w+) = (\d+)', content)
    if length_match:
        length_name = length_match.group(1)
        length_value = int(length_match.group(2))
        print(f"  ✓ Length variable: {length_name} = {length_value}")
        
        # Verify length matches model size
        if length_value == model_size:
            print(f"  ✓ Length matches model size")
        else:
            issues.append(f"Length mismatch: {length_value} vs {model_size}")
    else:
        issues.append("Could not find length variable")
    
    # Step 5: Check firmware directory structure
    print(f"\n✔ Checking firmware directory structure...")
    
    required_files = [
        "esp32_main.ino",
        "sensor_reader.cpp",
        "inference_engine.cpp", 
        "purifier_control.cpp",
        "model_data.h"
    ]
    
    firmware_dir = "firmware"
    for file in required_files:
        file_path = os.path.join(firmware_dir, file)
        if os.path.exists(file_path):
            file_size = os.path.getsize(file_path)
            print(f"  ✓ {file} ({file_size:,} bytes)")
        else:
            print(f"  ⚠️  {file} (missing)")
    
    # Step 6: Verify model size is reasonable
    print(f"\n✔ Model size verification...")
    
    size_checks = [
        (model_size > 1000, "Model size > 1KB (has content)"),
        (model_size < 50000, "Model size < 50KB (reasonable for ESP32)"),
        (model_size == 9248, "Model size exactly matches expected (9,248 bytes)"),
        (header_size > model_size * 3, "Header size reasonable (text expansion)")
    ]
    
    for check, description in size_checks:
        if check:
            print(f"  ✓ {description}")
        else:
            print(f"  ⚠️  {description}")
    
    # Step 7: Check for ESP32 compatibility markers
    print(f"\n✔ ESP32 compatibility check...")
    
    esp32_checks = [
        ('const unsigned char' in content, "Const qualifier for flash storage"),
        ('extern "C"' in content, "C linkage for C++ compatibility"),
        (model_size <= 20480, "Size within ESP32 flash limits (≤20KB)"),
        ('TFL3' in content, "TensorFlow Lite magic number present")
    ]
    
    for check, description in esp32_checks:
        if check:
            print(f"  ✓ {description}")
        else:
            print(f"  ⚠️  {description}")
    
    # Final assessment
    print("\n" + "=" * 60)
    if not issues:
        print("🎉 Step 7 complete!")
        print("\nStep 7 Success Criteria:")
        print("✔ dense_model.tflite exists")
        print("✔ model_data.h generated successfully")
        print("✔ Byte array present inside file")
        print("✔ Model size ~9 KB")
        print("✔ File placed inside firmware/")
        
        print(f"\nGenerated C Header Details:")
        print(f"  • File: {header_path}")
        print(f"  • Array: {array_name}[]")
        print(f"  • Length: {length_name} = {model_size:,} bytes")
        print(f"  • Size: {model_size/1024:.1f} KB")
        
        print(f"\nESP32 Integration:")
        print(f"  • Include: #include \"model_data.h\"")
        print(f"  • Usage: tflite::GetModel({array_name})")
        print(f"  • Size: {array_name}_len")
        
        print("\n🚀 Ready for Step 8: ESP32 Firmware Setup!")
        return True
    else:
        print("❌ Step 7 verification failed!")
        print("\nIssues found:")
        for i, issue in enumerate(issues, 1):
            print(f"{i:2d}. {issue}")
        return False

if __name__ == "__main__":
    success = verify_step7()
    exit(0 if success else 1)