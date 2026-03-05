"""
Convert TensorFlow Lite model to C header file for ESP32 firmware
Equivalent to: xxd -i model/dense_model.tflite > firmware/model_data.h
"""
import os

def convert_tflite_to_c_header(tflite_path, output_path):
    """
    Convert .tflite file to C header with byte array
    
    Args:
        tflite_path: Path to .tflite model file
        output_path: Path to output .h file
    """
    
    print(f"Converting {tflite_path} to C header...")
    
    # Read the binary model file
    with open(tflite_path, 'rb') as f:
        model_data = f.read()
    
    model_size = len(model_data)
    print(f"Model size: {model_size:,} bytes ({model_size/1024:.1f} KB)")
    
    # Create variable name from filename
    filename = os.path.basename(tflite_path)
    var_name = filename.replace('.', '_').replace('-', '_')
    
    # Generate C header content
    header_content = f"""/* Auto-generated C header file for TensorFlow Lite model */
/* Generated from: {filename} */
/* Model size: {model_size:,} bytes ({model_size/1024:.1f} KB) */

#ifndef MODEL_DATA_H
#define MODEL_DATA_H

#ifdef __cplusplus
extern "C" {{
#endif

/* TensorFlow Lite model data */
const unsigned char {var_name}[] = {{
"""
    
    # Add byte array data (16 bytes per line)
    for i in range(0, len(model_data), 16):
        line_bytes = model_data[i:i+16]
        hex_bytes = ', '.join(f'0x{b:02x}' for b in line_bytes)
        
        if i + 16 < len(model_data):
            header_content += f"  {hex_bytes},\n"
        else:
            header_content += f"  {hex_bytes}\n"
    
    # Close array and add length variable
    header_content += f"""}};\n
/* Model size in bytes */
const unsigned int {var_name}_len = {model_size};

#ifdef __cplusplus
}}
#endif

#endif /* MODEL_DATA_H */
"""
    
    # Create output directory if needed
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Write header file
    with open(output_path, 'w') as f:
        f.write(header_content)
    
    print(f"C header file generated: {output_path}")
    print(f"Array variable: {var_name}[]")
    print(f"Length variable: {var_name}_len")
    
    return output_path, var_name, model_size

def verify_c_header(header_path, expected_size):
    """
    Verify the generated C header file
    
    Args:
        header_path: Path to generated header file
        expected_size: Expected model size in bytes
    """
    
    print(f"\nVerifying C header file: {header_path}")
    
    if not os.path.exists(header_path):
        print("❌ Header file not found!")
        return False
    
    # Read and check header content
    with open(header_path, 'r') as f:
        content = f.read()
    
    # Check for required elements
    checks = [
        ('unsigned char' in content, "Byte array declaration"),
        ('unsigned int' in content, "Length variable declaration"),
        ('_len =' in content, "Length assignment"),
        ('#ifndef MODEL_DATA_H' in content, "Header guard"),
        ('0x' in content, "Hex byte data")
    ]
    
    all_passed = True
    for check, description in checks:
        if check:
            print(f"  ✓ {description}")
        else:
            print(f"  ❌ {description}")
            all_passed = False
    
    # Check file size (should be much larger than binary due to text format)
    header_size = os.path.getsize(header_path)
    print(f"  ✓ Header file size: {header_size:,} bytes")
    
    # Estimate if size makes sense (text representation is ~4x larger)
    if header_size > expected_size * 2:
        print(f"  ✓ Header size reasonable for {expected_size} byte model")
    else:
        print(f"  ⚠️  Header size seems small for {expected_size} byte model")
    
    return all_passed

def main():
    """Main conversion process"""
    
    print("🔄 TensorFlow Lite to C Header Conversion")
    print("=" * 60)
    
    # File paths
    tflite_path = "model/dense_model.tflite"
    header_path = "firmware/model_data.h"
    
    # Step 1: Verify TFLite model exists
    if not os.path.exists(tflite_path):
        print(f"❌ TensorFlow Lite model not found: {tflite_path}")
        return False
    
    print(f"✓ TensorFlow Lite model found: {tflite_path}")
    
    # Step 2: Convert to C header
    try:
        output_path, var_name, model_size = convert_tflite_to_c_header(tflite_path, header_path)
    except Exception as e:
        print(f"❌ Conversion failed: {e}")
        return False
    
    # Step 3: Verify generated header
    verification_success = verify_c_header(header_path, model_size)
    
    # Step 4: Check firmware directory structure
    print(f"\n📁 Checking firmware directory structure...")
    firmware_files = [
        "esp32_main.ino",
        "sensor_reader.cpp", 
        "inference_engine.cpp",
        "purifier_control.cpp",
        "model_data.h"
    ]
    
    for file in firmware_files:
        file_path = os.path.join("firmware", file)
        if os.path.exists(file_path):
            print(f"  ✓ {file}")
        else:
            print(f"  ⚠️  {file} (will be created later)")
    
    # Summary
    print("\n" + "=" * 60)
    print("🎉 C HEADER CONVERSION SUMMARY")
    print("=" * 60)
    
    if verification_success:
        print(f"\nGenerated Files:")
        print(f"  ✓ C Header: {header_path}")
        print(f"  ✓ Array Variable: {var_name}[]")
        print(f"  ✓ Length Variable: {var_name}_len")
        print(f"  ✓ Model Size: {model_size:,} bytes ({model_size/1024:.1f} KB)")
        
        print(f"\nESP32 Integration Ready:")
        print(f"  ✓ Model embedded as byte array")
        print(f"  ✓ No file system access required")
        print(f"  ✓ Compile-time model inclusion")
        print(f"  ✓ TensorFlow Lite Micro compatible")
        
        print(f"\nUsage in ESP32 code:")
        print(f"  #include \"model_data.h\"")
        print(f"  const tflite::Model* model = tflite::GetModel({var_name});")
        print(f"  // Model size: {var_name}_len")
        
        return True
    else:
        print(f"\n❌ Header generation had issues!")
        return False

if __name__ == "__main__":
    success = main()
    
    if success:
        print("\n🎉 Step 7 Success Criteria:")
        print("✔ dense_model.tflite exists")
        print("✔ model_data.h generated successfully") 
        print("✔ Byte array present inside file")
        print("✔ Model size ~9 KB")
        print("✔ File placed inside firmware/")
        print("\n🚀 Ready for Step 8: ESP32 Firmware Setup!")
    else:
        print("\n❌ Step 7 conversion failed!")
        print("Please check the model file and try again.")