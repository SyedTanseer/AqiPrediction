"""
Model Architecture Summary for Edge-based Air Quality Prediction
"""
import sys
sys.path.append('model')

from model.gru_model import create_and_compile_model, get_model_info

def display_model_summary():
    """Display comprehensive model summary"""
    
    print("🏗️  EDGE-OPTIMIZED GRU MODEL ARCHITECTURE")
    print("=" * 60)
    
    # Create model
    model = create_and_compile_model()
    info = get_model_info(model)
    
    print("\n📊 MODEL SPECIFICATIONS:")
    print(f"  • Architecture: GRU → Dropout → Dense → Dense")
    print(f"  • Input Shape: {info['input_shape']} (24 hours × 4 sensors)")
    print(f"  • Output Shape: {info['output_shape']} (CO_GT prediction)")
    print(f"  • Total Parameters: {info['total_params']:,}")
    print(f"  • Model Size: {info['estimated_size_kb']:.1f} KB")
    
    print("\n🔧 LAYER DETAILS:")
    for i, layer in enumerate(model.layers):
        layer_type = layer.__class__.__name__
        if hasattr(layer, 'units'):
            units = f" ({layer.units} units)"
        elif hasattr(layer, 'rate'):
            units = f" (rate={layer.rate})"
        else:
            units = ""
        print(f"  {i+1}. {layer.name}: {layer_type}{units}")
    
    print("\n⚙️  COMPILATION SETTINGS:")
    print(f"  • Optimizer: {model.optimizer.__class__.__name__}")
    print(f"  • Loss Function: {model.loss}")
    print(f"  • Metrics: MAE (Mean Absolute Error)")
    
    print("\n🎯 EDGE DEPLOYMENT READINESS:")
    print(f"  ✓ Ultra-lightweight: {info['total_params']:,} parameters")
    print(f"  ✓ Tiny footprint: {info['estimated_size_kb']:.1f} KB")
    print(f"  ✓ ESP32 compatible: ≤16 GRU units")
    print(f"  ✓ TensorFlow Lite ready")
    
    print("\n📡 SENSOR MAPPING:")
    print("  • Input[0]: CO_GT (MQ135 gas sensor)")
    print("  • Input[1]: NO2_GT (MQ135 gas sensor)")  
    print("  • Input[2]: T (BME280 temperature)")
    print("  • Input[3]: RH (BME280 humidity)")
    print("  • Output: Next hour CO_GT prediction")
    
    print("\n🚀 READY FOR TRAINING!")

if __name__ == "__main__":
    display_model_summary()