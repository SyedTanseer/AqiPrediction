"""
Create TensorFlow Lite compatible model for ESP32 deployment
Uses LSTM instead of GRU for better TFLite compatibility
"""
import tensorflow as tf
import numpy as np
import os
import sys
sys.path.append('model')

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam

def create_tflite_compatible_model(input_shape=(24, 4), lstm_units=16, dropout_rate=0.2):
    """
    Create LSTM model that's compatible with TensorFlow Lite
    
    Args:
        input_shape: Input tensor shape (timesteps, features)
        lstm_units: Number of LSTM units
        dropout_rate: Dropout rate for regularization
        
    Returns:
        Compiled Keras model
    """
    model = Sequential([
        # LSTM layer - more TFLite compatible than GRU
        LSTM(lstm_units, 
             input_shape=input_shape,
             return_sequences=False,
             name='lstm_layer'),
        
        # Dropout for regularization
        Dropout(dropout_rate, name='dropout'),
        
        # Dense layers
        Dense(8, activation='relu', name='dense_hidden'),
        Dense(1, activation='linear', name='output')
    ])
    
    # Compile model
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae']
    )
    
    return model

def train_lstm_model():
    """Train LSTM model quickly for TFLite conversion"""
    
    print("🚀 Creating TensorFlow Lite Compatible Model")
    print("=" * 60)
    
    # Load training data
    print("Loading training data...")
    X_train = np.load("data/train_X.npy")
    y_train = np.load("data/train_y.npy")
    X_val = np.load("data/val_X.npy")
    y_val = np.load("data/val_y.npy")
    
    print(f"  Train: X{X_train.shape}, y{y_train.shape}")
    print(f"  Val:   X{X_val.shape}, y{y_val.shape}")
    
    # Create LSTM model
    print("\nCreating LSTM model...")
    model = create_tflite_compatible_model()
    
    print("\nModel Architecture:")
    model.summary()
    
    # Quick training
    print(f"\nTraining LSTM model (10 epochs)...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=10,
        batch_size=32,
        verbose=1
    )
    
    # Evaluate
    val_loss = min(history.history['val_loss'])
    val_mae = min(history.history['val_mae'])
    
    print(f"\nTraining Results:")
    print(f"  Best Val Loss: {val_loss:.6f}")
    print(f"  Best Val MAE: {val_mae:.6f}")
    
    return model

def convert_lstm_to_tflite(model):
    """Convert LSTM model to TensorFlow Lite"""
    
    print("\n🔄 Converting LSTM to TensorFlow Lite...")
    
    # Load representative data
    X_train = np.load("data/train_X.npy")
    representative_data = X_train[:100]
    
    def representative_dataset():
        for i in range(100):
            yield [representative_data[i:i+1].astype(np.float32)]
    
    # Create converter
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    
    # Convert with fallback strategy
    try:
        # Try quantized conversion
        print("  Attempting quantized conversion...")
        tflite_model = converter.convert()
        quantization_type = "Quantized"
        
    except Exception as e:
        print(f"  Quantized conversion failed: {str(e)[:100]}...")
        print("  Trying basic conversion...")
        
        # Fallback to basic conversion
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        tflite_model = converter.convert()
        quantization_type = "Basic optimization"
    
    # Save model
    os.makedirs("model", exist_ok=True)
    tflite_path = "model/lstm_model.tflite"
    
    with open(tflite_path, 'wb') as f:
        f.write(tflite_model)
    
    model_size = len(tflite_model)
    model_size_kb = model_size / 1024
    
    print(f"  ✓ Model saved: {tflite_path}")
    print(f"  ✓ Size: {model_size:,} bytes ({model_size_kb:.1f} KB)")
    print(f"  ✓ Quantization: {quantization_type}")
    
    return tflite_path, model_size_kb

def validate_tflite_model(tflite_path):
    """Validate the TFLite model"""
    
    print(f"\n🔍 Validating TensorFlow Lite Model...")
    
    try:
        # Load interpreter
        interpreter = tf.lite.Interpreter(model_path=tflite_path)
        interpreter.allocate_tensors()
        
        # Get details
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        print(f"  ✓ Model loaded successfully")
        print(f"  ✓ Input shape: {input_details[0]['shape']}")
        print(f"  ✓ Input type: {input_details[0]['dtype']}")
        print(f"  ✓ Output shape: {output_details[0]['shape']}")
        print(f"  ✓ Output type: {output_details[0]['dtype']}")
        
        # Test inference
        X_test = np.load("data/test_X.npy")
        test_sample = X_test[0:1].astype(np.float32)
        
        interpreter.set_tensor(input_details[0]['index'], test_sample)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]['index'])
        
        print(f"  ✓ Test inference successful")
        print(f"  ✓ Sample prediction: {output[0][0]:.6f}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Validation failed: {e}")
        return False

def main():
    """Main pipeline for TFLite compatible model"""
    
    # Train LSTM model
    model = train_lstm_model()
    
    # Convert to TFLite
    tflite_path, model_size_kb = convert_lstm_to_tflite(model)
    
    # Validate
    validation_success = validate_tflite_model(tflite_path)
    
    # Summary
    print("\n" + "=" * 60)
    print("🎉 TFLITE COMPATIBLE MODEL SUMMARY")
    print("=" * 60)
    
    print(f"\nModel Details:")
    print(f"  • Architecture: LSTM (TFLite compatible)")
    print(f"  • Parameters: {model.count_params():,}")
    print(f"  • File: {tflite_path}")
    print(f"  • Size: {model_size_kb:.1f} KB")
    
    print(f"\nESP32 Deployment:")
    if model_size_kb <= 20:
        print(f"  ✓ Size within ESP32 limits")
    else:
        print(f"  ⚠️  Size may be tight for ESP32")
    
    if validation_success:
        print(f"  ✓ Model validation successful")
        print(f"  ✓ Ready for ESP32 deployment")
    else:
        print(f"  ⚠️  Validation issues detected")
    
    print(f"\nNote: LSTM model replaces GRU for TensorFlow Lite compatibility")
    print(f"Performance should be similar for time-series prediction tasks")
    
    return validation_success

if __name__ == "__main__":
    success = main()
    
    if success:
        print("\n🎉 TensorFlow Lite compatible model created successfully!")
        print("Ready for ESP32 firmware integration!")
    else:
        print("\n❌ Model creation had issues!")