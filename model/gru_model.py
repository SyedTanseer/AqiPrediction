"""
Dense neural network architecture for edge-based AQI prediction
Optimized for ESP32 deployment with TensorFlow Lite
Input: 6 pollutant features (PM2.5, PM10, NO2, SO2, CO, O3)
Output: AQI value (normalized)
"""
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
import numpy as np

def create_dense_model(input_shape=(6,), hidden_units=[32, 16, 8], dropout_rate=0.2):
    """
    Create lightweight Dense model for edge deployment
    
    Args:
        input_shape: Input tensor shape (6 features)
        hidden_units: List of hidden layer sizes
        dropout_rate: Dropout rate for regularization
        
    Returns:
        Compiled Keras model
    """
    model = Sequential([
        # Hidden layers
        Dense(hidden_units[0], activation='relu', input_shape=input_shape, name='dense1'),
        Dropout(dropout_rate, name='dropout1'),
        
        Dense(hidden_units[1], activation='relu', name='dense2'),
        Dropout(dropout_rate, name='dropout2'),
        
        Dense(hidden_units[2], activation='relu', name='dense3'),
        
        # Output layer - single AQI prediction
        Dense(1, activation='linear', name='output')
    ])
    
    return model

def compile_model(model, learning_rate=0.001):
    """
    Compile the model with appropriate optimizer and loss
    
    Args:
        model: Keras model to compile
        learning_rate: Learning rate for Adam optimizer
        
    Returns:
        Compiled model
    """
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='mse',  # Mean Squared Error for regression
        metrics=['mae']  # Mean Absolute Error for monitoring
    )
    
    return model

def get_model_info(model):
    """
    Get detailed model information for verification
    
    Args:
        model: Keras model
        
    Returns:
        Dictionary with model statistics
    """
    # Count parameters
    total_params = model.count_params()
    trainable_params = sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])
    
    # Estimate model size (rough approximation)
    # Each parameter is typically 4 bytes (float32)
    estimated_size_bytes = total_params * 4
    estimated_size_kb = estimated_size_bytes / 1024
    
    return {
        'total_params': total_params,
        'trainable_params': trainable_params,
        'estimated_size_kb': estimated_size_kb,
        'input_shape': model.input_shape,
        'output_shape': model.output_shape
    }

def create_and_compile_model():
    """
    Create and compile the complete Dense model
    
    Returns:
        Compiled model ready for training
    """
    print("Creating lightweight Dense model for edge AQI prediction...")
    
    # Create model
    model = create_dense_model()
    
    # Compile model
    model = compile_model(model)
    
    # Display model information
    print("\n" + "="*50)
    print("MODEL ARCHITECTURE")
    print("="*50)
    model.summary()
    
    # Get detailed info
    info = get_model_info(model)
    
    print("\n" + "="*50)
    print("MODEL SPECIFICATIONS")
    print("="*50)
    print(f"Input shape: {info['input_shape']}")
    print(f"Output shape: {info['output_shape']}")
    print(f"Total parameters: {info['total_params']:,}")
    print(f"Trainable parameters: {info['trainable_params']:,}")
    print(f"Estimated size: {info['estimated_size_kb']:.1f} KB")
    
    # Verify edge constraints
    print("\n" + "="*50)
    print("EDGE DEPLOYMENT VERIFICATION")
    print("="*50)
    
    if info['total_params'] <= 3000:
        print("✓ Parameter count within target (≤3K)")
    else:
        print(f"⚠️  Parameter count {info['total_params']:,} exceeds 3K target")
    
    if info['estimated_size_kb'] <= 20:
        print("✓ Estimated size within ESP32 target (≤20KB)")
    else:
        print(f"⚠️  Estimated size {info['estimated_size_kb']:.1f}KB may exceed ESP32 limit")
    
    # Verify architecture components
    layer_names = [layer.name for layer in model.layers]
    
    checks = [
        ('dense1' in layer_names, "Dense input layer present"),
        ('dropout1' in layer_names, "Dropout layer present"), 
        ('dense2' in layer_names, "Second dense layer present"),
        ('dense3' in layer_names, "Third dense layer present"),
        ('output' in layer_names, "Output layer present"),
        (model.input_shape == (None, 6), "Input shape is (6,) — 6 pollutant features"),
    ]
    
    for check, description in checks:
        if check:
            print(f"✓ {description}")
        else:
            print(f"❌ {description}")
    
    print("\nModel ready for training!")
    return model

def test_model_prediction():
    """
    Test model with dummy data to verify it works
    """
    print("\n" + "="*50)
    print("MODEL FUNCTIONALITY TEST")
    print("="*50)
    
    # Create model
    model = create_and_compile_model()
    
    # Create dummy input (batch_size=1, features=6)
    dummy_input = np.random.randn(1, 6)
    
    # Test prediction
    try:
        prediction = model.predict(dummy_input, verbose=0)
        print(f"✓ Model prediction successful")
        print(f"  Input shape: {dummy_input.shape}")
        print(f"  Output shape: {prediction.shape}")
        print(f"  Sample prediction: {prediction[0][0]:.6f}")
        return True
    except Exception as e:
        print(f"❌ Model prediction failed: {e}")
        return False

if __name__ == "__main__":
    print("Dense Model Architecture for Edge-based AQI Prediction")
    print("="*60)
    
    # Test model creation and functionality
    success = test_model_prediction()
    
    if success:
        print("\n🎉 Model Creation Success:")
        print("✔ Dense model file created")
        print("✔ Input shape = (6,) — PM2.5, PM10, NO2, SO2, CO, O3")
        print("✔ Dense layers: 32→16→8→1")
        print("✔ Dropout added")
        print("✔ Model compiles successfully")
        print("✔ Direct AQI prediction output")
        print("\n🚀 Ready for training!")
    else:
        print("\n❌ Model creation failed!")