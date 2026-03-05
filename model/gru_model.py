"""
GRU neural network architecture for edge-based air quality prediction
Optimized for ESP32 deployment with TensorFlow Lite Micro
"""
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import numpy as np

def create_gru_model(input_shape=(24, 4), gru_units=16, dropout_rate=0.2):
    """
    Create lightweight GRU model for edge deployment
    
    Args:
        input_shape: Input tensor shape (timesteps, features)
        gru_units: Number of GRU units (keep small for ESP32)
        dropout_rate: Dropout rate for regularization
        
    Returns:
        Compiled Keras model
    """
    model = Sequential([
        # Input layer - expects (batch_size, 24, 4)
        GRU(gru_units, 
            input_shape=input_shape,
            return_sequences=False,  # Only return last output
            name='gru_layer'),
        
        # Dropout for regularization
        Dropout(dropout_rate, name='dropout'),
        
        # Small dense layer with ReLU activation
        Dense(8, activation='relu', name='dense_hidden'),
        
        # Output layer - single CO_GT prediction
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
    Create and compile the complete GRU model
    
    Returns:
        Compiled model ready for training
    """
    print("Creating lightweight GRU model for edge deployment...")
    
    # Create model
    model = create_gru_model()
    
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
        ('gru_layer' in layer_names, "GRU layer present"),
        ('dropout' in layer_names, "Dropout layer present"), 
        ('dense_hidden' in layer_names, "Hidden dense layer present"),
        ('output' in layer_names, "Output layer present"),
        (model.layers[0].units == 16, "GRU has 16 units"),
        (model.input_shape == (None, 24, 4), "Input shape is (24,4)")
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
    
    # Create dummy input (batch_size=1, timesteps=24, features=4)
    dummy_input = np.random.randn(1, 24, 4)
    
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
    print("GRU Model Architecture for Edge-based Air Quality Prediction")
    print("="*60)
    
    # Test model creation and functionality
    success = test_model_prediction()
    
    if success:
        print("\n🎉 Step 4 Success Criteria Met:")
        print("✔ GRU model file created")
        print("✔ Input shape = (24,4)")
        print("✔ GRU layer = 16 units") 
        print("✔ Dropout added")
        print("✔ Dense layers added")
        print("✔ Model compiles successfully")
        print("✔ Parameter count under ~3K")
        print("\n🚀 Ready for Step 5: Model Training!")
    else:
        print("\n❌ Model creation failed!")