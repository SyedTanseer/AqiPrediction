"""
GRU model training pipeline for air quality prediction
Includes callbacks, monitoring, and visualization
"""
import numpy as np
import matplotlib.pyplot as plt
import json
import os
import sys
import random

# Add parent directory to path to import model
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from model.gru_model import create_and_compile_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import tensorflow as tf

# Set random seeds for reproducibility (Flaw 9 fix)
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

def load_training_data(data_dir="../data"):
    """
    Load preprocessed training and validation datasets
    
    Args:
        data_dir: Directory containing .npy files
        
    Returns:
        Tuple of (X_train, y_train, X_val, y_val)
    """
    print("Loading training datasets...")
    
    X_train = np.load(os.path.join(data_dir, "train_X.npy"))
    y_train = np.load(os.path.join(data_dir, "train_y.npy"))
    X_val = np.load(os.path.join(data_dir, "val_X.npy"))
    y_val = np.load(os.path.join(data_dir, "val_y.npy"))
    
    print(f"  Train: X{X_train.shape}, y{y_train.shape}")
    print(f"  Val:   X{X_val.shape}, y{y_val.shape}")
    
    return X_train, y_train, X_val, y_val

def create_callbacks(model_save_path="../model/best_gru_model.h5"):
    """
    Create training callbacks for monitoring and saving
    
    Args:
        model_save_path: Path to save best model
        
    Returns:
        List of callbacks
    """
    # Create model directory if it doesn't exist
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    
    # Early stopping to prevent overfitting
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True,
        verbose=1
    )
    
    # Model checkpoint to save best model
    model_checkpoint = ModelCheckpoint(
        filepath=model_save_path,
        monitor='val_loss',
        save_best_only=True,
        save_weights_only=False,
        verbose=1
    )
    
    return [early_stopping, model_checkpoint]

def train_model(epochs=50, batch_size=32, data_dir="../data", 
                model_save_path="../model/best_gru_model.h5"):
    """
    Train the GRU model with monitoring and callbacks
    
    Args:
        epochs: Maximum number of training epochs
        batch_size: Training batch size
        data_dir: Directory containing training data
        model_save_path: Path to save best model
        
    Returns:
        Tuple of (model, history)
    """
    print("Starting GRU model training...")
    print("=" * 50)
    
    # Load data
    X_train, y_train, X_val, y_val = load_training_data(data_dir)
    
    # Create model
    print("\nCreating GRU model...")
    model = create_and_compile_model()
    
    # Create callbacks
    callbacks = create_callbacks(model_save_path)
    
    # Training configuration
    print(f"\nTraining Configuration:")
    print(f"  Epochs: {epochs}")
    print(f"  Batch Size: {batch_size}")
    print(f"  Early Stopping: patience=5, monitor=val_loss")
    print(f"  Model Checkpoint: {model_save_path}")
    
    print(f"\nStarting training...")
    print("=" * 50)
    
    # Train model
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )
    
    print("\nTraining completed!")
    return model, history

def save_training_history(history, save_path="../reports/training_history.json"):
    """
    Save training history to JSON file
    
    Args:
        history: Keras training history object
        save_path: Path to save history JSON
    """
    # Create reports directory
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Convert history to serializable format
    history_dict = {
        'loss': history.history['loss'],
        'val_loss': history.history['val_loss'],
        'mae': history.history['mae'],
        'val_mae': history.history['val_mae'],
        'epochs': len(history.history['loss'])
    }
    
    # Save to JSON
    with open(save_path, 'w') as f:
        json.dump(history_dict, f, indent=2)
    
    print(f"Training history saved to: {save_path}")

def plot_training_curves(history, plots_dir="../reports/plots"):
    """
    Generate and save training curve plots
    
    Args:
        history: Keras training history object
        plots_dir: Directory to save plots
    """
    # Create plots directory
    os.makedirs(plots_dir, exist_ok=True)
    
    # Set up the plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss curve
    ax1.plot(history.history['loss'], label='Training Loss', color='blue')
    ax1.plot(history.history['val_loss'], label='Validation Loss', color='red')
    ax1.set_title('Model Loss Curve')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss (MSE)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # MAE curve
    ax2.plot(history.history['mae'], label='Training MAE', color='blue')
    ax2.plot(history.history['val_mae'], label='Validation MAE', color='red')
    ax2.set_title('Model MAE Curve')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Mean Absolute Error')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Save individual plots
    plt.tight_layout()
    
    # Save loss curve
    fig1, ax1_single = plt.subplots(figsize=(8, 6))
    ax1_single.plot(history.history['loss'], label='Training Loss', color='blue', linewidth=2)
    ax1_single.plot(history.history['val_loss'], label='Validation Loss', color='red', linewidth=2)
    ax1_single.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    ax1_single.set_xlabel('Epoch', fontsize=12)
    ax1_single.set_ylabel('Loss (MSE)', fontsize=12)
    ax1_single.legend(fontsize=11)
    ax1_single.grid(True, alpha=0.3)
    
    loss_path = os.path.join(plots_dir, "loss_curve.png")
    fig1.savefig(loss_path, dpi=300, bbox_inches='tight')
    plt.close(fig1)
    
    # Save MAE curve
    fig2, ax2_single = plt.subplots(figsize=(8, 6))
    ax2_single.plot(history.history['mae'], label='Training MAE', color='blue', linewidth=2)
    ax2_single.plot(history.history['val_mae'], label='Validation MAE', color='red', linewidth=2)
    ax2_single.set_title('Training and Validation MAE', fontsize=14, fontweight='bold')
    ax2_single.set_xlabel('Epoch', fontsize=12)
    ax2_single.set_ylabel('Mean Absolute Error', fontsize=12)
    ax2_single.legend(fontsize=11)
    ax2_single.grid(True, alpha=0.3)
    
    mae_path = os.path.join(plots_dir, "mae_curve.png")
    fig2.savefig(mae_path, dpi=300, bbox_inches='tight')
    plt.close(fig2)
    
    print(f"Loss curve saved to: {loss_path}")
    print(f"MAE curve saved to: {mae_path}")

def evaluate_final_model(model, data_dir="../data"):
    """
    Evaluate the trained model on test dataset
    
    Args:
        model: Trained Keras model
        data_dir: Directory containing test data
        
    Returns:
        Tuple of (test_loss, test_mae)
    """
    print("\nEvaluating model on test dataset...")
    
    # Load test data
    X_test = np.load(os.path.join(data_dir, "test_X.npy"))
    y_test = np.load(os.path.join(data_dir, "test_y.npy"))
    
    print(f"Test dataset: X{X_test.shape}, y{y_test.shape}")
    
    # Evaluate model
    test_loss, test_mae = model.evaluate(X_test, y_test, verbose=0)
    
    print(f"\nFinal Test Results:")
    print(f"  Test Loss (MSE): {test_loss:.6f}")
    print(f"  Test MAE: {test_mae:.6f}")
    
    return test_loss, test_mae

def main():
    """
    Main training pipeline
    """
    print("🚀 GRU Model Training Pipeline")
    print("=" * 60)
    
    # Train model
    model, history = train_model(
        epochs=50,
        batch_size=32,
        data_dir="../data",
        model_save_path="../model/best_gru_model.h5"
    )
    
    # Save training history
    save_training_history(history, "../reports/training_history.json")
    
    # Generate training curves
    plot_training_curves(history, "../reports/plots")
    
    # Final evaluation
    test_loss, test_mae = evaluate_final_model(model, "../data")
    
    # Training summary
    print("\n" + "=" * 60)
    print("🎉 TRAINING COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    
    print(f"\nTraining Summary:")
    print(f"  Total Epochs: {len(history.history['loss'])}")
    print(f"  Best Val Loss: {min(history.history['val_loss']):.6f}")
    print(f"  Best Val MAE: {min(history.history['val_mae']):.6f}")
    print(f"  Final Test Loss: {test_loss:.6f}")
    print(f"  Final Test MAE: {test_mae:.6f}")
    
    print(f"\nGenerated Files:")
    print(f"  ✓ model/best_gru_model.h5")
    print(f"  ✓ reports/training_history.json")
    print(f"  ✓ reports/plots/loss_curve.png")
    print(f"  ✓ reports/plots/mae_curve.png")
    
    print(f"\n🚀 Ready for Step 6: TensorFlow Lite Conversion!")

if __name__ == "__main__":
    main()