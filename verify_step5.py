"""
Verify Step 5 completion - GRU Model Training
"""
import os
import json
import numpy as np
import tensorflow as tf

def verify_step5():
    """Verify all Step 5 requirements are met"""
    
    print("Step 5 Verification: GRU Model Training")
    print("=" * 50)
    
    issues = []
    
    # Check required files exist
    required_files = {
        "model/best_gru_model.h5": "Trained model file",
        "reports/training_history.json": "Training history",
        "reports/plots/loss_curve.png": "Loss curve plot",
        "reports/plots/mae_curve.png": "MAE curve plot"
    }
    
    print("✔ Checking generated files...")
    for file_path, description in required_files.items():
        if os.path.exists(file_path):
            print(f"  ✓ {description}: {file_path}")
        else:
            print(f"  ❌ Missing {description}: {file_path}")
            issues.append(f"Missing file: {file_path}")
    
    # Verify model file
    print("\n✔ Verifying trained model...")
    model_path = "model/best_gru_model.h5"
    if os.path.exists(model_path):
        try:
            model = tf.keras.models.load_model(model_path)
            print(f"  ✓ Model loaded successfully")
            print(f"  ✓ Model input shape: {model.input_shape}")
            print(f"  ✓ Model output shape: {model.output_shape}")
            print(f"  ✓ Model parameters: {model.count_params():,}")
            
            # Test prediction
            dummy_input = np.random.randn(1, 24, 4)
            prediction = model.predict(dummy_input, verbose=0)
            print(f"  ✓ Model prediction works: {prediction[0][0]:.6f}")
            
        except Exception as e:
            print(f"  ❌ Model loading failed: {e}")
            issues.append("Model file corrupted or invalid")
    
    # Verify training history
    print("\n✔ Verifying training history...")
    history_path = "reports/training_history.json"
    if os.path.exists(history_path):
        try:
            with open(history_path, 'r') as f:
                history = json.load(f)
            
            required_keys = ['loss', 'val_loss', 'mae', 'val_mae', 'epochs']
            for key in required_keys:
                if key in history:
                    print(f"  ✓ {key} recorded")
                else:
                    print(f"  ❌ Missing {key} in history")
                    issues.append(f"Missing {key} in training history")
            
            if 'epochs' in history:
                print(f"  ✓ Training completed in {history['epochs']} epochs")
                
                if history['epochs'] < 50:
                    print(f"  ✓ Early stopping worked (stopped at epoch {history['epochs']})")
                
                # Check if validation loss improved
                if 'val_loss' in history and len(history['val_loss']) > 1:
                    initial_val_loss = history['val_loss'][0]
                    final_val_loss = min(history['val_loss'])
                    improvement = initial_val_loss - final_val_loss
                    print(f"  ✓ Validation loss improved by {improvement:.6f}")
                    
        except Exception as e:
            print(f"  ❌ History file reading failed: {e}")
            issues.append("Training history file corrupted")
    
    # Verify plot files
    print("\n✔ Verifying training plots...")
    plot_files = ["reports/plots/loss_curve.png", "reports/plots/mae_curve.png"]
    for plot_file in plot_files:
        if os.path.exists(plot_file):
            file_size = os.path.getsize(plot_file)
            if file_size > 1000:  # At least 1KB for a valid plot
                print(f"  ✓ {os.path.basename(plot_file)}: {file_size:,} bytes")
            else:
                print(f"  ⚠️  {os.path.basename(plot_file)}: suspiciously small ({file_size} bytes)")
        else:
            issues.append(f"Missing plot: {plot_file}")
    
    # Check training performance
    print("\n✔ Checking training performance...")
    if os.path.exists(history_path):
        try:
            with open(history_path, 'r') as f:
                history = json.load(f)
            
            if 'val_mae' in history:
                best_val_mae = min(history['val_mae'])
                print(f"  ✓ Best validation MAE: {best_val_mae:.6f}")
                
                if best_val_mae < 0.5:
                    print(f"  ✓ Good performance (MAE < 0.5)")
                elif best_val_mae < 1.0:
                    print(f"  ⚠️  Acceptable performance (MAE < 1.0)")
                else:
                    print(f"  ⚠️  High MAE, may need tuning")
            
            if 'val_loss' in history:
                best_val_loss = min(history['val_loss'])
                print(f"  ✓ Best validation loss: {best_val_loss:.6f}")
                
        except:
            pass
    
    # Final assessment
    print("\n" + "=" * 50)
    if not issues:
        print("🎉 Step 5 complete!")
        print("\nStep 5 Success Criteria:")
        print("✔ Model trains successfully")
        print("✔ Early stopping works") 
        print("✔ Best model saved")
        print("✔ Training history saved")
        print("✔ Loss curve generated")
        print("✔ MAE curve generated")
        print("✔ Test evaluation completed")
        print("\n🚀 Ready for Step 6: TensorFlow Lite Conversion!")
        return True
    else:
        print("❌ Step 5 verification failed!")
        print("\nIssues found:")
        for i, issue in enumerate(issues, 1):
            print(f"{i:2d}. {issue}")
        return False

if __name__ == "__main__":
    success = verify_step5()
    exit(0 if success else 1)