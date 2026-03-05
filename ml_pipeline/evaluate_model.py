"""
Standalone model evaluation module for air quality prediction
Evaluates both GRU and Dense models with comprehensive metrics

FIX: Flaw 17 - Previously an empty stub, now fully implemented
"""
import numpy as np
import json
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

def load_test_data(data_dir="../data"):
    """Load test datasets"""
    X_test = np.load(os.path.join(data_dir, "test_X.npy"))
    y_test = np.load(os.path.join(data_dir, "test_y.npy"))
    print(f"Test data loaded: X{X_test.shape}, y{y_test.shape}")
    return X_test, y_test

def compute_metrics(y_true, y_pred):
    """Compute comprehensive regression metrics"""
    mse = np.mean((y_true - y_pred) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(y_true - y_pred))
    
    # R² score
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0
    
    # Max error
    max_err = np.max(np.abs(y_true - y_pred))
    
    return {
        'mse': float(mse),
        'rmse': float(rmse),
        'mae': float(mae),
        'r2': float(r2),
        'max_error': float(max_err),
        'num_samples': int(len(y_true))
    }

def evaluate_keras_model(model_path, X_test, y_test, model_name="GRU"):
    """Evaluate a Keras .h5 model"""
    import tensorflow as tf
    
    print(f"\n{'='*50}")
    print(f"Evaluating {model_name} Model: {model_path}")
    print(f"{'='*50}")
    
    if not os.path.exists(model_path):
        print(f"  ⚠️  Model file not found: {model_path}")
        return None
    
    try:
        model = tf.keras.models.load_model(model_path)
        y_pred = model.predict(X_test, verbose=0).flatten()
        metrics = compute_metrics(y_test, y_pred)
        
        print(f"  MSE:       {metrics['mse']:.6f}")
        print(f"  RMSE:      {metrics['rmse']:.6f}")
        print(f"  MAE:       {metrics['mae']:.6f}")
        print(f"  R²:        {metrics['r2']:.6f}")
        print(f"  Max Error: {metrics['max_error']:.6f}")
        print(f"  Samples:   {metrics['num_samples']}")
        
        metrics['model_name'] = model_name
        metrics['model_path'] = model_path
        return metrics
        
    except Exception as e:
        print(f"  ❌ Evaluation failed: {e}")
        return None

def evaluate_tflite_model(tflite_path, X_test, y_test, model_name="Dense-TFLite"):
    """Evaluate a TFLite model"""
    import tensorflow as tf
    
    print(f"\n{'='*50}")
    print(f"Evaluating {model_name} Model: {tflite_path}")
    print(f"{'='*50}")
    
    if not os.path.exists(tflite_path):
        print(f"  ⚠️  Model file not found: {tflite_path}")
        return None
    
    try:
        interpreter = tf.lite.Interpreter(model_path=tflite_path)
        interpreter.allocate_tensors()
        
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        predictions = []
        for i in range(len(X_test)):
            sample = X_test[i:i+1].astype(np.float32)
            interpreter.set_tensor(input_details[0]['index'], sample)
            interpreter.invoke()
            pred = interpreter.get_tensor(output_details[0]['index'])
            predictions.append(pred[0][0])
        
        y_pred = np.array(predictions)
        metrics = compute_metrics(y_test, y_pred)
        
        print(f"  MSE:       {metrics['mse']:.6f}")
        print(f"  RMSE:      {metrics['rmse']:.6f}")
        print(f"  MAE:       {metrics['mae']:.6f}")
        print(f"  R²:        {metrics['r2']:.6f}")
        print(f"  Max Error: {metrics['max_error']:.6f}")
        print(f"  Samples:   {metrics['num_samples']}")
        
        # Model size info
        model_size = os.path.getsize(tflite_path)
        print(f"  File Size: {model_size:,} bytes ({model_size/1024:.1f} KB)")
        
        metrics['model_name'] = model_name
        metrics['model_path'] = tflite_path
        metrics['model_size_bytes'] = model_size
        return metrics
        
    except Exception as e:
        print(f"  ❌ Evaluation failed: {e}")
        return None

def run_full_evaluation(data_dir="../data", model_dir="../model", reports_dir="../reports"):
    """Run evaluation on all available models"""
    
    print("🔍 COMPREHENSIVE MODEL EVALUATION")
    print("=" * 60)
    
    X_test, y_test = load_test_data(data_dir)
    
    all_metrics = {}
    
    # Evaluate GRU model (.h5)
    gru_path = os.path.join(model_dir, "best_gru_model.h5")
    gru_metrics = evaluate_keras_model(gru_path, X_test, y_test, "GRU")
    if gru_metrics:
        all_metrics['gru'] = gru_metrics
    
    # Evaluate Dense model (.h5)
    dense_h5_path = os.path.join(model_dir, "best_dense_model.h5")
    dense_h5_metrics = evaluate_keras_model(dense_h5_path, X_test, y_test, "Dense-Keras")
    if dense_h5_metrics:
        all_metrics['dense_keras'] = dense_h5_metrics
    
    # Evaluate Dense TFLite model
    tflite_path = os.path.join(model_dir, "dense_model.tflite")
    tflite_metrics = evaluate_tflite_model(tflite_path, X_test, y_test, "Dense-TFLite")
    if tflite_metrics:
        all_metrics['dense_tflite'] = tflite_metrics
    
    # Comparison summary
    if len(all_metrics) > 1:
        print(f"\n{'='*60}")
        print("📊 MODEL COMPARISON")
        print(f"{'='*60}")
        print(f"{'Model':<20} {'MSE':>10} {'MAE':>10} {'R²':>10}")
        print(f"{'-'*50}")
        for key, m in all_metrics.items():
            print(f"{m['model_name']:<20} {m['mse']:>10.6f} {m['mae']:>10.6f} {m['r2']:>10.6f}")
    
    # Save all metrics
    os.makedirs(reports_dir, exist_ok=True)
    eval_path = os.path.join(reports_dir, "evaluation_results.json")
    with open(eval_path, 'w') as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\nEvaluation results saved to {eval_path}")
    
    return all_metrics

if __name__ == "__main__":
    run_full_evaluation()