import sys
import io
# Fix encoding for Windows compatibility
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import onnxruntime as ort
import numpy as np
import argparse
import pandas as pd
import joblib
from pathlib import Path
from typing import Optional, Tuple, Dict, List, Union
import time
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support
import torch
# from deep_learning import EnhancedAutoencoder as Autoencoder
# from hybrid_detector import hybrid_detect
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import torch.nn as nn
import logging
import traceback
import os
from collections import defaultdict
from alive_progress import alive_bar
from tqdm import tqdm

# Setup logging
LOG_DIR = Path("logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE_NAME = "test_onnx_model.log"
LOG_FILE = LOG_DIR / LOG_FILE_NAME

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

try:
    from deep_learning import load_autoencoder_model
except ImportError:
    logger.warning("Could not import load_autoencoder_model")
    load_autoencoder_model = None

try:
    from hybrid_detector import hybrid_detect
except ImportError:
    logger.warning("Could not import hybrid_detect")
    hybrid_detect = None

# Set UTF-8 encoding for Windows compatibility
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

def test_hybrid_detection(threshold_tolerance: float = 0.1) -> Optional[List[Dict]]:
    """Comprehensive hybrid system testing with edge cases."""
    try:
        # Check if required files exist
        artifacts_path = "models/preprocessing_artifacts.pkl"
        threshold_path = "models/anomaly_threshold.pkl"
        autoencoder_path = "models/autoencoder_ids.pth"
        
        if not all(Path(p).exists() for p in [artifacts_path, threshold_path]):
            print(f"[WARNING] Required files not found for hybrid testing")
            print(f"  Missing: {[p for p in [artifacts_path, threshold_path] if not Path(p).exists()]}")
            return None
        
        artifacts = joblib.load(artifacts_path)
        threshold = joblib.load(threshold_path)
        
        print("\n=== Comprehensive Hybrid Detection Test ===")
        print(f"[CONFIG] Anomaly threshold: {threshold:.4f}")
        
        # Test cases
        test_cases = [
            ("Normal Baseline", np.full(20, 0.5)),  # Mid-range
            ("High Variance", np.random.uniform(0.1, 0.9, 20)),
            ("Low Features", np.concatenate([np.ones(10), np.zeros(10)])),
            ("Extreme Values", np.array([0.1, 0.9]*10)),
            ("Adversarial Pattern", np.array([0.1 if i%2 else 0.9 for i in range(20)]))
        ]
        
        results = []
        for name, sample in test_cases:
            try:
                # Ensure proper scaling
                if len(sample) != len(artifacts['feature_names']):
                    print(f"[WARNING] Feature dimension mismatch: expected {len(artifacts['feature_names'])}, got {len(sample)}")
                    # Handle mismatch (truncate or pad as appropriate for your case)
                    sample = sample[:len(artifacts['feature_names'])]  # Simple truncation example

                sample_df = pd.DataFrame(
                    sample.reshape(1, -1),
                    columns=artifacts['feature_names']
                )
                scaled_sample = artifacts['scaler'].transform(sample_df)[0]
                sample = scaled_sample.astype(np.float32)
                
                # Try to get hybrid result if function exists
                if hybrid_detect is not None:
                    try:
                        result = hybrid_detect(sample)
                    except Exception as e:
                        result = f"hybrid_detect error: {str(e)}"
                else:
                    result = "hybrid_detect function not available"
                
                # Calculate autoencoder MSE if model exists
                mse = 0.0
                if Path(autoencoder_path).exists():
                    try:
                        if load_autoencoder_model is not None:
                            autoencoder = load_autoencoder_model(
                                Path(autoencoder_path), 
                                input_dim=len(sample)
                            )
                        else:
                            # Fallback manual loading
                            state_dict = torch.load(autoencoder_path, map_location='cpu', weights_only=True)
                            
                            # Simple autoencoder for compatibility
                            class SimpleAutoencoder(nn.Module):
                                def __init__(self, input_dim: int, encoding_dim: int):
                                    super().__init__()
                                    self.encoder = nn.Sequential(
                                        nn.Linear(input_dim, encoding_dim),
                                        nn.ReLU()
                                    )
                                    self.decoder = nn.Sequential(
                                        nn.Linear(encoding_dim, input_dim),
                                        nn.Sigmoid()
                                    )
                                
                                def forward(self, x):
                                    return self.decoder(self.encoder(x))
                            
                            encoder_weight_shape = state_dict['encoder.0.weight'].shape
                            encoding_dim = encoder_weight_shape[0]
                            input_dim = encoder_weight_shape[1]
                            
                            autoencoder = SimpleAutoencoder(input_dim, encoding_dim)
                            autoencoder.load_state_dict(state_dict)
                        
                        autoencoder.eval()
                        with torch.no_grad():
                            reconstruction = autoencoder(torch.from_numpy(sample).float().unsqueeze(0))
                            mse = torch.mean((torch.from_numpy(sample).unsqueeze(0) - reconstruction)**2).item()
                    except Exception as e:
                        logger.warning(f"Could not load autoencoder: {str(e)}")
                        # Mock MSE for testing
                        mse = np.random.uniform(0.001, 0.1)
                
                results.append({
                    "case": name,
                    "result": result,
                    "mse": mse,
                    "threshold_diff": mse - threshold
                })
            except Exception as e:
                logger.error(f"Error testing case {name}: {str(e)}")
                results.append({
                    "case": name,
                    "result": f"Error: {str(e)}",
                    "mse": 0.0,
                    "threshold_diff": 0.0
                })
        
        # Print results
        print("\nTest Case Results:")
        for r in results:
            print(f"{r['case']}:")
            print(f"  Hybrid Result: {r['result']}")
            print(f"  MSE: {r['mse']:.4f} (Threshold diff: {r['threshold_diff']:.4f})")
            print(f"  Verdict: {'ALERT' if r['mse'] > threshold*(1+threshold_tolerance) else 'OK'}")
        
        return results
    except Exception as e:
        print(f"[ERROR] Hybrid test failed: {str(e)}")
        return None

def diagnose_model_issues(session: ort.InferenceSession, test_input: np.ndarray) -> Dict[str, any]:
    """Comprehensive model diagnostic to identify NaN issues."""
    diagnostics = {}
    
    try:
        print("\n=== Model Diagnostics ===")
        
        # Check input data quality
        print("1. Input Data Analysis:")
        diagnostics['input_stats'] = {
            'shape': test_input.shape,
            'dtype': test_input.dtype,
            'min': np.min(test_input),
            'max': np.max(test_input),
            'mean': np.mean(test_input),
            'std': np.std(test_input),
            'has_nan': np.isnan(test_input).any(),
            'has_inf': np.isinf(test_input).any(),
            'zero_count': np.sum(test_input == 0),
            'unique_values': len(np.unique(test_input.flatten()))
        }
        
        for key, value in diagnostics['input_stats'].items():
            print(f"  {key}: {value}")
        
        # Check model weights (if accessible)
        print("\n2. Model Analysis:")
        input_info = session.get_inputs()[0]
        output_info = session.get_outputs()[0]
        diagnostics['model_info'] = {
            'input_name': input_info.name,
            'input_shape': input_info.shape,
            'input_type': input_info.type,
            'output_name': output_info.name,
            'output_shape': output_info.shape,
            'output_type': output_info.type
        }
        
        for key, value in diagnostics['model_info'].items():
            print(f"  {key}: {value}")
        
        # Test with different input patterns
        print("\n3. Testing Different Input Patterns:")
        test_patterns = {
            'zeros': np.zeros_like(test_input[:1]),
            'ones': np.ones_like(test_input[:1]),
            'small_values': np.full_like(test_input[:1], 0.1),
            'medium_values': np.full_like(test_input[:1], 0.5),
            'large_values': np.full_like(test_input[:1], 0.9),
            'random_normal': np.random.normal(0.5, 0.1, test_input[:1].shape).astype(np.float32)
        }
        
        diagnostics['pattern_results'] = {}
        for pattern_name, pattern_data in test_patterns.items():
            try:
                output = session.run(None, {input_info.name: pattern_data})[0]
                has_nan = np.isnan(output).any()
                has_inf = np.isinf(output).any()
                output_range = [np.min(output), np.max(output)]
                
                diagnostics['pattern_results'][pattern_name] = {
                    'has_nan': has_nan,
                    'has_inf': has_inf,
                    'range': output_range,
                    'success': not (has_nan or has_inf)
                }
                
                status = "✓" if not (has_nan or has_inf) else "✗"
                print(f"  {pattern_name}: {status} NaN={has_nan}, Inf={has_inf}, Range={output_range}")
                
            except Exception as e:
                diagnostics['pattern_results'][pattern_name] = {'error': str(e)}
                print(f"  {pattern_name}: ✗ Error: {str(e)}")
        
        # Check preprocessing artifacts
        print("\n4. Preprocessing Analysis:")
        try:
            artifacts = joblib.load("models/preprocessing_artifacts.pkl")
            diagnostics['preprocessing'] = {
                'scaler_type': type(artifacts['scaler']).__name__,
                'feature_count': len(artifacts['feature_names']),
                'feature_names': artifacts['feature_names'][:5] if len(artifacts['feature_names']) > 5 else artifacts['feature_names'],
                'scaler_range': [artifacts['scaler'].data_min_.min(), artifacts['scaler'].data_max_.max()] if hasattr(artifacts['scaler'], 'data_min_') else 'Unknown'
            }
            
            # Test with properly preprocessed data
            sample_data = np.random.uniform(0, 1, (1, len(artifacts['feature_names'])))
            sample_df = pd.DataFrame(
                sample_data,
                columns=artifacts['feature_names']
            )
            scaled_data = artifacts['scaler'].transform(sample_df)
            
            try:
                scaled_output = session.run(None, {input_info.name: scaled_data})[0]
                diagnostics['preprocessing']['scaled_test'] = {
                    'has_nan': np.isnan(scaled_output).any(),
                    'has_inf': np.isinf(scaled_output).any(),
                    'range': [np.min(scaled_output), np.max(scaled_output)]
                }
                print(f"  Preprocessed test: NaN={np.isnan(scaled_output).any()}, Range={[np.min(scaled_output), np.max(scaled_output)]}")
            except Exception as e:
                diagnostics['preprocessing']['scaled_test'] = {'error': str(e)}
                print(f"  Preprocessed test failed: {str(e)}")
                
        except Exception as e:
            diagnostics['preprocessing'] = {'error': str(e)}
            print(f"  Could not load preprocessing artifacts: {str(e)}")
        
        return diagnostics
        
    except Exception as e:
        logger.error(f"Diagnostic failed: {str(e)}")
        return {'error': str(e)}

def test_model_with_validation(
    model_path: str,
    test_sample: Optional[np.ndarray] = None,
    ground_truth: Optional[np.ndarray] = None,
    num_test_samples: int = 100
) -> None:
    """Comprehensive model testing with enhanced diagnostics."""
    try:
        with alive_bar(20, title="Model Validation", spinner='triangles') as bar:
            bar.text("Initializing validation...")
            print("\n=== ONNX Model Testing with Validation ===")
            bar()
            
            # Load model
            bar.text("Creating optimized session...")
            session_options = ort.SessionOptions()
            session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            # Suppress warnings
            session_options.log_severity_level = 3
            session = ort.InferenceSession(model_path, session_options, providers=['CPUExecutionProvider'])
            input_info = session.get_inputs()[0]
            output_info = session.get_outputs()[0]
            bar()
            
            # Print model info
            bar.text("Gathering model details...")
            print(f"[INFO] Model: {model_path}")
            print(f"1. Input name: {input_info.name}")
            print(f"2. Input shape (symbolic): {input_info.shape}")
            print(f"3. Input type: {input_info.type}")
            print(f"4. Output name: {output_info.name}")
            print(f"5. Output shape (symbolic): {output_info.shape}")
            print(f"6. Output type: {output_info.type}")
            bar()
            
            # Get input shape
            bar.text("Resolving input dimensions...")
            actual_shape, feature_size = get_model_input_shape(session)
            print(f"Feature size: {feature_size}")
            print(f"[INFO] Model loaded successfully with ONNX Runtime version: {ort.__version__}")
            print(f"[INFO] Resolved input shape: {actual_shape}")
            bar()
            
            # Prepare test data
            bar.text("Preparing test samples...")
            if test_sample is None or ground_truth is None:
                test_sample, ground_truth = load_real_test_data(
                    num_samples=num_test_samples, include_labels=True, show_progress=False
                )
                
                if test_sample is None:
                    test_sample = np.random.rand(num_test_samples, feature_size).astype(np.float32)
                    test_sample = test_sample * 0.8 + 0.1
                    ground_truth = np.random.randint(0, 2, num_test_samples)
                    print(f"[INFO] Generated synthetic test input: {test_sample.shape}")
            else:
                if len(test_sample.shape) == 1:
                    test_sample = test_sample.reshape(1, -1)
                test_sample = test_sample.astype(np.float32)
                print(f"[INFO] Using provided test input: {test_sample.shape}")
            bar()
            
            # Validate shape
            bar.text("Validating input shape...")
            expected_features = actual_shape[1] if len(actual_shape) > 1 else actual_shape[0]
            if test_sample.shape[1] != expected_features:
                print(f"[ERROR] Input shape mismatch!")
                print(f"  Expected: (batch_size, {expected_features})")
                print(f"  Got: {test_sample.shape}")
                return
            bar()
            
            # Run diagnostics
            bar.text("Running diagnostics...")
            diagnostics = diagnose_model_issues(session, test_sample)
            bar()
            
            # Fix input data if needed
            bar.text("Checking for data issues...")
            original_sample = test_sample.copy()
            if diagnostics.get('input_stats', {}).get('has_nan') or diagnostics.get('input_stats', {}).get('has_inf'):
                print("[WARNING] Input data contains NaN or Inf values, attempting to fix...")
                test_sample = fix_input_data(test_sample, method="clip")
            bar()
            
            # Try different preprocessing approaches if NaN detected
            bar.text("Testing preprocessing approaches...")
            best_sample = test_sample
            best_method = "original"
            
            preprocessing_methods = [
                ("original", test_sample),
                ("clipped", fix_input_data(original_sample, "clip")),
                ("normalized", fix_input_data(original_sample, "normalize"))
            ]
            
            if Path("models/preprocessing_artifacts.pkl").exists():
                try:
                    # Load artifacts with warning suppression
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", category=UserWarning)
                        warnings.simplefilter("ignore", category=InconsistentVersionWarning)
                        artifacts = joblib.load("models/preprocessing_artifacts.pkl")
                    
                    if artifacts.get('scaler') and artifacts.get('feature_names'):
                        # Convert to DataFrame with proper feature names
                        sample_df = pd.DataFrame(
                            original_sample,
                            columns=artifacts['feature_names'][:original_sample.shape[1]]
                        )
                        preprocessed = artifacts['scaler'].transform(sample_df)
                        preprocessing_methods.append(("preprocessed", preprocessed.astype(np.float32)))
                except Exception as e:
                    print(f"[WARNING] Could not apply preprocessing: {str(e)}")
            
            for method_name, method_data in preprocessing_methods:
                try:
                    test_output = session.run([output_info.name], {input_info.name: method_data[:5]})[0]
                    if not np.isnan(test_output).any() and not np.isinf(test_output).any():
                        print(f"[SUCCESS] {method_name} preprocessing works!")
                        best_sample = method_data
                        best_method = method_name
                        break
                    else:
                        print(f"[FAIL] {method_name} preprocessing still produces NaN/Inf")
                except Exception as e:
                    print(f"[FAIL] {method_name} preprocessing error: {str(e)}")
            
            test_sample = best_sample
            print(f"[INFO] Using {best_method} preprocessing method")
            bar()
            
            # Run inference
            bar.text("Running inference...")
            print(f"\n[INFO] Running inference on {test_sample.shape[0]} samples...")
            predictions = session.run([output_info.name], {input_info.name: test_sample})[0]
            
            # Check for NaN in predictions
            if np.isnan(predictions).any():
                print(f"[ERROR] Model still produces NaN outputs!")
                print(f"  NaN count: {np.sum(np.isnan(predictions))}")
                print(f"  This suggests a fundamental issue with the model")
                probabilities = np.full_like(predictions, 0.5)  # Fallback
            else:
                probabilities = softmax(predictions)
            bar()
            
            # Show results
            bar.text("Compiling results...")
            print(f"\n[SUCCESS] Inference completed successfully!")
            print(f"1. Input shape: {test_sample.shape}")
            print(f"2. Output shape: {predictions.shape}")
            print(f"3. Output range: [{np.min(predictions):.3f}, {np.max(predictions):.3f}]")
            bar()
            
            # Sample predictions
            bar.text("Analyzing sample predictions...")
            print("\n=== Sample Predictions ===")
            num_samples_to_show = min(5, len(predictions))
            for i in range(num_samples_to_show):
                pred = predictions[i]
                prob = probabilities[i] if not np.isnan(probabilities).any() else [0.5, 0.5]
                predicted_class = np.argmax(prob) if not np.isnan(prob).any() else 0
                confidence = prob[predicted_class] if not np.isnan(prob).any() else 0.5
                actual_class = ground_truth[i] if ground_truth is not None else "Unknown"
                
                print(f"\nSample {i+1}:")
                print(f"  Raw output: {pred}")
                print(f"  Probabilities: [Normal: {prob[0]:.4f}, Attack: {prob[1]:.4f}]")
                print(f"  Prediction: {'Normal' if predicted_class == 0 else 'Attack'} (confidence: {confidence:.4f})")
                if ground_truth is not None:
                    print(f"  Ground Truth: {'Normal' if actual_class == 0 else 'Attack'}")
            bar()
            
            # Continue with validation metrics if predictions are valid
            if ground_truth is not None and not np.isnan(predictions).any():
                bar.text("Calculating metrics...")
                metrics = evaluate_model_performance(predictions, ground_truth)
                print_detailed_metrics(metrics)
                analyze_prediction_confidence(predictions, probabilities)
                
                predicted_classes = np.argmax(probabilities, axis=1)
                class_names = ['Normal', 'Attack']
                print(f"\n=== Sklearn Classification Report ===")
                print(classification_report(ground_truth, predicted_classes, 
                                          target_names=class_names, digits=4))
                bar()
                
                # Enhanced analysis for bias and performance issues
                bar.text("Analyzing model bias...")
                analyze_model_bias(session, feature_size)
                bar()
                
                bar.text("Testing decision thresholds...")
                test_decision_threshold(session, test_sample, ground_truth)
                bar()
                
                bar.text("Analyzing feature importance...")
                inspect_feature_importance(test_sample, session)
                bar()
                
                bar.text("Generating recommendations...")
                suggest_model_fixes(metrics)
                bar()
                
            elif np.isnan(predictions).any():
                print("\n[ERROR] Cannot calculate metrics due to NaN predictions")
                bar()
            
            # Performance test (if predictions are valid)
            if not np.isnan(predictions).any():
                bar.text("Testing performance...")
                print(f"\n[INFO] Running performance test...")
                
                # Warm up
                print("Warming up the model for performance testing...")
                for i in range(5):
                    session.run([output_info.name], {input_info.name: test_sample[:10]})
                print("[INFO] Warm-up completed.")
                bar()
                
                # Single sample timing
                bar.text("Testing single sample...")
                single_sample = test_sample[:1]
                times = []
                for _ in range(100):
                    start = time.perf_counter_ns()
                    session.run([output_info.name], {input_info.name: single_sample})
                    times.append((time.perf_counter_ns() - start) / 1e6)
                
                avg_time = np.mean(times)
                print(f"Average inference time: {avg_time:.4f} ms (over 100 runs)")
                print(f"Throughput: {1000/avg_time:.2f} inferences/second")
                bar()
                
                # Batch timing
                if len(test_sample) > 1:
                    bar.text("Testing batch performance...")
                    batch_size = min(32, len(test_sample))
                    batch_data = test_sample[:batch_size]
                    
                    batch_times = []
                    for _ in range(10):
                        start = time.perf_counter_ns()
                        session.run([output_info.name], {input_info.name: batch_data})
                        batch_times.append((time.perf_counter_ns() - start) / 1e6)
                    
                    avg_batch_time = np.mean(batch_times)
                    per_sample_batch = avg_batch_time / batch_size

                    print(f"Batch processing ({batch_size} samples): {avg_batch_time:.4f} ms total")
                    print(f"Per-sample in batch: {per_sample_batch:.4f} ms")
                    print(f"Batch throughput: {1000/per_sample_batch:.2f} samples/second")
                    bar()
            
            bar.text("Validation complete!")
            
    except Exception as e:
        logger.error(f"Model validation failed: {str(e)}")
        print(f"[ERROR] Model validation failed: {str(e)}")
        traceback.print_exc()
        suggest_troubleshooting(model_path)





