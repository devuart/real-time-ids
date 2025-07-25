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
from deep_learning import Autoencoder
from hybrid_detector import hybrid_detect
from sklearn.preprocessing import StandardScaler
import logging
import traceback
import sys
import os
from collections import defaultdict

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

# Set UTF-8 encoding for Windows compatibility
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

def load_real_test_data(num_samples: int = 100, include_labels: bool = False) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Load balanced test data with equal class representation."""
    try:
        data_path = Path("models/preprocessed_dataset.csv")
        if not data_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {data_path}")
        
        df = pd.read_csv(data_path)
        
        # Stratified sampling to maintain class balance
        class_counts = df['Label'].value_counts()
        if len(class_counts) < 2:
            raise ValueError("Dataset must contain both normal and attack samples")
            
        min_samples = min(class_counts[0], class_counts[1], num_samples // 2)
        if min_samples == 0:
            raise ValueError("Insufficient samples for both classes")
            
        normal_samples = df[df['Label'] == 0].sample(n=min_samples, random_state=42)
        attack_samples = df[df['Label'] == 1].sample(n=min_samples, random_state=42)
        balanced_df = pd.concat([normal_samples, attack_samples])
        
        # Shuffle the dataset
        balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        # Separate features and labels
        X = balanced_df.drop(columns=["Label"]).values.astype(np.float32)
        y = balanced_df["Label"].values.astype(int) if include_labels else None
        
        logger.info(f"Loaded {X.shape[0]} balanced test samples")
        if include_labels:
            logger.info(f"Label distribution - Normal: {np.sum(y == 0)}, Attack: {np.sum(y == 1)}")
        
        return X, y
    except Exception as e:
        logger.error(f"Could not load balanced data: {str(e)}")
        return None, None

def create_adversarial_samples(base_samples: np.ndarray, noise_level: float = 0.1) -> np.ndarray:
    """Generate adversarial samples with realistic noise and perturbations."""
    try:
        # Gaussian noise
        noise = np.random.normal(0, noise_level, size=base_samples.shape)
        
        # Feature-specific perturbations (more aggressive on certain features)
        perturbations = np.zeros_like(base_samples)
        for i in range(perturbations.shape[1]):
            if i % 5 == 0:  # Perturb every 5th feature more aggressively
                perturbations[:, i] = np.random.uniform(-0.5, 0.5, size=perturbations.shape[0])
        
        # Create adversarial samples with clipping to valid range
        adversarial_samples = np.clip(
            base_samples + noise + perturbations,
            a_min=0.1,  # Min value from preprocessing
            a_max=0.9    # Max value from preprocessing
        )
        return adversarial_samples.astype(np.float32)
    except Exception as e:
        logger.error(f"Error creating adversarial samples: {str(e)}")
        raise

def test_adversarial_robustness(model_path: str, num_samples: int = 50) -> Tuple[np.ndarray, np.ndarray]:
    """Test model against adversarial samples with comprehensive metrics."""
    try:
        # Validate model path
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        # Create optimized inference session
        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        session = ort.InferenceSession(model_path, session_options)
        input_name = session.get_inputs()[0].name
        
        # Load or create clean samples
        clean_samples, _ = load_real_test_data(num_samples=num_samples)
        if clean_samples is None:
            logger.warning("Using synthetic samples as fallback")
            _, feature_size = get_model_input_shape(session)
            clean_samples = create_sample_input(feature_size=feature_size, num_samples=num_samples)
        
        # Generate adversarial samples
        adversarial_samples = create_adversarial_samples(clean_samples)
        
        # Run predictions
        clean_preds = session.run(None, {input_name: clean_samples})[0]
        adv_preds = session.run(None, {input_name: adversarial_samples})[0]
        
        # Calculate metrics
        clean_probs = softmax(clean_preds)
        adv_probs = softmax(adv_preds)
        
        clean_attack_rate = np.mean(np.argmax(clean_probs, axis=1))
        adv_attack_rate = np.mean(np.argmax(adv_probs, axis=1))
        
        # Calculate confidence changes
        clean_conf = np.max(clean_probs, axis=1)
        adv_conf = np.max(adv_probs, axis=1)
        conf_change = np.mean(adv_conf - clean_conf)
        
        print("\n=== Adversarial Robustness Test ===")
        print(f"Clean samples attack rate: {clean_attack_rate:.2%}")
        print(f"Adversarial samples attack rate: {adv_attack_rate:.2%}")
        print(f"Detection rate change: {adv_attack_rate - clean_attack_rate:+.2%}")
        print(f"Average confidence change: {conf_change:+.4f}")
        
        return clean_preds, adv_preds
    except Exception as e:
        logger.error(f"Adversarial test failed: {str(e)}")
        traceback.print_exc()
        return None, None

def test_hybrid_detection(threshold_tolerance: float = 0.1) -> Optional[List[Dict]]:
    """Comprehensive hybrid system testing with edge cases."""
    try:
        from hybrid_detector import hybrid_detect
        artifacts = joblib.load("models/preprocessing_artifacts.pkl")
        threshold = joblib.load("models/anomaly_threshold.pkl")
        
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
            # Ensure proper scaling
            sample = artifacts['scaler'].transform(sample.reshape(1, -1))[0]
            result = hybrid_detect(sample)
            
            # Calculate autoencoder MSE
            autoencoder = Autoencoder(input_dim=20)
            autoencoder.load_state_dict(torch.load("models/autoencoder_ids.pth"))
            with torch.no_grad():
                reconstruction = autoencoder(torch.from_numpy(sample).float())
                mse = torch.mean((torch.from_numpy(sample) - reconstruction)**2).item()
            
            results.append({
                "case": name,
                "result": result,
                "mse": mse,
                "threshold_diff": mse - threshold
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

def softmax(x: np.ndarray) -> np.ndarray:
    """Numerically stable softmax function."""
    e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e_x / e_x.sum(axis=1, keepdims=True)

def benchmark_performance(model_path: str, num_runs: int = 1000) -> Dict[int, Dict[str, float]]:
    """
    Comprehensive performance benchmarking.
    
    Args:
        model_path: Path to ONNX model file
        num_runs: Number of iterations to run for each batch size
        
    Returns:
        Dictionary containing performance metrics for each batch size
    """
    results = {}
    
    try:
        # Validate model path
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        # Create optimized inference session
        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        session = ort.InferenceSession(model_path, session_options)
        input_name = session.get_inputs()[0].name
        
        # Get model input requirements
        input_shape, feature_size = get_model_input_shape(session)
        logger.info(f"Benchmarking with input shape: {input_shape}")
        
        # Test with different batch sizes
        batch_sizes = [1, 8, 16, 32, 64]
        
        for bs in batch_sizes:
            try:
                # Create properly shaped test data
                test_data = create_sample_input(num_samples=bs, feature_size=feature_size)
                
                # Warm up (with progress indicator)
                print(f"\nWarming up for batch size {bs}...")
                for _ in range(10):
                    session.run(None, {input_name: test_data})
                    if _ % 2 == 0:
                        print(f"Warm-up progress:", end="", flush=True)
                print("Done.")
                
                # Benchmark with precise timing
                times = []
                logger.info(f"Starting benchmark for batch size {bs} with {num_runs} runs...")
                for _ in range(num_runs):
                    start = time.perf_counter_ns()
                    session.run(None, {input_name: test_data})
                    end = time.perf_counter_ns()
                    times.append((time.perf_counter_ns() - start) / 1e6)  # ms
                    
                    # Progress indicator
                    if _ % (num_runs // 10) == 0:
                        progress = (_ / num_runs) * 100
                        print(f"\rProgress: {progress:.1f}%", end="", flush=True)
                print("\nBenchmark completed.")
                logger.info(f"Batch size {bs} benchmarked successfully.")
                
                # Calculate comprehensive metrics
                times_ms = np.array(times)
                metrics = {
                    'avg_time': np.mean(times_ms),
                    'std_dev': np.std(times_ms),
                    'throughput': (bs * 1000) / np.mean(times_ms),
                    'time_per_sample': np.mean(times_ms) / bs,
                    'p95_time': np.percentile(times_ms, 95),
                    'min_time': np.min(times_ms),
                    'max_time': np.max(times_ms),
                    'samples_per_sec': (bs * 1e6) / np.mean(times_ms)
                }
                results[bs] = metrics
                
            except Exception as e:
                logger.error(f"Error benchmarking batch size {bs}: {str(e)}")
                results[bs] = {'error': str(e)}
        
        # Print comprehensive results
        print("\n=== Performance Benchmark Results ===")
        print(f"Model: {Path(model_path).name}")
        print(f"Input Shape: {input_shape}")
        print(f"Feature Size: {feature_size}")
        print(f"Test Config: {num_runs} iterations per batch size")
        print("\nBatch Size | Avg Time (ms) | Throughput (samples/s) | Time/Sample (ms) | 95th %ile (ms)")
        print("-"*100)
        for bs, metrics in sorted(results.items()):
            if 'error' in metrics:
                print(f"{bs:>11} | Error: {metrics['error']}")
            else:
                print(f"{bs:>11} | {metrics['avg_time']:>14.2f} | "
                      f"{metrics['throughput']:>20.2f} | "
                      f"{metrics['time_per_sample']:>16.4f} | "
                      f"{metrics['p95_time']:>14.2f}")
        return results
    
    except Exception as e:
        logger.error(f"Performance benchmark failed: {str(e)}")
        traceback.print_exc()
        return {'error': str(e)}

def get_model_input_shape(session: ort.InferenceSession) -> Tuple[Tuple[int, int], int]:
    """
    Extract and validate input dimensions from ONNX model.
    
    Args:
        session: Initialized ONNX Runtime inference session
        
    Returns:
        tuple: (full_input_shape, feature_size)
    """
    try:
        input_info = session.get_inputs()[0]
        shape = list(input_info.shape)  # Make mutable
        
        # Default values
        batch_size = 1
        feature_size = None
        
        # Process each dimension
        for i, dim in enumerate(shape):
            if isinstance(dim, str):  # Dynamic dimension
                if 'batch' in dim.lower():
                    shape[i] = batch_size
                else:
                    # Try to get feature size from preprocessing artifacts
                    try:
                        artifacts = joblib.load("models/preprocessing_artifacts.pkl")
                        feature_size = len(artifacts["feature_names"])
                        shape[i] = feature_size
                        logger.info(f"Detected feature size from artifacts: {feature_size}")
                    except:
                        logger.warning(f"Could not load preprocessing artifacts: {str(e)}")
                        # Fallback to default feature size
                        feature_size = 20  # Default expected size
                        shape[i] = feature_size
                        logger.warning(f"Using default feature size: {feature_size}")
            else:  # Static dimension
                if i == 1:  # Assume second dimension is features
                    feature_size = dim
        
        # Final validation and fallback
        if feature_size is None:
            feature_size = 20
            logger.warning(f"Could not determine feature size, using default: {feature_size}")
            if len(shape) > 1:
                shape[1] = feature_size
        
        # Ensure we have at least 2 dimensions (batch, features)
        if len(shape) < 2:
            shape = [batch_size, feature_size]
        
        return tuple(shape), feature_size
    
    except Exception as e:
        logger.error(f"Error getting model input shape: {str(e)}")
        # Return safe defaults
        return (1, 20), 20

def evaluate_model_performance(
    predictions: np.ndarray,
    ground_truth: np.ndarray,
    class_names: List[str] = ["Normal", "Attack"]
) -> Dict[str, Union[float, np.ndarray]]:
    """
    Evaluate model performance against ground truth labels.
    
    Args:
        predictions: Model output predictions (2D array)
        ground_truth: True labels (1D array)
        class_names: List of class names for labeling
        
    Returns:
        Dictionary containing all evaluation metrics
    """
    try:
        # Input validation
        if predictions.ndim != 2:
            raise ValueError("Predictions must be a 2D array, got {predictions.ndim}D")
        if ground_truth.ndim != 1:
            raise ValueError("Ground truth must be a 1D array, got {ground_truth.ndim}D")
        if predictions.shape[0] != ground_truth.shape[0]:
            raise ValueError(
                f"Predictions and ground truth must have the same number of samples, "
                f"got {predictions.shape[0]} vs {ground_truth.shape[0]}"
            )
        
        # Convert predictions to probabilities and class labels
        probs = softmax(predictions)
        pred_classes = np.argmax(probs, axis=1)
        
        # Calculate all metrics
        metrics = {
            'accuracy': accuracy_score(ground_truth, pred_classes),
            'confusion_matrix': confusion_matrix(ground_truth, pred_classes),
            'classification_report': classification_report(
                ground_truth, pred_classes, target_names=class_names, output_dict=True
            ),
            'predicted_classes': pred_classes,
            'probabilities': probs
        }
        
        # Per-class metrics
        precision, recall, f1, support = precision_recall_fscore_support(
            ground_truth, pred_classes, average=None, labels=[0, 1]
        )
        for i, name in enumerate(class_names):
            metrics.update({
                f'precision_{name.lower()}': precision[i],
                f'recall_{name.lower()}': recall[i],
                f'f1_{name.lower()}': f1[i],
                f'support_{name.lower()}': support[i]
            })
        
        # Weighted averages
        precision_avg, recall_avg, f1_avg, _ = precision_recall_fscore_support(
            ground_truth, pred_classes, average='weighted'
        )
        metrics.update({
            'precision_avg': precision_avg,
            'recall_avg': recall_avg,
            'f1_avg': f1_avg
        })
        
        # Additional metrics from confusion matrix
        tn, fp, fn, tp = metrics['confusion_matrix'].ravel()
        metrics.update({
            'true_negative_rate': tn / (tn + fp) if (tn + fp) > 0 else 0,
            'false_positive_rate': fp / (fp + tn) if (fp + tn) > 0 else 0,
            'true_positive_rate': tp / (tp + fn) if (tp + fn) > 0 else 0,
            'false_negative_rate': fn / (fn + tp) if (fn + tp) > 0 else 0
        })
        
        return metrics
    
    except Exception as e:
        logger.error(f"Error evaluating model performance: {str(e)}")
        return {'error': str(e)}

def print_detailed_metrics(metrics: dict) -> None:
    """Print detailed performance metrics."""
    print("\n=== Model Performance Metrics ===")
    print(f"Overall Accuracy: {metrics['accuracy']:.4f}")
    print(f"Weighted Precision: {metrics['precision_avg']:.4f}")
    print(f"Weighted Recall: {metrics['recall_avg']:.4f}")
    print(f"Weighted F1-Score: {metrics['f1_avg']:.4f}")
    
    print("\n--- Per-Class Metrics ---")
    print(f"Normal Traffic:")
    print(f"  Precision: {metrics['precision_normal']:.4f}")
    print(f"  Recall: {metrics['recall_normal']:.4f}")
    print(f"  F1-Score: {metrics['f1_normal']:.4f}")
    print(f"  Support: {metrics['support_normal']}")
    
    print(f"Attack Traffic:")
    print(f"  Precision: {metrics['precision_attack']:.4f}")
    print(f"  Recall: {metrics['recall_attack']:.4f}")
    print(f"  F1-Score: {metrics['f1_attack']:.4f}")
    print(f"  Support: {metrics['support_attack']}")
    
    print("\n--- Confusion Matrix ---")
    cm = metrics['confusion_matrix']
    print("Predicted ->  Normal  Attack")
    print(f"Normal       {cm[0,0]:6d}  {cm[0,1]:6d}")
    print(f"Attack       {cm[1,0]:6d}  {cm[1,1]:6d}")
    
    # Calculate additional metrics
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        false_positive_rate = fp / (fp + tn) if (fp + tn) > 0 else 0
        
        print(f"\n--- Additional Metrics ---")
        print(f"True Positive Rate (Sensitivity): {sensitivity:.4f}")
        print(f"True Negative Rate (Specificity): {specificity:.4f}")
        print(f"False Positive Rate: {false_positive_rate:.4f}")

def analyze_prediction_confidence(predictions: np.ndarray, probabilities: np.ndarray) -> None:
    """Analyze prediction confidence distribution."""
    confidence_scores = np.max(probabilities, axis=1)
    
    print("\n=== Prediction Confidence Analysis ===")
    print(f"Mean Confidence: {np.mean(confidence_scores):.4f}")
    print(f"Std Confidence: {np.std(confidence_scores):.4f}")
    print(f"Min Confidence: {np.min(confidence_scores):.4f}")
    print(f"Max Confidence: {np.max(confidence_scores):.4f}")
    
    # Confidence distribution
    high_conf = np.sum(confidence_scores > 0.9)
    med_conf = np.sum((confidence_scores >= 0.7) & (confidence_scores <= 0.9))
    low_conf = np.sum(confidence_scores < 0.7)
    
    print(f"\nConfidence Distribution:")
    print(f"High Confidence (>0.9): {high_conf} ({high_conf/len(confidence_scores)*100:.1f}%)")
    print(f"Medium Confidence (0.7-0.9): {med_conf} ({med_conf/len(confidence_scores)*100:.1f}%)")
    print(f"Low Confidence (<0.7): {low_conf} ({low_conf/len(confidence_scores)*100:.1f}%)")

def test_model_with_validation(
    model_path: str,
    test_sample: Optional[np.ndarray] = None,
    ground_truth: Optional[np.ndarray] = None,
    num_test_samples: int = 100
) -> None:
    """Run comprehensive model tests with ground truth validation
    Args:
        model_path: Path to the ONNX model file
        test_sample: Optional numpy array of test input data
        ground_truth: Optional numpy array of ground truth labels
        num_test_samples: Number of samples to use if test_sample is not provided
    """
    try:
        print("\n=== ONNX Model Testing with Validation ===")
        
        # Load model with optimized settings
        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        session = ort.InferenceSession(model_path, session_options)
        input_info = session.get_inputs()[0]
        output_info = session.get_outputs()[0]
        
        print(f"[INFO] Model: {model_path}")
        print(f"1. Input name: {input_info.name}")
        print(f"2. Input shape (symbolic): {input_info.shape}")
        print(f"3. Input type: {input_info.type}")
        print(f"4. Output name: {output_info.name}")
        print(f"5. Output shape (symbolic): {output_info.shape}")
        print(f"6. Output type: {output_info.type}")
        print(f"Feature size: {input_info.shape[1] if len(input_info.shape) > 1 else input_info.shape[0]}")
        print(f"[INFO] Model loaded successfully with ONNX Runtime version: {ort.__version__}")
        
        # Get actual input shape
        actual_shape, feature_size = get_model_input_shape(session)
        print(f"\n[INFO] Resolved input shape: {actual_shape}")
        
        # Prepare test input and ground truth
        if test_sample is None or ground_truth is None:
            # Load real data with labels for validation
            test_sample, ground_truth = load_real_test_data(
                num_samples=num_test_samples, include_labels=True
            )
            
            if test_sample is None:
                # Generate synthetic test data as fallback
                test_sample = np.random.rand(num_test_samples, feature_size).astype(np.float32)
                test_sample = test_sample * 0.8 + 0.1
                ground_truth = np.random.randint(0, 2, num_test_samples)
                print(f"[INFO] Generated synthetic test input: {test_sample.shape}")
            
        else:
            # Ensure provided sample has correct shape
            if len(test_sample.shape) == 1:
                test_sample = test_sample.reshape(1, -1)
            test_sample = test_sample.astype(np.float32)
            print(f"[INFO] Using provided test input: {test_sample.shape}")
        
        # Validate input shape
        expected_features = actual_shape[1] if len(actual_shape) > 1 else actual_shape[0]
        if test_sample.shape[1] != expected_features:
            print(f"[ERROR] Input shape mismatch!")
            print(f"  Expected: (batch_size, {expected_features})")
            print(f"  Got: {test_sample.shape}")
            return
        
        # Run inference
        print(f"\n[INFO] Running inference on {test_sample.shape[0]} samples...")
        predictions = session.run(
            [output_info.name],
            {input_info.name: test_sample}
        )[0]
        
        # Convert to probabilities
        probabilities = softmax(predictions)
        
        # Basic results
        print(f"\n[SUCCESS] Inference completed successfully!")
        print(f"1. Input shape: {test_sample.shape}")
        print(f"2. Output shape: {predictions.shape}")
        print(f"3. Output range: [{predictions.min():.4f}, {predictions.max():.4f}]")
        
        # Show sample predictions
        print("\n=== Sample Predictions ===")
        num_samples_to_show = min(5, len(predictions))
        for i in range(num_samples_to_show):
            pred = predictions[i]
            prob = probabilities[i]
            predicted_class = np.argmax(prob)
            confidence = prob[predicted_class]
            actual_class = ground_truth[i] if ground_truth is not None else "Unknown"
            correct = "[success]" if ground_truth is not None and predicted_class == ground_truth[i] else "[error]"
            
            print(f"\nSample {i+1}:")
            print(f"  Raw output: [{pred[0]:.4f}, {pred[1]:.4f}]")
            print(f"  Probabilities: [Normal: {prob[0]:.4f}, Attack: {prob[1]:.4f}]")
            print(f"  Prediction: {'Attack' if predicted_class == 1 else 'Normal'} (confidence: {confidence:.4f})")
            if ground_truth is not None:
                print(f"  Ground Truth: {'Attack' if actual_class == 1 else 'Normal'} {correct}")
        
        # Detailed validation if ground truth is available
        if ground_truth is not None:
            metrics = evaluate_model_performance(predictions, ground_truth)
            print_detailed_metrics(metrics)
            analyze_prediction_confidence(predictions, probabilities)
            
            # Generate classification report
            predicted_classes = np.argmax(probabilities, axis=1)
            class_names = ['Normal', 'Attack']
            print(f"\n=== Sklearn Classification Report ===")
            print(classification_report(ground_truth, predicted_classes, 
                                      target_names=class_names, digits=4))
        
        # Performance test
        print(f"\n[INFO] Running performance test...")
        
        # Warm up with progress indicator
        print("Warming up the model for performance testing...")
        for _ in range(10):
            session.run([output_info.name], {input_info.name: test_sample[:10]})
            if _ % 2 == 0:
                print(f"Warm-up progress:", end="", flush=True)
        print("[INFO] Warm-up completed.")
        
        # Measure inference time
        single_sample = test_sample[:1]
        times = []
        for _ in range(100):
            start = time.perf_counter_ns()
            session.run([output_info.name], {input_info.name: single_sample})
            times.append((time.perf_counter_ns() - start) / 1e6)  # Convert to milliseconds
        
        avg_time = np.mean(times)
        print(f"Average inference time: {avg_time:.4f} ms (over 100 runs)")
        print(f"Throughput: {1000/avg_time:.2f} inferences/second")
        
        # Batch processing performance
        if len(test_sample) > 1:
            batch_times = []
            batch_size = min(32, len(test_sample))
            batch_data = test_sample[:batch_size]
            
            for _ in range(10):
                start = time.perf_counter_ns()
                session.run([output_info.name], {input_info.name: batch_data})
                batch_times.append((time.perf_counter_ns() - start) / 1e6)
            
            avg_batch_time = np.mean(batch_times)
            per_sample_batch = avg_batch_time / batch_size

            print(f"Batch processing ({batch_size} samples): {avg_batch_time:.4f} ms total")
            print(f"Per-sample in batch: {per_sample_batch:.4f} ms")
            print(f"Batch throughput: {1000/per_sample_batch:.2f} samples/second")
        
    except Exception as e:
        # Log the error
        mod_validation_error = f"Model validation failed: {str(e)}"
        logger.error(mod_validation_error)
        print(f"[ERROR] {mod_validation_error}")
        
        # Print traceback for debugging
        print("\n=== Traceback ===")
        traceback.print_exc()
        
        # Suggest troubleshooting steps
        suggest_troubleshooting(model_path)

def create_sample_input(feature_size: int = 20, num_samples: int = 1) -> np.ndarray:
    """Create realistic sample input data."""
    samples = np.zeros((num_samples, feature_size), dtype=np.float32)
    
    for i in range(num_samples):
        # Generate realistic network traffic features
        samples[i, 0] = np.random.uniform(0.1, 0.9)  # Duration
        samples[i, 1] = np.random.choice([0.0, 0.33, 0.66, 1.0])  # Protocol
        samples[i, 2] = np.random.uniform(0.1, 0.8)  # Packet count
        samples[i, 3] = np.random.uniform(0.1, 0.7)  # Byte count
        samples[i, 4] = np.random.uniform(0.0, 1.0)  # Flags
        samples[i, 5:] = np.random.uniform(0.1, 0.9, feature_size - 5)  # Other features
    
    return samples

def suggest_troubleshooting(model_path: str) -> None:
    """Provide detailed troubleshooting suggestions."""
    print("\n=== Troubleshooting Guide ===")
    
    # Check model file
    if not Path(model_path).exists():
        print("[error] Model file not found")
        print("  [!] Verify the model path is correct")
        print("  [!] Check if you need to run model training/export first")
        return
    
    # Check ONNX runtime
    try:
        ort.InferenceSession(model_path)
    except Exception as e:
        print(f"[error] ONNX Runtime Error: {str(e)}")
        print("  [!] Ensure onnxruntime is properly installed")
        print("  [!] Check if the model file is corrupted")
        return
    
    # Check input data
    print("[success] Model loads successfully")
    print("\nCommon Issues:")
    print("1. Input Shape Mismatch:")
    print("  [!] Verify your input data matches the model's expected shape")
    print("  [!] Use get_model_input_shape() to check requirements")
    
    print("\n2. Preprocessing Issues:")
    print("  [!] Ensure input data is properly normalized/scaled")
    print("  [!] Check if preprocessing_artifacts.pkl exists")
    
    print("\n3. Performance Problems:")
    print("  [!] Try enabling ONNX runtime optimizations")
    print("  [!] Consider quantizing the model for better performance")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Enhanced ONNX Model Test Suite",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--model", default="models/ids_model.onnx", help="Path to ONNX model")
    parser.add_argument("--input", help="Path to numpy input file (.npy)")
    parser.add_argument("--samples", type=int, default=100, help="Number of test samples")
    parser.add_argument("--validate", action="store_true", help="Run validation tests")
    parser.add_argument("--adversarial", action="store_true", help="Test adversarial robustness")
    parser.add_argument("--hybrid", action="store_true", help="Test hybrid detection")
    parser.add_argument("--benchmark", action="store_true", help="Run performance benchmarks")
    parser.add_argument("--all", action="store_true", help="Run all test suites")
    
    args = parser.parse_args()
    
    if args.all:
        args.validate = True
        args.adversarial = True
        args.hybrid = True
        args.benchmark = True
    
    # Load test input if provided
    test_input = None
    if args.input:
        try:
            test_input = np.load(args.input)
            print(f"Loaded test input from {args.input}")
        except Exception as e:
            print(f"Could not load input file: {str(e)}")
    
    # Run selected tests
    if args.validate:
        print("\n" + "=" * 50)
        print("Running validation tests".center(50))
        print("=" * 50)
        test_model_with_validation(args.model, test_input, None, args.samples)
    
    if args.adversarial:
        print("\n" + "=" * 50)
        print("Testing adversarial robustness".center(50))
        print("=" * 50)
        test_adversarial_robustness(args.model, args.samples)
    
    if args.hybrid:
        print("\n" + "=" * 50)
        print("Testing hybrid detection system".center(50))
        print("=" * 50)
        test_hybrid_detection()
    
    if args.benchmark:
        print("\n" + "=" * 50)
        print("Running performance benchmarks".center(50))
        print("=" * 50)
        benchmark_performance(args.model)
    
    if not any([args.validate, args.adversarial, args.hybrid, args.benchmark, args.all]):
        test_model_with_validation(args.model, test_input, None, args.samples)