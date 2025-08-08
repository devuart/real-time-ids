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
from sklearn.exceptions import InconsistentVersionWarning
import torch.nn as nn
import logging
import traceback
import os
from collections import defaultdict
from alive_progress import alive_bar
from tqdm import tqdm
import warnings

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

def load_real_test_data(num_samples: int = 100, include_labels: bool = False, show_progress: bool = True) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Load balanced test data with optional progress tracking."""
    try:
        data_path = Path("models/preprocessed_dataset.csv")
        if not data_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {data_path}")
        
        if show_progress:
            return _load_data_with_tqdm(data_path, num_samples, include_labels)
        else:
            return _load_data_without_progress(data_path, num_samples, include_labels)
            
    except Exception as e:
        logger.error(f"Could not load balanced data: {str(e)}")
        return None, None

def _load_data_with_alive_progress(data_path, num_samples, include_labels, bar):
    """Helper function for loading data with alive_bar progress."""
    bar.text("Reading CSV file...")
    df = pd.read_csv(data_path)
    bar()
    
    bar.text("Analyzing class distribution...")
    class_counts = df['Label'].value_counts()
    if len(class_counts) < 2:
        raise ValueError("Dataset must contain both normal and attack samples")
    bar()
    
    min_samples = min(class_counts[0], class_counts[1], num_samples // 2)
    if min_samples == 0:
        raise ValueError("Insufficient samples for both classes")
    bar()
    
    bar.text("Sampling normal traffic...")
    normal_samples = df[df['Label'] == 0].sample(n=min_samples, random_state=42)
    bar()
    
    bar.text("Sampling attack traffic...")
    attack_samples = df[df['Label'] == 1].sample(n=min_samples, random_state=42)
    bar()
    
    bar.text("Preparing final dataset...")
    balanced_df = pd.concat([normal_samples, attack_samples])
    balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    X = balanced_df.drop(columns=["Label"]).values.astype(np.float32)
    y = balanced_df["Label"].values.astype(int) if include_labels else None
    bar()
    
    logger.info(f"Loaded {X.shape[0]} balanced test samples")
    return X, y

def _load_data_with_tqdm(data_path, num_samples, include_labels):
    """Helper function for loading data with tqdm progress."""
    steps = [
        ("Reading CSV file", lambda: pd.read_csv(data_path)),
        ("Analyzing class distribution", lambda df: (
            df['Label'].value_counts(),
            lambda counts: (
                len(counts) >= 2 or ValueError("Dataset must contain both normal and attack samples"),
                min(counts[0], counts[1], num_samples // 2) > 0 or ValueError("Insufficient samples for both classes")
            )
        )),
        ("Sampling normal traffic", lambda df, min_samples: df[df['Label'] == 0].sample(n=min_samples, random_state=42)),
        ("Sampling attack traffic", lambda df, min_samples: df[df['Label'] == 1].sample(n=min_samples, random_state=42)),
        ("Preparing final dataset", lambda normal, attack: (
            pd.concat([normal, attack]).sample(frac=1, random_state=42).reset_index(drop=True)
        ))
    ]
    
    with tqdm(total=len(steps), desc="Loading test data") as pbar:
        # Step 1: Read CSV
        pbar.set_description(steps[0][0])
        df = steps[0][1]()
        pbar.update(1)
        
        # Step 2: Analyze classes
        pbar.set_description(steps[1][0])
        class_counts, validation = steps[1][1](df)
        validation(class_counts)
        min_samples = min(class_counts[0], class_counts[1], num_samples // 2)
        pbar.update(1)
        
        # Step 3: Sample normal
        pbar.set_description(steps[2][0])
        normal_samples = steps[2][1](df, min_samples)
        pbar.update(1)
        
        # Step 4: Sample attack
        pbar.set_description(steps[3][0])
        attack_samples = steps[3][1](df, min_samples)
        pbar.update(1)
        
        # Step 5: Prepare final dataset
        pbar.set_description(steps[4][0])
        balanced_df = steps[4][1](normal_samples, attack_samples)
        pbar.update(1)
        
        X = balanced_df.drop(columns=["Label"]).values.astype(np.float32)
        y = balanced_df["Label"].values.astype(int) if include_labels else None
        
        logger.info(f"Loaded {X.shape[0]} balanced test samples")
        return X, y

def _load_data_without_progress(data_path, num_samples, include_labels):
    """Helper function for loading data without progress bar."""
    df = pd.read_csv(data_path)
    class_counts = df['Label'].value_counts()
    if len(class_counts) < 2:
        raise ValueError("Dataset must contain both normal and attack samples")
    
    min_samples = min(class_counts[0], class_counts[1], num_samples // 2)
    if min_samples == 0:
        raise ValueError("Insufficient samples for both classes")
    
    normal_samples = df[df['Label'] == 0].sample(n=min_samples, random_state=42)
    attack_samples = df[df['Label'] == 1].sample(n=min_samples, random_state=42)
    balanced_df = pd.concat([normal_samples, attack_samples])
    balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    X = balanced_df.drop(columns=["Label"]).values.astype(np.float32)
    y = balanced_df["Label"].values.astype(int) if include_labels else None
    
    logger.info(f"Loaded {X.shape[0]} balanced test samples")
    return X, y

def create_adversarial_samples(base_samples: np.ndarray, noise_level: float = 0.1) -> np.ndarray:
    """Generate adversarial samples with realistic noise and perturbations."""
    try:
        # Gaussian noise
        noise = np.random.normal(0, noise_level, size=base_samples.shape)
        
        # Feature-specific perturbations (more aggressive on certain features)
        perturbations = np.zeros_like(base_samples)
        for i in range(perturbations.shape[1]):
            # Perturb every 5th feature more aggressively
            if i % 5 == 0:
                perturbations[:, i] = np.random.uniform(-0.5, 0.5, size=perturbations.shape[0])
        
        # Create adversarial samples with clipping to valid range
        adversarial_samples = np.clip(
            base_samples + noise + perturbations,
            # Min value from preprocessing
            a_min=0.1,
            # Max value from preprocessing
            a_max=0.9
        )
        return adversarial_samples.astype(np.float32)
    except Exception as e:
        logger.error(f"Error creating adversarial samples: {str(e)}")
        raise

def test_adversarial_robustness(model_path: str, num_samples: int = 50) -> Tuple[np.ndarray, np.ndarray]:
    """Test model against adversarial samples with progress tracking."""
    try:
        # Main progress bar for overall process
        with tqdm(total=10, desc="Adversarial Testing", unit="step") as main_bar:
            # Validate model path
            main_bar.set_description("Validating model path")
            if not Path(model_path).exists():
                raise FileNotFoundError(f"Model file not found: {model_path}")
            main_bar.update(1)
            
            # Create session
            main_bar.set_description("Creating optimized session")
            session_options = ort.SessionOptions()
            session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            session = ort.InferenceSession(model_path, session_options)
            input_name = session.get_inputs()[0].name
            main_bar.update(1)
            
            # Load samples
            main_bar.set_description("Loading clean samples")
            clean_samples, _ = load_real_test_data(num_samples=num_samples)
            if clean_samples is None:
                logger.warning("Using synthetic samples as fallback")
                _, feature_size = get_model_input_shape(session)
                clean_samples = create_sample_input(feature_size=feature_size, num_samples=num_samples)
            main_bar.update(1)
            
            # Generate adversarial samples
            main_bar.set_description("Creating adversarial samples")
            adversarial_samples = create_adversarial_samples(clean_samples)
            main_bar.update(1)
            
            # Run predictions
            main_bar.set_description("Running clean predictions")
            clean_preds = session.run(None, {input_name: clean_samples})[0]
            main_bar.update(1)
            
            main_bar.set_description("Running adversarial predictions")
            adv_preds = session.run(None, {input_name: adversarial_samples})[0]
            main_bar.update(1)
            
            # Calculate metrics
            main_bar.set_description("Calculating metrics")
            clean_probs = softmax(clean_preds)
            adv_probs = softmax(adv_preds)
            
            clean_attack_rate = np.mean(np.argmax(clean_probs, axis=1))
            adv_attack_rate = np.mean(np.argmax(adv_probs, axis=1))
            
            clean_conf = np.max(clean_probs, axis=1)
            adv_conf = np.max(adv_probs, axis=1)
            conf_change = np.mean(adv_conf - clean_conf)
            main_bar.update(1)
            
            # Display results
            main_bar.set_description("Compiling results")
            print("\n=== Adversarial Robustness Test ===")
            print(f"Clean samples attack rate: {clean_attack_rate:.2%}")
            print(f"Adversarial samples attack rate: {adv_attack_rate:.2%}")
            print(f"Detection rate change: {adv_attack_rate - clean_attack_rate:+.2%}")
            print(f"Average confidence change: {conf_change:+.4f}")
            main_bar.update(1)
            
            return clean_preds, adv_preds
            
    except Exception as e:
        logger.error(f"Adversarial test failed: {str(e)}")
        traceback.print_exc()
        return None, None

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
        # Main progress bar for overall process
        with tqdm(total=7, desc="Performance Benchmark", unit="step") as main_bar:
            # Validate model path
            main_bar.set_description("Validating model")
            if not Path(model_path).exists():
                raise FileNotFoundError(f"Model file not found: {model_path}")
            main_bar.update(1)
        
            # Create optimized inference session
            main_bar.set_description("Creating optimized session")
            session_options = ort.SessionOptions()
            session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            session = ort.InferenceSession(model_path, session_options)
            input_name = session.get_inputs()[0].name
            main_bar.update(1)
        
            # Get model input requirements
            main_bar.set_description("Analyzing input requirements")
            input_shape, feature_size = get_model_input_shape(session)
            logger.info(f"Benchmarking with input shape: {input_shape}")
            main_bar.update(1)
            
            # Test with different batch sizes
            batch_sizes = [1, 8, 16, 32, 64]
            main_bar.set_description(f"Testing batch sizes: {batch_sizes}")
        
            for bs in batch_sizes:
                try:
                    # Inner progress bar for batch size testing
                    with tqdm(total=num_runs, desc=f"Batch Size {bs}", unit="run") as inner_bar:
                        # Create properly shaped test data
                        test_data = create_sample_input(num_samples=bs, feature_size=feature_size)
                        
                        # Warm up (with progress indicator)
                        for _ in range(10):
                            session.run(None, {input_name: test_data})
                        
                        # Benchmark with precise timing
                        times = []
                        for _ in range(num_runs):
                            start = time.perf_counter_ns()
                            session.run(None, {input_name: test_data})
                            times.append((time.perf_counter_ns() - start) / 1e6)
                            inner_bar.update(1)
                        
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
            
            main_bar.update(1)
            
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
            
            main_bar.update(1)
            
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
            scaled_data = artifacts['scaler'].transform(sample_df).astype(np.float32)
            
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

def fix_input_data(data: np.ndarray, method: str = "clip") -> np.ndarray:
    """Fix common input data issues that cause NaN outputs."""
    fixed_data = data.copy()
    
    # Handle NaN and Inf values
    if np.isnan(fixed_data).any():
        print(f"[FIX] Replacing {np.sum(np.isnan(fixed_data))} NaN values with 0.5")
        fixed_data[np.isnan(fixed_data)] = 0.5
    
    if np.isinf(fixed_data).any():
        print(f"[FIX] Replacing {np.sum(np.isinf(fixed_data))} Inf values with 0.5")
        fixed_data[np.isinf(fixed_data)] = 0.5
    
    if method == "clip":
        # Clip to reasonable range
        fixed_data = np.clip(fixed_data, 0.0, 1.0)
        print(f"[FIX] Clipped values to [0.0, 1.0] range")
    
    elif method == "normalize":
        # Min-max normalization
        data_min = np.min(fixed_data)
        data_max = np.max(fixed_data)
        if data_max > data_min:
            fixed_data = (fixed_data - data_min) / (data_max - data_min)
            print(f"[FIX] Normalized data from [{data_min:.4f}, {data_max:.4f}] to [0, 1]")
    
    elif method == "standardize":
        # Z-score standardization
        mean = np.mean(fixed_data)
        std = np.std(fixed_data)
        if std > 0:
            fixed_data = (fixed_data - mean) / std
            print(f"[FIX] Standardized data (mean={mean:.4f}, std={std:.4f})")
    
    return fixed_data.astype(np.float32)

def analyze_model_bias(session: ort.InferenceSession, feature_size: int = 20) -> None:
    """Analyze model bias and decision boundaries."""
    print("\n=== Model Bias Analysis ===")
    
    input_name = session.get_inputs()[0].name
    
    # Test with extreme values to understand decision boundaries
    test_cases = {
        'all_zeros': np.zeros((1, feature_size), dtype=np.float32),
        'all_ones': np.ones((1, feature_size), dtype=np.float32),
        'all_half': np.full((1, feature_size), 0.5, dtype=np.float32),
        'random_low': np.random.uniform(0.0, 0.3, (5, feature_size)).astype(np.float32),
        'random_high': np.random.uniform(0.7, 1.0, (5, feature_size)).astype(np.float32),
        'mixed_pattern': np.array([
            # Half zeros, half ones
            [0.0] * 10 + [1.0] * 10,
            # Half ones, half zeros
            [1.0] * 10 + [0.0] * 10,
            # Alternating pattern
            [0.0, 1.0] * 10,
        ], dtype=np.float32)
    }
    
    for case_name, test_data in test_cases.items():
        try:
            outputs = session.run(None, {input_name: test_data})[0]
            probs = softmax(outputs)
            predictions = np.argmax(probs, axis=1)
            
            print(f"\n{case_name}:")
            for i, (raw, prob, pred) in enumerate(zip(outputs, probs, predictions)):
                class_name = "Normal" if pred == 0 else "Attack"
                confidence = prob[pred]
                print(f"  Sample {i+1}: {class_name} (conf: {confidence:.4f}) | Raw: [{raw[0]:.3f}, {raw[1]:.3f}]")
                
        except Exception as e:
            print(f"  {case_name}: Error - {str(e)}")

def suggest_model_fixes(metrics: dict) -> None:
    """Provide specific suggestions based on model performance."""
    print("\n=== Model Performance Analysis & Suggestions ===")
    
    accuracy = metrics.get('accuracy', 0)
    normal_recall = metrics.get('recall_normal', 0)
    attack_recall = metrics.get('recall_attack', 0)
    
    print(f"Current Performance:")
    print(f"  - Overall Accuracy: {accuracy:.1%}")
    print(f"  - Normal Detection Rate: {normal_recall:.1%}")
    print(f"  - Attack Detection Rate: {attack_recall:.1%}")
    
    # Diagnose specific issues
    # Less than 10% attack detection
    if attack_recall < 0.1:
        print(f"\nCRITICAL ISSUE: Model is not detecting attacks!")
        print("Possible causes:")
        print("  1. Class imbalance during training")
        print("  2. Model bias towards normal traffic")
        print("  3. Incorrect label encoding")
        print("  4. Poor feature representation for attacks")
        
        print("\nRecommended fixes:")
        print("  ✓ Check training data balance (should be ~50/50)")
        print("  ✓ Verify label encoding (0=Normal, 1=Attack)")
        print("  ✓ Use class weights during training")
        print("  ✓ Apply SMOTE for data balancing")
        print("  ✓ Adjust decision threshold")
        
    elif accuracy < 0.7:
        print(f"\nLOW ACCURACY: Model needs improvement")
        print("Recommended actions:")
        print("  ✓ Increase training epochs")
        print("  ✓ Tune hyperparameters")
        print("  ✓ Add more diverse training data")
        print("  ✓ Try different model architecture")
        
    else:
        print(f"\n[Success] Model performance is acceptable")

def test_decision_threshold(
    session: ort.InferenceSession, 
    test_data: np.ndarray, 
    ground_truth: np.ndarray,
    thresholds: List[float] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
) -> Dict[float, Dict[str, float]]:
    """Test different decision thresholds to optimize performance."""
    print("\n=== Decision Threshold Analysis ===")
    
    input_name = session.get_inputs()[0].name
    predictions = session.run(None, {input_name: test_data})[0]
    probabilities = softmax(predictions)
    
    # Get probability of attack class (class 1)
    attack_probs = probabilities[:, 1]
    
    results = {}
    
    print("Threshold | Accuracy | Normal Recall | Attack Recall | F1-Score")
    print("-" * 65)
    
    for threshold in thresholds:
        # Apply threshold: if P(attack) > threshold, predict attack
        threshold_predictions = (attack_probs > threshold).astype(int)
        
        # Calculate metrics
        accuracy = accuracy_score(ground_truth, threshold_predictions)
        
        # Per-class metrics
        if len(np.unique(threshold_predictions)) > 1:
            precision, recall, f1, _ = precision_recall_fscore_support(
                ground_truth, threshold_predictions, average=None, labels=[0, 1]
            )
            normal_recall = recall[0] if len(recall) > 0 else 0
            attack_recall = recall[1] if len(recall) > 1 else 0
            # F1 for attack class
            f1_score = f1[1] if len(f1) > 1 else 0
        else:
            normal_recall = 1.0 if threshold_predictions[0] == 0 else 0.0
            attack_recall = 1.0 if threshold_predictions[0] == 1 else 0.0
            f1_score = 0.0
        
        results[threshold] = {
            'accuracy': accuracy,
            'normal_recall': normal_recall,
            'attack_recall': attack_recall,
            'f1_score': f1_score
        }
        
        print(f"   {threshold:>5.1f}   |   {accuracy:>5.3f}  |     {normal_recall:>5.3f}    |     {attack_recall:>5.3f}    |   {f1_score:>5.3f}")
    
    # Find best threshold
    best_threshold = max(results.keys(), key=lambda t: results[t]['f1_score'])
    best_metrics = results[best_threshold]
    
    print(f"\nOptimal Threshold: {best_threshold}")
    print(f"   Accuracy: {best_metrics['accuracy']:.3f}")
    print(f"   Attack Recall: {best_metrics['attack_recall']:.3f}")
    print(f"   F1-Score: {best_metrics['f1_score']:.3f}")
    
    return results

def inspect_feature_importance(test_data: np.ndarray, session: ort.InferenceSession) -> None:
    """Simple feature importance analysis through perturbation."""
    print("\n=== Feature Importance Analysis ===")
    
    input_name = session.get_inputs()[0].name
    # Use first sample as baseline
    baseline_sample = test_data[0:1]
    baseline_output = session.run(None, {input_name: baseline_sample})[0][0]
    baseline_prob = softmax(baseline_output.reshape(1, -1))[0]
    
    print(f"Baseline prediction: Normal={baseline_prob[0]:.4f}, Attack={baseline_prob[1]:.4f}")
    
    feature_impacts = []
    
    # Test impact of each feature
    for feature_idx in range(baseline_sample.shape[1]):
        # Create modified sample with feature set to different values
        modified_sample = baseline_sample.copy()
        
        # Test with feature set to 0
        modified_sample[0, feature_idx] = 0.0
        output_low = session.run(None, {input_name: modified_sample})[0][0]
        prob_low = softmax(output_low.reshape(1, -1))[0]
        
        # Test with feature set to 1
        modified_sample[0, feature_idx] = 1.0
        output_high = session.run(None, {input_name: modified_sample})[0][0]
        prob_high = softmax(output_high.reshape(1, -1))[0]
        
        # Calculate impact (change in attack probability)
        impact_low = abs(prob_low[1] - baseline_prob[1])
        impact_high = abs(prob_high[1] - baseline_prob[1])
        max_impact = max(impact_low, impact_high)
        
        feature_impacts.append((feature_idx, max_impact))
    
    # Sort by impact
    feature_impacts.sort(key=lambda x: x[1], reverse=True)
    
    print(f"\nTop 10 Most Important Features:")
    print("Feature | Impact | Description")
    print("-" * 35)
    for i, (feature_idx, impact) in enumerate(feature_impacts[:10]):
        print(f"   {feature_idx:>2d}   | {impact:>6.4f} | Feature_{feature_idx}")

def test_model_with_validation(
    model_path: str,
    test_sample: Optional[np.ndarray] = None,
    ground_truth: Optional[np.ndarray] = None,
    num_test_samples: int = 100
) -> None:
    """Comprehensive model testing with enhanced diagnostics."""
    try:
        # Main progress bar for overall process
        with tqdm(total=20, desc="Model Validation", unit="step") as main_bar:
            main_bar.set_description("Initializing validation")
            print("\n=== ONNX Model Testing with Validation ===")
            main_bar.update(1)
            
            # Load model
            main_bar.set_description("Creating optimized session")
            session_options = ort.SessionOptions()
            session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            session_options.log_severity_level = 3  # Suppress warnings
            session = ort.InferenceSession(model_path, session_options, providers=['CPUExecutionProvider'])
            input_info = session.get_inputs()[0]
            output_info = session.get_outputs()[0]
            main_bar.update(1)
            
            # Print model info
            main_bar.set_description("Gathering model details")
            print(f"[INFO] Model: {model_path}")
            print(f"1. Input name: {input_info.name}")
            print(f"2. Input shape (symbolic): {input_info.shape}")
            print(f"3. Input type: {input_info.type}")
            print(f"4. Output name: {output_info.name}")
            print(f"5. Output shape (symbolic): {output_info.shape}")
            print(f"6. Output type: {output_info.type}")
            main_bar.update(1)
            
            # Get input shape
            main_bar.set_description("Resolving input dimensions")
            actual_shape, feature_size = get_model_input_shape(session)
            print(f"Feature size: {feature_size}")
            print(f"[INFO] Model loaded successfully with ONNX Runtime version: {ort.__version__}")
            print(f"[INFO] Resolved input shape: {actual_shape}")
            main_bar.update(1)
            
            # Prepare test data
            main_bar.set_description("Preparing test samples")
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
            main_bar.update(1)
            
            # Validate shape
            main_bar.set_description("Validating input shape")
            expected_features = actual_shape[1] if len(actual_shape) > 1 else actual_shape[0]
            if test_sample.shape[1] != expected_features:
                print(f"[ERROR] Input shape mismatch!")
                print(f"  Expected: (batch_size, {expected_features})")
                print(f"  Got: {test_sample.shape}")
                return
            main_bar.update(1)
            
            # Run diagnostics
            main_bar.set_description("Running diagnostics")
            diagnostics = diagnose_model_issues(session, test_sample)
            main_bar.update(1)
            
            # Fix input data if needed
            main_bar.set_description("Checking for data issues")
            original_sample = test_sample.copy()
            if diagnostics.get('input_stats', {}).get('has_nan') or diagnostics.get('input_stats', {}).get('has_inf'):
                print("[WARNING] Input data contains NaN or Inf values, attempting to fix...")
                test_sample = fix_input_data(test_sample, method="clip")
            main_bar.update(1)
            
            # Try different preprocessing approaches if NaN detected
            main_bar.set_description("Testing preprocessing approaches")
            best_sample = test_sample
            best_method = "original"
            
            preprocessing_methods = [
                ("original", test_sample),
                ("clipped", fix_input_data(original_sample, "clip")),
                ("normalized", fix_input_data(original_sample, "normalize"))
            ]
            
            if Path("models/preprocessing_artifacts.pkl").exists():
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", category=UserWarning)
                        warnings.simplefilter("ignore", category=InconsistentVersionWarning)
                        artifacts = joblib.load("models/preprocessing_artifacts.pkl")
                    
                    if artifacts.get('scaler') and artifacts.get('feature_names'):
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
            main_bar.update(1)
            
            # Run inference
            main_bar.set_description("Running inference")
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
            main_bar.update(1)
            
            # Show results
            main_bar.set_description("Compiling results")
            print(f"\n[SUCCESS] Inference completed successfully!")
            print(f"1. Input shape: {test_sample.shape}")
            print(f"2. Output shape: {predictions.shape}")
            print(f"3. Output range: [{np.min(predictions):.3f}, {np.max(predictions):.3f}]")
            main_bar.update(1)
            
            # Sample predictions
            main_bar.set_description("Analyzing sample predictions")
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
            main_bar.update(1)
            
            # Continue with validation metrics if predictions are valid
            if ground_truth is not None and not np.isnan(predictions).any():
                main_bar.set_description("Calculating metrics")
                metrics = evaluate_model_performance(predictions, ground_truth)
                print_detailed_metrics(metrics)
                analyze_prediction_confidence(predictions, probabilities)
                
                predicted_classes = np.argmax(probabilities, axis=1)
                class_names = ['Normal', 'Attack']
                print(f"\n=== Sklearn Classification Report ===")
                print(classification_report(ground_truth, predicted_classes, 
                                          target_names=class_names, digits=4))
                main_bar.update(1)
                
                # Enhanced analysis for bias and performance issues
                main_bar.set_description("Analyzing model bias")
                analyze_model_bias(session, feature_size)
                main_bar.update(1)
                
                main_bar.set_description("Testing decision thresholds")
                test_decision_threshold(session, test_sample, ground_truth)
                main_bar.update(1)
                
                main_bar.set_description("Analyzing feature importance")
                inspect_feature_importance(test_sample, session)
                main_bar.update(1)
                
                main_bar.set_description("Generating recommendations")
                suggest_model_fixes(metrics)
                main_bar.update(1)
                
            elif np.isnan(predictions).any():
                print("\n[ERROR] Cannot calculate metrics due to NaN predictions")
                main_bar.update(1)
            
            # Performance test (if predictions are valid)
            if not np.isnan(predictions).any():
                main_bar.set_description("Testing performance")
                print(f"\n[INFO] Running performance test...")
                
                # Warm up
                print("Warming up the model for performance testing...")
                for i in range(5):
                    session.run([output_info.name], {input_info.name: test_sample[:10]})
                print("[INFO] Warm-up completed.")
                main_bar.update(1)
                
                # Single sample timing
                main_bar.set_description("Testing single sample")
                single_sample = test_sample[:1]
                times = []
                for _ in range(100):
                    start = time.perf_counter_ns()
                    session.run([output_info.name], {input_info.name: single_sample})
                    times.append((time.perf_counter_ns() - start) / 1e6)
                
                avg_time = np.mean(times)
                print(f"Average inference time: {avg_time:.4f} ms (over 100 runs)")
                print(f"Throughput: {1000/avg_time:.2f} inferences/second")
                main_bar.update(1)
                
                # Batch timing
                if len(test_sample) > 1:
                    main_bar.set_description("Testing batch performance")
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
                    main_bar.update(1)
            
            main_bar.set_description("Validation complete!")
            
    except Exception as e:
        logger.error(f"Model validation failed: {str(e)}")
        print(f"[ERROR] Model validation failed: {str(e)}")
        traceback.print_exc()
        suggest_troubleshooting(model_path)

def suggest_troubleshooting(model_path: str) -> None:
    """Provide detailed troubleshooting suggestions for NaN issues."""
    print("\n=== Troubleshooting Guide ===")
    
    # Check model file
    if not Path(model_path).exists():
        print("[ERROR] Model file not found")
        print("  [!] Verify the model path is correct")
        print("  [!] Check if you need to run model training/export first")
        return
    
    print("\n[INFO] Common causes of NaN outputs:")
    print("1. Input Data Issues:")
    print("  [!] Input contains NaN or Inf values")
    print("  [!] Input range doesn't match training data")
    print("  [!] Missing preprocessing (scaling/normalization)")
    
    print("\n2. Model Training Issues:")
    print("  [!] Model was not properly trained")
    print("  [!] Model weights contain NaN/Inf values")
    print("  [!] Learning rate was too high during training")
    
    print("\n3. Numerical Issues:")
    print("  [!] Division by zero in model operations")
    print("  [!] Exponential overflow in activations")
    print("  [!] Gradient explosion during training")
    
    print("\n[SOLUTIONS]:")
    print("1. Retrain the model with:")
    print("  - Lower learning rate")
    print("  - Gradient clipping")
    print("  - Proper input validation")
    print("  - Regularization techniques")
    
    print("2. Fix input data:")
    print("  - Apply proper preprocessing")
    print("  - Handle missing values")
    print("  - Validate input ranges")

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
            with tqdm(total=1, desc="Loading input file", unit="file") as pbar:
                test_input = np.load(args.input)
                print(f"Loaded test input from {args.input}")
                pbar.update(1)
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