import onnxruntime as ort
import numpy as np
import argparse
import pandas as pd
import joblib
from pathlib import Path
from typing import Optional, Tuple, Dict, List
import time
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support
import torch
from deep_learning import Autoencoder  # Assuming you have an autoencoder implementation

def load_real_test_data(num_samples: int = 100, include_labels: bool = False) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Load balanced test data with equal class representation."""
    try:
        df = pd.read_csv("models/preprocessed_dataset.csv")
        
        # Stratified sampling to maintain class balance
        min_samples = min(
            len(df[df['Label'] == 0]), 
            len(df[df['Label'] == 1]), 
            num_samples // 2
        )
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
        
        print(f"[INFO] Loaded {X.shape[0]} balanced test samples")
        if include_labels:
            print(f"[INFO] Label distribution - Normal: {np.sum(y == 0)}, Attack: {np.sum(y == 1)}")
        
        return X, y
    except Exception as e:
        print(f"[WARNING] Could not load balanced data: {str(e)}")
        return None, None

def create_adversarial_samples(base_samples: np.ndarray, noise_level: float = 0.1) -> np.ndarray:
    """Generate adversarial samples by adding noise and perturbations."""
    # Gaussian noise
    noise = np.random.normal(0, noise_level, size=base_samples.shape)
    
    # Feature-specific perturbations
    perturbations = np.zeros_like(base_samples)
    for i in range(perturbations.shape[1]):
        if i % 5 == 0:  # Perturb every 5th feature more aggressively
            perturbations[:, i] = np.random.uniform(-0.5, 0.5, size=perturbations.shape[0])
    
    adversarial_samples = np.clip(
        base_samples + noise + perturbations,
        a_min=0.1,  # Min value from your preprocessing
        a_max=0.9    # Max value from your preprocessing
    )
    return adversarial_samples.astype(np.float32)

def test_adversarial_robustness(model_path: str, num_samples: int = 50) -> Tuple[np.ndarray, np.ndarray]:
    """Test model against adversarial samples."""
    session = ort.InferenceSession(model_path)
    input_name = session.get_inputs()[0].name
    
    # Load or create clean samples
    clean_samples, _ = load_real_test_data(num_samples=num_samples)
    if clean_samples is None:
        clean_samples = create_sample_input(num_samples=num_samples)
    
    # Generate adversarial samples
    adversarial_samples = create_adversarial_samples(clean_samples)
    
    # Run predictions
    clean_preds = session.run(None, {input_name: clean_samples})[0]
    adv_preds = session.run(None, {input_name: adversarial_samples})[0]
    
    # Calculate detection rate difference
    clean_attack_rate = np.mean(np.argmax(clean_preds, axis=1))
    adv_attack_rate = np.mean(np.argmax(adv_preds, axis=1))
    
    print(f"\n=== Adversarial Robustness Test ===")
    print(f"Clean samples attack rate: {clean_attack_rate:.2%}")
    print(f"Adversarial samples attack rate: {adv_attack_rate:.2%}")
    print(f"Detection rate change: {adv_attack_rate - clean_attack_rate:.2%}")
    
    return clean_preds, adv_preds

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

def benchmark_performance(model_path: str, num_runs: int = 1000) -> Dict[int, Dict]:
    """Comprehensive performance benchmarking."""
    session = ort.InferenceSession(model_path)
    input_name = session.get_inputs()[0].name
    
    # Test with different batch sizes
    batch_sizes = [1, 8, 16, 32, 64]
    results = {}
    
    for bs in batch_sizes:
        # Create test data
        test_data = create_sample_input(num_samples=bs)
        
        # Warm up
        for _ in range(10):
            session.run(None, {input_name: test_data})
        
        # Benchmark
        times = []
        for _ in range(num_runs):
            start = time.perf_counter()
            session.run(None, {input_name: test_data})
            times.append(time.perf_counter() - start)
        
        # Calculate metrics
        avg_time = np.mean(times) * 1000  # ms
        throughput = bs / (np.mean(times))  # samples/sec
        
        results[bs] = {
            'avg_time': avg_time,
            'throughput': throughput,
            'time_per_sample': avg_time / bs
        }
    
    # Print results
    print("\n=== Performance Benchmark ===")
    print(f"Tested with {num_runs} iterations per batch size")
    print("Batch Size | Avg Time (ms) | Throughput (samples/sec) | Time/Sample (ms)")
    print("-"*65)
    for bs, metrics in results.items():
        print(f"{bs:9d} | {metrics['avg_time']:12.4f} | {metrics['throughput']:19.2f} | {metrics['time_per_sample']:15.4f}")
    
    return results

def get_model_input_shape(session: ort.InferenceSession) -> Tuple[Tuple[int, int], int]:
    """Extract actual input dimensions from ONNX model."""
    input_info = session.get_inputs()[0]
    shape = input_info.shape
    
    # Handle dynamic shapes - replace symbolic names with actual values
    batch_size = 1  # Default batch size for testing
    feature_size = None
    
    for i, dim in enumerate(shape):
        if isinstance(dim, str):
            if 'batch' in dim.lower():
                shape[i] = batch_size
            else:
                # For other symbolic dimensions, try to infer from preprocessing artifacts
                try:
                    artifacts = joblib.load("models/preprocessing_artifacts.pkl")
                    shape[i] = len(artifacts["feature_names"])
                    feature_size = shape[i]
                except:
                    # Fallback to known size
                    shape[i] = 14
                    feature_size = 14
        else:
            if i == 1:  # Assume second dimension is features
                feature_size = dim
    
    return tuple(shape), feature_size

def evaluate_model_performance(predictions: np.ndarray, ground_truth: np.ndarray) -> dict:
    """Evaluate model performance against ground truth labels."""
    # Convert predictions to class labels
    predicted_classes = np.argmax(predictions, axis=1)
    
    # Calculate metrics
    accuracy = accuracy_score(ground_truth, predicted_classes)
    precision, recall, f1, support = precision_recall_fscore_support(
        ground_truth, predicted_classes, average=None, labels=[0, 1]
    )
    
    # Overall metrics
    precision_avg, recall_avg, f1_avg, _ = precision_recall_fscore_support(
        ground_truth, predicted_classes, average='weighted'
    )
    
    # Confusion matrix
    cm = confusion_matrix(ground_truth, predicted_classes)
    
    results = {
        'accuracy': accuracy,
        'precision_normal': precision[0] if len(precision) > 0 else 0,
        'precision_attack': precision[1] if len(precision) > 1 else 0,
        'recall_normal': recall[0] if len(recall) > 0 else 0,
        'recall_attack': recall[1] if len(recall) > 1 else 0,
        'f1_normal': f1[0] if len(f1) > 0 else 0,
        'f1_attack': f1[1] if len(f1) > 1 else 0,
        'precision_avg': precision_avg,
        'recall_avg': recall_avg,
        'f1_avg': f1_avg,
        'confusion_matrix': cm,
        'predicted_classes': predicted_classes,
        'support_normal': support[0] if len(support) > 0 else 0,
        'support_attack': support[1] if len(support) > 1 else 0
    }
    
    return results

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

def test_model_with_validation(model_path: str, test_sample: np.ndarray = None, 
                              ground_truth: np.ndarray = None, num_test_samples: int = 100) -> None:
    """Run comprehensive model tests with ground truth validation."""
    try:
        print("\n=== ONNX Model Testing with Validation ===")
        
        # Load model
        session = ort.InferenceSession(model_path)
        input_info = session.get_inputs()[0]
        output_info = session.get_outputs()[0]
        
        print(f"[INFO] Model: {model_path}")
        print(f"1. Input name: {input_info.name}")
        print(f"2. Input shape (symbolic): {input_info.shape}")
        print(f"3. Input type: {input_info.type}")
        print(f"4. Output name: {output_info.name}")
        print(f"5. Output shape (symbolic): {output_info.shape}")
        print(f"6. Output type: {output_info.type}")
        
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
        probabilities = np.exp(predictions) / np.sum(np.exp(predictions), axis=1, keepdims=True)
        
        # Basic results
        print(f"\n[SUCCESS] Inference completed successfully!")
        print(f"1. Input shape: {test_sample.shape}")
        print(f"2. Output shape: {predictions.shape}")
        print(f"3. Output range: [{predictions.min():.4f}, {predictions.max():.4f}]")
        
        # Show sample predictions
        num_samples_to_show = min(5, len(predictions))
        for i in range(num_samples_to_show):
            pred = predictions[i]
            prob = probabilities[i]
            predicted_class = np.argmax(prob)
            confidence = prob[predicted_class]
            actual_class = ground_truth[i] if ground_truth is not None else "Unknown"
            correct = "✓" if ground_truth is not None and predicted_class == ground_truth[i] else "✗"
            
            print(f"\nSample {i+1}:")
            print(f"  Raw output: [{pred[0]:.4f}, {pred[1]:.4f}]")
            print(f"  Probabilities: [Normal: {prob[0]:.4f}, Attack: {prob[1]:.4f}]")
            print(f"  Prediction: {'Attack' if predicted_class == 1 else 'Normal'} (confidence: {confidence:.4f})")
            if ground_truth is not None:
                print(f"  Ground Truth: {'Attack' if actual_class == 1 else 'Normal'} {correct}")
        
        # Detailed validation if ground truth is available
        if ground_truth is not None:
            metrics = evaluate_model_performance(probabilities, ground_truth)
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
        
        # Warm up
        for _ in range(5):
            session.run([output_info.name], {input_info.name: test_sample[:10]})
        
        # Measure inference time
        single_sample = test_sample[:1]
        times = []
        for _ in range(100):
            start = time.time()
            session.run([output_info.name], {input_info.name: single_sample})
            times.append(time.time() - start)
        
        avg_time = np.mean(times) * 1000  # Convert to milliseconds
        print(f"Average inference time: {avg_time:.2f} ms (over 100 runs)")
        print(f"Throughput: {1000/avg_time:.1f} inferences/second")
        
        # Batch processing performance
        if len(test_sample) > 1:
            batch_times = []
            batch_size = min(32, len(test_sample))
            batch_data = test_sample[:batch_size]
            
            for _ in range(10):
                start = time.time()
                session.run([output_info.name], {input_info.name: batch_data})
                batch_times.append(time.time() - start)
            
            avg_batch_time = np.mean(batch_times) * 1000  # ms
            per_sample_batch = avg_batch_time / batch_size

            print(f"Batch processing ({batch_size} samples): {avg_batch_time:.2f} ms total")
            print(f"Per-sample in batch: {per_sample_batch:.4f} ms")

            if per_sample_batch > 0:
                print(f"Batch throughput: {1000/per_sample_batch:.1f} samples/second")
            else:
                print("Batch throughput: ∞ (too fast to measure reliably)")

        
    except Exception as e:
        print(f"\n[ERROR] Testing failed: {str(e)}")
        import traceback
        traceback.print_exc()
        suggest_troubleshooting(model_path)

def create_sample_input(feature_size: int = 14, num_samples: int = 3) -> np.ndarray:
    """Create realistic sample input data."""
    samples = []
    
    for i in range(num_samples):
        # Generate realistic network flow values
        sample = np.array([
            np.random.uniform(0.1, 0.9),  # Normalized flow features
            np.random.uniform(0.0, 1.0),  # Protocol indicators
            np.random.uniform(0.2, 0.8),  # Packet counts
            np.random.uniform(0.1, 0.7),  # Byte counts
            np.random.uniform(0.0, 1.0),  # Flags
            *np.random.uniform(0.1, 0.9, feature_size - 5)  # Additional features
        ], dtype=np.float32)
        samples.append(sample)
    
    return np.array(samples)

def suggest_troubleshooting(model_path: str) -> None:
    """Provide troubleshooting suggestions."""
    print("\n=== Troubleshooting Suggestions ===")
    if not Path(model_path).exists():
        print("[ERROR] Model file not found")
        print("   → Run: python convert_to_onnx.py")
    else:
        print("[INFO] Possible issues:")
        print("   → Check input shape matches model expectations")
        print("   → Verify ONNX runtime is properly installed: pip install onnxruntime")
        print("   → Ensure preprocessing artifacts exist: models/preprocessing_artifacts.pkl")
        print("   → Try with --input flag to provide custom input data")
        print("   → Check if preprocessed_dataset.csv exists for ground truth validation")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Enhanced ONNX Model Test Suite",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--model", default="models/ids_model.onnx", help="Path to ONNX model")
    parser.add_argument("--input", help="Path to numpy input file (.npy)")
    parser.add_argument("--samples", type=int, default=200, help="Number of test samples")
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
    if args.input and Path(args.input).exists():
        try:
            test_input = np.load(args.input)
            print(f"[INFO] Loaded test input from {args.input}")
        except Exception as e:
            print(f"[WARNING] Could not load input file: {str(e)}")
    
    if args.validate:
        print("\n" + "="*50)
        print("Running Validation Tests".center(50))
        print("="*50)
        test_model_with_validation(args.model, test_input, None, args.samples)
    
    if args.adversarial:
        print("\n" + "="*50)
        print("Running Adversarial Tests".center(50))
        print("="*50)
        test_adversarial_robustness(args.model)
    
    if args.hybrid:
        print("\n" + "="*50)
        print("Running Hybrid Detection Tests".center(50))
        print("="*50)
        test_hybrid_detection()
    
    if args.benchmark:
        print("\n" + "="*50)
        print("Running Performance Benchmarks".center(50))
        print("="*50)
        benchmark_performance(args.model)
    
    if not any([args.validate, args.adversarial, args.hybrid, args.benchmark, args.all]):
        test_model_with_validation(args.model, test_input, None, args.samples)