import sys
import io
import onnxruntime as ort
import numpy as np
import argparse
from pathlib import Path
from alive_progress import alive_bar
import time
from tqdm import tqdm
import warnings
import os
import logging

# Fix encoding for Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# Setup logging
LOG_DIR = Path("logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE_NAME = "verify.log"
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

def resolve_input_shape(session: ort.InferenceSession, show_progress: bool = True) -> tuple:
    """Resolve symbolic dimensions to actual values with progress tracking.
    
    Args:
        session: ONNX Runtime InferenceSession
        show_progress: Whether to display progress bars
        
    Returns:
        Tuple containing the resolved input shape
    """
    try:
        if not show_progress:
            return _resolve_shape_without_progress(session)
        return _resolve_shape_with_tqdm(session)
    except Exception as e:
        logger.error(f"Error resolving input shape: {str(e)}")
        raise

def _resolve_shape_with_alive_progress(session, bar):
    """Helper function for resolving shape with alive_bar progress."""
    try:
        bar.text("Getting input info...")
        input_info = session.get_inputs()[0]
        shape = input_info.shape
        bar()
        
        bar.text("Resolving dimensions...")
        resolved_shape = []
        for dim in shape:
            if isinstance(dim, str):
                if 'batch' in dim.lower():
                    resolved_shape.append(1)
                else:
                    # Default feature dimension
                    resolved_shape.append(14)
            else:
                resolved_shape.append(dim)
        bar()
        
        bar.text("Shape resolution complete")
        return tuple(resolved_shape)
    except Exception as e:
        logger.error(f"Error in _resolve_shape_with_alive_progress: {str(e)}")
        raise

def _resolve_shape_with_tqdm(session):
    """Helper function for resolving shape with tqdm progress."""
    try:
        steps = [
            ("Getting input info", lambda: session.get_inputs()[0]),
            ("Resolving dimensions", lambda input_info: [
                1 if isinstance(dim, str) and 'batch' in dim.lower() else 
                14 if isinstance(dim, str) else 
                dim for dim in input_info.shape
            ]),
            ("Finalizing shape", lambda shape: tuple(shape))
        ]
        
        with tqdm(total=len(steps), desc="Resolving input shape") as pbar:
            # Step 1: Get input info
            pbar.set_description(steps[0][0])
            input_info = steps[0][1]()
            pbar.update(1)
            
            # Step 2: Resolve dimensions
            pbar.set_description(steps[1][0])
            resolved_shape = steps[1][1](input_info)
            pbar.update(1)
            
            # Step 3: Finalize shape
            pbar.set_description(steps[2][0])
            final_shape = steps[2][1](resolved_shape)
            pbar.update(1)
            
            return final_shape
    except Exception as e:
        logger.error(f"Error in _resolve_shape_with_tqdm: {str(e)}")
        raise

def _resolve_shape_without_progress(session):
    """Helper function for resolving shape without progress bar."""
    try:
        input_info = session.get_inputs()[0]
        resolved_shape = []
        for dim in input_info.shape:
            if isinstance(dim, str):
                if 'batch' in dim.lower():
                    resolved_shape.append(1)
                else:
                    # Default feature dimension
                    resolved_shape.append(14)
            else:
                resolved_shape.append(dim)
        return tuple(resolved_shape)
    except Exception as e:
        logger.error(f"Error in _resolve_shape_without_progress: {str(e)}")
        raise

def quick_verify(model_path: str, show_progress: bool = True) -> None:
    """Perform quick model verification with progress tracking.
    
    Args:
        model_path: Path to the ONNX model file
        show_progress: Whether to display progress bars (default: True)
    """
    try:
        if not show_progress:
            return _quick_verify_without_progress(model_path)
        return _quick_verify_with_tqdm(model_path)
    except Exception as e:
        logger.error(f"Model verification failed: {str(e)}")
        _suggest_troubleshooting_without_progress(model_path)
        raise

def _quick_verify_with_alive_progress(model_path: str, bar) -> None:
    """Helper function for verification with alive_bar progress."""
    try:
        print(f"\n=== Quick Verification: {model_path} ===")
        
        # Step 1-3: Loading model
        bar.text("Loading ONNX model...")
        session = ort.InferenceSession(model_path)
        bar()
        time.sleep(0.2)
        
        input_info = session.get_inputs()[0]
        bar.text("Checking input specs...")
        bar()
        time.sleep(0.1)
        
        output_info = session.get_outputs()[0]
        bar.text("Checking output specs...")
        bar()
        time.sleep(0.1)
        
        # Step 4-6: Display basic info
        print(f"[INFO] Model loaded successfully")
        print(f"1. Input name: {input_info.name}")
        print(f"2. Input shape (symbolic): {input_info.shape}")
        print(f"3. Output name: {output_info.name}")
        print(f"4. Output shape (symbolic): {output_info.shape}")
        bar()
        
        # Step 7-8: Resolve shape
        bar.text("Resolving symbolic dimensions...")
        resolved_shape = resolve_input_shape(session)
        print(f"5. Resolved input shape: {resolved_shape}")
        bar()
        time.sleep(0.2)
        
        # Step 9: Generate test input
        bar.text("Generating test input...")
        test_input = np.random.rand(*resolved_shape).astype(np.float32)
        print(f"6. Test input generated: {test_input.shape}")
        bar()
        time.sleep(0.3)
        
        # Step 10: Run inference
        bar.text("Running inference...")
        predictions = session.run([output_info.name], {input_info.name: test_input})[0]
        bar()
        time.sleep(0.2)
        
        # Step 11: Calculate probabilities (with numerical stability fixes)
        bar.text("Calculating probabilities...")
        try:
            # Numerically stable softmax implementation
            def stable_softmax(x):
                z = x - np.max(x, axis=1, keepdims=True)
                numerator = np.exp(z)
                denominator = np.sum(numerator, axis=1, keepdims=True)
                return numerator / denominator
            
            probabilities = stable_softmax(predictions)
            
            # Check for invalid values
            if np.any(np.isnan(probabilities)):
                warnings.warn("Probabilities contain NaN values - model outputs may be too extreme")
                # Fallback to simple normalization if softmax fails
                probabilities = predictions / np.sum(np.abs(predictions), axis=1, keepdims=True)
        except Exception as e:
            logger.warning(f"Probability calculation failed: {str(e)}")
            probabilities = np.zeros_like(predictions)
            probabilities[:, 0] = 1.0  # Default to "Normal" class if calculation fails
        bar()
        time.sleep(0.1)
        
        # Step 12: Performance test
        bar.text("Running performance test...")
        start_time = time.time()
        for _ in range(10):
            session.run([output_info.name], {input_info.name: test_input})
        avg_time = (time.time() - start_time) / 10 * 1000
        bar()
        
        # Display results
        print(f"\n[SUCCESS] Model is operational!")
        print(f"[success] Input shape: {test_input.shape}")
        print(f"[success] Output shape: {predictions.shape}")
        print(f"[success] Raw prediction: [{predictions[0][0]:.4f}, {predictions[0][1]:.4f}]")
        
        # Handle potential NaN/inf values in display
        prob_normal = 0.0 if np.isnan(probabilities[0][0]) else probabilities[0][0]
        prob_attack = 0.0 if np.isnan(probabilities[0][1]) else probabilities[0][1]
        
        print(f"[success] Probabilities: [Normal: {prob_normal:.4f}, Attack: {prob_attack:.4f}]")
        
        # Determine predicted class safely
        if np.all(np.isnan(probabilities[0])):
            predicted_class = "Unknown (calculation failed)"
        else:
            predicted_class = 'Attack' if np.nanargmax(probabilities[0]) == 1 else 'Normal'
        
        print(f"[success] Predicted class: {predicted_class}")
        print(f"[success] Average inference time: {avg_time:.2f} ms")
        print(f"\n[RESULT] Model verification PASSED! [success]")
        
    except Exception as e:
        print(f"\n[ERROR] Verification failed: {str(e)}")
        import traceback
        print(f"[DEBUG] Full error traceback:")
        traceback.print_exc()
        suggest_troubleshooting(model_path)

def _quick_verify_with_tqdm(model_path: str) -> None:
    """Helper function for verification with tqdm progress."""
    try:
        steps = [
            ("Loading ONNX model", lambda: ort.InferenceSession(model_path)),
            ("Checking input specs", lambda session: session.get_inputs()[0]),
            ("Checking output specs", lambda session: session.get_outputs()[0]),
            ("Resolving symbolic dimensions", lambda session: resolve_input_shape(session, show_progress=False)),
            ("Generating test input", lambda shape: np.random.rand(*shape).astype(np.float32)),
            ("Running inference", lambda session, input_info, output_info, test_input: 
                session.run([output_info.name], {input_info.name: test_input})[0]),
            ("Analyzing predictions", lambda preds: analyze_predictions(preds)),
            ("Performance test", lambda session, input_info, output_info, test_input: (
                time.time(),
                [session.run([output_info.name], {input_info.name: test_input}) for _ in range(10)],
                time.time()
            ))
        ]
        
        with tqdm(total=len(steps), desc="Verifying model") as pbar:
            print(f"\n=== Quick Verification: {model_path} ===")
            
            # Step 1: Load model
            pbar.set_description(steps[0][0])
            session = steps[0][1]()
            pbar.update(1)
            
            # Step 2: Input specs
            pbar.set_description(steps[1][0])
            input_info = steps[1][1](session)
            pbar.update(1)
            
            # Step 3: Output specs
            pbar.set_description(steps[2][0])
            output_info = steps[2][1](session)
            pbar.update(1)
            
            # Display basic info
            print(f"[INFO] Model loaded successfully")
            print(f"1. Input name: {input_info.name}")
            print(f"2. Input shape (symbolic): {input_info.shape}")
            print(f"3. Output name: {output_info.name}")
            print(f"4. Output shape (symbolic): {output_info.shape}")
            
            # Step 4: Resolve shape
            pbar.set_description(steps[3][0])
            resolved_shape = steps[3][1](session)
            print(f"5. Resolved input shape: {resolved_shape}")
            pbar.update(1)
            
            # Step 5: Generate test input
            pbar.set_description(steps[4][0])
            test_input = steps[4][1](resolved_shape)
            print(f"6. Test input generated: {test_input.shape}")
            pbar.update(1)
            
            # Step 6: Run inference
            pbar.set_description(steps[5][0])
            predictions = steps[5][1](session, input_info, output_info, test_input)
            pbar.update(1)
            
            # Step 7: Analyze predictions
            pbar.set_description(steps[6][0])
            analysis = steps[6][1](predictions)
            pbar.update(1)
            
            # Step 8: Performance test
            pbar.set_description(steps[7][0])
            start_time, _, end_time = steps[7][1](session, input_info, output_info, test_input)
            avg_time = (end_time - start_time) / 10 * 1000
            pbar.update(1)
            
            # Display enhanced results
            print(f"\n[SUCCESS] Model is operational!")
            print(f"[success] Input shape: {test_input.shape}")
            print(f"[success] Output shape: {predictions.shape}")
            
            # Enhanced diagnostics
            print(f"\n[DIAGNOSTICS]")
            print(f"Raw output range: [{analysis['min']:.2f}, {analysis['max']:.2f}]")
            print(f"Output magnitude ratio: {analysis['magnitude_ratio']:.1e}")
            
            if analysis['output_scale_warning']:
                print("[warning] Extreme output values detected - consider:")
                print("   - Adding output layer normalization")
                print("   - Scaling model outputs during training")
                print("   - Using log-softmax instead of raw outputs")
            
            print(f"\n[RESULTS]")
            print(f"Probabilities: [Normal: {analysis['normal_prob']:.6f}, Attack: {analysis['attack_prob']:.6f}]")
            print(f"Predicted class: {analysis['predicted_class']}")
            
            if analysis['confidence_warning']:
                print("[warning] Extreme confidence detected - may indicate:")
                print("   - Overfitting to training data")
                print("   - Need for label smoothing")
                print("   - Class imbalance in training set")
            
            print(f"\n[PERFORMANCE]")
            print(f"Average inference time: {avg_time:.2f} ms")
            print(f"\n[VERDICT] Model verification PASSED with warnings! [success]")
            
    except Exception as e:
        print(f"\n[ERROR] Verification failed: {str(e)}")
        import traceback
        print(f"[DEBUG] Full error traceback:")
        traceback.print_exc()
        _suggest_troubleshooting_without_progress(model_path)

def analyze_predictions(predictions: np.ndarray) -> dict:
    """Comprehensive analysis of model predictions with enhanced diagnostics."""
    analysis = {
        'min': float(np.min(predictions)),
        'max': float(np.max(predictions)),
        'magnitude_ratio': 0.0,
        'output_scale_warning': False,
        'confidence_warning': False,
        'normal_prob': 0.5,
        'attack_prob': 0.5,
        'predicted_class': 'Unknown'
    }
    
    # Calculate magnitude ratio (max/min absolute values)
    abs_preds = np.abs(predictions)
    with np.errstate(divide='ignore', invalid='ignore'):
        analysis['magnitude_ratio'] = np.max(abs_preds) / np.min(abs_preds[abs_preds > 0])
    
    # Check for extreme output values
    if np.max(abs_preds) > 1000 or analysis['magnitude_ratio'] > 1e6:
        analysis['output_scale_warning'] = True
    
    # Calculate probabilities with multiple safety layers
    try:
        # First attempt: scaled softmax
        scale_factor = max(1, np.max(abs_preds)/100)
        scaled_preds = predictions / scale_factor
        probs = softmax(scaled_preds)
        
        # Second attempt: clipped softmax if first fails
        if np.any(np.isnan(probs)):
            clipped_preds = np.clip(predictions, -100, 100)
            probs = softmax(clipped_preds)
            
        # Final fallback: simple normalization
        if np.any(np.isnan(probs)):
            probs = abs_preds / np.sum(abs_preds, axis=1, keepdims=True)
        
        analysis['normal_prob'] = float(probs[0][0])
        analysis['attack_prob'] = float(probs[0][1])
        analysis['predicted_class'] = 'Attack' if analysis['attack_prob'] > analysis['normal_prob'] else 'Normal'
        
        # Confidence analysis
        confidence = max(analysis['normal_prob'], analysis['attack_prob'])
        if confidence > 0.99:
            analysis['confidence_warning'] = True
        elif confidence < 0.6:
            analysis['confidence_warning'] = True  # Also warn for low confidence
            
    except Exception as e:
        logger.warning(f"Prediction analysis failed: {str(e)}")
    
    return analysis

def softmax(x: np.ndarray) -> np.ndarray:
    """Enhanced numerically stable softmax with additional checks."""
    try:
        # Shift values for numerical stability
        shifted_x = x - np.max(x, axis=1, keepdims=True)
        # Clip exponents to prevent overflow
        exp_x = np.exp(np.clip(shifted_x, -50, 50))
        # Normalize with epsilon to avoid division by zero
        return exp_x / (np.sum(exp_x, axis=1, keepdims=True) + 1e-10)
    except:
        # Fallback to uniform distribution if softmax fails
        return np.full_like(x, 1/x.shape[1])

def _quick_verify_without_progress(model_path: str) -> None:
    """Helper function for verification without progress bar."""
    try:
        print(f"\n=== Quick Verification: {model_path} ===")
        
        session = ort.InferenceSession(model_path)
        input_info = session.get_inputs()[0]
        output_info = session.get_outputs()[0]
        
        # Display basic info
        print(f"[INFO] Model loaded successfully")
        print(f"1. Input name: {input_info.name}")
        print(f"2. Input shape (symbolic): {input_info.shape}")
        print(f"3. Output name: {output_info.name}")
        print(f"4. Output shape (symbolic): {output_info.shape}")
        
        # Resolve shape
        resolved_shape = resolve_input_shape(session, show_progress=False)
        print(f"5. Resolved input shape: {resolved_shape}")
        
        # Generate test input
        test_input = np.random.rand(*resolved_shape).astype(np.float32)
        print(f"6. Test input generated: {test_input.shape}")
        
        # Run inference
        predictions = session.run([output_info.name], {input_info.name: test_input})[0]
        
        # Calculate probabilities with numerical stability
        def safe_softmax(x):
            try:
                x = np.clip(x, -100, 100)  # Prevent extreme values
                e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
                return e_x / e_x.sum(axis=1, keepdims=True)
            except:
                return np.full_like(x, 0.5)  # Fallback
        
        probabilities = safe_softmax(predictions)
        
        # Performance test
        start_time = time.time()
        for _ in range(10):
            session.run([output_info.name], {input_info.name: test_input})
        avg_time = (time.time() - start_time) / 10 * 1000
        
        # Display results with safety checks
        print(f"\n[SUCCESS] Model is operational!")
        print(f"[success] Input shape: {test_input.shape}")
        print(f"[success] Output shape: {predictions.shape}")
        print(f"[success] Raw prediction: [{predictions[0][0]:.4f}, {predictions[0][1]:.4f}]")
        
        # Handle potential numerical issues
        normal_prob = probabilities[0][0] if not np.isnan(probabilities[0][0]) else 0.5
        attack_prob = probabilities[0][1] if not np.isnan(probabilities[0][1]) else 0.5
        
        print(f"[success] Probabilities: [Normal: {normal_prob:.6f}, Attack: {attack_prob:.6f}]")
        print(f"[success] Predicted class: {'Attack' if attack_prob > normal_prob else 'Normal'}")
        print(f"[success] Average inference time: {avg_time:.2f} ms")
        
        # Output diagnostics
        print(f"[info] Output magnitude: min={np.min(predictions):.2f}, max={np.max(predictions):.2f}")
        
        print(f"\n[RESULT] Model verification PASSED! [success]")
        
    except Exception as e:
        logger.error(f"Error during verification: {str(e)}")
        print(f"\n[ERROR] Verification failed: {str(e)}")
        suggest_troubleshooting(model_path)

def suggest_troubleshooting(model_path: str, show_progress: bool = True) -> None:
    """Provide troubleshooting suggestions with progress tracking.
    
    Args:
        model_path: Path to the ONNX model file
        show_progress: Whether to display progress bars (default: True)
    """
    try:
        # Never use progress bars in troubleshooting (called during error handling)
        return _suggest_troubleshooting_without_progress(model_path)
    except Exception as e:
        logger.error(f"Troubleshooting failed: {str(e)}")
        raise

def _suggest_troubleshooting_with_alive_progress(model_path: str, bar) -> None:
    """Helper function for troubleshooting with alive_bar progress."""
    try:
        print(f"\n=== Troubleshooting Suggestions ===")
        
        # Check file existence
        bar.text("Checking file existence...")
        if not Path(model_path).exists():
            print("[error] Model file not found")
            print("   + Run: python convert_to_onnx.py")
            bar()
            return
        bar()
        
        # Check file size
        bar.text("Checking file size...")
        file_size = Path(model_path).stat().st_size
        print(f"[info] Model file exists ({file_size:,} bytes)")
        bar()
        
        # Check ONNX runtime
        bar.text("Checking ONNX runtime...")
        try:
            import onnxruntime as ort
            print(f"[success] ONNX Runtime version: {ort.__version__}")
        except ImportError:
            print("[error] ONNX Runtime not installed")
            print("   + Run: pip install onnxruntime")
            bar()
            return
        bar()
        
        # Try to load model
        bar.text("Checking model integrity...")
        try:
            session = ort.InferenceSession(model_path)
            print("[success] Model file can be loaded")
        except Exception as e:
            print(f"[error] Model file corrupted: {str(e)}")
            print("   + Re-run: python convert_to_onnx.py")
            bar()
            return
        bar()
        
        bar.text("Compiling suggestions...")
        print("\n[info] Additional suggestions:")
        print("   + Check if model was converted properly")
        print("   + Verify input data preprocessing is correct")
        print("   + Try running: python test_onnx_model.py for detailed testing")
        print("   + Check model input/output shapes match expectations")
        bar()
        
    except Exception as e:
        logger.error(f"Error during troubleshooting: {str(e)}")
        raise

def _suggest_troubleshooting_with_tqdm(model_path: str) -> None:
    """Helper function for troubleshooting with tqdm progress."""
    try:
        steps = [
            ("Checking file existence", lambda: Path(model_path).exists()),
            ("Checking file size", lambda exists: (
                Path(model_path).stat().st_size if exists else None
            )),
            ("Checking ONNX runtime", lambda: (
                __import__("onnxruntime"), 
                __import__("onnxruntime").__version__
            )),
            ("Checking model integrity", lambda ort: (
                ort.InferenceSession(model_path)
            )),
            ("Compiling suggestions", lambda: None)
        ]
        
        with tqdm(total=len(steps), desc="Diagnosing issues") as pbar:
            print(f"\n=== Troubleshooting Suggestions ===")
            
            # Step 1: Check file existence
            pbar.set_description(steps[0][0])
            exists = steps[0][1]()
            if not exists:
                print("[error] Model file not found")
                print("   + Run: python convert_to_onnx.py")
                return
            pbar.update(1)
            
            # Step 2: Check file size
            pbar.set_description(steps[1][0])
            file_size = steps[1][1](exists)
            print(f"[info] Model file exists ({file_size:,} bytes)")
            pbar.update(1)
            
            # Step 3: Check ONNX runtime
            pbar.set_description(steps[2][0])
            try:
                ort, version = steps[2][1]()
                print(f"[success] ONNX Runtime version: {version}")
            except ImportError:
                print("[error] ONNX Runtime not installed")
                print("   + Run: pip install onnxruntime")
                return
            pbar.update(1)
            
            # Step 4: Check model integrity
            pbar.set_description(steps[3][0])
            try:
                session = steps[3][1](ort)
                print("[success] Model file can be loaded")
            except Exception as e:
                print(f"[error] Model file corrupted: {str(e)}")
                print("   + Re-run: python convert_to_onnx.py")
                return
            pbar.update(1)
            
            # Step 5: Compile suggestions
            pbar.set_description(steps[4][0])
            print("\n[info] Additional suggestions:")
            print("   + Check if model was converted properly")
            print("   + Verify input data preprocessing is correct")
            print("   + Try running: python test_onnx_model.py for detailed testing")
            print("   + Check model input/output shapes match expectations")
            pbar.update(1)
            
    except Exception as e:
        logger.error(f"Error during troubleshooting: {str(e)}")
        raise

def _suggest_troubleshooting_without_progress(model_path: str) -> None:
    """Helper function for troubleshooting without progress bar."""
    try:
        print(f"\n=== Troubleshooting Suggestions ===")
        
        # Check file existence
        if not Path(model_path).exists():
            print("[error] Model file not found")
            print("   + Run: python convert_to_onnx.py")
            return
        
        # Check file size
        file_size = Path(model_path).stat().st_size
        print(f"[info] Model file exists ({file_size:,} bytes)")
        
        # Check ONNX runtime
        try:
            import onnxruntime as ort
            print(f"[success] ONNX Runtime version: {ort.__version__}")
        except ImportError:
            print("[error] ONNX Runtime not installed")
            print("   + Run: pip install onnxruntime")
            return
        
        # Try to load model
        try:
            session = ort.InferenceSession(model_path)
            print("[success] Model file can be loaded")
        except Exception as e:
            print(f"[error] Model file corrupted: {str(e)}")
            print("   + Re-run: python convert_to_onnx.py")
            return
        
        print("\n[info] Additional suggestions:")
        print("   + Check if model was converted properly")
        print("   + Verify input data preprocessing is correct")
        print("   + Try running: python test_onnx_model.py for detailed testing")
        print("   + Check model input/output shapes match expectations")
        
    except Exception as e:
        logger.error(f"Error during troubleshooting: {str(e)}")
        raise

def check_model_details(model_path: str, show_progress: bool = True) -> None:
    """Show detailed model information with progress tracking.
    
    Args:
        model_path: Path to the ONNX model file
        show_progress: Whether to display progress bars (default: True)
    """
    try:
        if not show_progress:
            return _check_model_details_without_progress(model_path)
        return _check_model_details_with_tqdm(model_path)
    except Exception as e:
        logger.error(f"Failed to check model details: {str(e)}")
        raise

def _check_model_details_with_alive_progress(model_path: str, bar) -> None:
    """Helper function for checking details with alive_bar progress."""
    try:
        bar.text("Initializing session...")
        session = ort.InferenceSession(model_path)
        bar()
        
        print(f"\n=== Model Details ===")
        print(f"Model: {model_path}")
        bar()
        
        # Input details
        bar.text("Checking inputs...")
        inputs = session.get_inputs()
        print("\nInputs:")
        for i, input_info in enumerate(inputs):
            print(f"  Input {i+1}: {input_info.name}")
            print(f"    Shape: {input_info.shape}")
            print(f"    Type: {input_info.type}")
        bar()
        
        # Output details
        bar.text("Checking outputs...")
        outputs = session.get_outputs()
        print("\nOutputs:")
        for i, output_info in enumerate(outputs):
            print(f"  Output {i+1}: {output_info.name}")
            print(f"    Shape: {output_info.shape}")
            print(f"    Type: {output_info.type}")
        bar()
        
        # Providers and metadata
        bar.text("Checking providers...")
        print("\nProviders:")
        print(f"  Available: {session.get_providers()}")
        print(f"  Current: {session.get_provider_options()}")
        bar()
        
    except Exception as e:
        logger.error(f"Error checking model details: {str(e)}")
        raise

def _check_model_details_with_tqdm(model_path: str) -> None:
    """Helper function for checking details with tqdm progress."""
    try:
        steps = [
            ("Initializing session", lambda: ort.InferenceSession(model_path)),
            ("Checking inputs", lambda session: session.get_inputs()),
            ("Checking outputs", lambda session: session.get_outputs()),
            ("Checking providers", lambda session: (
                session.get_providers(),
                session.get_provider_options()
            ))
        ]
        
        with tqdm(total=len(steps), desc="Loading model details") as pbar:
            print(f"\n=== Model Details ===")
            print(f"Model: {model_path}")
            
            # Step 1: Initialize session
            pbar.set_description(steps[0][0])
            session = steps[0][1]()
            pbar.update(1)
            
            # Step 2: Check inputs
            pbar.set_description(steps[1][0])
            inputs = steps[1][1](session)
            print("\nInputs:")
            for i, input_info in enumerate(inputs):
                print(f"  Input {i+1}: {input_info.name}")
                print(f"    Shape: {input_info.shape}")
                print(f"    Type: {input_info.type}")
            pbar.update(1)
            
            # Step 3: Check outputs
            pbar.set_description(steps[2][0])
            outputs = steps[2][1](session)
            print("\nOutputs:")
            for i, output_info in enumerate(outputs):
                print(f"  Output {i+1}: {output_info.name}")
                print(f"    Shape: {output_info.shape}")
                print(f"    Type: {output_info.type}")
            pbar.update(1)
            
            # Step 4: Check providers
            pbar.set_description(steps[3][0])
            providers, provider_options = steps[3][1](session)
            print("\nProviders:")
            print(f"  Available: {providers}")
            print(f"  Current: {provider_options}")
            pbar.update(1)
            
    except Exception as e:
        logger.error(f"Error checking model details: {str(e)}")
        raise

def _check_model_details_without_progress(model_path: str) -> None:
    """Helper function for checking details without progress bar."""
    try:
        print(f"\n=== Model Details ===")
        print(f"Model: {model_path}")
        
        session = ort.InferenceSession(model_path)
        
        # Input details
        print("\nInputs:")
        inputs = session.get_inputs()
        for i, input_info in enumerate(inputs):
            print(f"  Input {i+1}: {input_info.name}")
            print(f"    Shape: {input_info.shape}")
            print(f"    Type: {input_info.type}")
        
        # Output details
        print("\nOutputs:")
        outputs = session.get_outputs()
        for i, output_info in enumerate(outputs):
            print(f"  Output {i+1}: {output_info.name}")
            print(f"    Shape: {output_info.shape}")
            print(f"    Type: {output_info.type}")
        
        # Providers and metadata
        print("\nProviders:")
        print(f"  Available: {session.get_providers()}")
        print(f"  Current: {session.get_provider_options()}")
        
    except Exception as e:
        logger.error(f"Error checking model details: {str(e)}")
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="ONNX Model Quick Verification",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--model",
        default="models/ids_model.onnx",
        help="Path to ONNX model file"
    )
    parser.add_argument(
        "--details",
        action="store_true",
        help="Show detailed model information"
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable progress bars"
    )
    
    args = parser.parse_args()
    
    # Show model details if requested
    if args.details:
        check_model_details(args.model, not args.no_progress)
    
    # Run verification
    quick_verify(args.model, not args.no_progress)