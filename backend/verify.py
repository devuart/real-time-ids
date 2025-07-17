import onnxruntime as ort
import numpy as np
import argparse
from pathlib import Path

def resolve_input_shape(session: ort.InferenceSession) -> tuple:
    """Resolve symbolic dimensions to actual values."""
    input_info = session.get_inputs()[0]
    shape = input_info.shape
    
    # Convert symbolic dimensions to actual values
    resolved_shape = []
    for dim in shape:
        if isinstance(dim, str):
            if 'batch' in dim.lower():
                resolved_shape.append(1)  # Use batch size of 1 for testing
            else:
                # For other symbolic dimensions, use the known feature size
                resolved_shape.append(14)  # Known feature size from our model
        else:
            resolved_shape.append(dim)
    
    return tuple(resolved_shape)

def quick_verify(model_path: str) -> None:
    """Perform quick model verification."""
    try:
        print(f"\n=== Quick Verification: {model_path} ===")
        
        # Load model
        session = ort.InferenceSession(model_path)
        input_info = session.get_inputs()[0]
        output_info = session.get_outputs()[0]
        
        print(f"[INFO] Model loaded successfully")
        print(f"1. Input name: {input_info.name}")
        print(f"2. Input shape (symbolic): {input_info.shape}")
        print(f"3. Output name: {output_info.name}")
        print(f"4. Output shape (symbolic): {output_info.shape}")
        
        # Resolve symbolic dimensions
        resolved_shape = resolve_input_shape(session)
        print(f"5. Resolved input shape: {resolved_shape}")
        
        # Generate test input with resolved shape
        test_input = np.random.rand(*resolved_shape).astype(np.float32)
        print(f"6. Test input generated: {test_input.shape}")
        
        # Run inference
        predictions = session.run([output_info.name], {input_info.name: test_input})[0]
        
        # Apply softmax to get probabilities
        exp_preds = np.exp(predictions)
        probabilities = exp_preds / np.sum(exp_preds, axis=1, keepdims=True)
        
        print(f"\n[SUCCESS] Model is operational!")
        print(f"[success] Input shape: {test_input.shape}")
        print(f"[success] Output shape: {predictions.shape}")
        print(f"[success] Raw prediction: [{predictions[0][0]:.4f}, {predictions[0][1]:.4f}]")
        print(f"[success] Probabilities: [Normal: {probabilities[0][0]:.4f}, Attack: {probabilities[0][1]:.4f}]")
        print(f"[success] Predicted class: {'Attack' if np.argmax(probabilities[0]) == 1 else 'Normal'}")
        
        # Quick performance test
        import time
        start_time = time.time()
        for _ in range(10):
            session.run([output_info.name], {input_info.name: test_input})
        avg_time = (time.time() - start_time) / 10 * 1000  # Convert to ms
        
        print(f"[success] Average inference time: {avg_time:.2f} ms")
        print(f"\n[RESULT] Model verification PASSED! [success]")
        
    except Exception as e:
        print(f"\n[ERROR] Verification failed: {str(e)}")
        import traceback
        print(f"[DEBUG] Full error traceback:")
        traceback.print_exc()
        suggest_troubleshooting(model_path)

def suggest_troubleshooting(model_path: str) -> None:
    """Provide troubleshooting suggestions."""
    print(f"\n=== Troubleshooting Suggestions ===")
    
    # Check file existence
    if not Path(model_path).exists():
        print("[error] Model file not found")
        print("   + Run: python convert_to_onnx.py")
        return
    
    # Check file size
    file_size = Path(model_path).stat().st_size
    print(f"[info] Model file exists ({file_size} bytes)")
    
    # Check ONNX runtime
    try:
        import onnxruntime as ort
        print(f"[success] ONNX Runtime version: {ort.__version__}")
    except ImportError:
        print("[error] ONNX Runtime not installed")
        print("   + Run: pip install onnxruntime")
        return
    
    # Try to load model to check corruption
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

def check_model_details(model_path: str) -> None:
    """Show detailed model information."""
    try:
        session = ort.InferenceSession(model_path)
        
        print(f"\n=== Model Details ===")
        print(f"Model: {model_path}")
        
        # Input details
        for i, input_info in enumerate(session.get_inputs()):
            print(f"Input {i+1}: {input_info.name} | Shape: {input_info.shape} | Type: {input_info.type}")
        
        # Output details
        for i, output_info in enumerate(session.get_outputs()):
            print(f"Output {i+1}: {output_info.name} | Shape: {output_info.shape} | Type: {output_info.type}")
        
        # Providers
        print(f"Available providers: {session.get_providers()}")
        
    except Exception as e:
        print(f"Could not load model details: {str(e)}")

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
    
    args = parser.parse_args()
    
    # Show model details if requested
    if args.details:
        check_model_details(args.model)
    
    # Run verification
    quick_verify(args.model)