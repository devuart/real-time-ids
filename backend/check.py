import onnxruntime as ort
import argparse
from pathlib import Path

def verify_model_input(model_path: str) -> None:
    """Verify and display model input specifications."""
    try:
        session = ort.InferenceSession(model_path)
        input_info = session.get_inputs()[0]
        
        print("\n=== Model Input Verification ===")
        print(f"[SUCCESS] Model loaded: {model_path}")
        print(f"1. Input name: {input_info.name}")
        print(f"2. Shape: {input_info.shape}")
        print(f"3. Type: {input_info.type}")
        
    except Exception as e:
        print(f"\n[ERROR] Verification failed: {str(e)}")
        suggest_troubleshooting(model_path)

def suggest_troubleshooting(model_path: str) -> None:
    """Provide troubleshooting suggestions."""
    print("\nTroubleshooting:")
    if not Path(model_path).exists():
        print("- Model file not found. Run convert_to_onnx.py first")
    else:
        print("- Model may be corrupted. Try regenerating with convert_to_onnx.py")
        print("- Check ONNX runtime compatibility (pip install onnxruntime)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ONNX Model Input Verification")
    parser.add_argument(
        "--model",
        default="models/ids_model.onnx",
        help="Path to ONNX model"
    )
    args = parser.parse_args()
    
    verify_model_input(args.model)