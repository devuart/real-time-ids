import onnxruntime as ort
import argparse
from pathlib import Path

def verify_model_output(model_path: str) -> None:
    """Verify and display model output specifications."""
    try:
        session = ort.InferenceSession(model_path)
        output_info = session.get_outputs()[0]
        
        print("\n=== Model Output Verification ===")
        print(f"[SUCCESS] Model loaded: {model_path}")
        print(f"1. Output name: {output_info.name}")
        print(f"2. Shape: {output_info.shape}")
        print(f"3. Type: {output_info.type}")
        
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
        print("- Check ONNX runtime version compatibility")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ONNX Model Output Verification")
    parser.add_argument(
        "--model",
        default="models/ids_model.onnx",
        help="Path to ONNX model"
    )
    args = parser.parse_args()
    
    verify_model_output(args.model)