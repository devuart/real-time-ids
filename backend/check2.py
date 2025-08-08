import sys
import io
import onnxruntime as ort
import argparse
from pathlib import Path
from alive_progress import alive_bar
import time

# Fix encoding for Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

def verify_model_output(model_path: str) -> None:
    """Verify and display model output specifications."""
    try:
        with alive_bar(4, title="Verifying model outputs", spinner='classic', bar='blocks') as bar:
            # Step 1: Initializing verification
            time.sleep(0.3)
            bar.text("Loading ONNX session...")
            bar()
            
            # Step 2: Create inference session
            session = ort.InferenceSession(model_path)
            time.sleep(0.4)
            bar.text("Analyzing outputs...")
            bar()
            
            # Step 3: Get output info
            output_info = session.get_outputs()[0]
            time.sleep(0.3)
            bar.text("Compiling results...")
            bar()
            
            # Step 4: Display results
            print("\n=== Model Output Verification ===")
            print(f"[SUCCESS] Model loaded: {model_path}")
            print(f"1. Output name: {output_info.name}")
            print(f"2. Shape: {output_info.shape}")
            print(f"3. Type: {output_info.type}")
            
            time.sleep(0.2)
            bar.text("Verification complete!")
            bar()
            
    except Exception as e:
        print(f"\n[ERROR] Verification failed: {str(e)}")
        suggest_troubleshooting(model_path)

def suggest_troubleshooting(model_path: str) -> None:
    """Provide troubleshooting suggestions."""
    with alive_bar(1, title="Diagnosing issues", spinner='vertical') as bar:
        print("\nTroubleshooting:")
        if not Path(model_path).exists():
            print("- Model file not found. Run convert_to_onnx.py first")
        else:
            print("- Model may be corrupted. Try regenerating with convert_to_onnx.py")
            print("- Check ONNX runtime version compatibility")
        bar()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ONNX Model Output Verification")
    parser.add_argument(
        "--model",
        default="models/ids_model.onnx",
        help="Path to ONNX model"
    )
    args = parser.parse_args()
    
    verify_model_output(args.model)

