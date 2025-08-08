import torch
import torch.onnx
import os
import sys
import argparse
from pathlib import Path
from train_model import IDSModel
import joblib
import subprocess
import pandas as pd
import pickle
from sklearn.preprocessing import MinMaxScaler, StandardScaler

def prompt_user(prompt: str, default: bool = True) -> bool:
    """Interactive user prompt with default handling."""
    while True:
        response = input(f"{prompt} [{'Y/n' if default else 'y/N'}]: ").strip().lower()
        if not response:
            return default
        if response in ('y', 'yes'):
            return True
        if response in ('n', 'no'):
            return False
        print("Please answer yes/y or no/n")

def check_dependencies() -> bool:
    """Verify all required files exist."""
    required_files = [
        "models/ids_model.pth",
        "models/preprocessing_artifacts.pkl"
    ]
    missing = [f for f in required_files if not Path(f).exists()]
    if missing:
        print("\n[WARNING] Missing required files:")
        for f in missing:
            print(f"- {f}")
        return False
    return True

def run_training_if_needed() -> bool:
    """Offer to run training if model is missing."""
    print("\n[WARNING] Trained model not found!")
    if prompt_user("Run model training now?", default=True):
        try:
            result = subprocess.run(
                [sys.executable, "train_model.py"],
                check=True,
                capture_output=True,
                text=True
            )
            print(result.stdout)
            return Path("models/ids_model.pth").exists()
        except subprocess.CalledProcessError as e:
            print(f"[ERROR] Training failed: {e.stderr}")
            return False
    return False

def get_model_io_dims() -> tuple:
    """Get input/output dimensions with comprehensive fallback strategy."""
    input_size = None
    output_size = 2  # Binary classification is standard
    
    # Method 1: Try to get dimensions from preprocessed data
    try:
        df = pd.read_csv("models/preprocessed_dataset.csv")
        actual_input_size = df.shape[1] - 1  # Subtract 1 for Label column
        print(f"[INFO] Detected input dimensions from preprocessed data: {actual_input_size}")
        return actual_input_size, output_size
    except Exception as e1:
        print(f"[WARNING] Could not read preprocessed data: {str(e1)}")
    
    # Method 2: Try to get dimensions from preprocessing artifacts
    try:
        artifacts = joblib.load("models/preprocessing_artifacts.pkl")
        if "feature_names" in artifacts:
            input_size = len(artifacts["feature_names"])
            print(f"[INFO] Using feature count from artifacts: {input_size}")
            return input_size, output_size
        elif "input_size" in artifacts:
            input_size = artifacts["input_size"]
            print(f"[INFO] Using stored input size from artifacts: {input_size}")
            return input_size, output_size
    except Exception as e2:
        print(f"[WARNING] Could not read preprocessing artifacts: {str(e2)}")
    
    # Method 3: Try to infer from saved model if it exists
    try:
        checkpoint = torch.load("models/ids_model.pth", map_location='cpu', weights_only=True)
        first_layer_weight = checkpoint['net.0.weight']
        input_size = first_layer_weight.shape[1]
        print(f"[INFO] Inferred input size from saved model: {input_size}")
        return input_size, output_size
    except Exception as e3:
        print(f"[WARNING] Could not infer from model file: {str(e3)}")
    
    # Method 4: Check if there's a feature specification file
    try:
        feature_spec_files = [
            "models/feature_specification.txt",
            "feature_config.json",
            "preprocessing_config.json"
        ]
        for spec_file in feature_spec_files:
            if Path(spec_file).exists():
                with open(spec_file, 'r') as f:
                    content = f.read()
                    # Try to extract feature count from content
                    import re
                    feature_match = re.search(r'feature.*?(\d+)', content.lower())
                    if feature_match:
                        input_size = int(feature_match.group(1))
                        print(f"[INFO] Found feature count in {spec_file}: {input_size}")
                        return input_size, output_size
    except Exception as e4:
        print(f"[WARNING] Could not read feature specification files: {str(e4)}")
    
    print("[ERROR] Could not determine model dimensions from any source")
    return None, None

def validate_model_dimensions(model_path: str, expected_input_size: int) -> bool:
    """Validate that the saved model matches expected dimensions."""
    try:
        #checkpoint = torch.load(model_path, map_location='cpu', pickle_module=pickle, weights_only=True)
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=True)
        first_layer_weight = checkpoint['net.0.weight']
        actual_input_size = first_layer_weight.shape[1]
        
        if actual_input_size != expected_input_size:
            print(f"[ERROR] Dimension mismatch:")
            print(f"  Expected: {expected_input_size}")
            print(f"  Actual model: {actual_input_size}")
            return False
        return True
    except Exception as e:
        print(f"[WARNING] Could not validate model dimensions: {str(e)}")
        return True  # Assume valid if we can't check

def inspect_preprocessing_artifacts():
    """Debug function to inspect preprocessing artifacts."""
    try:
        print("\n[DEBUG] Inspecting preprocessing artifacts...")
        
        # Check preprocessed data
        try:
            df = pd.read_csv("models/preprocessed_dataset.csv")
            print(f"Preprocessed data shape: {df.shape}")
            print(f"Columns ({len(df.columns)}): {list(df.columns)}")
            print(f"Feature columns: {len(df.columns) - 1} (excluding Label)")
        except Exception as e:
            print(f"Could not read preprocessed data: {str(e)}")
        
        # Check artifacts
        try:
            artifacts = joblib.load("models/preprocessing_artifacts.pkl")
            print(f"\nArtifacts content:")
            for key, value in artifacts.items():
                if key == "feature_names":
                    print(f"  {key}: {len(value)} features")
                    print(f"    Features: {value}")
                else:
                    print(f"  {key}: {type(value)}")
        except Exception as e:
            print(f"Could not read artifacts: {str(e)}")
        
        # Check model file
        try:
            checkpoint = torch.load("models/ids_model.pth", map_location='cpu', weights_only=True)
            print(f"\nModel structure:")
            for key, tensor in checkpoint.items():
                if hasattr(tensor, 'shape'):
                    print(f"  {key}: {tensor.shape}")
            first_layer_weight = checkpoint['net.0.weight']
            print(f"\nModel expects {first_layer_weight.shape[1]} input features")
        except Exception as e:
            print(f"Could not read model file: {str(e)}")
        
    except Exception as e:
        print(f"[DEBUG] Inspection failed: {str(e)}")

def save_feature_specification(input_size: int, output_size: int):
    """Save feature specification for future reference."""
    try:
        spec_content = f"""# Feature Specification
# Generated by convert_to_onnx.py

INPUT_FEATURES={input_size}
OUTPUT_CLASSES={output_size}
MODEL_TYPE=binary_classification

# This file helps maintain consistency across model operations
"""
        with open("models/feature_specification.txt", "w") as f:
            f.write(spec_content)
        print(f"[INFO] Saved feature specification to models/feature_specification.txt")
    except Exception as e:
        print(f"[WARNING] Could not save feature specification: {str(e)}")

def convert_to_onnx(args):
    """Main conversion pipeline with comprehensive dynamic sizing."""
    print("\n=== PyTorch to ONNX Converter ===")
    
    # Verify dependencies
    if not check_dependencies():
        if not args.non_interactive and run_training_if_needed():
            pass  # Dependencies now exist
        else:
            print("\n[ERROR] Cannot proceed without trained model")
            sys.exit(1)
    
    # Get model dimensions with improved fallback strategy
    input_size, output_size = get_model_io_dims()
    
    if input_size is None:
        print("\n[WARNING] Could not determine input dimensions automatically")
        if not args.non_interactive:
            if prompt_user("Would you like to specify input size manually?", default=True):
                while True:
                    try:
                        manual_input = input("Enter number of input features: ").strip()
                        input_size = int(manual_input)
                        if input_size > 0:
                            output_size = 2  # Default binary classification
                            print(f"[INFO] Using manual input size: {input_size}")
                            break
                        else:
                            print("Please enter a positive number")
                    except ValueError:
                        print("Please enter a valid number")
            else:
                print("[ERROR] Cannot proceed without input dimensions")
                sys.exit(1)
        else:
            # Non-interactive fallback to common default
            input_size, output_size = 14, 2
            print(f"[WARNING] Using default dimensions (input={input_size}, output={output_size})")
    
    # Validate model dimensions before loading
    if not validate_model_dimensions("models/ids_model.pth", input_size):
        print("\n[ERROR] Model dimension mismatch detected!")
        
        # Show debug information
        if not args.non_interactive:
            if prompt_user("Show detailed debug information?", default=True):
                inspect_preprocessing_artifacts()
            
            if prompt_user("Retrain model with correct dimensions?", default=True):
                if run_training_if_needed():
                    # Re-check dimensions after retraining
                    input_size, output_size = get_model_io_dims()
                    if input_size is None:
                        print("\n[ERROR] Still cannot determine dimensions after retraining")
                        sys.exit(1)
                else:
                    print("\n[ERROR] Retraining failed")
                    sys.exit(1)
            else:
                print("\n[ERROR] Cannot proceed with dimension mismatch")
                sys.exit(1)
        else:
            print("\n[ERROR] Dimension mismatch in non-interactive mode")
            sys.exit(1)
    
    print(f"[INFO] Using model dimensions: input={input_size}, output={output_size}")
    
    # Save feature specification for future reference
    save_feature_specification(input_size, output_size)
    
    # Load model
    try:
        model = IDSModel(input_size, output_size)
        model.load_state_dict(torch.load("models/ids_model.pth", map_location='cpu'))
        model.eval()
        print("[INFO] Model loaded successfully")

        # Apply quantization if requested
        if args.quantize:
            print("[INFO] Applying dynamic quantization to model")
            model = torch.quantization.quantize_dynamic(
                model,
                {torch.nn.Linear},
                dtype=torch.qint8
            )
    except Exception as e:
        print(f"[ERROR] Model loading failed: {str(e)}")
        if not args.non_interactive:
            print("\nTroubleshooting suggestions:")
            print("1. Check if model file is corrupted")
            print("2. Verify preprocessing artifacts are correct")
            print("3. Try retraining the model")
            print("4. Check if input dimensions match model architecture")
        sys.exit(1)
    
    # Prepare dummy input
    dummy_input = torch.randn(args.batch_size, input_size, dtype=torch.float32)
    print(f"[INFO] Created dummy input with shape: {dummy_input.shape}")
    
    # Ensure output directory exists
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    
    # Export to ONNX
    try:
        print(f"[INFO] Exporting to ONNX format...")
        
        # Prepare dynamic axes based on whether batch size should be dynamic
        dynamic_axes = {}
        if args.dynamic_batch:
            dynamic_axes = {
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        
        torch.onnx.export(
            model,
            dummy_input,
            args.output,
            opset_version=args.opset,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes=dynamic_axes,
            export_params=True,
            do_constant_folding=True,
            verbose=args.verbose
        )
        
        print(f"\n[SUCCESS] Model converted to ONNX: {args.output}")
        print(f"1. Input shape: [{'batch_size' if args.dynamic_batch else args.batch_size}, {input_size}]")
        print(f"2. Output shape: [{'batch_size' if args.dynamic_batch else args.batch_size}, {output_size}]")
        print(f"3. Opset version: {args.opset}")
        print(f"4. Quantized: {args.quantize}")
        print(f"5. Dynamic batch: {args.dynamic_batch}")
        print(f"6. File size: {Path(args.output).stat().st_size / 1024:.1f} KB")
        
        # Verify the ONNX model
        try:
            import onnx
            onnx_model = onnx.load(args.output)
            onnx.checker.check_model(onnx_model)
            print("[VALIDATION] ONNX model is valid")
            
            # Print model info
            print(f"\nModel Information:")
            print(f"- Inputs: {len(onnx_model.graph.input)}")
            print(f"- Outputs: {len(onnx_model.graph.output)}")
            print(f"- Nodes: {len(onnx_model.graph.node)}")
            
            # Test inference if onnxruntime is available
            try:
                import onnxruntime as ort
                session = ort.InferenceSession(args.output)
                test_input = dummy_input.numpy()
                result = session.run(None, {"input": test_input})
                print(f"[TEST] Inference test passed - output shape: {result[0].shape}")
            except ImportError:
                print("[INFO] Install onnxruntime (pip install onnxruntime) for inference testing")
            except Exception as e:
                print(f"[WARNING] Inference test failed: {str(e)}")
                
        except ImportError:
            print("[INFO] Install onnx package (pip install onnx) for model validation")
        except Exception as e:
            print(f"[WARNING] ONNX validation failed: {str(e)}")
            
    except Exception as e:
        print(f"[ERROR] Conversion failed: {str(e)}")
        if not args.non_interactive:
            print("\nTroubleshooting suggestions:")
            print("1. Try a different opset version (--opset 11 or --opset 14)")
            print("2. Disable quantization if enabled")
            print("3. Check available disk space")
            print("4. Try without dynamic batch sizing (--no-dynamic-batch)")
        sys.exit(1)

if __name__ == "__main__":
    # Create models directory if needed
    Path("models").mkdir(exist_ok=True)
    
    # Argument parsing
    parser = argparse.ArgumentParser(
        description="PyTorch to ONNX model converter with comprehensive dynamic sizing",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=13,
        help="ONNX opset version to use for conversion"
    )
    parser.add_argument(
        "--output",
        default="models/ids_model.onnx",
        help="Path to save the ONNX model"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size for the dummy input tensor"
    )
    parser.add_argument(
        "--quantize",
        action="store_true",
        help="Apply dynamic quantization before conversion"
    )
    parser.add_argument(
        "--non-interactive",
        action="store_true",
        help="Disable all interactive prompts"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Show detailed debug information"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose ONNX export"
    )
    parser.add_argument(
        "--no-dynamic-batch",
        dest="dynamic_batch",
        action="store_false",
        default=True,
        help="Disable dynamic batch sizing"
    )
    
    args = parser.parse_args()
    
    # Show debug info if requested
    if args.debug:
        inspect_preprocessing_artifacts()
    
    # Run conversion
    convert_to_onnx(args)