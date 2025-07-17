import pandas as pd
import joblib
import json
from pathlib import Path
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, LabelEncoder
import sys
import warnings
import subprocess
import os
from typing import Optional, Dict, Any, List, Tuple

# Constants for hybrid system
HYBRID_FEATURE_COUNT = 20
SCALER_RANGE = (0.1, 0.9)
MEMORY_SAFETY_FACTOR = 0.7

def prompt_user(prompt: str, default: Optional[bool] = None) -> bool:
    """Interactive user prompt with default handling and validation."""
    while True:
        response = input(f"{prompt} [{'Y/n' if default else 'y/N'}]: ").strip().lower()
        if not response:
            if default is not None:
                return default
        elif response in ('y', 'yes'):
            return True
        elif response in ('n', 'no'):
            return False
        print("Please answer yes/y or no/n")

def run_test_max_rows(config_path: str) -> Optional[Dict[str, Any]]:
    """Execute test_max_rows.py and return its results."""
    print("\n[INFO] Running system capacity test...")
    try:
        result = subprocess.run(
            [sys.executable, "test_max_rows.py", "--output", os.path.dirname(config_path)],
            check=True,
            capture_output=True,
            text=True
        )
        print(result.stdout)
        
        summary_path = Path(os.path.dirname(config_path)) / "testing_summary.json"
        if summary_path.exists():
            with open(summary_path) as f:
                return json.load(f)
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Test failed: {e.stderr}")
    except Exception as e:
        print(f"[ERROR] Unexpected error: {str(e)}")
    return None

def get_memory_config(
    config_path: str = "results/testing_summary.json",
    interactive: bool = True
) -> Tuple[int, int]:
    """
    Get memory configuration with interactive fallback.
    Returns (max_rows, chunk_size)
    """
    try:
        if Path(config_path).exists():
            with open(config_path) as f:
                config = json.load(f)
                max_rows = config['dataset_info']['total_rows']
                chunk_size = int(config['performance_metrics']['max_chunk_size'] * MEMORY_SAFETY_FACTOR)
                print(f"[CONFIG] Using tested configuration:")
                print(f"  - Max rows: {max_rows:,}")
                print(f"  - Safe chunk size: {chunk_size:,}")
                return max_rows, chunk_size
    except Exception as e:
        warnings.warn(f"Config load error: {str(e)}")

    if not interactive:
        return 1000000, 100000

    print("\n[WARNING] System capacity data not available!")
    print("Running a capacity test ensures optimal performance and prevents memory errors.")
    
    if prompt_user("Run system capacity test now?", default=True):
        if config := run_test_max_rows(config_path):
            try:
                max_rows = config['dataset_info']['total_rows']
                chunk_size = int(config['performance_metrics']['max_chunk_size'] * MEMORY_SAFETY_FACTOR)
                return max_rows, chunk_size
            except KeyError:
                print("[WARNING] Test didn't produce valid results")

    print("\n[WARNING] Using default values may cause memory issues!")
    if prompt_user("Specify custom limits?", default=False):
        while True:
            try:
                max_rows = int(input("Enter max rows to process (0 for default): "))
                chunk_size = int(input("Enter chunk size (0 for default): "))
                if max_rows > 0 and chunk_size > 0:
                    return max_rows, chunk_size
                if max_rows == 0 and chunk_size == 0:
                    break
                print("Please enter positive integers")
            except ValueError:
                print("Invalid number format")

    default_max = 1000000
    default_chunk = 100000
    warnings.warn(f"Using default values - Max: {default_max:,}, Chunk: {default_chunk:,}")
    return default_max, default_chunk

def select_and_pad_features(df: pd.DataFrame, all_features: list, target_count: int = HYBRID_FEATURE_COUNT) -> Tuple[pd.DataFrame, List[str]]:
    """Select and pad features to reach target count for hybrid system compatibility."""
    current_features = [col for col in all_features if col in df.columns]
    
    if len(current_features) >= target_count:
        selected_features = current_features[:target_count]
        print(f"[FEATURE_SELECTION] Using {len(selected_features)} features (truncated from {len(current_features)})")
        return df[selected_features], selected_features
    else:
        missing_count = target_count - len(current_features)
        print(f"[FEATURE_PADDING] Adding {missing_count} synthetic features to reach {target_count}")
        
        synthetic_features = []
        for i in range(missing_count):
            synthetic_col = f"synthetic_feature_{i}"
            df[synthetic_col] = 0.0
            synthetic_features.append(synthetic_col)
        
        final_features = current_features + synthetic_features
        return df[final_features], final_features

def safe_label_encode(series: pd.Series, encoder: LabelEncoder) -> pd.Series:
    """Handle unseen labels in LabelEncoder by assigning them to a special category."""
    # Ensure string comparison and prevent Categorical dtype issues
    series_str = series.astype(str)
    encoder_classes = list(encoder.classes_)

    try:
        return encoder.transform(series_str)
    except ValueError:
        # Handle unseen labels by assigning them to a new category
        extended_classes = encoder_classes + ['UNSEEN_LABEL']
        extended_encoder = LabelEncoder()
        extended_encoder.fit(extended_classes)

        clean_series = series_str.where(series_str.isin(encoder_classes), 'UNSEEN_LABEL')
        return extended_encoder.transform(clean_series)

def process_chunk(
    df: pd.DataFrame,
    output_path: Path,
    encoders: Dict[str, Any],
    scaler: Optional[MinMaxScaler],
    label_col: str = "Label"
) -> Tuple[pd.DataFrame, List[str], MinMaxScaler]:
    """Process a single chunk of data with all transformations."""
    initial_rows = len(df)
    df.dropna(inplace=True)
    if initial_rows != len(df):
        print(f"[CLEANING] Removed {initial_rows - len(df)} rows with missing values")

    labels = df[label_col].astype('category').cat.codes
    df.drop(columns=[label_col], inplace=True)

    categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()

    # IP Address Encoding with unseen label handling
    ip_cols = ["IPV4_SRC_ADDR", "IPV4_DST_ADDR"]
    for col in ip_cols:
        if col in df.columns and col in encoders:
            df[col] = safe_label_encode(df[col], encoders[col])

    other_categorical = list(set(categorical_cols) - set(ip_cols))
    encoded_feature_names = []
    
    if other_categorical and 'one_hot' in encoders:
        encoded_features = encoders['one_hot'].transform(df[other_categorical])
        encoded_feature_names = encoders['one_hot'].get_feature_names_out(other_categorical).tolist()
        encoded_df = pd.DataFrame(encoded_features, columns=encoded_feature_names)
        df = df.drop(columns=other_categorical)
        df = pd.concat([df, encoded_df], axis=1)

    remaining_numeric = [col for col in numeric_cols if col not in ip_cols]
    ip_encoded_cols = [col for col in ip_cols if col in df.columns]
    all_feature_names = remaining_numeric + ip_encoded_cols + encoded_feature_names

    df_features, selected_features = select_and_pad_features(df, all_feature_names)

    if scaler is None:
        scaler = MinMaxScaler(feature_range=SCALER_RANGE)
        scaler.fit(df_features[selected_features])
    
    df_features[selected_features] = scaler.transform(df_features[selected_features])
    df_features[label_col] = labels

    return df_features, selected_features, scaler

def preprocess_data(
    filepath: str,
    output_dir: str = "models",
    config_path: str = "results/testing_summary.json",
    interactive: bool = True,
    verbose: bool = False
) -> None:
    """Main preprocessing pipeline with memory-aware chunked processing."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    max_rows, chunk_size = get_memory_config(config_path, interactive)
    print(f"\n[CONFIG] Processing limit: {max_rows:,} rows")
    print(f"[CONFIG] Chunk size: {chunk_size:,} rows")
    print(f"[HYBRID] Target feature count: {HYBRID_FEATURE_COUNT}")

    # Initialize encoders on a larger sample to reduce chance of unseen labels
    print("\n[INIT] Initializing encoders...")
    sample_size = min(50000, chunk_size)  # Larger sample for better coverage
    sample_df = pd.read_csv(
        filepath,
        nrows=sample_size,
        low_memory=False,
        dtype={col: "category" for col in ["IPV4_SRC_ADDR", "IPV4_DST_ADDR"]},
        on_bad_lines='warn'
    )
    
    encoders = {}
    ip_cols = ["IPV4_SRC_ADDR", "IPV4_DST_ADDR"]
    for col in ip_cols:
        if col in sample_df.columns:
            le = LabelEncoder()
            le.fit(sample_df[col])
            encoders[col] = le
            joblib.dump(le, output_path / f"{col}_label_encoder.pkl")

    categorical_cols = sample_df.select_dtypes(include=["object", "category"]).columns.tolist()
    other_categorical = list(set(categorical_cols) - set(ip_cols))
    
    if other_categorical:
        encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        encoder.fit(sample_df[other_categorical])
        encoders['one_hot'] = encoder
        joblib.dump(encoder, output_path / "one_hot_encoder.pkl")

    print("\n[PROCESSING] Starting chunked processing...")
    chunk_reader = pd.read_csv(
        filepath,
        chunksize=chunk_size,
        nrows=max_rows,
        low_memory=False,
        dtype={col: "category" for col in ["IPV4_SRC_ADDR", "IPV4_DST_ADDR"]},
        on_bad_lines='warn'
    )

    processed_chunks = []
    selected_features = []
    scaler = None
    total_rows = 0

    for chunk_idx, df_chunk in enumerate(chunk_reader, 1):
        print(f"\n[CHUNK {chunk_idx}] Processing {len(df_chunk):,} rows...")
        
        try:
            df_processed, features, scaler = process_chunk(
                df_chunk,
                output_path,
                encoders,
                scaler
            )
            
            if not selected_features:
                selected_features = features
            
            processed_chunks.append(df_processed)
            total_rows += len(df_processed)
            
            print(f"[CHUNK {chunk_idx}] Completed - {total_rows:,} total rows processed")
            if verbose:
                print(f"  Features: {features[:5]}{'...' if len(features) > 5 else ''}")

        except Exception as e:
            print(f"[ERROR] Failed to process chunk {chunk_idx}: {str(e)}")
            if prompt_user("Continue processing?", default=True):
                continue
            else:
                break

    if not processed_chunks:
        print("\n[ERROR] No data was processed successfully")
        return

    artifacts = {
        "scaler": scaler,
        "feature_names": selected_features,
        "total_rows": total_rows,
        "chunks_processed": chunk_idx,
        "original_features": list(set(
            sample_df.select_dtypes(include=["object", "category", "int64", "float64"]).columns.tolist()
        )),
        "hybrid_feature_count": HYBRID_FEATURE_COUNT,
        "input_size": len(selected_features)
    }
    joblib.dump(artifacts, output_path / "preprocessing_artifacts.pkl")

    final_df = pd.concat(processed_chunks)
    processed_path = output_path / "preprocessed_dataset.csv"
    final_df.to_csv(processed_path, index=False)

    print("\n[SUCCESS] Preprocessing complete (Hybrid-ready)")
    print(f"1. Total rows processed: {total_rows:,}")
    print(f"2. Chunks processed: {chunk_idx}")
    print(f"3. Final feature count: {len(selected_features)}")
    print(f"4. Output directory: {output_path.resolve()}")
    if verbose:
        print(f"5. Sample features: {selected_features[:10]}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Memory-Aware Hybrid-Compatible Data Preprocessing Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--input",
        default="datasets/NF-CSE-CIC-IDS2018.csv",
        help="Path to input CSV file"
    )
    parser.add_argument(
        "--output",
        default="models",
        help="Output directory for processed data"
    )
    parser.add_argument(
        "--config",
        default="results/testing_summary.json",
        help="Path to capacity test results"
    )
    parser.add_argument(
        "--non-interactive",
        action="store_true",
        help="Disable all interactive prompts"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show detailed processing information"
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        pd.set_option('display.max_columns', None)
    warnings.filterwarnings("once")
    
    preprocess_data(
        filepath=args.input,
        output_dir=args.output,
        config_path=args.config,
        interactive=not args.non_interactive,
        verbose=args.verbose
    )