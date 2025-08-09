import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import joblib
import argparse
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset
import sys
import warnings
import json
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import GradScaler, autocast
import onnx
import onnxruntime as ort
import optuna
from typing import List, Optional, Dict, Tuple, Union, Any
import logging
import pandas as pd
import optuna.visualization as vis
import plotly
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import platform
import shutil

# Setup logging
LOG_DIR = Path("logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE_NAME = "deep_learning.log"
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

# Configure directories
DEFAULT_MODEL_DIR = Path("models")
DEFAULT_MODEL_DIR.mkdir(exist_ok=True)

CONFIG_DIR = Path("config")
CONFIG_DIR.mkdir(exist_ok=True)
CONFIG_FILE_NAME = "deep_learning_config.json"
CONFIG_FILE = CONFIG_DIR / CONFIG_FILE_NAME

REPORTS_DIR = Path("reports")
REPORTS_DIR.mkdir(exist_ok=True)

# Declare the class names that will be defined later
class SimpleAutoencoder:
    # Forward declaration
    pass

class EnhancedAutoencoder:
    # Forward declaration
    pass

class AutoencoderEnsemble:
    # Forward declaration
    pass

# Configuration Constants
DEFAULT_BATCH_SIZE = 64
DEFAULT_EPOCHS = 10
EARLY_STOPPING_PATIENCE = 100
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-4
GRADIENT_CLIP = 1.0
GRADIENT_ACCUMULATION_STEPS = 4
MIXED_PRECISION = True

# Model Architecture Constants
DEFAULT_ENCODING_DIM = 10
HIDDEN_LAYER_SIZES = [128, 64]
DROPOUT_RATES = [0.2, 0.15]
ACTIVATION = 'leaky_relu'
ACTIVATION_PARAM = 0.2
NORMALIZATION = 'batch'
USE_BATCH_NORM = True
USE_LAYER_NORM = False
DIVERSITY_FACTOR = 0.1
MIN_FEATURES = 5
NUM_MODELS = 3
FEATURES = 20
NORMALIZATION_OPTIONS = ['batch', 'layer', None]
NORMAL_SAMPLES = 8000
ATTACK_SAMPLES = 2000
ANOMALY_FACTOR = 1.5
RANDOM_STATE = 42

# Security Constants
DEFAULT_PERCENTILE = 95
DEFAULT_ATTACK_THRESHOLD = 0.3
FALSE_NEGATIVE_COST = 2.0
SECURITY_METRICS = True

NUM_WORKERS = min(4, os.cpu_count() or 1)

# Configuration for Testing Different Architectures
STABILITY_CONFIG = {
    'model_type': 'SimpleAutoencoder',
    'use_batch_norm': True,
    'dropout_rates': [0.3, 0.25],
    'gradient_clip': 1.0,
    'learning_rate': 1e-3,
    'weight_decay': 1e-4,
    'fn_cost': 2.0
}

PERFORMANCE_CONFIG = {
    'model_type': 'AutoencoderEnsemble',
    'num_models': 3,
    'use_batch_norm': True,
    'dropout_rates': [0.2, 0.15],
    'gradient_clip': 0.5,
    'learning_rate': 5e-4,
    'weight_decay': 1e-5
}

# Model Architecture Options
MODEL_VARIANTS = {
    'SimpleAutoencoder': SimpleAutoencoder,
    'EnhancedAutoencoder': EnhancedAutoencoder,
    'AutoencoderEnsemble': AutoencoderEnsemble
}

# Preset Configurations
PRESET_CONFIGS = {
    'stability': {
        'training': {
            'batch_size': 64,
            'epochs': 50,
            'learning_rate': 1e-3,
            'patience': 8,
            'gradient_clip': 1.0
        },
        'model': {
            'model_type': 'SimpleAutoencoder',
            'encoding_dim': 8,
            'use_batch_norm': True,
            'dropout_rates': [0.3, 0.25],
            'activation': 'leaky_relu',
            'activation_param': 0.1
        },
        'security': {
            'percentile': 97,
            'false_negative_cost': 2.5
        }
    },
    'performance': {
        'training': {
            'batch_size': 128,
            'epochs': 100,
            'learning_rate': 5e-4,
            'patience': 12,
            'gradient_clip': 0.5
        },
        'model': {
            'model_type': 'AutoencoderEnsemble',
            'encoding_dim': 12,
            'num_models': 3,
            'use_batch_norm': True,
            'dropout_rates': [0.2, 0.15],
            'activation': 'leaky_relu',
            'activation_param': 0.2
        },
        'security': {
            'percentile': 95,
            'false_negative_cost': 2.0
        }
    },
    'baseline': {
        'training': {
            'batch_size': 64,
            'epochs': 75,
            'learning_rate': 1e-3,
            'patience': 10,
            'gradient_clip': 1.0
        },
        'model': {
            'model_type': 'EnhancedAutoencoder',
            'encoding_dim': 10,
            'hidden_dims': [128, 64],
            'use_batch_norm': True,
            'dropout_rates': [0.2, 0.15],
            'activation': 'leaky_relu',
            'activation_param': 0.2
        },
        'security': {
            'percentile': 95,
            'false_negative_cost': 1.5
        }
    },
    'debug': {
        'training': {
            'batch_size': 32,
            'epochs': 10,
            'learning_rate': 1e-3,
            'patience': 5,
            'gradient_clip': 0.5
        },
        'model': {
            'model_type': 'SimpleAutoencoder',
            'encoding_dim': 5,
            'use_batch_norm': True,
            'dropout_rates': [0.2],
            'activation': 'relu'
        },
        'security': {
            'percentile': 90,
            'false_negative_cost': 1.0
        }
    }
}

def compare_model_architectures(input_dim: int = 20) -> Dict[str, Dict]:
    """Compare parameter counts and complexity of different model architectures"""
    results = {}
    
    # Initialize model variants if empty
    if not MODEL_VARIANTS:
        initialize_model_variants()
    
    for model_name, model_class in MODEL_VARIANTS.items():
        try:
            if model_name == 'SimpleAutoencoder':
                model = model_class(
                    input_dim=input_dim,
                    encoding_dim=DEFAULT_ENCODING_DIM
                )
            elif model_name == 'EnhancedAutoencoder':
                model = model_class(
                    input_dim=input_dim,
                    encoding_dim=DEFAULT_ENCODING_DIM,
                    hidden_dims=HIDDEN_LAYER_SIZES,
                    dropout_rates=DROPOUT_RATES,
                    activation=ACTIVATION,
                    activation_param=ACTIVATION_PARAM,
                    normalization=NORMALIZATION
                )
            elif model_name == 'AutoencoderEnsemble':
                model = model_class(
                    input_dim=input_dim,
                    num_models=NUM_MODELS,
                    encoding_dim=DEFAULT_ENCODING_DIM,
                    diversity_factor=DIVERSITY_FACTOR
                )
            else:
                continue
                
            # Calculate parameters
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            
            # Estimate memory usage (rough approximation)
            param_memory_mb = total_params * 4 / (1024 * 1024)  # 4 bytes per float32
            
            results[model_name] = {
                'total_params': total_params,
                'trainable_params': trainable_params,
                'memory_mb': param_memory_mb,
                'model_class': model_class.__name__
            }
            
        except Exception as e:
            results[model_name] = {'error': str(e)}
    
    return results

def display_model_comparison():
    """Display model architecture comparison in a formatted table"""
    results = compare_model_architectures()
    
    print("\nModel Architecture Comparison:")
    print("-" * 80)
    print(f"{'Model':<20} | {'Params':>12} | {'Trainable':>12} | {'Memory (MB)':>12} | Status")
    print("-" * 80)
    
    for model_name, stats in results.items():
        if 'error' in stats:
            status = f"Error: {stats['error'][:30]}..."
            print(f"{model_name:<20} | {'N/A':>12} | {'N/A':>12} | {'N/A':>12} | {status}")
        else:
            # Determine complexity level
            params = stats['total_params']
            if params < 10000:
                complexity = "Low"
            elif params < 100000:
                complexity = "Medium"
            else:
                complexity = "High"
            
            print(f"{model_name:<20} | {stats['total_params']:>12,} | {stats['trainable_params']:>12,} | {stats['memory_mb']:>12.1f} | {complexity} complexity")
    
    print("-" * 80)
    
    # Show recommendations
    print("\nRecommendations:")
    print("- SimpleAutoencoder: Fastest training, good for debugging")
    print("- EnhancedAutoencoder: Balanced performance and stability")
    print("- AutoencoderEnsemble: Highest accuracy, longer training")

def get_current_config() -> Dict[str, Any]:
    """Return comprehensive configuration with all constants and metadata.
    
    Returns:
        Nested dictionary containing all configuration parameters with metadata
    """
    return {
        "metadata": {
            "config_version": "1.2",
            "config_type": "autoencoder",
            "created": datetime.now().isoformat(),
            "system": {
                "python_version": platform.python_version(),
                "pytorch_version": torch.__version__,
                "cuda_available": torch.cuda.is_available()
            }
        },
        "training": {
            "batch_size": DEFAULT_BATCH_SIZE,
            "epochs": DEFAULT_EPOCHS,
            "learning_rate": LEARNING_RATE,
            "patience": EARLY_STOPPING_PATIENCE,
            "weight_decay": WEIGHT_DECAY,
            "gradient_clip": GRADIENT_CLIP,
            "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
            "mixed_precision": MIXED_PRECISION,
            "num_workers": NUM_WORKERS
        },
        "model": {
            "encoding_dim": DEFAULT_ENCODING_DIM,
            "hidden_dims": HIDDEN_LAYER_SIZES,
            "dropout_rates": DROPOUT_RATES,
            "activation": ACTIVATION,
            "activation_param": ACTIVATION_PARAM,
            "normalization": NORMALIZATION,
            "use_batch_norm": USE_BATCH_NORM,
            "use_layer_norm": USE_LAYER_NORM,
            "diversity_factor": DIVERSITY_FACTOR,
            "min_features": MIN_FEATURES,
            "num_models": NUM_MODELS,
            "model_types": list(MODEL_VARIANTS.keys()),
            "available_activations": ["relu", "leaky_relu", "gelu"],
            "available_normalizations": ["batch", "layer", None]
        },
        "security": {
            "percentile": DEFAULT_PERCENTILE,
            "attack_threshold": DEFAULT_ATTACK_THRESHOLD,
            "false_negative_cost": FALSE_NEGATIVE_COST,
            "enable_security_metrics": SECURITY_METRICS,
            "anomaly_threshold_strategy": "percentile"
        },
        "presets": {
            "available_presets": list(PRESET_CONFIGS.keys()),
            "current_preset": None,
            "preset_configs": {k: v["metadata"]["description"] 
                              for k, v in PRESET_CONFIGS.items() 
                              if "metadata" in v and "description" in v["metadata"]}
        },
        "data": {
            "normal_samples": NORMAL_SAMPLES,
            "attack_samples": ATTACK_SAMPLES,
            "features": FEATURES,
            "normalization_options": NORMALIZATION_OPTIONS,
            "anomaly_factor": ANOMALY_FACTOR,
            "random_state": RANDOM_STATE,
            "validation_split": 0.2,
            "test_split": 0.2
        },
        "hardware": {
            "recommended_gpu_memory": 8,
            "minimum_system_requirements": {
                "cpu_cores": 4,
                "ram_gb": 8,
                "disk_space": 10
            }
        }
    }

def deep_update(original: Dict[str, Any], update: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively update a dictionary."""
    for key, value in update.items():
        if isinstance(value, dict) and key in original:
            original[key] = deep_update(original[key], value)
        else:
            original[key] = value
    return original

def save_config(config: Dict, config_path: Path = CONFIG_FILE) -> None:
    """Save config with metadata."""
    try:
        full_config = {
            "metadata": {
                "created": datetime.now().isoformat(),
                "modified": datetime.now().isoformat(),
                "version": "1.0",
                "system": {
                    "python_version": platform.python_version(),
                    "hostname": platform.node(),
                    "os": platform.system()
                }
            },
            "config": config
        }
        
        # Create backup if exists
        if config_path.exists():
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = config_path.parent / f"{config_path.stem}_backup_{timestamp}{config_path.suffix}"
            shutil.copy(config_path, backup_path)
            logger.info(f"Backup created at {backup_path}")
        
        # Atomic write
        temp_path = config_path.with_suffix(".tmp")
        with open(temp_path, 'w') as f:
            json.dump(full_config, f, indent=4)
        temp_path.replace(config_path)
        logger.info(f"Config saved to {config_path}")

    except Exception as e:
        logger.error(f"Failed to save config: {e}")
        raise

def load_config(config_path: Path = CONFIG_FILE) -> Dict[str, Any]:
    """Load config file with error handling."""
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        logger.info(f"Loaded config from {config_path}")
        return config.get("config", {})
    except FileNotFoundError:
        logger.warning(f"Config file not found: {config_path}")
        return {}
    except json.JSONDecodeError as e:
        logger.error(f"Invalid config file: {str(e)}")
        return {}

def initialize_config(config_path: Path = CONFIG_FILE) -> Dict[str, Any]:
    """Initialize or load configuration."""
    loaded = load_config(config_path)
    if loaded:
        merged = deep_update(get_current_config(), loaded)
        save_config(merged, config_path)
        return merged
    else:
        default_config = get_current_config()
        save_config(default_config, config_path)
        return default_config

def update_global_config(config: Dict[str, Any]) -> None:
    """Update module-level constants from config with validation and logging.
    
    Args:
        config: Configuration dictionary to update from
        
    Raises:
        ValueError: If any configuration values are invalid
        TypeError: If any configuration values are of incorrect type
        KeyError: If required configuration sections are missing
    """
    # Validate config structure
    required_sections = ['training', 'model', 'security']
    for section in required_sections:
        if section not in config:
            raise KeyError(f"Missing required configuration section: {section}")
    
    # Initialize change tracking
    changes = {
        'training': {},
        'model': {},
        'security': {}
    }
    
    # Training configuration
    training = config.get("training", {})
    global DEFAULT_BATCH_SIZE, DEFAULT_EPOCHS, LEARNING_RATE, EARLY_STOPPING_PATIENCE
    global WEIGHT_DECAY, GRADIENT_CLIP, GRADIENT_ACCUMULATION_STEPS, MIXED_PRECISION
    
    # Validate and update training parameters
    try:
        if 'batch_size' in training:
            if not isinstance(training['batch_size'], int) or training['batch_size'] < 1:
                raise ValueError("batch_size must be a positive integer")
            if DEFAULT_BATCH_SIZE != training['batch_size']:
                changes['training']['batch_size'] = (DEFAULT_BATCH_SIZE, training['batch_size'])
                DEFAULT_BATCH_SIZE = training['batch_size']
        
        if 'epochs' in training:
            if not isinstance(training['epochs'], int) or training['epochs'] < 1:
                raise ValueError("epochs must be a positive integer")
            if DEFAULT_EPOCHS != training['epochs']:
                changes['training']['epochs'] = (DEFAULT_EPOCHS, training['epochs'])
                DEFAULT_EPOCHS = training['epochs']
        
        if 'learning_rate' in training:
            if not isinstance(training['learning_rate'], (int, float)) or training['learning_rate'] <= 0:
                raise ValueError("learning_rate must be a positive number")
            if LEARNING_RATE != training['learning_rate']:
                changes['training']['learning_rate'] = (LEARNING_RATE, training['learning_rate'])
                LEARNING_RATE = training['learning_rate']
        
        if 'patience' in training:
            if not isinstance(training['patience'], int) or training['patience'] < 1:
                raise ValueError("patience must be a positive integer")
            if EARLY_STOPPING_PATIENCE != training['patience']:
                changes['training']['patience'] = (EARLY_STOPPING_PATIENCE, training['patience'])
                EARLY_STOPPING_PATIENCE = training['patience']
        
        if 'weight_decay' in training:
            if not isinstance(training['weight_decay'], (int, float)) or training['weight_decay'] < 0:
                raise ValueError("weight_decay must be a non-negative number")
            if WEIGHT_DECAY != training['weight_decay']:
                changes['training']['weight_decay'] = (WEIGHT_DECAY, training['weight_decay'])
                WEIGHT_DECAY = training['weight_decay']
        
        if 'gradient_clip' in training:
            if not isinstance(training['gradient_clip'], (int, float)) or training['gradient_clip'] < 0:
                raise ValueError("gradient_clip must be a non-negative number")
            if GRADIENT_CLIP != training['gradient_clip']:
                changes['training']['gradient_clip'] = (GRADIENT_CLIP, training['gradient_clip'])
                GRADIENT_CLIP = training['gradient_clip']
        
        if 'gradient_accumulation_steps' in training:
            if not isinstance(training['gradient_accumulation_steps'], int) or training['gradient_accumulation_steps'] < 1:
                raise ValueError("gradient_accumulation_steps must be a positive integer")
            if GRADIENT_ACCUMULATION_STEPS != training['gradient_accumulation_steps']:
                changes['training']['gradient_accumulation_steps'] = (GRADIENT_ACCUMULATION_STEPS, training['gradient_accumulation_steps'])
                GRADIENT_ACCUMULATION_STEPS = training['gradient_accumulation_steps']
        
        if 'mixed_precision' in training:
            if not isinstance(training['mixed_precision'], bool):
                raise TypeError("mixed_precision must be a boolean")
            if MIXED_PRECISION != training['mixed_precision']:
                changes['training']['mixed_precision'] = (MIXED_PRECISION, training['mixed_precision'])
                MIXED_PRECISION = training['mixed_precision']
    
    except Exception as e:
        logger.error("Failed to update training configuration")
        raise ValueError(f"Training configuration error: {str(e)}") from e
    
    # Model architecture configuration
    model = config.get("model", {})
    global DEFAULT_ENCODING_DIM, HIDDEN_LAYER_SIZES, DROPOUT_RATES
    global ACTIVATION, ACTIVATION_PARAM, NORMALIZATION
    global USE_BATCH_NORM, USE_LAYER_NORM, DIVERSITY_FACTOR, MIN_FEATURES, NUM_MODELS
    
    try:
        if 'encoding_dim' in model:
            if not isinstance(model['encoding_dim'], int) or model['encoding_dim'] < 1:
                raise ValueError("encoding_dim must be a positive integer")
            if DEFAULT_ENCODING_DIM != model['encoding_dim']:
                changes['model']['encoding_dim'] = (DEFAULT_ENCODING_DIM, model['encoding_dim'])
                DEFAULT_ENCODING_DIM = model['encoding_dim']
        
        if 'hidden_dims' in model:
            if not isinstance(model['hidden_dims'], list) or not all(isinstance(x, int) and x > 0 for x in model['hidden_dims']):
                raise ValueError("hidden_dims must be a list of positive integers")
            if HIDDEN_LAYER_SIZES != model['hidden_dims']:
                changes['model']['hidden_dims'] = (HIDDEN_LAYER_SIZES, model['hidden_dims'])
                HIDDEN_LAYER_SIZES = model['hidden_dims']
        
        if 'dropout_rates' in model:
            if not isinstance(model['dropout_rates'], list) or not all(isinstance(x, (int, float)) and 0 <= x < 1 for x in model['dropout_rates']):
                raise ValueError("dropout_rates must be a list of numbers between 0 and 1")
            if DROPOUT_RATES != model['dropout_rates']:
                changes['model']['dropout_rates'] = (DROPOUT_RATES, model['dropout_rates'])
                DROPOUT_RATES = model['dropout_rates']
        
        if 'activation' in model:
            if model['activation'] not in ['relu', 'leaky_relu', 'gelu']:
                raise ValueError("activation must be one of: 'relu', 'leaky_relu', 'gelu'")
            if ACTIVATION != model['activation']:
                changes['model']['activation'] = (ACTIVATION, model['activation'])
                ACTIVATION = model['activation']
        
        if 'activation_param' in model:
            if not isinstance(model['activation_param'], (int, float)):
                raise ValueError("activation_param must be a number")
            if ACTIVATION_PARAM != model['activation_param']:
                changes['model']['activation_param'] = (ACTIVATION_PARAM, model['activation_param'])
                ACTIVATION_PARAM = model['activation_param']
        
        if 'normalization' in model:
            if model['normalization'] not in ['batch', 'layer', None]:
                raise ValueError("normalization must be one of: 'batch', 'layer', None")
            if NORMALIZATION != model['normalization']:
                changes['model']['normalization'] = (NORMALIZATION, model['normalization'])
                NORMALIZATION = model['normalization']
        
        if 'diversity_factor' in model:
            if not isinstance(model['diversity_factor'], (int, float)) or not 0 <= model['diversity_factor'] <= 1:
                raise ValueError("diversity_factor must be a number between 0 and 1")
            if DIVERSITY_FACTOR != model['diversity_factor']:
                changes['model']['diversity_factor'] = (DIVERSITY_FACTOR, model['diversity_factor'])
                DIVERSITY_FACTOR = model['diversity_factor']
        
        if 'min_features' in model:
            if not isinstance(model['min_features'], int) or model['min_features'] < 1:
                raise ValueError("min_features must be a positive integer")
            if MIN_FEATURES != model['min_features']:
                changes['model']['min_features'] = (MIN_FEATURES, model['min_features'])
                MIN_FEATURES = model['min_features']
        
        if 'num_models' in model:
            if not isinstance(model['num_models'], int) or model['num_models'] < 1:
                raise ValueError("num_models must be a positive integer")
            if NUM_MODELS != model['num_models']:
                changes['model']['num_models'] = (NUM_MODELS, model['num_models'])
                NUM_MODELS = model['num_models']
    
    except Exception as e:
        logger.error("Failed to update model configuration")
        raise ValueError(f"Model configuration error: {str(e)}") from e
    
    # Security configuration
    security = config.get("security", {})
    global DEFAULT_PERCENTILE, DEFAULT_ATTACK_THRESHOLD, FALSE_NEGATIVE_COST, SECURITY_METRICS
    
    try:
        if 'percentile' in security:
            if not isinstance(security['percentile'], (int, float)) or not 0 <= security['percentile'] <= 100:
                raise ValueError("percentile must be a number between 0 and 100")
            if DEFAULT_PERCENTILE != security['percentile']:
                changes['security']['percentile'] = (DEFAULT_PERCENTILE, security['percentile'])
                DEFAULT_PERCENTILE = security['percentile']
        
        if 'attack_threshold' in security:
            if not isinstance(security['attack_threshold'], (int, float)) or security['attack_threshold'] < 0:
                raise ValueError("attack_threshold must be a non-negative number")
            if DEFAULT_ATTACK_THRESHOLD != security['attack_threshold']:
                changes['security']['attack_threshold'] = (DEFAULT_ATTACK_THRESHOLD, security['attack_threshold'])
                DEFAULT_ATTACK_THRESHOLD = security['attack_threshold']
        
        if 'false_negative_cost' in security:
            if not isinstance(security['false_negative_cost'], (int, float)) or security['false_negative_cost'] < 0:
                raise ValueError("false_negative_cost must be a non-negative number")
            if FALSE_NEGATIVE_COST != security['false_negative_cost']:
                changes['security']['false_negative_cost'] = (FALSE_NEGATIVE_COST, security['false_negative_cost'])
                FALSE_NEGATIVE_COST = security['false_negative_cost']
        
        if 'enable_security_metrics' in security:
            if not isinstance(security['enable_security_metrics'], bool):
                raise TypeError("enable_security_metrics must be a boolean")
            if SECURITY_METRICS != security['enable_security_metrics']:
                changes['security']['enable_security_metrics'] = (SECURITY_METRICS, security['enable_security_metrics'])
                SECURITY_METRICS = security['enable_security_metrics']
    
    except Exception as e:
        logger.error("Failed to update security configuration")
        raise ValueError(f"Security configuration error: {str(e)}") from e
    
    # Handle preset application
    presets = config.get("presets", {})
    if "current_preset" in presets and presets["current_preset"] in PRESET_CONFIGS:
        logger.info(f"Applying preset configuration: {presets['current_preset']}")
        try:
            preset_config = PRESET_CONFIGS[presets["current_preset"]]
            update_global_config(preset_config)
        except Exception as e:
            logger.error(f"Failed to apply preset {presets['current_preset']}")
            raise ValueError(f"Preset configuration error: {str(e)}") from e
    
    # Log configuration changes
    if any(changes.values()):
        logger.info("Configuration changes applied:")
        for section, section_changes in changes.items():
            if section_changes:
                logger.info(f"  {section}:")
                for param, (old_val, new_val) in section_changes.items():
                    logger.info(f"    {param}: {old_val} -> {new_val}")
    else:
        logger.debug("No configuration changes detected")

def initialize_system() -> Dict[str, Any]:
    """Initialize the complete system with comprehensive setup."""
    # Setup logging (already done at module level)
    
    # Check hardware with detailed reporting
    hw_info = check_hardware()
    logger.info("\n[System Hardware Configuration]")
    for k, v in hw_info.items():
        logger.info(f"{k:>20}: {v}")
    
    # Initialize configuration system
    config = initialize_config()
    
    # Validate configuration
    try:
        validate_config(config)
    except ValueError as e:
        logger.error(f"Configuration validation failed: {e}")
        logger.info("Falling back to default configuration")
        config = get_current_config()
    
    # Update global constants from config
    update_global_config(config)
    
    # Initialize model variants with validation
    initialize_model_variants()
    if not MODEL_VARIANTS:
        logger.warning("No model variants found - check model class definitions")
    
    # Log available models and their status
    logger.info("\n[Available Model Variants]")
    comparison = compare_model_architectures()
    for model_name, stats in comparison.items():
        if 'error' in stats:
            logger.warning(f"{model_name:>20}: Failed to initialize - {stats['error']}")
        else:
            logger.info(f"{model_name:>20}: {stats['total_params']:,} parameters")
    
    # Return system information
    return {
        'system': {
            'hardware': hw_info,
            'python_version': sys.version,
            'pytorch_version': torch.__version__,
            'timestamp': datetime.now().isoformat()
        },
        'config': {
            'active_config': config,
            'available_presets': list(PRESET_CONFIGS.keys()),
            'model_variants': list(MODEL_VARIANTS.keys())
        },
        'status': {
            'config_loaded': bool(config),
            'models_initialized': bool(MODEL_VARIANTS),
            'gpu_available': hw_info.get('gpu_available', False)
        }
    }

def validate_config(config: Dict[str, Any]) -> None:
    """Validate configuration structure and values with enhanced error reporting."""
    required_sections = ['training', 'model', 'security']
    for section in required_sections:
        if section not in config:
            raise ValueError(f"Missing configuration section: {section}")
    
    # Validate training parameters
    training = config['training']
    if not isinstance(training.get('batch_size', 1), int) or training.get('batch_size', 1) < 1:
        raise ValueError("Invalid batch_size in training config")
    
    # Validate model architecture with detailed checking
    model = config['model']
    hidden_dims = model.get('hidden_dims', HIDDEN_LAYER_SIZES)
    dropout_rates = model.get('dropout_rates', DROPOUT_RATES)
    
    # Fix mismatched lengths by adjusting dropout_rates
    if len(hidden_dims) != len(dropout_rates):
        logger.warning(f"Mismatch: hidden_dims length ({len(hidden_dims)}) != dropout_rates length ({len(dropout_rates)})")
        
        if len(dropout_rates) < len(hidden_dims):
            # Extend dropout_rates to match hidden_dims
            last_dropout = dropout_rates[-1] if dropout_rates else 0.2
            while len(dropout_rates) < len(hidden_dims):
                dropout_rates.append(last_dropout * 0.9)  # Gradually decrease
            logger.info(f"Extended dropout_rates to: {dropout_rates}")
        else:
            # Truncate dropout_rates to match hidden_dims
            dropout_rates = dropout_rates[:len(hidden_dims)]
            logger.info(f"Truncated dropout_rates to: {dropout_rates}")
        
        # Update the config
        config['model']['dropout_rates'] = dropout_rates
    
    # Validate security parameters
    security = config['security']
    percentile = security.get('percentile', DEFAULT_PERCENTILE)
    if not 0 < percentile <= 100:
        raise ValueError(f"percentile must be between 0 and 100, got {percentile}")

def initialize_model_variants() -> None:
    """Initialize MODEL_VARIANTS dictionary with validation and error recovery."""
    global MODEL_VARIANTS
    
    # Clear existing variants
    MODEL_VARIANTS = {}
    
    # Ensure dropout_rates and hidden_dims are compatible
    compatible_hidden_dims = HIDDEN_LAYER_SIZES.copy()
    compatible_dropout_rates = DROPOUT_RATES.copy()
    
    # Fix length mismatch if it exists
    if len(compatible_hidden_dims) != len(compatible_dropout_rates):
        if len(compatible_dropout_rates) < len(compatible_hidden_dims):
            # Extend dropout_rates
            last_dropout = compatible_dropout_rates[-1] if compatible_dropout_rates else 0.2
            while len(compatible_dropout_rates) < len(compatible_hidden_dims):
                compatible_dropout_rates.append(max(0.1, last_dropout * 0.8))
        else:
            # Truncate dropout_rates
            compatible_dropout_rates = compatible_dropout_rates[:len(compatible_hidden_dims)]
        
        logger.info(f"Adjusted dropout_rates for compatibility: {compatible_dropout_rates}")
    
    # Define expected model classes and their initialization parameters
    model_definitions = {
        'SimpleAutoencoder': {
            'class': SimpleAutoencoder,
            'params': {
                'input_dim': 20,  # Test input size
                'encoding_dim': DEFAULT_ENCODING_DIM
            }
        },
        'EnhancedAutoencoder': {
            'class': EnhancedAutoencoder,
            'params': {
                'input_dim': 20,
                'encoding_dim': DEFAULT_ENCODING_DIM,
                'hidden_dims': compatible_hidden_dims,
                'dropout_rates': compatible_dropout_rates,
                'activation': ACTIVATION,
                'activation_param': ACTIVATION_PARAM,
                'normalization': NORMALIZATION
            }
        },
        'AutoencoderEnsemble': {
            'class': AutoencoderEnsemble,
            'params': {
                'input_dim': 20,
                'num_models': NUM_MODELS,
                'encoding_dim': DEFAULT_ENCODING_DIM,
                'diversity_factor': DIVERSITY_FACTOR
            }
        }
    }
    
    # Initialize each model variant with validation
    for name, definition in model_definitions.items():
        if definition['class'] is None:
            logger.debug(f"Model class not found: {name}")
            continue
        
        try:
            # Test instantiation
            model = definition['class'](**definition['params'])
            MODEL_VARIANTS[name] = definition['class']
            logger.debug(f"Successfully initialized model variant: {name}")
        except Exception as e:
            logger.warning(f"Failed to initialize model variant {name}: {str(e)}")
            logger.debug(f"Parameters used: {definition['params']}")

class SimpleAutoencoder(nn.Module):
    """Simple autoencoder with enhanced initialization, mixed precision support, and CPU/GPU awareness.
    
    Args:
        input_dim: Dimension of input features
        encoding_dim: Size of latent representation (default: DEFAULT_ENCODING_DIM)
        mixed_precision: Enable mixed precision training (auto-disabled for CPU) (default: MIXED_PRECISION)
        min_features: Minimum allowed input dimension (default: MIN_FEATURES)
    """
    def __init__(
        self,
        input_dim: int,
        encoding_dim: int = DEFAULT_ENCODING_DIM,
        mixed_precision: bool = MIXED_PRECISION,
        min_features: int = MIN_FEATURES
    ):
        super().__init__()
        
        # Input validation
        if input_dim < min_features:
            raise ValueError(f"Input dimension must be at least {min_features}")
        
        # Architecture definition
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, encoding_dim),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, input_dim),
            nn.Sigmoid()
        )
        
        # Mixed precision handling
        self._mixed_precision_requested = mixed_precision
        self.mixed_precision = mixed_precision and torch.cuda.is_available()
        
        # Logging configuration
        logger.debug(f"SimpleAutoencoder initialized with: "
                    f"input_dim={input_dim}, encoding_dim={encoding_dim}, "
                    f"mixed_precision={self.mixed_precision} (requested={mixed_precision})")
        
        self._initialize_weights()
    
    def _initialize_weights(self) -> None:
        """Initialize weights using Xavier/Glorot initialization with proper scaling."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=nn.init.calculate_gain('relu'))
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with automatic mixed precision support.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            
        Returns:
            Reconstructed tensor of same shape as input
        """
        with torch.cuda.amp.autocast(enabled=self.mixed_precision):
            encoded = self.encoder(x)
            decoded = self.decoder(encoded)
            return decoded
    
    @property
    def original_mixed_precision_setting(self) -> bool:
        """Returns the originally requested mixed precision setting."""
        return self._mixed_precision_requested
    
    def get_config(self) -> Dict[str, Any]:
        """Returns the configuration of the autoencoder."""
        return {
            "input_dim": self.encoder[0].in_features,
            "encoding_dim": self.encoder[0].out_features,
            "mixed_precision": self.mixed_precision,
            "min_features": MIN_FEATURES,
            "architecture": "simple",
            "initialized_with_cuda": torch.cuda.is_available()
        }

class EnhancedAutoencoder(nn.Module):
    """Enhanced autoencoder with configurable architecture, mixed precision support, and advanced features.
    
    Args:
        input_dim: Dimension of input features
        encoding_dim: Size of latent representation (default: DEFAULT_ENCODING_DIM)
        hidden_dims: List of hidden layer dimensions (default: HIDDEN_LAYER_SIZES)
        dropout_rates: Dropout rates for each layer (default: DROPOUT_RATES)
        activation: Activation function ('relu', 'leaky_relu', 'gelu') (default: ACTIVATION)
        activation_param: Parameter for activation (e.g., slope for LeakyReLU) (default: ACTIVATION_PARAM)
        normalization: Normalization type ('batch', 'layer', None) (default: NORMALIZATION)
        legacy_mode: Use simple architecture if True (default: False)
        skip_connection: Enable skip connection if True (default: True)
        min_features: Minimum allowed input dimension (default: MIN_FEATURES)
        mixed_precision: Enable mixed precision training (auto-disabled for CPU) (default: MIXED_PRECISION)
    """
    
    def __init__(
        self,
        input_dim: int,
        encoding_dim: int = DEFAULT_ENCODING_DIM,
        hidden_dims: List[int] = None,
        dropout_rates: Optional[List[float]] = None,
        activation: str = ACTIVATION,
        activation_param: float = ACTIVATION_PARAM,
        normalization: str = NORMALIZATION,
        legacy_mode: bool = False,
        skip_connection: bool = True,
        min_features: int = MIN_FEATURES,
        mixed_precision: bool = MIXED_PRECISION
    ):
        super().__init__()
        
        # Set defaults if None
        if hidden_dims is None:
            hidden_dims = HIDDEN_LAYER_SIZES.copy()
        if dropout_rates is None:
            dropout_rates = DROPOUT_RATES.copy()
            
        # Input validation
        if input_dim < min_features:
            raise ValueError(f"Input dimension must be at least {min_features}")
        
        # Fix length mismatch between hidden_dims and dropout_rates
        if len(hidden_dims) != len(dropout_rates):
            logger.warning(f"Length mismatch: hidden_dims({len(hidden_dims)}) vs dropout_rates({len(dropout_rates)})")
            
            if len(dropout_rates) < len(hidden_dims):
                # Extend dropout_rates
                last_dropout = dropout_rates[-1] if dropout_rates else 0.2
                while len(dropout_rates) < len(hidden_dims):
                    dropout_rates.append(max(0.1, last_dropout * 0.8))
                logger.info(f"Extended dropout_rates to: {dropout_rates}")
            else:
                # Truncate dropout_rates
                dropout_rates = dropout_rates[:len(hidden_dims)]
                logger.info(f"Truncated dropout_rates to: {dropout_rates}")

        # Store all parameters as instance attributes
        self.input_dim = input_dim
        self.encoding_dim = encoding_dim
        self.hidden_dims = hidden_dims
        self.dropout_rates = dropout_rates
        self.activation = activation
        self.activation_param = activation_param
        self.normalization = normalization
        self.legacy_mode = legacy_mode
        self.skip_connection = skip_connection
        self.min_features = min_features
        self._mixed_precision_requested = mixed_precision
        self.mixed_precision = mixed_precision and torch.cuda.is_available()
        
        # Logging configuration
        logger.debug(f"EnhancedAutoencoder initialized with: input_dim={input_dim}, "
                    f"encoding_dim={encoding_dim}, hidden_dims={hidden_dims}, "
                    f"dropout_rates={dropout_rates}, mixed_precision={self.mixed_precision}")

        if legacy_mode:
            # Simple architecture for backward compatibility
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, encoding_dim),
                nn.ReLU()
            )
            self.decoder = nn.Sequential(
                nn.Linear(encoding_dim, input_dim),
                nn.Sigmoid()
            )
            self.skip = None
        else:
            # Build encoder and decoder networks with proper dropout handling
            encoder_dims = hidden_dims + [encoding_dim]
            # Add dropout for encoding layer
            encoder_dropouts = dropout_rates + [0.1]
            
            decoder_dims = hidden_dims[::-1] + [input_dim]
            # Add dropout for final layer
            decoder_dropouts = dropout_rates[::-1] + [0.1]
            
            self.encoder = self._build_network(
                input_dim=input_dim,
                layer_dims=encoder_dims,
                dropout_rates=encoder_dropouts,
                activation=activation,
                activation_param=activation_param,
                normalization=normalization,
                final_activation="tanh"
            )
            
            self.decoder = self._build_network(
                input_dim=encoding_dim,
                layer_dims=decoder_dims,
                dropout_rates=decoder_dropouts,
                activation=activation,
                activation_param=activation_param,
                normalization=normalization,
                final_activation="sigmoid"
            )
            
            # Skip connection (only when dimensions match)
            self.skip = (
                nn.Linear(input_dim, input_dim)
                # More flexible condition
                if skip_connection and input_dim <= encoding_dim * 2
                else None
            )
        
        self._initialize_weights()

    def _build_network(
        self,
        input_dim: int,
        layer_dims: List[int],
        dropout_rates: List[float],
        activation: str,
        activation_param: float,
        normalization: str,
        final_activation: Optional[str] = None
    ) -> nn.Sequential:
        """Construct encoder/decoder networks with robust dropout handling."""
        layers = []
        prev_dim = input_dim
        
        # Ensure dropout_rates matches layer_dims length
        if len(dropout_rates) != len(layer_dims):
            if len(dropout_rates) < len(layer_dims):
                # Extend with last value
                last_dropout = dropout_rates[-1] if dropout_rates else 0.2
                dropout_rates = dropout_rates + [last_dropout] * (len(layer_dims) - len(dropout_rates))
            else:
                # Truncate
                dropout_rates = dropout_rates[:len(layer_dims)]
        
        for i, (h_dim, dropout) in enumerate(zip(layer_dims, dropout_rates)):
            # Linear layer
            layers.append(nn.Linear(prev_dim, h_dim))
            
            # Normalization (skip for final layer to avoid conflicts)
            if i < len(layer_dims) - 1:
                if normalization == "batch" and h_dim > 1:
                    layers.append(nn.BatchNorm1d(h_dim))
                elif normalization == "layer":
                    layers.append(nn.LayerNorm(h_dim))
            
            # Activation (skip for final layer if final_activation is specified)
            if i < len(layer_dims) - 1 or final_activation is None:
                if activation == "leaky_relu":
                    layers.append(nn.LeakyReLU(negative_slope=activation_param))
                elif activation == "gelu":
                    layers.append(nn.GELU())
                else:  # Default to ReLU
                    layers.append(nn.ReLU())
            
            # Dropout (skip for final layer to avoid output corruption)
            if dropout > 0 and i < len(layer_dims) - 1:
                layers.append(nn.Dropout(dropout))
            
            prev_dim = h_dim
        
        # Final activation if specified
        if final_activation == "tanh":
            layers.append(nn.Tanh())
        elif final_activation == "sigmoid":
            layers.append(nn.Sigmoid())
        
        return nn.Sequential(*layers)

    def _initialize_weights(self) -> None:
        """Initialize weights using appropriate methods based on activation."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if self.activation == "leaky_relu":
                    nn.init.kaiming_normal_(
                        m.weight, 
                        mode='fan_in', 
                        nonlinearity='leaky_relu',
                        a=self.activation_param
                    )
                else:
                    nn.init.xavier_normal_(m.weight, gain=nn.init.calculate_gain('relu'))
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm1d, nn.LayerNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with mixed precision and optional skip connection."""
        with torch.cuda.amp.autocast(enabled=self.mixed_precision):
            encoded = self.encoder(x)
            decoded = self.decoder(encoded)
            if self.skip is not None and not self.legacy_mode:
                decoded = decoded + self.skip(x)
            return decoded

    @property
    def original_mixed_precision_setting(self) -> bool:
        """Returns the originally requested mixed precision setting."""
        return self._mixed_precision_requested

    def get_config(self) -> Dict[str, Any]:
        """Returns the configuration of the autoencoder."""
        return {
            "input_dim": self.input_dim,
            "encoding_dim": self.encoding_dim,
            "hidden_dims": self.hidden_dims,
            "dropout_rates": self.dropout_rates,
            "activation": self.activation,
            "activation_param": self.activation_param,
            "normalization": self.normalization,
            "skip_connection": self.skip is not None,
            "mixed_precision": self.mixed_precision,
            "legacy_mode": self.legacy_mode,
            "min_features": self.min_features,
            "architecture": "enhanced"
        }

class AutoencoderEnsemble(nn.Module):
    """Ensemble of autoencoders with configurable diversity and mixed precision support.
    
    Args:
        input_dim: Dimension of input features
        num_models: Number of autoencoders in ensemble (default: NUM_MODELS)
        encoding_dim: Base size of latent representation (default: DEFAULT_ENCODING_DIM)
        diversity_factor: Scale factor for varying architectures (default: DIVERSITY_FACTOR)
        mixed_precision: Enable mixed precision training (auto-disabled for CPU) (default: MIXED_PRECISION)
        min_features: Minimum allowed input dimension (default: MIN_FEATURES)
    """
    def __init__(
        self,
        input_dim: int,
        num_models: int = NUM_MODELS,
        encoding_dim: int = DEFAULT_ENCODING_DIM,
        diversity_factor: float = DIVERSITY_FACTOR,
        mixed_precision: bool = MIXED_PRECISION,
        min_features: int = MIN_FEATURES
    ):
        super().__init__()
        # Input validation
        if input_dim < min_features:
            raise ValueError(f"Input dimension must be at least {min_features}")
        if num_models < 1:
            raise ValueError("Number of models must be at least 1")
        if not 0 <= diversity_factor <= 1:
            raise ValueError("Diversity factor must be between 0 and 1")

        # Mixed precision handling
        self._mixed_precision_requested = mixed_precision
        self.mixed_precision = mixed_precision and torch.cuda.is_available()
        
        # Initialize ensemble models with architectural diversity
        self.models = nn.ModuleList([
            EnhancedAutoencoder(
                input_dim=input_dim,
                encoding_dim=max(4, int(encoding_dim * (1 + (i - num_models//2) * diversity_factor))),
                hidden_dims=[
                    max(32, int(128 * (1 + (i - num_models//2) * diversity_factor * 0.5))),
                    max(16, int(64 * (1 + (i - num_models//2) * diversity_factor * 0.5)))
                ],
                dropout_rates=[0.2 + i*0.05, 0.15 + i*0.05],
                skip_connection=(i % 2 == 0),
                mixed_precision=self.mixed_precision,
                normalization=NORMALIZATION if i % 2 == 0 else None,
                activation=ACTIVATION,
                activation_param=ACTIVATION_PARAM if i % 2 == 0 else 0.1,
                legacy_mode=(i == 0 and num_models == 1),
                min_features=min_features
            )
            for i in range(num_models)
        ])
        
        # Log initialization details
        logger.debug(f"AutoencoderEnsemble initialized with: "
                    f"num_models={num_models}, encoding_dim={encoding_dim}, "
                    f"mixed_precision={self.mixed_precision} (requested={mixed_precision}), "
                    f"diversity_factor={diversity_factor}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with mixed precision support.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            
        Returns:
            Averaged reconstruction from all ensemble members
        """
        with torch.cuda.amp.autocast(enabled=self.mixed_precision):
            reconstructions = [model(x) for model in self.models]
            return torch.stack(reconstructions).mean(dim=0)

    @property
    def original_mixed_precision_setting(self) -> bool:
        """Returns the originally requested mixed precision setting."""
        return self._mixed_precision_requested

    def get_config(self) -> Dict[str, Any]:
        """Returns the configuration of the ensemble."""
        return {
            "input_dim": self.models[0].encoder[0].in_features if not self.models[0].legacy_mode 
                         else self.models[0].encoder[0][0].in_features,
            "num_models": len(self.models),
            "encoding_dim": DEFAULT_ENCODING_DIM,
            "diversity_factor": DIVERSITY_FACTOR,
            "mixed_precision": self.mixed_precision,
            "min_features": MIN_FEATURES,
            "architecture": "ensemble",
            "model_types": [type(m).__name__ for m in self.models]
        }

def load_autoencoder_model(
    model_path: Path,
    input_dim: Optional[int] = None,
    encoding_dim: int = DEFAULT_ENCODING_DIM,
    config: Optional[Dict] = None
) -> Union[EnhancedAutoencoder, AutoencoderEnsemble]:
    """Load autoencoder with automatic architecture detection and config handling.
    
    Args:
        model_path: Path to the saved model file
        input_dim: Expected input dimension (optional, will be inferred if None)
        encoding_dim: Default encoding dimension if not found in state_dict
        config: Optional configuration dictionary for model parameters
        
    Returns:
        Loaded model instance (EnhancedAutoencoder or AutoencoderEnsemble)
        
    Raises:
        FileNotFoundError: If model file doesn't exist
        RuntimeError: If model loading fails
        ValueError: If architecture parameters are invalid
    """
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    # Load the state dict to inspect its structure
    state_dict = torch.load(model_path, map_location='cpu')
    
    # Check if this is an ensemble model
    is_ensemble = any(k.startswith('models.') for k in state_dict.keys())
    legacy_keys = ['encoder.0.weight', 'encoder.0.bias', 'decoder.0.weight', 'decoder.0.bias']
    is_legacy = all(key in state_dict for key in legacy_keys)
    
    # Try to extract model parameters from state_dict if not provided
    if input_dim is None:
        if is_legacy:
            input_dim = state_dict['encoder.0.weight'].shape[1]
        else:
            # Look for first encoder layer weight in any model architecture
            for k in state_dict:
                if 'encoder.net.0.weight' in k or 'encoder.0.weight' in k or 'models.0.encoder.net.0.weight' in k:
                    input_dim = state_dict[k].shape[1]
                    break
            if input_dim is None:
                raise ValueError("Could not infer input_dim from state_dict")
    
    # Initialize default config if not provided
    if config is None:
        config = {
            'hidden_dims': HIDDEN_LAYER_SIZES,
            'dropout_rates': DROPOUT_RATES,
            'activation': ACTIVATION,
            'activation_param': ACTIVATION_PARAM,
            'normalization': NORMALIZATION,
            'skip_connection': True,
            'min_features': MIN_FEATURES,
            'num_models': NUM_MODELS,
            'diversity_factor': DIVERSITY_FACTOR
        }
    
    if is_ensemble:
        logger.info("Loading AutoencoderEnsemble model")
        # Try to get encoding_dim from state_dict if available
        for k in state_dict:
            if 'models.0.decoder.net.0.weight' in k:
                encoding_dim = state_dict[k].shape[1]
                break
        
        model = AutoencoderEnsemble(
            input_dim=input_dim,
            num_models=config.get('num_models', NUM_MODELS),
            encoding_dim=encoding_dim,
            diversity_factor=config.get('diversity_factor', DIVERSITY_FACTOR)
        )
    elif is_legacy:
        # Extract dimensions from the saved model
        encoding_dim = state_dict['encoder.0.weight'].shape[0]
        logger.info(f"Loading legacy autoencoder: input_dim={input_dim}, encoding_dim={encoding_dim}")
        model = EnhancedAutoencoder(
            input_dim=input_dim,
            encoding_dim=encoding_dim,
            legacy_mode=True
        )
    else:
        # Try to get encoding_dim from state_dict if available
        for k in state_dict:
            if 'decoder.net.0.weight' in k:
                encoding_dim = state_dict[k].shape[1]
                break
        
        logger.info(f"Loading enhanced autoencoder: input_dim={input_dim}, encoding_dim={encoding_dim}")
        model = EnhancedAutoencoder(
            input_dim=input_dim,
            encoding_dim=encoding_dim,
            hidden_dims=config.get('hidden_dims', HIDDEN_LAYER_SIZES),
            dropout_rates=config.get('dropout_rates', DROPOUT_RATES),
            activation=config.get('activation', ACTIVATION),
            activation_param=config.get('activation_param', ACTIVATION_PARAM),
            normalization=config.get('normalization', NORMALIZATION),
            skip_connection=config.get('skip_connection', True),
            min_features=config.get('min_features', MIN_FEATURES),
            legacy_mode=False
        )
    
    try:
        # Handle potential key mismatches
        model_state_dict = model.state_dict()
        
        # Filter out unexpected keys and keep only matching ones
        filtered_state_dict = {
            k: v for k, v in state_dict.items()
            if k in model_state_dict and v.shape == model_state_dict[k].shape
        }
        
        # Load the filtered state dict
        model.load_state_dict(filtered_state_dict, strict=False)
        
        # Log any missing or unexpected keys
        missing_keys = [k for k in model_state_dict if k not in state_dict]
        unexpected_keys = [k for k in state_dict if k not in model_state_dict]
        
        if missing_keys:
            logger.warning(f"Missing keys in state_dict: {missing_keys}")
        if unexpected_keys:
            logger.warning(f"Unexpected keys in state_dict: {unexpected_keys}")
        
        logger.info("Successfully loaded autoencoder model")
        return model
    except Exception as e:
        logger.error(f"Failed to load model state_dict: {str(e)}")
        raise RuntimeError(f"Model loading failed: {str(e)}")

def check_hardware() -> Dict[str, Union[str, bool, int, float]]:
    """Check and report available hardware resources with validation."""
    try:
        info = {
            "pytorch_version": torch.__version__,
            "gpu_available": torch.cuda.is_available(),
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            "cpu_threads": torch.get_num_threads(),
            "system": platform.system(),
            "cpu_count": os.cpu_count() or 1,
            "python_version": platform.python_version(),
            "hostname": platform.node()
        }
        
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            info.update({
                "gpu_name": props.name,
                "cuda_capability": f"{props.major}.{props.minor}",
                "gpu_memory_gb": round(props.total_memory / (1024**3), 2),
                "cuda_version": torch.version.cuda
            })
        
        return info
    except Exception as e:
        logger.error(f"Hardware check failed: {str(e)}")
        return {
            "pytorch_version": "unknown",
            "gpu_available": False,
            "device": "cpu",
            "gpu_count": 0,
            "cpu_threads": 1,
            "system": "unknown",
            "cpu_count": 1
        }

def load_and_validate_data(
    data_path: Path = DEFAULT_MODEL_DIR / "preprocessed_dataset.csv",
    artifacts_path: Path = DEFAULT_MODEL_DIR / "preprocessing_artifacts.pkl"
) -> Dict[str, np.ndarray]:
    """Load and validate preprocessed data with comprehensive checks."""
    try:
        if not data_path.exists():
            raise FileNotFoundError(f"Data file not found: {data_path}")
        if not artifacts_path.exists():
            raise FileNotFoundError(f"Artifacts file not found: {artifacts_path}")
            
        logger.info(f"Loading data from {data_path} and artifacts from {artifacts_path}")
        
        df = pd.read_csv(data_path)
        artifacts = joblib.load(artifacts_path)
        
        # Validate data
        if "Label" not in df.columns:
            raise ValueError("Dataset missing 'Label' column")
            
        feature_names = artifacts.get("feature_names", [])
        if not feature_names:
            raise ValueError("No feature names found in artifacts")
            
        if len(feature_names) < MIN_FEATURES:
            raise ValueError(f"Too few features ({len(feature_names)}), need at least {MIN_FEATURES}")
            
        # Apply scaling if available
        scaler = artifacts.get("scaler")
        X = df[feature_names].values.astype(np.float32)
        if scaler:
            logger.info("Applying feature scaling from artifacts")
            X = scaler.transform(X)
        
        # Split data
        normal_mask = df["Label"] == 0
        X_normal = X[normal_mask]
        X_attack = X[~normal_mask]
        
        # Balance dataset
        min_samples = min(len(X_normal), len(X_attack))
        if min_samples == 0:
            raise ValueError("One class has zero samples")
            
        logger.info(f"Loaded dataset with {len(X_normal)} normal and {len(X_attack)} attack samples")
        logger.info(f"Using balanced subset with {min_samples} samples per class")
            
        return {
            "X_train": X_normal[:min_samples],
            "X_val": X_normal[min_samples:min_samples + min_samples//4],  # 25% for validation
            "X_test": X_attack[:min_samples],
            "feature_names": feature_names
        }
    except Exception as e:
        logger.error(f"Data loading failed: {str(e)}")
        raise

def generate_synthetic_data(
    normal_samples: int = NORMAL_SAMPLES,
    attack_samples: int = ATTACK_SAMPLES,
    features: int = FEATURES,
    anomaly_factor: float = ANOMALY_FACTOR,
    random_state: int = RANDOM_STATE
) -> Dict[str, np.ndarray]:
    """Generate realistic synthetic data with proper anomaly separation."""
    np.random.seed(random_state)
    logger.info(f"Generating synthetic data with {normal_samples} normal and {attack_samples} attack samples")
    
    # Generate normal data (clustered around 0.5)
    X_normal = np.random.normal(0.5, 0.1, (normal_samples, features))
    X_normal = np.clip(X_normal, 0.1, 0.9)
    
    # Generate anomalies (higher variance and shifted means)
    X_attack = np.random.normal(0.5, 0.3, (attack_samples//2, features)) * anomaly_factor
    X_attack = np.clip(X_attack, 0.1, 0.9)
    
    # Add some extreme anomalies
    X_extreme = np.random.uniform(0, 1, (attack_samples//2, features))
    X_extreme[:, ::2] *= anomaly_factor  # Make some features more extreme
    
    return {
        "X_train": X_normal,
        "X_val": X_normal[:normal_samples//5],  # 20% for validation
        "X_test": np.vstack([X_attack, X_extreme]),
        "feature_names": [f"feature_{i}" for i in range(features)]
    }

def create_dataloaders(
    data: Dict[str, np.ndarray],
    batch_size: int = DEFAULT_BATCH_SIZE,
    shuffle: bool = True,
    num_workers: Optional[int] = None
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create properly configured dataloaders with pinned memory."""
    if num_workers is None:
        num_workers = min(4, os.cpu_count() or 1)
    
    train_data = TensorDataset(torch.tensor(data["X_train"], dtype=torch.float32))
    val_data = TensorDataset(torch.tensor(data["X_val"], dtype=torch.float32))
    test_data = TensorDataset(torch.tensor(data["X_test"], dtype=torch.float32))
    
    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=True,
        num_workers=num_workers,
        persistent_workers=num_workers > 0
    )
    
    val_loader = DataLoader(
        val_data,
        batch_size=batch_size * 2,  # Larger batches for validation
        shuffle=False,
        pin_memory=True,
        num_workers=num_workers,
        persistent_workers=num_workers > 0
    )
    
    test_loader = DataLoader(
        test_data,
        batch_size=batch_size * 2,
        shuffle=False,
        pin_memory=True,
        num_workers=num_workers,
        persistent_workers=num_workers > 0
    )
    
    logger.info(f"Created dataloaders with batch sizes: train={batch_size}, val/test={batch_size*2}")
    return train_loader, val_loader, test_loader

def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    grad_clip: float = GRADIENT_CLIP,
    accumulation_steps: int = GRADIENT_ACCUMULATION_STEPS,
    mixed_precision: bool = MIXED_PRECISION
) -> float:
    """Train model for one epoch with gradient clipping and mixed precision."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    mixed_precision = mixed_precision and torch.cuda.is_available()
    #scaler = torch.cuda.amp.GradScaler(enabled=mixed_precision)
    
    try:
        optimizer.zero_grad()
        
        if mixed_precision:
            mixed_precision = True
            device = torch.device("cuda")
            scaler = torch.cuda.amp.GradScaler(enabled=mixed_precision)
        else:
            mixed_precision = False
            device = torch.device("cpu")
            scaler = GradScaler(enabled=mixed_precision)
        
        for i, batch in enumerate(loader):
            inputs = batch[0].to(device)
            
            with torch.cuda.amp.autocast(enabled=mixed_precision):
                outputs = model(inputs)
                loss = criterion(outputs, inputs) / accumulation_steps
            
            # Backpropagation with gradient accumulation
            if mixed_precision:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # Gradient accumulation and clipping
            if (i + 1) % accumulation_steps == 0:
                if mixed_precision and grad_clip > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    # Enhanced gradient clipping
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    optimizer.step()
                    optimizer.zero_grad()
            
            # Metrics
            total_loss += loss.item() * accumulation_steps
            predicted = torch.max(outputs.data, 1)[1]
            total += loss.item() * accumulation_steps
            correct += (predicted == inputs).sum().item()
            
            # Log progress every 10 batches
            if i % 10 == 0 and i > 0:
                logger.debug(f"Batch {i}/{len(loader)} - Loss: {loss.item() * accumulation_steps:.4f} - Accuracy: {correct / ((i + 1)) * loader.batch_size:.4f}")
        
        # Memory cleanup
        torch.cuda.empty_cache()
        
        avg_loss = total_loss / len(loader)
        accuracy = correct / (len(loader) * loader.batch_size)
        logger.debug(f"Epoch complete - Average Loss: {avg_loss:.4f}",
                    f"Accuracy: {accuracy:.4f}")
        return avg_loss, accuracy

    except Exception as e:
        logger.error(f"Training failed at batch {i}: {str(e)}")
        raise RuntimeError("Training epoch failed") from e

def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Tuple[float, np.ndarray]:
    """Validate model and return MSE values for threshold calculation."""
    model.eval()
    total_loss = 0.0
    all_mse = []
    
    with torch.no_grad():
        for batch in loader:
            inputs = batch[0].to(device)
            outputs = model(inputs)
            loss = criterion(outputs, inputs)
            total_loss += loss.item()
            
            # Calculate per-sample MSE
            mse = torch.mean((inputs - outputs)**2, dim=1).cpu().numpy()
            all_mse.extend(mse)
    
    avg_loss = total_loss / len(loader)
    logger.debug(f"Validation complete - Average Loss: {avg_loss:.4f}")
    return avg_loss, np.array(all_mse)

def calculate_threshold(
    model: nn.Module,
    loader: DataLoader,
    percentile: int = DEFAULT_PERCENTILE,
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
) -> float:
    """Calculate anomaly threshold based on reconstruction error."""
    model.eval()
    mse_values = []
    
    with torch.no_grad():
        for batch in loader:
            inputs = batch[0].to(device)
            outputs = model(inputs)
            mse = torch.mean((inputs - outputs)**2, dim=1).cpu().numpy()
            mse_values.extend(mse)
    
    if not mse_values:
        raise ValueError("No MSE values calculated for threshold")
    
    threshold = np.percentile(mse_values, percentile)
    logger.info(f"Calculated anomaly threshold (P{percentile}): {threshold:.6f}")
    return threshold

def export_to_onnx(
    model: nn.Module,
    input_dim: int,
    device: torch.device,
    model_dir: Path = DEFAULT_MODEL_DIR,
    opset_version: int = 14
) -> Path:
    """Export model to ONNX format with proper configuration."""
    model.eval()
    onnx_path = model_dir / "autoencoder_ids.onnx"
    logger.info(f"Exporting model to ONNX format at {onnx_path}")
    
    try:
        dummy_input = torch.randn(1, input_dim).to(device)
        
        torch.onnx.export(
            model,
            dummy_input,
            onnx_path,
            opset_version=opset_version,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            },
            do_constant_folding=True,
            export_params=True,
            verbose=False
        )
        
        # Verify ONNX model
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)
        
        # Additional validation
        ort_session = ort.InferenceSession(onnx_path)
        outputs = ort_session.run(None, {'input': dummy_input.cpu().numpy()})
        if outputs[0].shape[1] != input_dim:
            raise RuntimeError("ONNX model output dimension mismatch")
        
        logger.info(f"Successfully exported and validated ONNX model: {onnx_path}")
        return onnx_path
    except Exception as e:
        logger.error(f"ONNX export failed: {str(e)}")
        if onnx_path.exists():
            onnx_path.unlink()
        raise RuntimeError(f"ONNX export failed: {str(e)}")

def train_model(args: argparse.Namespace) -> Dict[str, float]:
    """Main training pipeline with comprehensive features and config integration."""
    # Initialize and validate configuration system
    config = initialize_config()
    try:
        validate_config(config)
    except ValueError as e:
        logger.error(f"Configuration validation failed: {e}")
        if hasattr(args, 'non_interactive') and args.non_interactive:
            config = get_current_config()
        elif hasattr(args, 'non_interactive'):
            if prompt_user("Continue with default config?", default=True):
                config = get_current_config()
            else:
                raise
        else:
            config = get_current_config()

    update_global_config(config)
    
    # Apply configuration with command line args taking precedence
    training_config = config.get('training', {})
    model_config = config.get('model', {})
    data_config = config.get('data', {})
    security_config = config.get('security', {})
    
    # Ensure all required attributes exist with proper defaults
    # Training parameters
    args.batch_size = getattr(args, 'batch_size', None) or training_config.get('batch_size', DEFAULT_BATCH_SIZE)
    args.epochs = getattr(args, 'epochs', None) or training_config.get('epochs', DEFAULT_EPOCHS)
    args.lr = getattr(args, 'lr', None) or training_config.get('learning_rate', LEARNING_RATE)
    args.patience = getattr(args, 'patience', None) or training_config.get('patience', EARLY_STOPPING_PATIENCE)
    args.weight_decay = getattr(args, 'weight_decay', None) or training_config.get('weight_decay', WEIGHT_DECAY)
    args.grad_clip = getattr(args, 'grad_clip', None) or training_config.get('gradient_clip', GRADIENT_CLIP)
    
    # Model architecture parameters
    args.encoding_dim = getattr(args, 'encoding_dim', None) or model_config.get('encoding_dim', DEFAULT_ENCODING_DIM)
    args.features = getattr(args, 'features', None) or data_config.get('features', FEATURES)
    args.hidden_dims = model_config.get('hidden_dims', HIDDEN_LAYER_SIZES)
    args.dropout_rates = model_config.get('dropout_rates', DROPOUT_RATES)
    args.activation = model_config.get('activation', ACTIVATION)
    args.activation_param = model_config.get('activation_param', ACTIVATION_PARAM)
    args.normalization = model_config.get('normalization', NORMALIZATION)
    
    # Data parameters - THIS IS THE KEY FIX
    args.normal_samples = getattr(args, 'normal_samples', None) or data_config.get('normal_samples', NORMAL_SAMPLES)
    args.attack_samples = getattr(args, 'attack_samples', None) or data_config.get('attack_samples', ATTACK_SAMPLES)
    args.percentile = getattr(args, 'percentile', None) or security_config.get('percentile', DEFAULT_PERCENTILE)
    
    # Other parameters
    args.use_real_data = getattr(args, 'use_real_data', False) or data_config.get('use_real_data', False)
    args.preset = getattr(args, 'preset', None)
    args.debug = getattr(args, 'debug', False)
    args.export_onnx = getattr(args, 'export_onnx', False)
    args.model_dir = getattr(args, 'model_dir', DEFAULT_MODEL_DIR)
    
    # Ensure model_dir is a Path object
    if not isinstance(args.model_dir, Path):
        args.model_dir = Path(args.model_dir)
    
    # Log the resolved parameters
    logger.info(f"Resolved parameters:")
    logger.info(f"  normal_samples: {args.normal_samples}")
    logger.info(f"  attack_samples: {args.attack_samples}")
    logger.info(f"  features: {args.features}")
    logger.info(f"  batch_size: {args.batch_size}")
    logger.info(f"  epochs: {args.epochs}")
    
    # Setup experiment tracking
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"run_{timestamp}"
    writer = SummaryWriter(log_dir=LOG_DIR / run_id)
    
    # Hardware setup
    hw = check_hardware()
    device = torch.device(hw["device"])
    
    logger.info("\n[System Configuration]")
    logger.info(f"Using device: {device}")
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"Configuration source: {'command line' if args.preset is None else f'preset: {args.preset}'}")
    
    try:
        # Data preparation with fallback handling
        logger.info("\n[Data Preparation]")
        if args.use_real_data:
            try:
                data = load_and_validate_data()
                logger.info("Using real preprocessed data")
                args.features = len(data["feature_names"])  # Update from actual data
            except Exception as e:
                logger.warning(f"Failed to load real data: {str(e)}")
                if args.non_interactive or prompt_user("Use synthetic data instead?", True):
                    data = generate_synthetic_data(
                        normal_samples=args.normal_samples,
                        attack_samples=args.attack_samples,
                        features=args.features
                    )
                    logger.info("Using synthetic data as fallback")
                else:
                    raise
        else:
            data = generate_synthetic_data(
                normal_samples=args.normal_samples,
                attack_samples=args.attack_samples,
                features=args.features
            )
            logger.info("Using synthetic data by request")
        
        # Create dataloaders with optimal settings
        train_loader, val_loader, test_loader = create_dataloaders(
            data,
            batch_size=args.batch_size,
            shuffle=not args.debug  # Disable shuffling in debug mode for reproducibility
        )
        
        # Model initialization with config validation
        logger.info("\n[Model Initialization]")
        model_class = MODEL_VARIANTS.get(model_config.get('model_type', 'EnhancedAutoencoder'), EnhancedAutoencoder)
        logger.info(f"Using model class: {model_class.__name__}")
        
        model = model_class(
            input_dim=args.features,
            encoding_dim=args.encoding_dim,
            hidden_dims=args.hidden_dims,
            dropout_rates=args.dropout_rates,
            activation=args.activation,
            activation_param=args.activation_param,
            normalization=args.normalization
        ).to(device)
        
        # Log model architecture
        logger.info(f"Model architecture:\n{model}")
        total_params = sum(p.numel() for p in model.parameters())
        logger.info(f"Total parameters: {total_params:,}")
        
        # Optimizer and scheduler setup
        optimizer = optim.AdamW(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay
        )
        
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            patience=training_config.get('lr_patience', 2),
            factor=training_config.get('lr_factor', 0.5)
        )
        
        criterion = nn.MSELoss()
        
        # Training loop with comprehensive logging
        logger.info("\n[Training Started]")
        logger.info(f"Training for {args.epochs} epochs with batch size {args.batch_size}")
        logger.info(f"Initial learning rate: {args.lr:.2e}")
        
        best_val_loss = float('inf')
        patience_counter = 0
        metrics = {
            'train_loss': [],
            'val_loss': [],
            'learning_rate': []
        }
        
        for epoch in range(args.epochs):
            # Training phase
            model.train()
            epoch_train_loss = 0.0
            
            for batch_idx, batch in enumerate(train_loader):
                inputs = batch[0].to(device)
                optimizer.zero_grad()
                
                outputs = model(inputs)
                loss = criterion(outputs, inputs)
                loss.backward()
                
                # Gradient clipping
                if args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                
                optimizer.step()
                epoch_train_loss += loss.item()
                
                # Progress reporting
                if batch_idx % 10 == 0:
                    logger.debug(f"Epoch {epoch+1} | Batch {batch_idx} | Loss: {loss.item():.4f}")
            
            # Validation phase
            model.eval()
            epoch_val_loss = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    inputs = batch[0].to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, inputs)
                    epoch_val_loss += loss.item()
            
            # Calculate epoch metrics
            epoch_train_loss /= len(train_loader)
            epoch_val_loss /= len(val_loader)
            current_lr = optimizer.param_groups[0]['lr']
            
            # Update metrics tracking
            metrics['train_loss'].append(epoch_train_loss)
            metrics['val_loss'].append(epoch_val_loss)
            metrics['learning_rate'].append(current_lr)
            
            # Learning rate scheduling
            scheduler.step(epoch_val_loss)
            
            # Logging
            writer.add_scalar("Loss/train", epoch_train_loss, epoch)
            writer.add_scalar("Loss/val", epoch_val_loss, epoch)
            writer.add_scalar("Learning Rate", current_lr, epoch)
            
            logger.info(
                f"Epoch {epoch+1}/{args.epochs} | "
                f"Train Loss: {epoch_train_loss:.4f} | "
                f"Val Loss: {epoch_val_loss:.4f} | "
                f"LR: {current_lr:.2e}"
            )
            
            # Early stopping and model checkpointing
            if epoch_val_loss < best_val_loss:
                best_val_loss = epoch_val_loss
                patience_counter = 0
                torch.save(model.state_dict(), args.model_dir / "best_model.pth")
                logger.debug("New best model saved")
            else:
                patience_counter += 1
                if patience_counter >= args.patience:
                    logger.info(f"Early stopping triggered at epoch {epoch+1}")
                    break
        
        # Load best model for final evaluation
        model.load_state_dict(torch.load(args.model_dir / "best_model.pth"))
        
        # Calculate anomaly threshold
        threshold = calculate_threshold(
            model,
            val_loader,
            args.percentile,
            device
        )
        logger.info(f"\nAnomaly threshold (P{args.percentile}): {threshold:.6f}")
        
        # Final evaluation on test set
        test_loss, test_mse = validate(model, test_loader, criterion, device)
        anomaly_rate = (test_mse > threshold).mean()
        
        logger.info("\n[Final Evaluation]")
        logger.info(f"Test Loss: {test_loss:.4f}")
        logger.info(f"Anomaly Detection Rate: {anomaly_rate:.2%}")
        
        # Save all artifacts
        args.model_dir.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), args.model_dir / "autoencoder_ids.pth")
        joblib.dump(threshold, args.model_dir / "anomaly_threshold.pkl")
        
        # Export to ONNX if requested
        if args.export_onnx:
            export_to_onnx(model, args.features, args.model_dir, device)
        
        # Prepare comprehensive training metadata
        training_meta = {
            "config": {
                "training": {
                    "batch_size": args.batch_size,
                    "epochs": args.epochs,
                    "learning_rate": args.lr,
                    "weight_decay": args.weight_decay,
                    "patience": args.patience,
                    "gradient_clip": args.grad_clip,
                    "actual_epochs_trained": epoch + 1
                },
                "model": {
                    "type": model_class.__name__,
                    "encoding_dim": args.encoding_dim,
                    "hidden_dims": args.hidden_dims,
                    "dropout_rates": args.dropout_rates,
                    "activation": args.activation,
                    "activation_param": args.activation_param,
                    "normalization": args.normalization,
                    "total_parameters": total_params
                },
                "data": {
                    "features": args.features,
                    "normal_samples": len(data["X_train"]),
                    "attack_samples": len(data["X_test"]),
                    "source": "real" if args.use_real_data else "synthetic"
                },
                "security": {
                    "percentile": args.percentile,
                    "threshold": float(threshold)
                }
            },
            "results": {
                "best_val_loss": float(best_val_loss),
                "test_loss": float(test_loss),
                "anomaly_detection_rate": float(anomaly_rate),
                "training_metrics": metrics
            },
            "system": {
                "hardware": hw,
                "software": {
                    "python_version": sys.version,
                    "pytorch_version": torch.__version__
                },
                "timestamp": timestamp,
                "duration_seconds": (datetime.now() - datetime.fromisoformat(timestamp)).total_seconds()
            }
        }
        
        # Save metadata
        with open(args.model_dir / "training_metadata.json", "w") as f:
            json.dump(training_meta, f, indent=2)
        
        logger.info("\n[Training Complete]")
        logger.info(f"Model saved to: {args.model_dir / 'autoencoder_ids.pth'}")
        logger.info(f"Threshold saved to: {args.model_dir / 'anomaly_threshold.pkl'}")
        logger.info(f"Metadata saved to: {args.model_dir / 'training_metadata.json'}")
        
        return training_meta['results']
        
    except Exception as e:
        logger.error(f"\n[Training Failed] Error: {str(e)}")
        raise
    finally:
        writer.close()
        if 'model' in locals():
            del model  # Clean up GPU memory
        torch.cuda.empty_cache()

def hyperparameter_search(trial: optuna.Trial, base_args: argparse.Namespace) -> float:
    """Enhanced Optuna hyperparameter optimization with config integration."""
    # Define search space with conditional parameters
    params = {
        "encoding_dim": trial.suggest_int("encoding_dim", 4, min(64, getattr(base_args, 'features', 20)//2)),
        "lr": trial.suggest_float("lr", 1e-5, 1e-2, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128, 256]),
        "percentile": trial.suggest_int("percentile", 90, 99),
        "dropout_rate": trial.suggest_float("dropout_base", 0.1, 0.5),
        "num_hidden_layers": trial.suggest_int("num_hidden_layers", 1, 3)
    }
    
    # Generate architecture parameters
    params["dropout_rates"] = [
        params["dropout_rate"] * (1 - 0.1 * i) 
        for i in range(params["num_hidden_layers"])
    ]
    
    params["hidden_dims"] = [
        max(32, int(128 * (0.5 ** i)))
        for i in range(params["num_hidden_layers"])
    ]

    # Create and use trial-specific configuration
    trial_config = {
        "training": {
            "batch_size": params["batch_size"],
            "learning_rate": params["lr"],
            "patience": 3,
            "weight_decay": getattr(base_args, 'weight_decay', WEIGHT_DECAY),
            "gradient_clip": getattr(base_args, 'grad_clip', GRADIENT_CLIP)
        },
        "model": {
            "encoding_dim": params["encoding_dim"],
            "hidden_dims": params["hidden_dims"],
            "dropout_rates": params["dropout_rates"],
            "activation": getattr(base_args, 'activation', ACTIVATION),
            "activation_param": getattr(base_args, 'activation_param', ACTIVATION_PARAM),
            "normalization": getattr(base_args, 'normalization', NORMALIZATION)
        },
        "security": {
            "percentile": params["percentile"],
            "false_negative_cost": getattr(base_args, 'false_negative_cost', FALSE_NEGATIVE_COST)
        }
    }

    # Create trial args by combining base_args with trial parameters
    search_args = argparse.Namespace(**vars(base_args))
    for key, value in params.items():
        setattr(search_args, key, value)
    
    # Set trial-specific parameters
    search_args.model_dir = DEFAULT_MODEL_DIR / f"hpo_trial_{trial.number}"
    search_args.epochs = 30  # Shorter trials for HPO
    search_args.patience = 5
    search_args.non_interactive = True
    search_args.export_onnx = False

    try:
        # Save trial config for reference
        search_args.trial_config = trial_config
        
        # Run training with these parameters
        trial_result = train_model(search_args)
        
        # Load the validation metrics
        trial_meta_path = search_args.model_dir / "training_metadata.json"
        if trial_meta_path.exists():
            with open(trial_meta_path) as f:
                trial_meta = json.load(f)
            
            # Track additional metrics
            trial.set_user_attr("test_loss", trial_result["test_loss"])
            trial.set_user_attr("anomaly_rate", trial_result["anomaly_detection_rate"])
            trial.set_user_attr("num_params", trial_meta["config"]["model"]["total_parameters"])
            
            # Save trial config in metadata for reference
            trial.set_user_attr("config", trial_config)
        
        # Clean up trial directory if not in debug mode
        if not getattr(base_args, 'debug', False):
            for f in search_args.model_dir.glob("*"):
                try:
                    f.unlink()
                except Exception as e:
                    logger.warning(f"Could not delete {f}: {str(e)}")
            try:
                search_args.model_dir.rmdir()
            except Exception as e:
                logger.warning(f"Could not remove directory {search_args.model_dir}: {str(e)}")
            
        return trial_result["best_val_loss"]
        
    except Exception as e:
        logger.error(f"Trial {trial.number} failed: {str(e)}")
        trial.set_user_attr("error", str(e))
        return float('inf')

def setup_hyperparameter_optimization(args: argparse.Namespace) -> Dict[str, Any]:
    """Configure and run comprehensive hyperparameter optimization with enhanced logging and visualization.
    
    Args:
        args: Namespace object containing configuration parameters
        
    Returns:
        Dictionary containing the best parameters found during optimization
    """
    # Initialize configuration
    config = initialize_config()
    update_global_config(config)
    
    logger.info("\n" + "="*80)
    logger.info(f"Starting Hyperparameter Optimization ({args.hpo_trials} trials)")
    logger.info("="*80)
    
    # Setup study with enhanced configuration
    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(
            seed=42,
            consider_prior=True,
            prior_weight=1.0,
            consider_magic_clip=True,
            consider_endpoints=False,
            n_startup_trials=10,
            n_ei_candidates=24
        ),
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=5,
            n_warmup_steps=10,
            interval_steps=1
        ),
        study_name=f"autoencoder_hpo_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        load_if_exists=False
    )
    
    # Configure logging for Optuna
    optuna_logger = optuna.logging.get_logger("optuna")
    optuna_logger.setLevel(logging.INFO)
    optuna_logger.addHandler(logging.StreamHandler(sys.stdout))
    
    # Add custom callback for enhanced logging
    def log_callback(study: optuna.Study, frozen_trial: optuna.trial.FrozenTrial) -> None:
        logger.info(f"Trial {frozen_trial.number} finished with value: {frozen_trial.value:.4f}")
        logger.debug(f"Trial params: {frozen_trial.params}")
    
    # Run optimization
    try:
        study.optimize(
            lambda trial: hyperparameter_search(trial, args),
            n_trials=args.hpo_trials,
            timeout=args.hpo_timeout if args.hpo_timeout > 0 else None,
            gc_after_trial=True,
            callbacks=[log_callback],
            show_progress_bar=True
        )
        
        # Log and visualize results
        logger.info("\n" + "="*80)
        logger.info("Hyperparameter Optimization Complete")
        logger.info("="*80)
        
        # Report best trial
        best_trial = study.best_trial
        logger.info(f"\nBest trial (Value: {best_trial.value:.5f}):")
        for key, value in best_trial.params.items():
            logger.info(f"  {key:>20}: {value}")
        
        # Report important user attributes
        if best_trial.user_attrs:
            logger.info("\nAdditional metrics from best trial:")
            for key, value in best_trial.user_attrs.items():
                if key not in ["config", "params"]:  # These are logged separately
                    logger.info(f"  {key:>20}: {value}")
        
        # Save study for future reference
        study_path = DEFAULT_MODEL_DIR / "hpo_studies" / f"study_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
        study_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(study, study_path)
        logger.info(f"\nStudy saved to: {study_path}")
        
        # Visualization
        try:
            if hasattr(vis, "plot_optimization_history"):
                fig = vis.plot_optimization_history(study)
                fig.write_html(REPORTS_DIR / "hpo_optimization_history.html")
                logger.info("Saved optimization history plot")
            
            if hasattr(vis, "plot_parallel_coordinate"):
                fig = vis.plot_parallel_coordinate(study)
                fig.write_html(REPORTS_DIR / "hpo_parallel_coordinate.html")
                logger.info("Saved parallel coordinate plot")
            
            if hasattr(vis, "plot_param_importances"):
                fig = vis.plot_param_importances(study)
                fig.write_html(REPORTS_DIR / "hpo_param_importances.html")
                logger.info("Saved parameter importance plot")
        except Exception as e:
            logger.warning(f"Visualization failed: {str(e)}")
        
        # Return best parameters in a format ready for training
        best_params = {
            "training": {
                "batch_size": best_trial.params.get("batch_size", args.batch_size),
                "learning_rate": best_trial.params["lr"],
                "patience": args.patience  # Keep original patience for final training
            },
            "model": {
                "encoding_dim": best_trial.params["encoding_dim"],
                "hidden_dims": [
                    max(32, int(128 * (0.5 ** i)))
                    for i in range(best_trial.params["num_hidden_layers"])
                ],
                "dropout_rates": [
                    best_trial.params["dropout_base"] * (1 - 0.1 * i)
                    for i in range(best_trial.params["num_hidden_layers"])
                ],
                "percentile": best_trial.params["percentile"]
            }
        }
        
        # Update with any additional parameters from the trial
        if "config" in best_trial.user_attrs:
            best_params = deep_update(best_params, best_trial.user_attrs["config"])
        
        return best_params
        
    except Exception as e:
        logger.error(f"Hyperparameter optimization failed: {str(e)}")
        raise
    finally:
        # Clean up any resources
        optuna_logger.handlers.clear()

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

def main():
    """Main entry point with argument parsing and system orchestration."""
    # Initialize system first to set up logging and configuration
    system_info = initialize_system()
    #logger.info("System initialized with configuration:")
    #logger.info(json.dumps(system_info, indent=2))
    
    parser = argparse.ArgumentParser(
        description="Enhanced Anomaly Detection Model Training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Training configuration
    training_group = parser.add_argument_group('Training')
    training_group.add_argument(
        "--model-dir",
        type=Path,
        default=DEFAULT_MODEL_DIR,
        help="Directory to save model artifacts"
    )
    training_group.add_argument(
        "--epochs",
        type=int,
        help=f"Maximum number of training epochs (default: {DEFAULT_EPOCHS})"
    )
    training_group.add_argument(
        "--patience",
        type=int,
        help=f"Early stopping patience in epochs (default: {EARLY_STOPPING_PATIENCE})"
    )
    training_group.add_argument(
        "--batch-size",
        type=int,
        help=f"Training batch size (default: {DEFAULT_BATCH_SIZE})"
    )
    training_group.add_argument(
        "--lr",
        type=float,
        help=f"Learning rate (default: {LEARNING_RATE})"
    )
    training_group.add_argument(
        "--weight-decay",
        type=float,
        default=WEIGHT_DECAY,
        help=f"Weight decay for optimizer (default: {WEIGHT_DECAY})"
    )
    training_group.add_argument(
        "--grad-clip",
        type=float,
        default=GRADIENT_CLIP,
        help=f"Gradient clipping value (default: {GRADIENT_CLIP})"
    )
    
    # Model configuration
    model_group = parser.add_argument_group('Model')
    model_group.add_argument(
        "--features",
        type=int,
        help=f"Number of input features (default: {FEATURES})"
    )
    model_group.add_argument(
        "--encoding-dim",
        type=int,
        help=f"Encoder hidden dimension (default: {DEFAULT_ENCODING_DIM})"
    )
    model_group.add_argument(
        "--model-type",
        choices=list(MODEL_VARIANTS.keys()),
        default="EnhancedAutoencoder",
        help="Type of model architecture to use"
    )
    model_group.add_argument(
        "--num-models",
        type=int,
        default=NUM_MODELS,
        help=f"Number of models in ensemble (default: {NUM_MODELS})"
    )
    
    # Data configuration
    data_group = parser.add_argument_group('Data')
    data_group.add_argument(
        "--normal-samples",
        type=int,
        help=f"Normal training samples for synthetic data (default: {NORMAL_SAMPLES})"
    )
    data_group.add_argument(
        "--attack-samples",
        type=int,
        help=f"Anomalous test samples for synthetic data (default: {ATTACK_SAMPLES})"
    )
    data_group.add_argument(
        "--percentile",
        type=int,
        help=f"Percentile for anomaly threshold (default: {DEFAULT_PERCENTILE})"
    )
    data_group.add_argument(
        "--use-real-data",
        action="store_true",
        help="Use preprocessed data instead of synthetic"
    )
    data_group.add_argument(
        "--data-path",
        type=Path,
        default=DEFAULT_MODEL_DIR / "preprocessed_dataset.csv",
        help="Path to preprocessed data file"
    )
    data_group.add_argument(
        "--artifacts-path",
        type=Path,
        default=DEFAULT_MODEL_DIR / "preprocessing_artifacts.pkl",
        help="Path to preprocessing artifacts file"
    )
    
    # System configuration
    system_group = parser.add_argument_group('System')
    system_group.add_argument(
        "--export-onnx",
        action="store_true",
        help="Export model to ONNX format"
    )
    system_group.add_argument(
        "--non-interactive",
        action="store_true",
        help="Disable all interactive prompts"
    )
    system_group.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging"
    )
    system_group.add_argument(
        "--num-workers",
        type=int,
        default=min(4, os.cpu_count() or 1),
        help="Number of workers for data loading"
    )
    system_group.add_argument(
        "--mixed-precision",
        action="store_true",
        default=MIXED_PRECISION,
        help="Enable mixed precision training"
    )
    
    # Hyperparameter optimization
    hpo_group = parser.add_argument_group('HPO')
    hpo_group.add_argument(
        "--hpo",
        action="store_true",
        help="Enable hyperparameter optimization"
    )
    hpo_group.add_argument(
        "--hpo-trials",
        type=int,
        default=0,
        help="Number of hyperparameter optimization trials"
    )
    hpo_group.add_argument(
        "--hpo-timeout",
        type=int,
        default=3600,
        help="Timeout for hyperparameter optimization in seconds (0 for no timeout)"
    )
    
    # Configuration management
    config_group = parser.add_argument_group('Configuration')
    config_group.add_argument(
        "--show-config",
        action="store_true",
        help="Show current configuration"
    )
    config_group.add_argument(
        "--preset",
        choices=list(PRESET_CONFIGS.keys()),
        help="Use a preset configuration"
    )
    config_group.add_argument(
        "--save-config",
        action="store_true",
        help="Save current arguments as new config"
    )
    config_group.add_argument(
        "--compare-models",
        action="store_true",
        help="Compare available model architectures"
    )
    config_group.add_argument(
        "--reset-config",
        action="store_true",
        help="Reset configuration to defaults"
    )
    
    args = parser.parse_args()
    
    # Handle configuration commands first
    if args.reset_config:
        if args.non_interactive or prompt_user("Reset configuration to defaults?", default=False):
            default_config = get_current_config()
            save_config(default_config)
            logger.info("Configuration reset to defaults")
        return
    
    if args.show_config:
        print(json.dumps(get_current_config(), indent=2))
        return
        
    if args.compare_models:
        display_model_comparison()
        return
        
    # Apply preset configuration if specified
    if args.preset:
        if args.preset not in PRESET_CONFIGS:
            logger.error(f"Invalid preset: {args.preset}. Available presets: {list(PRESET_CONFIGS.keys())}")
            sys.exit(1)
            
        preset_config = PRESET_CONFIGS[args.preset]
        save_config(preset_config)
        logger.info(f"Loaded preset configuration: {args.preset}")
        
        # Update args from preset, preserving any explicitly provided command line args
        for group in ['training', 'model', 'data']:
            if group in preset_config:
                for key, value in preset_config[group].items():
                    if not hasattr(args, key) or getattr(args, key) is None:
                        setattr(args, key, value)
        
    # Configure logging level
    if args.debug:
        logger.setLevel(logging.DEBUG)
        logging.getLogger("torch").setLevel(logging.DEBUG)
        logging.getLogger("optuna").setLevel(logging.DEBUG)
    else:
        logging.getLogger("optuna").setLevel(logging.WARNING)
    
    # Save config if requested
    if args.save_config:
        current_config = get_current_config()
        
        # Map command line args to config structure
        arg_mapping = {
            'batch_size': 'training.batch_size',
            'epochs': 'training.epochs',
            'lr': 'training.learning_rate',
            'patience': 'training.patience',
            'weight_decay': 'training.weight_decay',
            'grad_clip': 'training.gradient_clip',
            'encoding_dim': 'model.encoding_dim',
            'model_type': 'model.model_type',
            'num_models': 'model.num_models',
            'features': 'data.features',
            'normal_samples': 'data.normal_samples',
            'attack_samples': 'data.attack_samples',
            'percentile': 'security.percentile',
            'mixed_precision': 'training.mixed_precision'
        }
        
        for arg_name, config_path in arg_mapping.items():
            arg_value = getattr(args, arg_name)
            if arg_value is not None:
                keys = config_path.split('.')
                target = current_config
                for key in keys[:-1]:
                    target = target.setdefault(key, {})
                target[keys[-1]] = arg_value
        
        # Mark the preset used if applicable
        if args.preset:
            current_config['presets'] = current_config.get('presets', {})
            current_config['presets']['current_preset'] = args.preset
        
        save_config(current_config)
        logger.info(f"Saved current configuration to {CONFIG_FILE}")
    
    # Run hyperparameter optimization if requested
    if args.hpo_trials > 0:
        logger.info(f"Starting hyperparameter optimization with {args.hpo_trials} trials")
        best_params = setup_hyperparameter_optimization(args)
        
        # Update args with best parameters
        for section, params in best_params.items():
            for key, value in params.items():
                setattr(args, key, value)
        
        logger.info("\nTraining final model with best parameters...")
        logger.info(json.dumps(best_params, indent=2))
    
    # Run training if not just doing config operations
    if not (args.show_config or args.compare_models or args.save_config or args.reset_config):
        try:
            # Ensure model directory exists
            args.model_dir.mkdir(parents=True, exist_ok=True)
            
            # Log all configuration parameters
            logger.info("\nFinal training configuration:")
            config_summary = {
                'training': {
                    'batch_size': args.batch_size,
                    'epochs': args.epochs,
                    'learning_rate': args.lr,
                    'patience': args.patience,
                    'weight_decay': args.weight_decay,
                    'grad_clip': args.grad_clip,
                    'mixed_precision': args.mixed_precision
                },
                'model': {
                    'model_type': args.model_type,
                    'encoding_dim': args.encoding_dim,
                    'num_models': args.num_models,
                    'features': args.features
                },
                'data': {
                    'use_real_data': args.use_real_data,
                    'normal_samples': args.normal_samples,
                    'attack_samples': args.attack_samples,
                    'percentile': args.percentile
                },
                'system': {
                    'num_workers': args.num_workers,
                    'export_onnx': args.export_onnx
                }
            }
            logger.info(json.dumps(config_summary, indent=2))
            
            train_model(args)
        except Exception as e:
            logger.error(f"Fatal error during training: {str(e)}", exc_info=args.debug)
            sys.exit(1)
        finally:
            # Clean up CUDA memory
            torch.cuda.empty_cache()

if __name__ == "__main__":
    # Configure warnings and logging before anything else
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
    
    # Ensure required directories exist
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    DEFAULT_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}", exc_info=True)
        sys.exit(1)