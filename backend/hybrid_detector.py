import numpy as np
import onnxruntime as ort
import torch
import joblib
from pathlib import Path
from typing import Optional
import logging
from deep_learning import EnhancedAutoencoder

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HybridDetector:
    def __init__(self):
        """Initialize hybrid detector with dynamic configuration."""
        self.models_loaded = False
        self.supervised_model = None
        self.autoencoder = None
        self.threshold = None
        self.feature_size = None
        
        try:
            self._load_models()
            self.models_loaded = True
        except Exception as e:
            logger.error(f"Failed to load models: {str(e)}")

    def _load_models(self) -> None:
        """Load all required models and artifacts with validation."""
        # Load preprocessing artifacts to get feature size
        artifacts_path = Path("models/preprocessing_artifacts.pkl")
        if not artifacts_path.exists():
            raise FileNotFoundError(f"Preprocessing artifacts not found at {artifacts_path}")
            
        artifacts = joblib.load(artifacts_path)
        self.feature_size = len(artifacts["feature_names"])
        
        # Load supervised model
        supervised_path = Path("models/ids_model.onnx")
        if not supervised_path.exists():
            raise FileNotFoundError(f"Supervised model not found at {supervised_path}")
            
        self.supervised_model = ort.InferenceSession(
            supervised_path,
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
        )
        
        # Validate supervised model input shape
        model_input_shape = self.supervised_model.get_inputs()[0].shape
        if len(model_input_shape) != 2 or model_input_shape[1] != self.feature_size:
            raise ValueError(
                f"Model expects input shape (batch, {model_input_shape[1]}), "
                f"but data has {self.feature_size} features"
            )
        
        # Load autoencoder
        autoencoder_path = Path("models/autoencoder_ids.pth")
        if not autoencoder_path.exists():
            raise FileNotFoundError(f"Autoencoder not found at {autoencoder_path}")
            
        self.autoencoder = Autoencoder(
            input_dim=self.feature_size,
            encoding_dim=max(10, self.feature_size // 2)  # Dynamic encoding size
        )
        self.autoencoder.load_state_dict(torch.load(autoencoder_path))
        self.autoencoder.eval()
        
        # Load threshold
        threshold_path = Path("models/anomaly_threshold.pkl")
        if not threshold_path.exists():
            raise FileNotFoundError(f"Threshold file not found at {threshold_path}")
            
        self.threshold = joblib.load(threshold_path)

    def validate_input(self, features: np.ndarray) -> None:
        """Validate input features before processing."""
        if not isinstance(features, np.ndarray):
            raise TypeError("Features must be a numpy array")
            
        if features.ndim != 1:
            raise ValueError("Features must be a 1D array")
            
        if len(features) != self.feature_size:
            raise ValueError(
                f"Expected {self.feature_size} features, got {len(features)}"
            )
            
        if np.isnan(features).any():
            raise ValueError("Input contains NaN values")

    def hybrid_detect(self, features: np.ndarray) -> str:
        """
        Combine supervised and unsupervised detection with robust error handling.
        
        Args:
            features: Input feature array of shape (feature_size,)
            
        Returns:
            One of: "Known Attack", "Unknown Anomaly", "Normal", or "Error"
        """
        if not self.models_loaded:
            return "Error: Models not loaded"
            
        try:
            self.validate_input(features)
            
            # Convert to float32 and normalize if needed
            features = features.astype(np.float32)
            
            # --- Supervised Detection ---
            supervised_input = features.reshape(1, -1)
            try:
                outputs = self.supervised_model.run(
                    None, 
                    {"input": supervised_input}
                )[0][0]
                
                if np.isnan(outputs).any():
                    logger.warning("Supervised model returned NaN outputs")
                    return "Error: Invalid model output"
                    
                prob_normal, prob_attack = outputs
                
                if prob_attack > 0.7:  # Configurable threshold
                    return "Known Attack"
                    
            except Exception as e:
                logger.error(f"Supervised detection failed: {str(e)}")
                # Fall through to unsupervised detection
                
            # --- Unsupervised Detection ---
            try:
                features_tensor = torch.from_numpy(features).float().unsqueeze(0)
                with torch.no_grad():
                    reconstruction = self.autoencoder(features_tensor)
                    mse = torch.mean((features_tensor - reconstruction) ** 2).item()
                    
                    if np.isnan(mse):
                        logger.warning("Autoencoder returned NaN MSE")
                        return "Error: Invalid autoencoder output"
                        
                    if mse > self.threshold * 1.1:  # 10% buffer zone
                        return "Unknown Anomaly"
                    else:
                        return "Normal"
                        
            except Exception as e:
                logger.error(f"Unsupervised detection failed: {str(e)}")
                return "Error: Detection failed"
                
        except Exception as e:
            logger.error(f"Detection error: {str(e)}")
            return f"Error: {str(e)}"

# Singleton instance for easy import
detector = HybridDetector()

def hybrid_detect(features: np.ndarray) -> str:
    """Legacy function wrapper for backward compatibility."""
    return detector.hybrid_detect(features)