import numpy as np
import onnxruntime as ort
import torch
import joblib
from deep_learning import Autoencoder  # Import your Autoencoder class

# Load models
supervised_model = ort.InferenceSession("models/ids_model.onnx")
autoencoder = Autoencoder(input_dim=20, encoding_dim=10)  # Match dimensions from deep_learning.py
autoencoder.load_state_dict(torch.load("models/autoencoder_ids.pth"))
autoencoder.eval()
threshold = joblib.load("models/anomaly_threshold.pkl")

def hybrid_detect(features: np.ndarray) -> str:
    """Combine supervised and unsupervised detection."""
    # --- Supervised Detection ---
    supervised_input = features.astype(np.float32).reshape(1, -1)
    prob_normal, prob_attack = supervised_model.run(None, {"input": supervised_input})[0][0]
    
    if prob_attack > 0.7:  # Threshold for "known attack"
        return "Known Attack"
    
    # --- Unsupervised Detection ---
    features_tensor = torch.from_numpy(features).float().unsqueeze(0)
    reconstruction = autoencoder(features_tensor)
    mse = torch.mean((features_tensor - reconstruction) ** 2).item()
    
    if mse > threshold:
        return "Unknown Anomaly"
    else:
        return "Normal"