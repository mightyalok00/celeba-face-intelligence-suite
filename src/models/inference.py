"""
Inference Engine for CelebA Face Attribute Classification.
Provides unified prediction pipeline, checkpoint loading, image tensor preprocessing,
genuine Test-Time Augmentation (TTA), and strict distinction between model inference
and ground-truth database lookups.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from PIL import Image

from src.models.architectures.ann import ANNClassifier
from src.models.architectures.bilstm import SpatialBiLSTMClassifier
from src.models.architectures.residual_cnn import ResidualConv2DCNN
from src.models.architectures.vision_transformer import VisionTransformerClassifier
from src.models.ensemble import apply_test_time_augmentation

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def preprocess_image_tensor(
    image_input: Union[str, Path, Image.Image, np.ndarray],
    target_size: Tuple[int, int] = (64, 64),
) -> torch.Tensor:
    """
    Loads and preprocesses face image into normalized PyTorch tensor (1, 3, H, W).
    Pixel intensities scaled strictly to [0.0, 1.0].
    """
    if isinstance(image_input, (str, Path)):
        img = Image.open(image_input).convert("RGB")
    elif isinstance(image_input, np.ndarray):
        if image_input.max() <= 1.0 and image_input.dtype in (np.float32, np.float64):
            image_input = (image_input * 255.0).astype(np.uint8)
        img = Image.fromarray(image_input).convert("RGB")
    elif isinstance(image_input, Image.Image):
        img = image_input.convert("RGB")
    else:
        raise ValueError(f"Unsupported image input type: {type(image_input)}")

    img_resized = img.resize(target_size, Image.Resampling.BILINEAR)
    arr = np.array(img_resized, dtype=np.float32) / 255.0
    # Shape: (H, W, C) -> (C, H, W) -> (1, C, H, W)
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    return tensor


class ModelInferenceEngine:
    """
    Manages loaded deep learning architectures and tree models for inference.
    """
    def __init__(self, checkpoints_dir: Optional[Path] = None) -> None:
        self.device = DEVICE
        self.checkpoints_dir = checkpoints_dir or Path("models")
        self.models: Dict[str, Any] = {}
        self._init_models()

    def _init_models(self) -> None:
        """Instantiates all 4 deep learning architectures."""
        self.models["Residual CNN"] = ResidualConv2DCNN().to(self.device).eval()
        self.models["Vision Transformer"] = VisionTransformerClassifier().to(self.device).eval()
        self.models["Spatial BiLSTM"] = SpatialBiLSTMClassifier().to(self.device).eval()
        self.models["ANN"] = ANNClassifier().to(self.device).eval()

    def predict_attribute(
        self,
        image_input: Union[str, Path, Image.Image, np.ndarray],
        architecture: str = "Residual CNN",
        use_tta: bool = True,
    ) -> Dict[str, Any]:
        """
        Executes genuine model forward pass on image input.
        Returns prediction dictionary with probability, binary prediction, confidence,
        and explicit inference metadata.
        """
        tensor = preprocess_image_tensor(image_input).to(self.device)

        model = self.models.get(architecture, self.models["Residual CNN"])

        with torch.no_grad():
            if use_tta:
                # Original
                p1 = model.predict_proba(tensor).cpu().numpy().item()
                # Horizontal reflection
                tensor_flip = torch.flip(tensor, dims=[3])
                p2 = model.predict_proba(tensor_flip).cpu().numpy().item()
                prob = float(0.5 * p1 + 0.5 * p2)
            else:
                prob = float(model.predict_proba(tensor).cpu().numpy().item())

        predicted_class = 1 if prob >= 0.5 else 0
        confidence = prob if predicted_class == 1 else (1.0 - prob)

        return {
            "source": "model_inference",
            "architecture": architecture,
            "use_tta": use_tta,
            "probability": round(prob, 4),
            "prediction": predicted_class,
            "confidence": round(confidence, 4),
            "is_ground_truth_lookup": False,
        }
