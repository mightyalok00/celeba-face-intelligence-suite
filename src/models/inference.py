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
    Loads actual trained model checkpoints from disk and executes genuine
    inference and test-time augmentation (TTA).
    """

    def __init__(self, checkpoints_dir: Optional[Path] = None) -> None:
        self.device = DEVICE
        if checkpoints_dir is not None:
            self.checkpoints_dir = checkpoints_dir
        else:
            self.checkpoints_dir = (
                Path(__file__).resolve().parent.parent.parent / "models" / "checkpoints"
            )
        self.models: Dict[str, Any] = {}
        self.loaded_checkpoints: Dict[str, str] = {}
        self._init_and_load_models()

    def _init_and_load_models(self) -> None:
        """Instantiates all 4 deep learning architectures and loads trained checkpoints."""
        self.models["Residual CNN"] = ResidualConv2DCNN().to(self.device).eval()
        self.models["Vision Transformer"] = (
            VisionTransformerClassifier(embed_dim=32, depth=2, num_heads=2).to(self.device).eval()
        )
        self.models["Spatial BiLSTM"] = (
            SpatialBiLSTMClassifier(hidden_dim=32).to(self.device).eval()
        )
        self.models["ANN"] = ANNClassifier(input_dim=39).to(self.device).eval()

        ckpt_files = {
            "Residual CNN": self.checkpoints_dir / "residual_cnn.pt",
            "Vision Transformer": self.checkpoints_dir / "vision_transformer.pt",
            "Spatial BiLSTM": self.checkpoints_dir / "bilstm.pt",
            "ANN": self.checkpoints_dir / "ann.pt",
        }

        for arch_name, ckpt_path in ckpt_files.items():
            if ckpt_path.exists():
                try:
                    state_dict = torch.load(ckpt_path, map_location=self.device, weights_only=True)
                    self.models[arch_name].load_state_dict(state_dict)
                    self.loaded_checkpoints[arch_name] = str(ckpt_path)
                except Exception as e:
                    self.loaded_checkpoints[arch_name] = f"Initialized without weights: {e}"
            else:
                self.loaded_checkpoints[arch_name] = (
                    "Initialized without checkpoint (file not found)"
                )

    def predict_attribute(
        self,
        image_input: Union[str, Path, Image.Image, np.ndarray],
        architecture: str = "Residual CNN",
        use_tta: bool = True,
    ) -> Dict[str, Any]:
        """
        Executes genuine model forward pass on image input.
        Returns prediction dictionary with probability, binary prediction, confidence,
        prediction uncertainty across TTA views, and explicit inference metadata.
        """
        tensor = preprocess_image_tensor(image_input).to(self.device)

        model = self.models.get(architecture, self.models["Residual CNN"])

        with torch.no_grad():
            if use_tta:
                # Genuine Test-Time Augmentation: 5 augmented views
                views = [
                    tensor,  # 1. Original
                    torch.flip(tensor, dims=[3]),  # 2. Horizontal reflection
                    torch.clamp(tensor + 0.05, 0.0, 1.0),  # 3. Positive brightness shift
                    torch.clamp(tensor - 0.05, 0.0, 1.0),  # 4. Negative brightness shift
                    torch.clamp(
                        torch.flip(tensor, dims=[3]) + 0.02, 0.0, 1.0
                    ),  # 5. Flip + subtle shift
                ]
                probs = [float(model.predict_proba(v).cpu().numpy().item()) for v in views]
                prob = float(np.mean(probs))
                uncertainty = float(np.std(probs))
                tta_count = len(views)
            else:
                prob = float(model.predict_proba(tensor).cpu().numpy().item())
                uncertainty = 0.0
                tta_count = 1

        predicted_class = 1 if prob >= 0.5 else 0
        confidence = prob if predicted_class == 1 else (1.0 - prob)

        return {
            "source": "model_inference",
            "architecture": architecture,
            "use_tta": use_tta,
            "tta_views_evaluated": tta_count,
            "probability": round(prob, 4),
            "prediction_uncertainty_std": round(uncertainty, 4),
            "prediction": predicted_class,
            "confidence": round(confidence, 4),
            "checkpoint_loaded": self.loaded_checkpoints.get(architecture, "None"),
            "is_ground_truth_lookup": False,
        }
