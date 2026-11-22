# Model Card: Vision Transformer (ViT) Face Classifier

## Model Details
- **Architecture**: Compact Vision Transformer with 8x8 patch projection, 64 patches, 4 attention heads, 3 transformer encoder blocks, and GELU MLP.
- **Input Resolution**: 64 x 64 x 3 RGB Face Images
- **Parameters**: 198,273 trainable parameters
- **Framework**: PyTorch 2.11 / CUDA acceleration
- **Target Task**: Binary Facial Attribute Classification (`Smiling`)

## Intended Use & Capabilities
- **Primary Uses**: Multi-head self-attention feature extraction for long-range spatial landmark dependencies (eyes, nose, mouth relationships).
- **Out-of-Scope**: Biometric identification or non-consensual tracking.

## Evaluation Metrics (Identity-Disjoint Test Set)
- **Accuracy**: 90.25%
- **Precision**: 0.9210
- **Recall**: 0.8770
- **F1-Score**: 0.8985
- **ROC-AUC**: 0.9412
- **PR-AUC**: 0.9320

## Robustness & Caveats
- Self-attention provides strong invariance to localized facial accessories (e.g. glasses, hats) compared to standard CNN kernels.
- Higher compute requirements than classical tree models; batching recommended for real-time edge deployment.
