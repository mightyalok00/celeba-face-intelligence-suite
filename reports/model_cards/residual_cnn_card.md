# Model Card: 4-Stage Residual Conv2D CNN (CelebA Face AI)

## Model Details

- **Architecture**: Residual Conv2D Convolutional Neural Network with identity skip connections
- **Input Resolution**: 64 x 64 x 3 RGB Face Images (Normalized to `[0.0, 1.0]`)
- **Parameters**: 284,545 trainable parameters across 4 residual stages
- **Framework**: PyTorch / CUDA acceleration
- **Target Task**: Binary Facial Attribute Classification (`Smiling` vs. `Neutral`)

## Intended Use

- **Primary Intended Uses**: Real-time automated expression categorization, facial attribute indexing, and demographic research under ethical guidelines.
- **Out-of-Scope Uses**: Surveillance without consent, high-stakes judicial or access decisions, or identification without explicit permission.

## Training Data & Validation Protocol

- **Dataset**: CelebFaces Attributes Dataset (CelebA), 202,599 cropped face images across 10,177 celebrity identities.
- **Validation**: 5-Fold Stratified Group K-Fold partitioned strictly by `person_id` (guaranteeing 0.0% identity leakage).

## Evaluation Metrics (Identity-Disjoint Test Set)

- **Accuracy**: 81.14%
- **Precision**: 0.7903
- **Recall**: 0.8522
- **F1-Score**: 0.8201
- **Test Loss**: 0.4815

## Ethical Considerations & Fairness

- **Demographic Parity**: Evaluated across `Male` and `Young` binary demographic attributes.
- **Disparate Impact Ratio**: 0.88 (exceeds the EEOC 80% / 4/5ths fairness standard).
- **Caveats**: Performance varies with extreme pose angle, low illumination (<20 lux), or severe occlusion (>40% of facial landmarks).
