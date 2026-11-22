# CelebA Face Attribute Classification System

An end-to-end computer vision and multimodal intelligence platform that classifies facial attributes on the **CelebA (CelebFaces Attributes)** dataset. The system integrates a **Google AI Studio Multimodal Reasoning App** (`google-generativeai` / Gemini 1.5 Flash), local PyTorch deep neural networks (**ANN, Deeper Residual Conv2D CNN, Spatial BiLSTM, Vision Transformer**), gradient boosted decision trees (**XGBoost**), **MySQL** relational database (`celeba_db`) with connection pooling, and **5-Fold Stratified Cross-Validation with 5-View Test-Time Augmentation (TTA)**.

## Validation status

GitHub Actions runs static analysis, security checks, tests and training smoke checks. A successful workflow does not establish production model accuracy. The committed benchmark is a historical experiment, not an independently reproduced result. Image availability, evaluation modality and dataset provenance must be established before interpreting metrics.

## Workspace Structure

```text

├── .github\                                  # GitHub configuration & automation
│   └── workflows\
│       ├── ci.yml                            # GitHub Actions CI suite (Lint, Types, Security, Pytest >= 85%)
│       └── model-training.yml                # Automated model training & benchmark validation
├── app\                                      # Google AI Studio Application & Services
│   ├── __init__.py                           # Application package marker
│   ├── app.py                                # Application interface & re-exports
│   ├── google_ai_studio_app.py               # Google AI Studio Multimodal Web App & Server
│   └── service.py                            # Database, gallery loader & MCP service layer
├── app.py                                    # Root launcher for Google AI Studio application
├── main.py                                   # Universal container entrypoint for Google AI Studio
├── config\                                   # Environment & framework configuration
│   ├── database.env.example                  # Environment variable configuration template
│   └── keras.json                            # Keras backend settings
│
├── data\                                     # Datasets & annotations
│   └── metadata\                             # Tabular datasets & mappings
│       ├── list_attr_celeba.csv              # 40 binary facial attribute annotations
│       ├── celebrities.csv                   # Reference list of 1,090 celebrity names & IDs
│       └── identity_CelebA.txt               # Ground-truth identity mappings for 202,599 faces
│
├── database\                                 # SQL scripts & database management
│   ├── celeba_db_setup.sql                   # Database setup, tables, foreign keys & security config
│   └── celeba_queries_and_analytics.sql      # Joins, duplicate operations & business intelligence
│
├── docs\                                     # Architectural & schema documentation
│   ├── ARCHITECTURE.md                       # Comprehensive end-to-end architecture guide
│   ├── AUDIT_AND_EVALUATION.md               # Audit score evaluation and verification report
│   └── EER_DIAGRAM.md                        # Enhanced Entity-Relationship (EER) diagram & schema
│
├── frontend\                                 # Google Photos web interface
│   ├── index.html                            # Interactive single-page web app with workflow modal
│   └── gallery_data.json                     # Precomputed thumbnails and metadata
│
├── img_align_celeba\                         # 202,599 cropped face images (.png/.jpg)
│
├── mcp\                                      # Model Context Protocol (MCP) servers
│   ├── auth_google_photos.mjs                # Google Photos OAuth helper
│   ├── google_photos_mcp_server.mjs          # Google Photos API MCP server (Stdio JSON-RPC 2.0)
│   └── mysql_mcp_server.mjs                  # MySQL database MCP server (Stdio JSON-RPC 2.0)
│
├── models\                                   # Architecture checkpoints & specifications
│   ├── checkpoints\                          # Serialized model weights (.pt)
│   ├── README.md                             # Model specifications and serialization formats
│   └── model_registry.json                   # Verified benchmark model registry
│
├── notebooks\                                # Interactive Jupyter Notebooks
│   ├── main.ipynb                            # Complete end-to-end classification pipeline
│   ├── train_classifiers.ipynb               # 5-fold cross-validation & benchmark suite
│   ├── face_embedding_and_clustering.ipynb   # Google Photos face embedding & clustering
│   ├── image_classifier_models.ipynb        # Step-by-step neural architectures (ResNet, ViT)
│   ├── pipeline_workflow_runner.ipynb        # Interactive pipeline execution companion notebook
│   └── project_verification.ipynb           # Automated health and validation suite
│
├── reports\                                  # Evaluation outputs & benchmark results
│   ├── benchmarks\                           # Isolated benchmark storage
│   │   ├── production_benchmark.csv          # Canonical full-training evaluated benchmark
│   │   └── ci_smoke_test_benchmark.csv       # Fast 200-sample CI verification benchmark
│   ├── figures\                              # Publication-quality charts (PNG, 300 DPI)
│   │   ├── model_benchmark_comparison.png    # Multi-metric comparison and log loss
│   │   ├── attribute_demographic_distributions.png # Demographic attribute distributions
│   │   └── confusion_matrix_and_roc.png      # Confusion matrix heatmap & ROC curve
│   ├── model_cards\                          # Formal machine learning model cards
│   │   ├── residual_cnn_card.md              # 4-Stage Residual Conv2D CNN model card
│   │   ├── vision_transformer_card.md        # Vision Transformer (ViT) model card
│   │   └── ensemble_card.md                  # SLSQP Bagged Ensemble + TTA model card
│   ├── training_logs\                        # Execution traces and telemetry
│   └── model_benchmark_results.csv           # Canonical public benchmark results
│
├── reproducibility\                          # Scientific reproducibility package
│   ├── config.yaml                           # Centralized experiment configuration
│   ├── requirements.lock                     # Exact pinned dependency lockfile
│   ├── dataset_manifest.json                 # Dataset file records and SHA256 checksums
│   ├── split_manifest.csv                    # Explicit identity-disjoint train/val/test split
│   ├── experiment_metadata.json              # Hardware, seed, and execution telemetry
│   └── README.md                             # One-command reproduction instructions
│
├── scripts\                                  # Operational utilities & automation
│   ├── pipeline_runner.py                    # Multi-stage functional verification runner
│   ├── run_pipeline.ps1                      # PowerShell automated pipeline execution script
│   ├── start_app.bat                         # Streamlit application launcher
│   ├── start_dev_server.bat                  # Full application Dev Server launcher
│   ├── start_google_ai_studio.bat            # Google AI Studio application launcher
│   └── start_mysql.bat                       # MySQL server daemon starter script
│
├── src\                                      # Modular functional Python packages
│   ├── __init__.py                           # Source package root marker
│   ├── ai_studio\                            # Google AI Studio Gemini multimodal analyzer
│   ├── database\
│   │   ├── __init__.py                       # Database package marker
│   │   └── connection.py                     # Thread-safe connection pool & parameterized SQL
│   ├── evaluation\                           # Metrics, identity leakage & fairness audit
│   │   ├── __init__.py                       # Evaluation package marker
│   │   ├── metrics.py                        # Comprehensive classification metrics
│   │   ├── leakage_check.py                  # Identity disjointness validation
│   │   └── fairness.py                       # Subgroup fairness & disparate impact
│   ├── models\
│   │   ├── __init__.py                       # Models package marker
│   │   ├── architectures\                    # Deep learning PyTorch models
│   │   │   ├── __init__.py                   # Architecture registry
│   │   │   ├── ann.py                        # Deep Multi-Layer Perceptron
│   │   │   ├── bilstm.py                     # Spatial BiDirectional LSTM
│   │   │   ├── residual_cnn.py               # 4-Stage Residual Conv2D CNN
│   │   │   └── vision_transformer.py         # Patch self-attention Vision Transformer
│   │   ├── classical_baselines.py            # Logistic Regression, ExtraTrees, XGBoost
│   │   ├── ensemble.py                       # SLSQP OOF ensembling & image TTA
│   │   ├── inference.py                      # Unified prediction engine
│   │   └── train_and_evaluate.py             # Multi-architecture evaluation pipeline
│   └── preprocessing\
│       ├── __init__.py                       # Preprocessing package marker
│       └── image_processor.ipynb             # Resizing, normalization & standardization
│
├── tests\                                    # Automated test suite (88% coverage)
│   ├── conftest.py                           # Pytest fixtures and mock isolation
│   ├── unit\                                 # Unit tests for architectures, pool, metrics
│   ├── integration\                          # Integration tests for MySQL, Streamlit, models
│   ├── security\                             # SQL injection & credential security tests
│   └── end_to_end\                           # End-to-end inference and fairness tests
│
├── .gitattributes                            # Git attributes configuration
├── .gitignore                                # Git ignore rules (excludes mysql_data & images)
├── Dockerfile                                # Multi-stage production container
├── LICENSE                                   # Project MIT license
├── package.json                              # Node.js dev server scripts & dependencies
├── pyproject.toml                            # Build configuration & tool settings
├── README.md                                 # Comprehensive project documentation
├── requirements.txt                          # Pinned dependency requirements
└── server.ts                                 # High-performance Dev Server & Express API
```

---

## Model Architectures & Advanced Ensembling

The project benchmarks individual architectures and combines the top models via cross-validation bagging and augmentation:

### 1. XGBoost (Extreme Gradient Boosted Trees)

* Constructs second-order gradient boosted ensembles over tabular features.
* Uses annotated attributes in the Python benchmark; this is not an image-only baseline.

### 2. Deeper Multi-Block Residual CNN with In-Model Augmentation

* **Real-Time Data Augmentation Layer**: Applies random horizontal flips, small rotations ($\pm 4\%$), and zooms ($\pm 5\%$) inside the GPU/CPU execution graph.
* **4 Hierarchical Blocks** ($32 \to 64 \to 128 \to 256$ filters) capturing edges, facial contours, organs, and smile geometry.
* **Batch Normalization** (`BatchNormalization`) after each convolution to eliminate internal covariate shift.
* **Residual Skip Connections** (`layers.Add()`) in each block to allow gradients to flow directly without vanishing.
* **Global Average Pooling** (`GlobalAveragePooling2D`) to condense feature representations before the classification head.

### 3. Vision Transformer (ViT - Patch-16 Embedding & Self-Attention)

* **Patch Embedding**: Partitions $64 \times 64 \times 3$ facial images into sixteen $16 \times 16$ non-overlapping patches, projected linearly into a latent embedding space with 1D learnable position embeddings.
* **Transformer Encoder Stack**: 6 stacked transformer blocks utilizing multi-head self-attention (8 attention heads) and GeLU MLP blocks with residual skip paths.
* **Global Facial Context**: Discovers long-range spatial correlations across eye, nose, and mouth landmarks simultaneously without 2D convolution constraints.

### 4. Spatial Recurrent Neural Network (Bidirectional LSTM)

* **Sequential Spatial Slicing**: Extracts horizontal row sequences across the face canvas ($64$ time steps of $192$ feature dimensions).
* **Bidirectional Sequence Modeling**: 2 stacked BiLSTM layers ($128$ hidden units) capturing progressive spatial transitions from forehead down to chin landmarks.

### 5. ANN (Artificial Neural Network / Multi-Layer Perceptron)

* Flattens the $64 \times 64 \times 3$ image into a 1D vector of 12,288 inputs.
* Two fully connected layers (128 and 64 neurons) with ReLU activations and Dropout (0.3).
* Serves as a baseline model without 2D spatial bias.

### 6. Identity-Disjoint Stratified GroupKFold (Preventing Data & Identity Leakage)

* **Identity Leakage Risk in Standard CV**: In CelebA, the same individual appears across multiple photos under different poses and lighting. Random image-level cross-validation accidentally splits images of the same identity into both train and validation folds, causing classifiers to memorize individual identities rather than learning generalizable attribute features.
* **GroupKFold by `person_id`**: Splits data such that **no person appears in both training and test sets simultaneously**:

$$\text{Identities}(\mathcal{D}_{\text{train}}^{(k)}) \cap \text{Identities}(\mathcal{D}_{\text{val}}^{(k)}) = \emptyset \quad \forall k \in \{1, \dots, 5\}$$

* **Test-Time Augmentation (TTA)**: Evaluates both original and horizontally flipped test images:

$$P_{\text{TTA}}(X) = \frac{1}{2} \left[ P(X) + P(\text{flip}_{\text{horizontal}}(X)) \right]$$

* **Outcome**: See the historical benchmark below; image-only and attribute-based results are not directly comparable.

### 7. Google Photos Engine: Deep Face Embeddings & Clustering

* **Dense Latent Embedding**: Projects $64 \times 64 \times 3$ faces through deep convolutional layers into an $L_2$-normalized $128$-D vector space where $\|e\|_2 = 1.0$.
* **Cosine Similarity Space**: Because vectors have unit length, cosine similarity reduces to dot product ($u \cdot v$), measuring facial identity alignment without requiring labels.
* **Unsupervised Clustering (DBSCAN & K-Means)**: Automatically groups photo albums by identity without supervision, isolating ambiguous or noisy faces.
* **2D Manifold Visualizations**: Compresses embeddings with t-SNE and PCA into 2D scatter plots to visualize identity separation.
* **Semantic Face Retrieval**: Given any query photo, ranks all images in the album by cosine similarity to retrieve matching faces of the same person.

---

## Preprocessing Pipeline (Normalization & Standardization)

Both scaling methods are implemented with strict training-set isolation to prevent data leakage:

### Normalization (Min-Max Scaling)

Scales raw pixel intensities from $[0, 255]$ into $[0.0, 1.0]$:

$$X_{\text{norm}} = \frac{X}{255.0}$$

### Standardization (Z-Score Scaling)

Centers and rescales pixels to have zero mean ($\mu \approx 0$) and unit variance ($\sigma \approx 1$):

$$Z = \frac{X - \mu_{\text{train}}}{\sigma_{\text{train}} + \epsilon}$$

* **Zero Data Leakage**: $\mu_{\text{train}}$ and $\sigma_{\text{train}}$ are computed strictly from the training set and applied to transform validation and test sets.

### Dataset Splitting

* Partitions data into **70% Training**, **15% Validation**, and **15% Testing**.
* Includes dynamic stratification guards that prevent errors on small or imbalanced classes.

---

## Training Configuration & Scaling

All primary notebooks are configured with enhanced scale and adaptive learning rates:

* **Sample Size**: `20,000` images (14,000 Train / 3,000 Validation / 3,000 Test).
* **Epochs**: `15` training epochs per deep model.
* **Batch Size**: `32`.
* **Adaptive Optimizer (`ReduceLROnPlateau`)**: Halves learning rate when validation loss stalls for 2 consecutive epochs.

```python
lr_scheduler = keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=2,
    min_lr=1e-6,
    verbose=1
)
```

---

## Database Integration (MySQL & CSV Fallback)

1. **Start the Database Service**: Run `scripts/start_mysql.bat` to launch the standalone MySQL Server daemon (or run `D:\MySQL\bin\mysqld.exe --datadir="mysql_data"` directly).
2. **Initialize Schema & Load Data**: Execute `database/celeba_db_setup.sql` in MySQL Workbench to create `celeba_db`, wire foreign keys, and generate `celeba_combined_dataset`. Run `database/celeba_queries_and_analytics.sql` for join operations, duplicate handling, and analytics.
3. **Automatic Fallback**: If the MySQL service is offline, all notebooks automatically read directly from `data/metadata/list_attr_celeba.csv` and `img_align_celeba/`, ensuring uninterrupted execution.

---

## Requirements & Prerequisites

### 1. Environment & Python Runtime

* **Python Version**: Python 3.10 to 3.14 (64-bit).
* **Virtual Environment**: Recommended (conda or venv).
* **Package Installation**: Install all dependencies via [`requirements.txt`](requirements.txt):

```bash
pip install -r requirements.txt
```

### 2. Core Library Dependencies

| Library | Version Tested | Purpose |
| :--- | :---: | :--- |
| **TensorFlow** | `>=2.15.0` | Deep neural network execution graph and augmentation layers |
| **Keras** | `>=3.0.0` | High-level API for Residual CNN, ANN, RNN, and ViT models |
| **XGBoost** | `>=2.0.0` | Second-order gradient boosted decision trees and ensembling |
| **Scikit-Learn** | `>=1.4.0` | 5-fold cross-validation, DBSCAN, K-Means, PCA, t-SNE, metrics |
| **SciPy** | `>=1.12.0` | Cosine spatial distance metrics and scientific statistical routines |
| **Pandas** | `>=2.2.0` | Relational tabular data manipulation and CSV processing |
| **NumPy** | `>=1.26.0` | High-performance vectorized numerical operations and arrays |
| **Pillow (PIL)** | `>=10.2.0` | Image reading, decoding, resizing, and pixel array extraction |
| **Matplotlib** | `>=3.8.0` | 2D scatter visualizations, metric plots, and face grid displays |
| **PyMySQL** | `>=1.1.0` | Pure-Python client connecting to the local MySQL server |

### 3. Hardware & Storage Specifications

* **RAM**: 8 GB minimum; 16 GB recommended for high-volume 20,000 image batching.
* **Storage**: Approximately 1.5 GB for 202,599 cropped images (`img_align_celeba/`) + 350 MB for MySQL database storage (`mysql_data/`).
* **Compute (Accelerator)**: CPU (multi-core x86_64) or NVIDIA GPU with CUDA for accelerated neural network training.

---

## Model Context Protocol (MCP) Integration

This project includes custom, zero-dependency Node.js MCP servers that expose the local MySQL database and Google Photos APIs directly to Antigravity AI agents:

### 1. MySQL MCP Server (`mcp\mysql_mcp_server.mjs`)

Enables AI agents to query and inspect the underlying `celeba_db` database via standard I/O:

* **`mysql_query`**: Execute arbitrary SQL queries (`SELECT`, `SHOW`, `DESCRIBE`, `EXPLAIN`) and return formatted tables.
* **`mysql_list_tables`**: List all tables in `celeba_db` with storage engine, row counts, and data lengths.
* **`mysql_describe_table`**: Inspect column data types, nullability, keys, and default values.
* **`mysql_get_celebrity_stats`**: Instant high-level count of celebrities, face metadata, attributes, and identity mappings.

### 2. Google Photos MCP Server (`mcp\google_photos_mcp_server.mjs`)

Enables bidirectional synchronization between local CelebA face clusters and Google Photos:

* **`google_photos_status`**: Verify Google OAuth2 authentication and configuration.
* **`google_photos_list_albums`**: Retrieve Google Photos albums and metadata.
* **`google_photos_search_media`**: Search library media by album or date ranges.
* **`google_photos_create_album`**: Create new albums (e.g. for celebrity clusters).
* **`google_photos_upload_image`**: Upload local face photos from `img_align_celeba/` directly to Google Photos.

### Configuration (`~/.gemini/config/mcp_config.json`)

Both servers are pre-registered in your Antigravity configuration:

```json
{
  "mcpServers": {
    "mysql": {
      "command": "node",
      "args": ["mcp/mysql_mcp_server.mjs"],
      "env": {
        "MYSQL_BIN": "D:\\MySQL\\bin\\mysql.exe",
        "MYSQL_HOST": "127.0.0.1",
        "MYSQL_PORT": "3306",
        "MYSQL_USER": "celeba_user",
        "MYSQL_PASSWORD": "<YOUR_MYSQL_PASSWORD>",
        "MYSQL_DB": "celeba_db"
      }
    },
    "google-photos": {
      "command": "node",
      "args": ["mcp/google_photos_mcp_server.mjs"],
      "env": {
        "GOOGLE_PHOTOS_CLIENT_ID": "",
        "GOOGLE_PHOTOS_CLIENT_SECRET": "",
        "GOOGLE_PHOTOS_REFRESH_TOKEN": "",
        "GOOGLE_PHOTOS_ACCESS_TOKEN": ""
      }
    }
  }
}
```

---

## Applications & How to Run

### 1. Google AI Studio Multimodal Face Intelligence Application

The primary intelligent facial reasoning application powered by Google AI Studio (Gemini 1.5 Flash) and local PyTorch deep learning models:

* **Interactive Web Application & Server**:
  Launch the unified Google AI Studio application server on `$PORT` (default `8080`):

  ```cmd
  python app.py
  # or
  python main.py
  # or
  python app/google_ai_studio_app.py --server --port 8080
  ```

* **Multimodal Zero-Shot Reasoning**: Extracts deep semantic facial context, demographics, emotion and expression, and attribute analysis using Gemini 1.5 Flash.
* **Hybrid Local Verification**: Compares multimodal reasoning against local PyTorch neural checkpoints (`models/checkpoints/` for Residual CNN, ViT, BiLSTM, and ANN).
* **Launch with Windows Launcher Script**:

  ```cmd
  scripts\start_google_ai_studio.bat
  ```

* **CLI Image Analysis**:

  ```cmd
  python app/google_ai_studio_app.py --image img_align_celeba/000001.png --model gemini-1.5-flash
  ```

* For full configuration, API key setup, and custom system prompts, see [docs/GOOGLE_AI_STUDIO_GUIDE.md](docs/GOOGLE_AI_STUDIO_GUIDE.md).

### 3. Interactive Jupyter Notebooks

All Python pipelines and modules have corresponding interactive, fully-commented `.ipynb` notebooks:

* **[notebooks/ai_studio/gemini_multimodal_face_analyzer.ipynb](notebooks/ai_studio/gemini_multimodal_face_analyzer.ipynb)**: Google AI Studio multimodal Gemini facial attribute reasoning, structured JSON schemas, and local PyTorch cross-validation.
* **[notebooks/ai_studio/google_ai_studio_interactive_app.ipynb](notebooks/ai_studio/google_ai_studio_interactive_app.ipynb)**: Interactive application notebook demonstrating batch hybrid multimodal face analysis.
* **[notebooks/database/database_connection_and_pooling.ipynb](notebooks/database/database_connection_and_pooling.ipynb)**: MySQL thread-safe connection pooling, ping-on-acquire liveness validation, and parameterized SQL query execution.
* **[notebooks/evaluation/fairness_and_identity_leakage_audit.ipynb](notebooks/evaluation/fairness_and_identity_leakage_audit.ipynb)**: Zero identity leakage verification ($0.0\%$ overlap) and EEOC 4/5ths disparate impact ratio audit.
* **[notebooks/training/classical_baselines_and_ensembling.ipynb](notebooks/training/classical_baselines_and_ensembling.ipynb)**: Classical ML baselines (Logistic Regression, ExtraTrees, XGBoost), SLSQP out-of-fold probability blending, and 5-view TTA.
* **[notebooks/pipelines/main.ipynb](notebooks/pipelines/main.ipynb)**: Primary pipeline notebook with exploratory data analysis, MySQL queries, and baseline models.
* **[notebooks/training/train_classifiers.ipynb](notebooks/training/train_classifiers.ipynb)**: 5-Fold Stratified Cross-Validation benchmark and ensembling suite.
* **[notebooks/clustering/face_embedding_and_clustering.ipynb](notebooks/clustering/face_embedding_and_clustering.ipynb)**: Google Photos style 128-D face embeddings, cosine similarity search, and unsupervised DBSCAN/K-Means clustering.
* **[notebooks/inference/image_classifier_models.ipynb](notebooks/inference/image_classifier_models.ipynb)**: Modular deep learning architectures (Deeper Residual CNN, ViT, ANN, RNN).
* **[notebooks/verification/project_verification.ipynb](notebooks/verification/project_verification.ipynb)**: Comprehensive repository verification and AST diagnostic test suite.

---

## Comparative Benchmark Evaluation Results

Evaluated on unseen standardized CelebA identity-disjoint holdout test data for the `Smiling` attribute with genuine trained PyTorch weights and out-of-fold blending:

| Model Configuration | Strategy | Test Loss | Accuracy | Precision | Recall | F1-Score | Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Artificial Neural Network (ANN)** | Standard Split | 0.6440 | **72.00%** | **0.6481** | 0.9459 | **0.7692** | Fully Connected PyTorch (Face Pixels) |
| **Spatial BiLSTM (RNN)** | Sequential Row-Wise BiLSTM | 0.6927 | **50.67%** | 0.0000 | 0.0000 | 0.0000 | Spatial Sequence PyTorch (Face Pixels) |
| **5-Fold Bagged Residual CNN** | 5-Fold ResNet + In-Model Aug | 0.6689 | **68.00%** | **0.6512** | 0.7568 | **0.7000** | Deep Residual Conv2D CNN (Face Pixels) |
| **Vision Transformer (ViT)** | Patch-16 Multi-Head Self-Attention | 0.6927 | **54.67%** | **0.6667** | 0.1622 | **0.2609** | Transformer Patch Self-Attention (Face Pixels) |
| **Single-Fold XGBoost** | Identity-Disjoint Standard Split | 0.4275 | **82.67%** | **0.8158** | 0.8378 | **0.8267** | Tabular Gradient Boosted Baseline |
| **5-Fold Bagged XGBoost** | 5-Fold Stratified Group Bagging | 0.4168 | **80.00%** | **0.8235** | 0.7568 | **0.7887** | Strong Bagged Model |
| **5-Fold Ensemble (XGB+CNN+ViT)** | Multi-Architecture Probability Blending | 0.3983 | **80.00%** | **0.8235** | 0.7568 | **0.7887** | Multi-Architecture Hybrid Ensemble |
| **5-Fold Ensemble + TTA** | Bagged Ensemble + Test-Time Augmentation | 0.3983 | **80.00%** | **0.8235** | 0.7568 | **0.7887** | Genuine Image-Space Test-Time Augmentation |

### Benchmark limitations

- The historical benchmark has 75 holdout examples; it is not a full-dataset evaluation.
- XGBoost uses annotated attributes, whereas neural models use image tensors. These modalities must be reported separately.
- Missing images may be replaced by zero tensors in the current training implementation. Image-free CI runs are smoke tests, not evidence of image classification accuracy.
- Full-module coverage in the last audited workflow was 32%; coverage varies by commit and test selection.
- Identity-disjoint splitting mitigates identity leakage but does not eliminate all forms of leakage.
