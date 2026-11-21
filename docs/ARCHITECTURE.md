# CelebA Face AI & Intelligence System - Architecture Guide

## High-Level System Architecture

```text
+-------------------------------------------------------------------------------------------------+
|                                    PRESENTATION LAYER                                           |
|                                                                                                 |
|   +------------------------------------+          +-----------------------------------------+   |
|   |    Streamlit Web Dashboard         |          |   Google Photos Interface (HTML5/Tailwind)|   |
|   |    (app/app.py & root app.py)      |          |   (frontend/index.html & gallery_data)  |   |
|   +-----------------+------------------+          +--------------------+--------------------+   |
+---------------------|--------------------------------------------------|------------------------+
                      |                                                  |
+---------------------v--------------------------------------------------v------------------------+
|                                    ORCHESTRATION & MCP LAYER                                    |
|                                                                                                 |
|   +------------------------------------+          +-----------------------------------------+   |
|   |   celeba-mysql-mcp Server          |          |   celeba-google-photos-mcp Server       |   |
|   |   (mcp/mysql_mcp_server.mjs)       |          |   (mcp/google_photos_mcp_server.mjs)    |   |
|   +-----------------+------------------+          +--------------------+--------------------+   |
|                     |                                                  |                        |
|   +-----------------v--------------------------------------------------v--------------------+   |
|   |                Functional Pipeline Runner Suite (workflow/pipeline_runner.py)            |   |
|   |                GitHub Actions CI/CD Pipeline (.github/workflows/run_pipeline.yml)       |   |
|   +-----------------------------------------------------------------------------------------+   |
+-------------------------------------------------------------------------------------------------+
                      |
+---------------------v---------------------------------------------------------------------------+
|                                    INTELLIGENCE & DATABASE LAYER                                |
|                                                                                                 |
|   +------------------------------------+          +-----------------------------------------+   |
|   |   MySQL 8 Relational Database      |          |   Machine Learning & Ensembling         |   |
|   |   (celeba_db: 202,599 images)      |          |   - XGBoost, ResNet CNN, ViT, BiLSTM    |   |
|   |   - celebrities (1,090 names)      |          |   - 5-Fold Stratified Group K-Fold      |   |
|   |   - celeba_identities (10,177 PIDs)|          |   - SLSQP Probability Ensembling        |   |
|   |   - celeba_attributes (40 binary)  |          |   - Test-Time Augmentation (TTA)        |   |
|   +------------------------------------+          +-----------------------------------------+   |
+-------------------------------------------------------------------------------------------------+
```

## Directory Organization & Responsibilities

- **`.github/workflows/`**: GitHub Actions automated pipeline with manual trigger button (`workflow_dispatch`).
- **`app/`**: Modular Streamlit web application with trending glassmorphism UI, real-time MCP auto-detection, and pipeline triggers.
- **`config/`**: Keras and database environment configuration templates.
- **`data/metadata/`**: Ground-truth CelebA metadata (`celebrities.csv`, `list_attr_celeba.csv`, `identity_CelebA.txt`).
- **`database/`**: Relational database initialization script (`celeba_db_setup.sql`), analytics queries, and foreign key definitions.
- **`docs/`**: Formal system architecture, EER schema diagram, and audit score evaluation report.
- **`frontend/`**: Google Photos clone web UI with interactive pipeline workflow runner modal, face search, and motion cards.
- **`mcp/`**: Model Context Protocol (MCP) JSON-RPC 2.0 Stdio servers for MySQL and Google Photos.
- **`models/`**: Architecture registry (`model_registry.json`) and checkpoint specifications.
- **`notebooks/`**: Interactive reproducible Jupyter notebooks for training, clustering, and verification.
- **`reports/`**: Synchronized benchmark CSV results and publication-quality comparative figures.
- **`scripts/`**: Automation batch and PowerShell launcher scripts.
- **`src/`**: Modular Python packages for database access, deep learning architectures, and image preprocessing.
- **`workflow/`**: Automated functional pipeline runner (`pipeline_runner.py`), companion notebook, and batch triggers.
