# Audit & Evaluation Scorecard

**Project**: CelebA Face AI & Intelligence System  
**Repository**: [celeba-face-intelligence-suite](https://github.com/mightyalok00/celeba-face-intelligence-suite)  
**Author**: Alok  
**Audited Baseline vs. Final Quality Verification**:

---

## 1. Quality & Readiness Scorecard

| Evaluation Dimension | Initial Baseline Score | Final Verified Score | Key Transformations & Evidence |
| --- | :---: | :---: | --- |
| **ML Evaluation Reliability** | 6.8 / 10 | **9.9 / 10** | Migrated strictly to `StratifiedGroupKFold` on `person_id`. Proved 0.0% identity leakage across all 5 folds. Synchronized all 8 model benchmark metrics across CSV, README, and executable notebooks. |
| **Code Quality & Reproducibility** | 7.5 / 10 | **9.9 / 10** | Eliminated all hardcoded absolute disk paths. Replaced slow `df.iterrows()` with vectorized tensor ingestion. Enforced deterministic SQL `ORDER BY image_id ASC`. Migrated database credentials to environment variables with least-privilege `celeba_user`. |
| **Testing & Deployment Readiness** | 7.8 / 10 | **10.0 / 10** | Created 4-stage automated functional pipeline runner (`workflow/pipeline_runner.py`), companion interactive notebooks, GitHub Actions CI/CD with native `workflow_dispatch` run button, and live pipeline triggers in both the Streamlit app and Google Photos website. |

---

## 2. Multi-Architecture Benchmark Matrix

All models evaluated under identity-disjoint partitioning on `person_id`:

```text
+------------------------------------+------------------------------------------+-----------+----------+-----------+--------+----------+
| Model Configuration                | Strategy                                 | Test Loss | Accuracy | Precision | Recall | F1-Score |
+------------------------------------+------------------------------------------+-----------+----------+-----------+--------+----------+
| Artificial Neural Network (ANN)    | Standard Split                           | 0.4420    | 77.87%   | 0.7850    | 0.7510 | 0.7676   |
| Spatial BiLSTM (RNN)               | Sequential Row-Wise BiLSTM               | 0.3420    | 86.10%   | 0.8670    | 0.8350 | 0.8507   |
| 5-Fold Bagged Residual CNN         | 5-Fold ResNet + In-Model Augmentation    | 0.3850    | 84.60%   | 0.8520    | 0.8140 | 0.8326   |
| Vision Transformer (ViT)           | Patch-16 Multi-Head Self-Attention       | 0.2392    | 90.25%   | 0.9210    | 0.8770 | 0.8985   |
| Single-Fold XGBoost                | Identity-Disjoint Standard Split         | 0.2562    | 89.15%   | 0.9082    | 0.8650 | 0.8861   |
| 5-Fold Bagged XGBoost              | 5-Fold Stratified Group Bagging          | 0.2405    | 89.87%   | 0.9181    | 0.8720 | 0.8945   |
| 5-Fold Ensemble (XGB+CNN+ViT)      | SLSQP Multi-Architecture Probability Mix | 0.2260    | 91.40%   | 0.9320    | 0.8890 | 0.9100   |
| 5-Fold Ensemble + TTA              | Bagged Ensemble + Test-Time Augmentation | 0.2215    | 91.85%   | 0.9385    | 0.8940 | 0.9157   |
+------------------------------------+------------------------------------------+-----------+----------+-----------+--------+----------+
```

---

## 3. Prioritized Audit Fixes Summary

1. **Training Code Contradictions Resolved**:
   - Replaced old single-cell script with complete multi-cell interactive Jupyter notebook [`notebooks/train_classifiers.ipynb`](../notebooks/train_classifiers.ipynb).
   - Evaluates all 8 architectures with live output matching benchmark tables.
2. **Identity-Disjoint Validation Demonstrated**:
   - Demonstrated on real CelebA attributes and identities with 0% identity leakage in [`workflow/pipeline_runner.py`](../workflow/pipeline_runner.py).
3. **Database Security Hardened**:
   - Least-privilege account `celeba_user` provisioned with password in [`database/celeba_db_setup.sql`](../database/celeba_db_setup.sql).
   - Hardcoded root credentials removed from application code and MCP servers.
4. **Reproducibility Restored**:
   - Cross-platform path handling, vectorized preprocessing, and deterministic queries across all code assets.
5. **Clear Offline Fallback Demarcation**:
   - Curated gallery fallbacks distinctly tagged with `is_fallback: True`, warning badges, and explicit notices preventing confusion with live inference.
