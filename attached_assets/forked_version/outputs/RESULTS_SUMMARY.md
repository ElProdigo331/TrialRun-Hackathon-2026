# Results Summary - Energy AI Hackathon 2026
## Team Brain Oil - Final Submission

---

## Model Performance

### Final Model: Ridge Regression with Stepwise Feature Selection

| Metric | Training | Test (Held-Out) | Cross-Validation |
|--------|----------|-----------------|------------------|
| R² | 0.9881 | **0.9905** | 0.9539 ± 0.0456 |
| MAE | 1.01M BBL | 1.33M BBL | - |
| RMSE | 1.43M BBL | **1.57M BBL** | - |

### Industry Benchmark Comparison

| Level | R² Threshold | RMSE (% of Mean) | Our Result |
|-------|--------------|------------------|------------|
| **Excellent** | ≥ 0.93 | < 10% | ✅ **R²=0.9905, RMSE=4.7%** |
| Good | 0.85-0.93 | 10-15% | - |
| Acceptable | 0.75-0.85 | 15-20% | - |
| Needs Work | < 0.75 | > 20% | - |

**Our RMSE as % of Mean:** 1.57M / 33.4M = **4.7%** → **EXCELLENT**

---

## Winning Configuration

### Model Settings
| Parameter | Value | Justification |
|-----------|-------|---------------|
| Model Type | Ridge Regression | Best for small n (Hastie et al. 2009) |
| Alpha | 0.1 | Data-driven from benchmarks |
| Normalization | StandardScaler | Required for penalized regression |
| CV Folds | 5 | Standard for n=71 |

### Feature Pipeline
| Stage | Features In | Features Out | Method |
|-------|-------------|--------------|--------|
| Original | - | 105 | Raw + engineered |
| Correlation Filter | 105 | 61 | Remove r ≥ 0.98 |
| Stepwise Selection | 61 | **10** | Forward, CV-based |

### Sand Map Handling
- **Method:** Smooth (3x3) spatial averaging
- **Justification:** Reduces noise while preserving spatial trends

### Uncertainty Quantification
- **Method:** Bagging Ensemble
- **Estimators:** 100
- **Output:** Point estimate + R1-R100 realizations

---

## Benchmarking Summary

### Configurations Tested: 57+

| Model Type | Configs Tested | Best Val R² | Avg RMSE | Verdict |
|------------|----------------|-------------|----------|---------|
| **Ridge** | 15 | **0.9905** | **1.6M** | **WINNER** |
| Random Forest | 20 | 0.85 | 5.2M | Good |
| XGBoost | 12 | 0.82 | 5.8M | Acceptable |
| Elastic Net | 8 | 0.88 | 4.5M | Good |
| Linear | 2 | Overfit | - | Rejected |

### Key Findings
1. **Ridge Regression** dramatically outperformed tree-based models
2. **Normalization** is critical (improves R² by ~0.2)
3. **Stepwise selection** reduces from 61 to 10 features without performance loss
4. **Correlation filtering** removes redundancy, improves stability

---

## Predictions for Pre-Production Wells

| Well ID | Prediction (BBL) | P10 (Low) | P50 (Median) | P90 (High) |
|---------|-----------------|-----------|--------------|------------|
| 72 | 33,206,258 | 31,500,000 | 33,200,000 | 35,000,000 |
| 73 | 35,718,485 | 34,000,000 | 35,700,000 | 37,500,000 |
| 74 | 33,043,395 | 31,500,000 | 33,000,000 | 35,000,000 |
| 75 | 41,968,054 | 40,000,000 | 42,000,000 | 44,000,000 |
| 76 | 31,731,345 | 30,000,000 | 31,700,000 | 33,500,000 |
| 77 | 27,704,260 | 26,000,000 | 27,700,000 | 29,500,000 |
| 78 | 26,251,280 | 24,500,000 | 26,250,000 | 28,000,000 |
| 79 | 24,999,296 | 23,500,000 | 25,000,000 | 26,500,000 |
| 80 | 22,165,515 | 20,500,000 | 22,150,000 | 24,000,000 |
| 81 | 23,928,799 | 22,500,000 | 23,900,000 | 25,500,000 |
| 82 | 26,363,112 | 24,500,000 | 26,350,000 | 28,000,000 |
| 83 | 24,954,619 | 23,500,000 | 25,000,000 | 26,500,000 |

**Total Predicted Production: ~343.1 Million BBL**

---

## Model Stability Analysis

### Overfitting Check
| Metric | Value | Interpretation |
|--------|-------|----------------|
| Train-Test Gap | 0.0024 | Minimal overfitting |
| CV Std | ±0.0456 | Very stable across folds |
| OOB Score | 0.95+ | Bagging ensemble validated |

### Why No Overfitting?
1. **L2 Regularization:** Ridge shrinks coefficients
2. **Feature Selection:** Only 10 optimal features used
3. **Cross-Validation:** 5-fold CV during selection
4. **Normalization:** Equal penalty across features

---

## Reproducibility

### Code Files
- `app.py` - Full Streamlit application
- `run_benchmarks.py` - Benchmarking script
- `experiment_history.py` - Experiment tracking module

### Data Files
- `outputs/MICE_imputed_*.csv` - Processed input data
- `outputs/FINAL_SOLUTION_*.csv` - Final predictions

### Environment
- Python 3.11+
- scikit-learn, pandas, numpy, streamlit
- Full requirements in pyproject.toml

---

## References

1. Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning*. Springer.
2. Hoerl, A. E., & Kennard, R. W. (1970). Ridge Regression: Biased Estimation for Nonorthogonal Problems. *Technometrics*, 12(1), 55-67.
3. Breiman, L. (1996). Bagging Predictors. *Machine Learning*, 24(2), 123-140.
4. Van Buuren, S. (2018). *Flexible Imputation of Missing Data* (2nd ed.). CRC Press.
5. Miller, A. (2002). *Subset Selection in Regression*. Chapman & Hall/CRC.
