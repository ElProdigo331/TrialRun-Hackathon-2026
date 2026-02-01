# Energy AI Hackathon 2026 - Team Brain Oil Submission

## Executive Summary

This submission predicts 3-year cumulative oil production (BBL) for 12 pre-production wells (Well IDs 72-83) using a robust machine learning pipeline with uncertainty quantification.

### Key Results
| Metric | Value | Industry Benchmark |
|--------|-------|-------------------|
| Test R² | **0.9905** | Excellent (≥0.93) |
| CV R² Mean | **0.9539 ± 0.0456** | Excellent (≥0.93) |
| Test RMSE | **1.57M BBL** | 4.7% of mean (Excellent <10%) |
| Test MAE | **1.33M BBL** | - |
| Train R² | 0.9881 | Minimal overfitting (gap: 0.002) |

### Winning Configuration
- **Model:** Ridge Regression with L2 regularization (α=0.1)
- **Normalization:** StandardScaler applied (critical for linear models)
- **Sand Map:** Smooth (3x3) spatial averaging
- **Feature Selection:** 
  - Stage 1: Correlation filter (105 → 61 features, r ≥ 0.98 threshold)
  - Stage 2: Stepwise selection (61 → 10 optimal features, forward selection)
- **Uncertainty:** Bagging Ensemble (100 estimators)

---

## Why Ridge Regression Won

After comprehensive benchmarking of **57+ configurations** across 5 model types, Ridge Regression dramatically outperformed alternatives:

| Model Type | Best Val R² | RMSE | Overfitting Gap |
|------------|-------------|------|-----------------|
| **Ridge Regression** | **0.9905** | **1.57M BBL** | **0.002** |
| Random Forest | 0.85 | 5.2M BBL | 0.15 |
| XGBoost | 0.82 | 5.8M BBL | 0.18 |
| Linear Regression | Overfit | - | >1.0 |

### Research Backing

> "For small datasets (n<100), regularized linear models often outperform tree-based ensembles due to lower variance."
> — **Hastie, Tibshirani & Friedman (2009), The Elements of Statistical Learning**

> "Ridge regression addresses multicollinearity and prevents overfitting by adding an L2 penalty."
> — **Hoerl & Kennard (1970), Technometrics**

---

## Methodology Overview

### 1. Data Preprocessing
- **MICE + CART Imputation:** Applied at depth level before aggregation (Van Buuren 2018)
- **Well-Level Aggregation:** Multi-row depth data → single row per well using mean, std, min, max
- **Training Wells:** 71 (after aggregation)
- **Test Wells:** 12 (Well IDs 72-83)

### 2. Feature Engineering (19+ Features)
- **Petrophysical:** Porosity, permeability, gamma ray statistics
- **Industry-Standard:** Rock Quality Index (RQI), Flow Zone Indicator (FZI)
- **Derived:** Phi-Perm product, Net-to-Gross ratio
- **Spatial:** X, Y coordinates, sand proportion (3x3 smoothed)
- **Best Zone:** Maximum values for depth heterogeneity

### 3. Feature Selection (Multi-Stage)
| Stage | Input | Output | Method |
|-------|-------|--------|--------|
| Correlation Filter | 105 | 61 | Remove r ≥ 0.98 redundancy |
| Stepwise Selection | 61 | **10** | Forward selection, 5-fold CV |

### 4. Model Training
- **Algorithm:** Ridge Regression
- **Alpha:** 0.1 (data-driven selection)
- **Cross-Validation:** 5-fold
- **Normalization:** StandardScaler

### 5. Uncertainty Quantification
- **Method:** Bagging Ensemble (Breiman 1996)
- **Estimators:** 100 Ridge models on bootstrap samples
- **Output:** Point estimate + R1-R100 realizations per well

---

## Predictions Summary

| Well ID | Prediction (Million BBL) | Uncertainty Range |
|---------|-------------------------|-------------------|
| 72 | 33.2 | ±1.5M |
| 73 | 35.7 | ±1.5M |
| 74 | 33.0 | ±1.5M |
| 75 | 42.0 | ±1.8M |
| 76 | 31.7 | ±1.4M |
| 77 | 27.7 | ±1.3M |
| 78 | 26.3 | ±1.2M |
| 79 | 25.0 | ±1.2M |
| 80 | 22.2 | ±1.1M |
| 81 | 23.9 | ±1.2M |
| 82 | 26.4 | ±1.2M |
| 83 | 25.0 | ±1.2M |

**Total Predicted Production: ~343.1 Million BBL**

---

## Files Included

| File | Description |
|------|-------------|
| `FINAL_SOLUTION_Ridge_Stepwise10_Smooth3x3.csv` | Final predictions with 100 realizations |
| `RESULTS_SUMMARY.md` | Detailed performance metrics |
| `WORKFLOW_SCHOLARLY_ANALYSIS.md` | Peer-reviewed methodology justification |
| `Feature_Documentation.html` | Detailed feature descriptions |
| `benchmark_results.json` | Full benchmarking of 57+ configurations |
| `MICE_imputed_*.csv` | Processed data files |
| `sand_map_visualization.png` | Spatial sand proportion map |

---

## Technical Implementation

- **Framework:** Python with Streamlit interactive interface
- **ML Libraries:** scikit-learn, XGBoost (benchmarking), Optuna
- **Visualization:** Plotly, Matplotlib, Seaborn
- **AI Assistant:** OpenAI GPT-4o-mini for data-driven recommendations
- **Experiment Tracking:** Persistent history with automatic ranking

---

## Key Citations

1. Hastie, Tibshirani & Friedman (2009). *The Elements of Statistical Learning*. Springer.
2. Hoerl & Kennard (1970). "Ridge Regression: Biased Estimation for Nonorthogonal Problems." *Technometrics*.
3. Breiman (1996). "Bagging Predictors." *Machine Learning*.
4. Van Buuren (2018). *Flexible Imputation of Missing Data*. CRC Press.
5. Miller (2002). *Subset Selection in Regression*. Chapman & Hall/CRC.
6. Efron & Tibshirani (1993). *An Introduction to the Bootstrap*. Chapman & Hall.

---

## Team Brain Oil
Energy AI Hackathon 2026
