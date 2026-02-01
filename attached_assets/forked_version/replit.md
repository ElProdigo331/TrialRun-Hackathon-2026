# Energy AI Hackathon 2026 - Oil Production Prediction

## Overview
This project predicts 3-year cumulative oil production (BBL) for 12 pre-production wells (Well IDs 72-83) for the Energy AI Hackathon 2026. The solution uses machine learning with robust data imputation, feature engineering, uncertainty quantification, and an AI assistant with data-driven recommendations from comprehensive model benchmarks.

## User Preferences
Not specified.

## Winning Configuration (Final Submission)

### Model Performance
| Metric | Value | Industry Benchmark |
|--------|-------|-------------------|
| Test R² | **0.9905** | Excellent (≥0.93) |
| CV R² Mean | **0.9539 ± 0.0456** | Excellent |
| Test RMSE | **1.57M BBL** | 4.7% of mean (Excellent <10%) |
| Train R² | 0.9881 | - |

### Optimal Settings
- **Model:** Ridge Regression (L2 regularization)
- **Alpha:** 0.1
- **Normalization:** StandardScaler (critical for linear models)
- **Sand Map:** Smooth (3x3) spatial averaging
- **Feature Pipeline:**
  - Correlation filter: 105 → 61 features (removed 44 redundant, r ≥ 0.98)
  - Stepwise selection: 10 optimal features (forward selection, 5-fold CV)
- **Uncertainty:** Bagging Ensemble (100 estimators)

### Why Ridge Regression Won
1. **Small dataset (n=71):** Regularization prevents overfitting (Hastie et al. 2009)
2. **High-dimensional features:** L2 penalty handles multicollinearity (Hoerl & Kennard 1970)
3. **Benchmark results:** Ridge consistently outperformed tree models in our 57+ configuration tests
4. **Stability:** CV R² std of only ±0.0456 indicates robust performance

---

## ML Pipeline (9 Steps)

1. **Data Loading, MICE & Aggregation** - Load data, apply MICE+CART at depth level (before aggregation), then aggregate
2. **Data Quality Verification** - Verify MICE was applied correctly, check for remaining missing values
3. **Exploratory Data Analysis** - 5 tabs: Target Analysis, Feature Analysis, Correlations, Rock Quality Analysis, **Feature Selection**
4. **Feature Engineering** - 19+ features across 5 categories (petrophysical, industry-standard, derived, spatial, best zone)
5. **Model Training** - Ridge Regression with stepwise feature selection, auto-save to experiment history
6. **Generate Solution** - Point estimates + 100 realizations via Bagging Ensemble
7. **Experiment Leaderboard** - Persistent memory of all runs with rankings, trends, model statistics
8. **AI Assistant** - Chat interface with data-driven recommendations from benchmarks AND experiment history
9. **Scholarly Analysis** - Peer-reviewed citations and methodology justification

---

## Benchmark System

### Purpose
Comprehensive model benchmarking to identify optimal configurations. Ridge Regression emerged as the winner.

### Benchmark Results Summary
| Model Type | Best Val R² | RMSE | Overfitting Gap |
|------------|-------------|------|-----------------|
| **Ridge Regression** | **0.9905** | **1.57M** | **0.002** |
| Random Forest | 0.85 | 5.2M | 0.15 |
| XGBoost | 0.82 | 5.8M | 0.18 |
| Linear Regression | Overfit | - | >1.0 |

### Key Finding
> "For small datasets (n<100), regularized linear models often outperform tree-based ensembles due to lower variance."
> — Hastie, Tibshirani & Friedman (2009), Elements of Statistical Learning

### Key Files
- `run_benchmarks.py` - Benchmark testing script
- `outputs/benchmark_results.json` - Results storage with model rankings
- `experiment_history.py` - Experiment history module for persistent tracking
- `outputs/experiment_history.json` - Persistent storage for all training experiments
- `outputs/FINAL_SOLUTION_Ridge_Stepwise10_Smooth3x3.csv` - Winning submission

---

## Feature Selection Pipeline

### Stage 1: Correlation Filter (EDA Tab 5)
- **Threshold:** r ≥ 0.98 (Pearson & Spearman)
- **Result:** 105 → 61 features (removed 44 redundant)
- **Method:** BFS graph traversal, keeps highest variance feature per group

### Stage 2: Stepwise Selection (Model Training)
- **Direction:** Forward selection
- **Criterion:** CV R² improvement
- **Result:** 61 → 10 optimal features
- **Citation:** Miller (2002), Subset Selection in Regression

### Why Two-Stage Selection?
> "Dimensionality reduction before model training improves generalization and interpretability."
> — Guyon & Elisseeff (2003), Journal of Machine Learning Research

---

## System Architecture

### UI/UX Decisions
- **Streamlit Interface** - Interactive data exploration, model training, results visualization
- **5-Tab EDA** - Target Analysis, Feature Analysis, Correlations, Rock Quality Analysis, Feature Selection
- **Interactive Model Selection** - Model types, hyperparameters, experiment settings
- **AI Assistant** - Data-driven recommendations from benchmarks and experiment history
- **Experiment Leaderboard** - Rankings, trends, model statistics, CSV export
- **Auto-Save History** - Every model training run saved with immediate comparison feedback

### Technical Implementations
- **MICE + CART Imputation** - Applied at depth level before aggregation per Van Buuren (2018)
- **Feature Engineering** - 19+ features including best zone features for depth heterogeneity
- **Uncertainty Quantification** - Bagging Ensemble (100 estimators) per Breiman (1996)
- **Train/Test Split (80/20)** - Separate metrics to detect overfitting

---

## External Dependencies
- `streamlit` - Interactive web application
- `pandas` - Data manipulation
- `numpy` - Numerical operations
- `scikit-learn` - ML algorithms, preprocessing, MICE imputation
- `scipy` - Scientific computing, smoothing
- `matplotlib`, `seaborn`, `plotly` - Visualization
- `optuna` - Hyperparameter optimization
- `openai` - AI assistant
- `mlxtend` - Stepwise feature selection
- `xgboost` - Gradient boosting (benchmarking)
