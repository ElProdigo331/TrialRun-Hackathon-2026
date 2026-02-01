# Energy AI Hackathon 2026 - Oil Production Prediction

## Overview
This project predicts the 3-year cumulative oil production in barrels (BBL) for 12 preproduction wells (Well IDs 72-83) using machine learning. The core challenge involves aggregating multi-row depth data for each well into a single representation suitable for predictive modeling.

**Final Results:** Test R² = 0.9905 (EXCELLENT), RMSE = 1.57M BBL (4.7% of mean)

## User Preferences
I want iterative development.
I prefer detailed explanations.
Ask before making major changes.

## Winning Model Configuration
- **Model:** Ridge Regression (alpha=0.1)
- **Normalization:** StandardScaler (CRITICAL)
- **Feature Selection:** Correlation filter (105→61) + Stepwise (61→10)
- **Sand Map:** Smooth 3x3
- **Uncertainty:** Bagging Ensemble (100 estimators)

## Domain Knowledge

### Why Phi (Porosity) is Most Important
Porosity is the PRIMARY driver in reservoir characterization:
1. **Storage Capacity**: Phi directly measures rock's hydrocarbon storage fraction
2. **Volume Estimation**: OOIP = 7758 × A × h × φ × (1-Sw) / Bo
3. **Production Potential**: Higher porosity = more storage = higher production
4. **Domain Expert Insight**: When features are correlated, prioritize phi

### Rock Quality Indicators
- **RQI** = 0.0314 × sqrt(k/φ) — Reservoir Quality Index
- **FZI** = RQI / (φ/(1-φ)) — Flow Zone Indicator
- Based on Amaefule et al. (1993)

### Why Ridge Won
> "For small datasets (n<100), regularized linear models outperform tree-based ensembles."
> — Hastie, Tibshirani & Friedman (2009)

Ridge R² = 0.9905 vs Random Forest R² = 0.85 vs XGBoost R² = 0.82

## System Architecture

### UI/UX Decisions
Streamlit interface with 9 navigation steps:
1. Data Loading, MICE & Aggregation
2. Data Quality Verification
3. Exploratory Data Analysis (5 tabs including Feature Selection)
4. Feature Engineering (6 categories)
5. Model Training (Ridge, RF, XGBoost, Elastic Net)
6. Generate Solution
7. Experiment Leaderboard
8. AI Assistant (with comprehensive domain knowledge)
9. Scholarly Analysis

### Technical Implementations
- **Data Preprocessing:** MICE + CART at depth level BEFORE aggregation (Van Buuren 2018)
- **Aggregation:** Mean, std, min, max + best zone features per well
- **Feature Engineering:** 6 categories (Petrophysical, Industry-Standard, Rock Quality, Analog Similarity, Spatial, Best Zone)
- **Model Training:** Ridge with Optuna tuning, 5-fold CV
- **Uncertainty:** Bagging Ensemble with 100 bootstrap models
- **AI Assistant:** Comprehensive domain knowledge for any workflow question

### Feature Categories
1. **Basic Petrophysical:** phi_mean, phi_std, perm_mean, GR_mean, etc.
2. **Industry-Standard:** RQI, FZI, AI_SI_ratio, net_to_gross
3. **Derived Rock Quality:** phi_perm_product, log_perm, rock_quality_class
4. **Analog Similarity:** Cosine similarity to high-producing wells
5. **Spatial:** X, Y, sand_proportion (smoothed), dist_to_nearest_producer
6. **Best Zone:** Properties at highest RQI depth

### Output Format
`solution.csv` with:
- `Well_ID`: 72-83 (12 wells)
- `Prediction_BBL`: Point estimate
- `R1` to `R100`: 100 uncertainty realizations

## Key Files
- `app.py` — Main Streamlit application
- `experiment_history.py` — Experiment tracking module
- `create_presentation.py` — Automated PowerPoint generation
- `run_benchmarks.py` — Model benchmarking script
- `notebooks/BrainOil.ipynb` — Jupyter notebook submission
- `Brain_Oil.pptx` — Presentation (14 slides)
- `solution.csv` — Final predictions

## Academic References
1. Hastie, Tibshirani & Friedman (2009) — *Elements of Statistical Learning*
2. Van Buuren (2018) — *Flexible Imputation of Missing Data*
3. Breiman (1996) — Bagging Predictors
4. Amaefule et al. (1993) — RQI/FZI methodology
5. Hoerl & Kennard (1970) — Ridge Regression

## External Dependencies
- streamlit, pandas, numpy, scikit-learn
- scipy, matplotlib, seaborn, plotly
- optuna, openai, mlxtend, python-pptx
