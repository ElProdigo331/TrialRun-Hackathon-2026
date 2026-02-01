# Energy AI Hackathon 2026 - Oil Production Prediction

**Team:** Brain Oil (Kailasadatta Boggaram, Jayanth Damodaran, Bilal Shihab, Carlos Fabela)

## Live Demo

**[Click here to try the web app](https://EnergyGladiators-adaptive-data-science.replit.app)**

## Overview

This is a complete machine learning workflow application for the Energy AI Hackathon 2026. It predicts **3-year cumulative oil production (BBL)** for 12 preproduction wells using well log data from a clastic/sandstone reservoir. The solution includes point estimates and 100 uncertainty realizations (R1-R100) per prediction.

## Our Results

After testing **57+ configurations**, we achieved:

| Metric | Value | Industry Benchmark |
|--------|-------|-------------------|
| **Test R²** | **0.9905** | Excellent (≥0.93) |
| **CV R²** | **0.9539 ± 0.0456** | Very stable |
| **RMSE** | **1.57M BBL (4.7%)** | Excellent (<10%) |

### Why Ridge Regression Won

| Model | Test R² | RMSE | Verdict |
|-------|---------|------|---------|
| **Ridge Regression** | **0.9905** | **1.57M BBL** | **WINNER** |
| Random Forest | 0.85 | 5.2M BBL | Good |
| XGBoost | 0.82 | 5.8M BBL | Acceptable |

> "For small datasets (n<100), regularized linear models often outperform tree-based ensembles due to lower variance."
> — Hastie, Tibshirani & Friedman (2009), *The Elements of Statistical Learning*

## Quick Start

### Option 1: Run on Replit (Recommended)
1. Import this project to Replit
2. Click "Run" - the app will start automatically
3. Open the Webview to access the application

### Option 2: Run Locally
**Requires: Python 3.11+**

```bash
# Install dependencies
pip install streamlit pandas numpy scikit-learn matplotlib seaborn plotly scipy optuna xgboost

# Run the application
python -m streamlit run app.py --server.port 5000
```

Then open your browser to: **http://localhost:5000**

## Winning Configuration

| Setting | Value | Justification |
|---------|-------|---------------|
| Model | Ridge Regression | Best for n=71 (Hastie et al. 2009) |
| Alpha | 1.0 | Data-driven from benchmarks |
| Normalization | StandardScaler | CRITICAL for penalized regression |
| Sand Map | Smooth (3x3) | Reduces noise, preserves trends |
| Feature Selection | Forward Stepwise | 105 → 10 optimal features |
| Uncertainty | Bagging Ensemble (100) | Captures model uncertainty |

## Notebook Template (Official Format)

Our `BrainOil.ipynb` follows the official hackathon template:

### Executive Summary (4 Short Sentences)
1. **The Problem:** Predict 3-year cumulative oil production for 12 preproduction wells (IDs 72-83) with uncertainty quantification.
2. **Our Solution:** Ridge Regression (alpha=1.0) with MICE imputation, stepwise feature selection (105→10), and Bagging Ensemble for 100 uncertainty realizations.
3. **What We Learned:** For small datasets (n=71), regularized linear models outperform complex tree-based ensembles; porosity (φ) is the primary production driver.
4. **Recommendation:** Use domain-driven feature engineering with simple, interpretable models for subsurface prediction problems.

### Workflow Goal
Develop a reproducible machine learning workflow to predict cumulative 3-year oil production with uncertainty for 12 preproduction wells.

### Workflow Steps
1. **Data Loading** - Load well logs (71 train, 12 test) and 2D sand proportion map
2. **MICE + CART Imputation** - Fill missing values at depth level before aggregation
3. **Well-Level Aggregation** - Convert multi-row depth data to one row per well
4. **Feature Engineering** - Create 105 features including RQI, FZI, spatial features
5. **Feature Selection** - Correlation filter + Stepwise selection (105→10)
6. **Model Training** - Ridge Regression with StandardScaler (alpha=1.0)
7. **Uncertainty Quantification** - Bagging Ensemble (100 estimators)
8. **Model Validation** - 5-fold CV, train/test split, benchmark comparison

## The 9-Step Streamlit Workflow

1. **Data Loading & MICE Imputation** - Load data, apply MICE+CART at depth level
2. **Data Quality Verification** - Check imputation results
3. **Exploratory Data Analysis** - 5 tabs including Feature Selection
4. **Feature Engineering** - 19 industry-standard features
5. **Model Training** - Interactive with experiment tracking
6. **Generate Solution** - Point estimates + R1-R100 realizations
7. **Experiment Leaderboard** - Compare all configurations
8. **AI Assistant** - Data-backed recommendations
9. **Scholarly Analysis** - Peer-reviewed citations

## Project Structure

```
/
├── app.py                    # Main Streamlit application
├── experiment_history.py     # Experiment tracking module
├── create_presentation.py    # Auto-generate PowerPoint
├── run_benchmarks.py         # Benchmarking script
├── data/                     # Data files
│   ├── Well_log_data_production_wells.csv
│   ├── Well_log_data_preproduction_wells.csv
│   ├── Production_history_production_wells.csv
│   └── 2d_sand_proportion.npy
├── outputs/                  # Generated outputs
│   ├── FINAL_SOLUTION_*.csv  # Final predictions
│   ├── RESULTS_SUMMARY.md    # Performance summary
│   └── experiment_history.json
├── notebooks/
│   └── BrainOil.ipynb        # Submission notebook
├── Brain_Oil.pptx            # Presentation (6 slides)
├── Presentation_Walkthrough.pdf  # Presenter's guide
└── Laymans_Guide_to_ML_Workflow.md  # Plain-English guide
```

## Data Description

### Training Data
- **71 production wells** with known 3-year cumulative oil production
- **~21 depth measurements** per well (Z=19-39)
- **Aggregation required:** Multi-row → single row per well

### Features Used
| Category | Examples |
|----------|----------|
| Petrophysical | phi, perm, GR, AI, SI, Vp, Vs |
| Industry-Standard | RQI, FZI, Vp/Vs ratio |
| Rock Quality | phi_perm_product, net_to_gross |
| Spatial | X, Y, sand_proportion |
| Best Zone | best_zone_phi, best_zone_perm |

### Target Variable
- **3-year cumulative oil production (BBL)**
- Range: 8.2M - 74.0M BBL
- Mean: 33.4M BBL

## Solution File Format

| Column | Description |
|--------|-------------|
| Well_ID | Well identifier (72-83) |
| Prediction_BBL | Point estimate |
| R1 - R100 | 100 uncertainty realizations |

## Key References

1. Hastie, Tibshirani & Friedman (2009). *The Elements of Statistical Learning*. Springer.
2. Hoerl & Kennard (1970). "Ridge Regression." *Technometrics*.
3. Van Buuren (2018). *Flexible Imputation of Missing Data*. CRC Press.
4. Breiman (1996). "Bagging Predictors." *Machine Learning*.
5. Amaefule et al. (1993). RQI & FZI methodology.

## Dependencies

- streamlit, pandas, numpy, scikit-learn
- scipy, matplotlib, seaborn, plotly
- optuna, xgboost, openai (optional)

## Team Brain Oil

- **Kailasadatta Boggaram**
- **Jayanth Damodaran**
- **Bilal Shihab**
- **Carlos Fabela**

Energy AI Hackathon 2026 | The University of Texas at Austin
