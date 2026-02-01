# Energy AI Hackathon 2026 - Oil Production Prediction

**Team:** Brain Oil (Kailasadatta Boggaram, Jayanth Damodaran, Bilal Shihab, Carlos Fabela)

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
| Alpha | 0.1 | Data-driven from benchmarks |
| Normalization | StandardScaler | CRITICAL for penalized regression |
| Sand Map | Smooth (3x3) | Reduces noise, preserves trends |
| Feature Selection | Forward Stepwise | 105 → 10 optimal features |
| Uncertainty | Bagging Ensemble (100) | Captures model uncertainty |

## The 9-Step Workflow

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
├── Brain_Oil.pptx            # Presentation
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
