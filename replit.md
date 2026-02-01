# Energy AI Hackathon 2026 - Oil Production Prediction

## Team: Brain Oil
**Deadline:** February 1st, 2026 at 12:00 noon
**Submission Files:** BrainOil.ipynb, BrainOil.pptx, solution.csv

---

## Problem Statement (2026)
Predict **3-year cumulative oil production (BBL)** for **12 preproduction wells** (Well IDs 72-83).

### Key Differences from 2025:
| 2025 | 2026 |
|------|------|
| Energy consumption (Grid, Diesel, CNG) | Oil production (BBL) |
| Frac fleet operations | Petrophysical well logs |
| 50 wells, single row each | 12 wells, multi-row depth data |
| Multiple targets | Single target |

---

## Data Structure

### Files:
| File | Rows | Description |
|------|------|-------------|
| Well_log_data_production_wells.csv | 1,491 | Training features (71 wells × ~21 depths) |
| Well_log_data_preproduction_wells.csv | 252 | Test features (12 wells × ~21 depths) |
| Production_history_production_wells.csv | 5,517 | Target source (monthly cumulative) |
| 2d_sand_proportion.npy | 200×200 | Spatial sand map |
| solution.csv | 12 | Submission template |

### Multi-Row Data Challenge:
Each well has ~21 rows (depth measurements Z=19-39). Must aggregate to one row per well:
- Mean, std, min, max of each numeric feature
- Facies distribution percentages
- Depth range

---

## Solution Format

| Column | Description |
|--------|-------------|
| Well_ID | Well identifier (72-83) |
| Prediction_BBL | Point estimate |
| R1 - R100 | 100 uncertainty realizations |

**Note:** Column names are `R1`, `R2`, ..., `R100` (NOT `Real_1`, etc.)

---

## ML Pipeline (7 Steps)

1. **Data Loading, MICE & Aggregation** - Load data, apply MICE+CART at depth level (before aggregation), then aggregate
2. **Data Quality Verification** - Verify MICE was applied correctly, check for remaining missing values
3. **Exploratory Data Analysis** - 4 tabs: Target Analysis, Feature Analysis, Correlations, **Rock Quality Analysis**
4. **Feature Engineering** - 19 features across 5 categories (see below)
5. **Model Training** - Interactive model selection with full hyperparameters, metrics (R², MAE, RMSE, OOB), SHAP analysis
6. **Generate Solution** - Point estimates + 100 realizations, experiment tracking & comparison
7. **AI Assistant** - Enhanced chat interface with industry benchmarks, session context awareness, and actionable recommendations

**Note:** MICE is now applied at the depth level BEFORE aggregation per Van Buuren (2018) and Hallam et al. (2022). This preserves correlations and ensures all depth measurements contribute.

---

## AI Assistant Features

The AI ML Assistant provides expert guidance with:

### Industry Benchmarks (Built-In Knowledge)
| Performance Level | R² Range | RMSE (% of mean) | For This Dataset |
|-------------------|----------|------------------|------------------|
| Excellent | ≥ 0.93 | < 10% | RMSE < 3.3M BBL |
| Good | 0.85-0.93 | 10-15% | RMSE 3.3-5M BBL |
| Acceptable | 0.75-0.85 | 15-20% | RMSE 5-6.7M BBL |
| Needs Improvement | < 0.75 | > 20% | RMSE > 6.7M BBL |

### Session Context Awareness
The assistant automatically knows your current:
- Model type and experiment name
- CV R² scores and best hyperparameters
- Top feature importances
- Training results

### Suggested Questions
- "What's the best model to start with?"
- "Is my R² of 0.85 good enough?"
- "How can I improve my model's accuracy?"
- "Which uncertainty method is better?"

### Optimal Settings Recommendations
Based on SPE publications and industry studies:
- **Best model:** XGBoost (R² 0.95-0.98 in published studies)
- **Features:** Use stepwise selection (15-25 features)
- **Normalize:** Always YES
- **Sand map:** Smooth 3x3
- **Uncertainty:** Bagging ensemble for model uncertainty

---

## Model Training Features (Per Hackathon Checklist)

| Requirement | Status | Implementation |
|-------------|--------|----------------|
| Aggregation (mean, std, min, max) | ✅ Done | Well log aggregation function |
| MICE imputation | ✅ Done | MICE+CART at depth level |
| Hyperparameter tuning with Optuna | ✅ Done | 30 trials by default |
| n_estimators | ✅ Done | 50-300 search range |
| max_depth | ✅ Done | 3-20 search range |
| min_samples_split | ✅ Done | 2-20 search range |
| min_samples_leaf | ✅ Done | 1-10 search range |
| max_features | ✅ Done | sqrt, log2, 0.5 options |
| MAE and MSE metrics | ✅ Done | Displayed after training |
| OOB samples | ✅ Done | oob_score=True for RF |
| Residual bootstrapping | ✅ Done | 100 realizations (R1-R100) |
| SHAP values | ✅ Done | TreeExplainer + summary plots |
| Train/Test split evaluation | ✅ Done | 80/20 split with separate metrics |
| Uncertainty calibration check | ✅ Done | 5-fold CV with coverage analysis |
| Stepwise feature selection | ✅ Done | Forward/Backward with mlxtend |
| Bagging ensemble uncertainty | ✅ Done | sklearn BaggingRegressor (100 estimators) |

---

## Stepwise Feature Selection

**Location:** Step 5 (Model Training) → Optional expander

| Option | Description |
|--------|-------------|
| **Direction** | Forward (add best) or Backward (remove worst) |
| **Target Features** | 3-30 features to select |
| **CV Folds** | Cross-validation folds for scoring |

Uses `mlxtend.feature_selection.SequentialFeatureSelector` with Ridge as base model. Selected features can be used for subsequent model training.

---

## Uncertainty Quantification Methods

| Method | How R1-R100 are Generated |
|--------|---------------------------|
| **Residual Bootstrap** | prediction + random_historical_error |
| **Bagging Ensemble** | Each of 100 estimators gives its own prediction |

### Bagging Ensemble (New)
Uses `sklearn.ensemble.BaggingRegressor`:
- Creates 100 bootstrap samples of training data
- Trains separate model on each sample
- Each estimator's prediction = one realization
- Reports OOB R² score for validation
- Works with all model types (Linear, Ridge, Elastic Net, RF, XGBoost)

---

## Model Evaluation Features

### Train/Test Split (80/20)
Shows separate metrics for training and held-out test data:
- **Training Set:** R², MAE, RMSE on 80% of data
- **Test Set:** R², MAE, RMSE on held-out 20%
- **Overfitting Detection:** Warning if Train R² >> Test R²

### Uncertainty Calibration Check
Validates if the R1-R100 prediction intervals are reliable:
- **90% Coverage:** Should be ~90% (actual values within P5-P95 range)
- **50% Coverage:** Should be ~50% (actual values within P25-P75 range)
- **Calibration Plot:** Visual showing which wells are within predicted ranges

### RMSE Interpretation Guide
How to evaluate if your RMSE is good:

| Baseline | Value | Purpose |
|----------|-------|---------|
| Target Mean | 33.4M BBL | RMSE as % of mean |
| Target Std Dev | 14.1M BBL | Null model baseline |
| Target Range | 8.2M - 74.0M BBL | Context for error magnitude |

**Null Model:** Predicting the mean for every well gives RMSE ≈ 14.1M BBL (the standard deviation).

| RMSE % of Mean | Interpretation |
|----------------|----------------|
| < 10% | Excellent |
| 10-20% | Good |
| 20-30% | Acceptable |
| > 30% | Needs improvement |

**Note:** Linear Regression may show Train MAE/RMSE = 0 when features outnumber samples (overfitting). Use Ridge or Random Forest instead.

---

## Industry Expert Insights (Incorporated)

From 4 industry heads at the hackathon:

| Insight | Implementation |
|---------|----------------|
| "Good rock = more oil" | Rock Quality Analysis tab, Good/Bad rock classification |
| "Time & location matter" | X, Y spatial correlations, sand_proportion map |
| "Format features correctly" | StandardScaler normalization, derived rock quality metrics |

---

## Key Techniques (Host Recommended)

1. **MICE Imputation** - Multivariate Imputation by Chained Equations
   - Uses DecisionTreeRegressor (CART) per SPE 218890
   - Applied at depth level before aggregation

2. **Feature Normalization** - StandardScaler (Dr. Pyrcz's advice)
   - Equalizes feature scales for fair comparison

3. **Rock Quality Analysis** - Industry expert recommendation
   - Good vs Bad rock classification
   - Production correlation by rock class

---

## Interactive Experiment Framework (Step 5)

| Option | Choices | Purpose |
|--------|---------|---------|
| **Model Type** | Linear Regression, Ridge, Random Forest, XGBoost | Try simple first (Dr. Pyrcz) |
| **Normalize Features** | Checkbox (default: ON) | StandardScaler |
| **Sand Map Handling** | Include, Exclude, Smooth (3x3), Smooth (5x5) | Handle noisy sand map (Dinghan Wang) |
| **Experiment Name** | Auto-generated | Labels output files for comparison |

**Experiment outputs saved to:** `outputs/solution_{experiment_name}.csv`

---

## Running the Application
```bash
streamlit run app.py --server.port 5000
```

---

## Project Structure
```
/
├── app.py                    # Main Streamlit application (2026 version)
├── data/                     # Data files
│   ├── Well_log_data_production_wells.csv
│   ├── Well_log_data_preproduction_wells.csv
│   ├── Production_history_production_wells.csv
│   ├── 2d_sand_proportion.npy
│   └── solution.csv
├── outputs/                  # Generated outputs (experiment CSVs)
├── notebooks/
│   └── BrainOil.ipynb       # Submission notebook
├── presentations/
│   └── BrainOil.md          # Presentation outline
└── hackathon2026_data/      # Original downloaded data
```

---

## Features Used

### Petrophysical (Aggregated):
- **phi** - Porosity
- **perm** - Permeability
- **GR** - Gamma Ray
- **AI, SI** - Acoustic/Shear Impedance
- **Vp, Vs** - P-wave/S-wave velocity
- **rho_b, rho_f, rho_m** - Bulk/Fluid/Matrix density
- **K0, Kdry, Kf, Ksat** - Bulk modulus variants
- **G0, Gdry, Gsat** - Shear modulus variants
- **facies** - Rock type (1-6, encoded as distribution)

### Industry-Standard Features (SPE Literature):
| Feature | Formula | Reference |
|---------|---------|-----------|
| **RQI** | 0.0314 × √(k/φ) | Reservoir Quality Index (Amaefule et al. 1993) |
| **FZI** | RQI / [φ/(1-φ)] | Flow Zone Indicator - hydraulic flow unit classification |
| **Vp_Vs_ratio** | Vp / Vs | Lithology & fluid indicator (rock physics) |

### Derived Rock Quality Features:
| Feature | Formula | Interpretation |
|---------|---------|----------------|
| **phi_perm_product** | phi × log(perm) | Flow productivity (higher = better) |
| **rock_quality** | phi / GR | Clean sand index (higher = cleaner) |
| **impedance_ratio** | AI / SI | Lithology contrast |
| **net_to_gross** | 1 - facies_5% - facies_6% | Sand vs shale ratio |
| **storage_capacity** | phi × depth_range | Total pore volume proxy |
| **flow_quality** | log(perm) / GR | Flow per unit shaliness |

### Analog Well Similarity (Nataly's Insight):
| Feature | Description |
|---------|-------------|
| **analog_similarity** | 1 / (1 + min_distance_to_good_producer) |
| **analog_production_proxy** | Weighted avg production of similar good wells |

Nataly (Hackathon Architect): "Look for correlation of known wells in good sand/rock that historically produced oil to the training wells."

### Spatial Proximity Features:
| Feature | Description |
|---------|-------------|
| **proximity_to_high_prod** | 1 / (1 + dist_to_high_prod_centroid) |
| **left_region_score** | 1 / (1 + dist_from_left_edge) - left side produces more |
| **spatial_production_proxy** | Inverse-distance weighted avg of nearby production |

User observation: Left side of map (low X) has higher production, especially bottom-left and top-left corners.

### Best Zone Features (Preserve Depth Heterogeneity):
| Feature | Description |
|---------|-------------|
| **best_zone_phi** | Porosity at best rock quality depth |
| **best_zone_perm** | Permeability at best rock quality depth |
| **best_zone_GR** | Gamma Ray at best rock quality depth |
| **best_zone_quality** | Rock quality score at best depth |
| **zone_quality_contrast** | best_zone_phi - worst_zone_phi |

Note: Instead of just averaging all depths, we extract features from the "pay zone" (best rock quality depth) to preserve important depth-level heterogeneity.

### Spatial:
- **X, Y** - Well coordinates
- **sand_proportion** - From 2D seismic map
- **depth_range** - Z_max - Z_min

---

## Target Details

- **Target:** 3-year cumulative oil production
- **Range:** 8,242,955 to 74,021,408 BBL
- **Mean:** 33,378,139 BBL
- **Std:** 14,148,917 BBL

---

## Dependencies
- streamlit
- pandas
- numpy
- scikit-learn
- scipy (for MICE, smoothing)
- matplotlib
- seaborn
- plotly
- optuna
- openai (for AI assistant)

---

## Scholarly Analysis Page

Available at sidebar option "Scholarly Analysis" - includes:
- 15+ peer-reviewed citations
- Methodology justification
- Downloadable HTML for print-to-PDF
