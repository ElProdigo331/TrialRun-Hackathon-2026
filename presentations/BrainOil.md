# Energy AI Hackathon 2026 Presentation Outline

## SLIDE 1: Title
**Predicting 3-Year Cumulative Oil Production**

Brain Oil

Team Members:
- [Member 1] - [Affiliation]
- [Member 2] - [Affiliation]
- [Member 3] - [Affiliation]
- [Member 4] - [Affiliation]

Energy AI Hackathon 2026 | February 1, 2026

---

## SLIDE 2: Problem Statement

**The Challenge**

Predict 3-year cumulative oil production (BBL) for 12 preproduction wells (Well IDs 72-83) using:
- Petrophysical well log data (multiple depth measurements per well)
- Production history from 71 analog wells
- 2D seismic sand proportion map

**Plus: Quantify Uncertainty**
- 100 probabilistic realizations (R1-R100) per prediction
- Enables risk-aware investment decisions

**Why This Matters**
- Well prioritization for development
- Capital allocation optimization
- Reserve estimation confidence intervals

---

## SLIDE 3: Data Overview

**Training Data**
- 71 production wells with historical data
- ~21 depth measurements per well (1,491 total rows)
- 17 petrophysical features at each depth

**Key Challenge: Multi-Row Data**

Each well has multiple depth measurements (Z = 19 to 39):
```
Well_ID | X | Y | Z | phi | perm | GR | AI | facies | ...
   1    | 50| 30| 19| 0.15| 120  | 45 | 8.2|   2    |
   1    | 50| 30| 20| 0.14| 115  | 48 | 8.1|   2    |
   ...
```

**Solution:** Aggregate to one row per well (mean, std, min, max of each feature)

---

## SLIDE 4: Our Approach

**Complete ML Pipeline:**

```
1. Well Log Aggregation
   └── Multi-row → One row per well
   └── Statistics: mean, std, min, max

2. Target Calculation
   └── 3-year cumulative oil from production history

3. Feature Engineering
   └── Sand proportion from seismic map
   └── Facies distribution percentages
   └── Derived features (phi×log(perm), rock quality)

4. MICE Imputation
   └── Handle 7-10% missing values
   └── Preserves feature correlations

5. Random Forest + Optuna
   └── Automated hyperparameter tuning

6. Uncertainty: Residual Bootstrapping
   └── 100 realizations per prediction
```

---

## SLIDE 5: Feature Engineering

**Petrophysical Features (Aggregated per Well)**

| Feature Type | Examples | Aggregation |
|--------------|----------|-------------|
| Porosity | phi | mean, std, min, max |
| Permeability | perm | mean, std, min, max |
| Acoustic | AI, SI, Vp, Vs | mean, std, min, max |
| Lithology | GR, facies | mean, distribution |
| Density | rho_b, rho_f, rho_m | mean, std |
| Moduli | K, G (dry, sat) | mean, std |

**Derived Features:**
- `phi_perm_product` = phi × log(perm) - productivity indicator
- `rock_quality` = phi / GR - reservoir quality index
- `sand_proportion` - from 2D seismic map using X,Y coordinates

---

## SLIDE 6: MICE Imputation

**Why MICE? (Recommended by Hackathon Host)**

Missing data: 7-10% across petrophysical features

| Simple Imputation | MICE |
|-------------------|------|
| Uses median/mean only | Uses ALL features to predict missing |
| Ignores correlations | Preserves correlations |
| Can distort relationships | Maintains data structure |

**How MICE Works:**
1. Initialize missing values with median
2. For each feature with missing values:
   - Fit regression using all OTHER features
   - Predict and replace missing values
3. Iterate until convergence

**Result:** More realistic imputed values that respect petrophysical relationships

---

## SLIDE 7: Model Training

**Random Forest with Optuna Auto-Tuning**

Optuna searches parameter space:
- `n_estimators`: 50-300
- `max_depth`: 3-20
- `min_samples_split`: 2-20

**Cross-Validation Results:**
- CV R²: [Value from training]
- Number of features: ~70+ after aggregation

**Top 10 Most Important Features:**
1. [Feature 1]
2. [Feature 2]
3. [Feature 3]
... (from model.feature_importances_)

---

## SLIDE 8: Uncertainty Quantification

**Method: Residual Bootstrapping**

1. Train model and compute residuals: `r = y_actual - y_predicted`
2. For each test prediction:
   - Sample 100 residuals with replacement
   - Add sampled residual to point estimate
   - Ensure non-negative values

**Result: 100 Realizations Per Well**

```
Well_ID | Prediction_BBL | R1 | R2 | ... | R100
   72   |   25,000,000   | .. | .. | ... | ..
   73   |   38,000,000   | .. | .. | ... | ..
```

This captures prediction uncertainty from model limitations.

---

## SLIDE 9: Results

**Predictions for Wells 72-83**

| Well_ID | Prediction (BBL) | P10 | P50 | P90 |
|---------|------------------|-----|-----|-----|
| 72 | [Value] | [Value] | [Value] | [Value] |
| 73 | [Value] | [Value] | [Value] | [Value] |
| ... | ... | ... | ... | ... |

**Uncertainty Distribution:**
[Box plot showing prediction spread for each well]

**Key Insights:**
- Predictions align with training data distribution (8M - 74M BBL)
- Higher uncertainty for wells in underrepresented regions
- Spatial patterns consistent with sand proportion map

---

## SLIDE 10: Value Proposition

**Why Our Solution?**

| Traditional Decline Curves | Our ML Approach |
|---------------------------|-----------------|
| Needs production history | Works with well logs only |
| Point estimates | Uncertainty quantification |
| Manual parameter selection | Automated optimization |
| Single analog well | Learns from 71 wells |

**Novel Data Analytics:**
1. **MICE Imputation** - Recommended by host, preserves feature correlations
2. **Well Log Aggregation** - Handles complex multi-row structure
3. **Spatial Integration** - Incorporates 2D seismic sand map
4. **Automated Tuning** - Optuna finds optimal model

---

## SLIDE 11: Conclusion

**Summary:**
- Complete ML pipeline for oil production prediction
- Handles multi-row well log data through aggregation
- MICE imputation for missing values (host recommendation)
- Optuna-tuned Random Forest model
- 100 uncertainty realizations per prediction

**Files Submitted:**
1. `BrainOil.ipynb` - Complete reproducible workflow
2. `BrainOil.pptx` - This presentation
3. `solution.csv` - Predictions for Wells 72-83

**Team Brain Oil**
Energy AI Hackathon 2026

---

## APPENDIX: Technical Details

**Data Files:**
- `Well_log_data_production_wells.csv` (1,491 rows, 71 wells)
- `Well_log_data_preproduction_wells.csv` (252 rows, 12 wells)
- `Production_history_production_wells.csv` (5,517 rows)
- `2d_sand_proportion.npy` (200×200 spatial map)

**Feature List (Post-Aggregation):**
- 17 petrophysical features × 4 aggregations = 68 features
- + Spatial (X, Y, depth_range)
- + Facies distribution (6 facies types)
- + Sand proportion
- + Derived features (3)
- Total: ~75-80 features

**Compute Time:**
- Data loading & aggregation: ~2 seconds
- MICE imputation: ~5 seconds
- Optuna tuning (30 trials): ~30 seconds
- Total: < 1 minute

---

## APPENDIX: Solution Format

**Column Definitions:**

| Column | Description |
|--------|-------------|
| Well_ID | Well identifier (72-83) |
| Prediction_BBL | Point estimate (3-year cumulative oil) |
| R1 - R100 | 100 uncertainty realizations |

**All values in BBL (barrels of oil)**
