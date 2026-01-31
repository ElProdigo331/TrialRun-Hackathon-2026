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

## SLIDE 4: Industry Expert Insights

**From Industry Experts at the Hackathon:**

| Expert | Insight | How We Applied It |
|--------|---------|------------------|
| **Industry Heads** | "Good rock = more oil" | Rock quality features: High phi, high perm, low GR |
| **Industry Heads** | "Time & location matter" | X, Y coordinates + sand proportion map |
| **Dr. Pyrcz** | "Format features correctly" | StandardScaler normalization |
| **Nataly** | "Correlate to good producers" | Analog well similarity feature |

**Rock Quality Classification:**
- **Good Rock:** High porosity (φ), Low Gamma Ray (GR), High permeability
- **Poor Rock:** Low φ, High GR, Low permeability

**Result:** Good rock wells produce significantly more oil than poor rock wells!

---

## SLIDE 5: Our Approach

**Complete ML Pipeline:**

```
1. Data Loading & MICE Imputation (Depth Level)
   └── Handle ~7.3% missing values BEFORE aggregation
   └── Uses DecisionTreeRegressor (CART) per SPE 218890

2. Well Log Aggregation
   └── Multi-row → One row per well
   └── Statistics: mean, std, min, max

3. Target Calculation
   └── 3-year cumulative oil from production history

4. Feature Engineering (Rock Quality Focus)
   └── 6 derived rock quality features
   └── Sand proportion from seismic map

5. Model Training (Interactive Experiments)
   └── Linear Regression, Ridge, Random Forest
   └── StandardScaler normalization
   └── Sand map: Include / Exclude / Smooth options

6. Uncertainty: Residual Bootstrapping
   └── 100 realizations per prediction
```

---

## SLIDE 6: Feature Engineering

**Petrophysical Features (Aggregated per Well)**

| Feature Type | Examples | Aggregation |
|--------------|----------|-------------|
| Porosity | phi | mean, std, min, max |
| Permeability | perm | mean, std, min, max |
| Acoustic | AI, SI, Vp, Vs | mean, std, min, max |
| Lithology | GR, facies | mean, distribution |
| Density | rho_b, rho_f, rho_m | mean, std |
| Moduli | K, G (dry, sat) | mean, std |

**Derived Rock Quality Features (Industry Expert Advice):**

| Feature | Formula | Interpretation |
|---------|---------|----------------|
| **phi_perm_product** | phi × log(perm) | Flow productivity |
| **rock_quality** | phi / GR | Clean sand index |
| **impedance_ratio** | AI / SI | Lithology contrast |
| **net_to_gross** | 1 - facies_5% - facies_6% | Sand vs shale ratio |
| **storage_capacity** | phi × depth_range | Pore volume proxy |
| **flow_quality** | log(perm) / GR | Flow per unit shaliness |

**Analog Well Similarity (Nataly's Insight):**

| Feature | Description |
|---------|-------------|
| **analog_similarity** | 1 / (1 + min_distance_to_good_producer) |
| **analog_production_proxy** | Weighted avg production of similar good wells |

*"Look for correlation of known wells in good sand/rock that historically produced oil to the training wells."*

---

## SLIDE 7: MICE Imputation

**Why MICE? (Recommended by Hackathon Host)**

Missing data: ~7.3% across petrophysical features

| Simple Imputation | MICE + CART |
|-------------------|-------------|
| Uses median/mean only | Uses ALL features to predict missing |
| Ignores correlations | Preserves correlations |
| Can distort relationships | Maintains data structure |
| One estimate | Per SPE 218890 (Abdulkhaleq et al. 2024) |

**Key Innovation: MICE at Depth Level**
- Applied BEFORE aggregation (per Van Buuren 2018)
- All 21 depth measurements contribute to well statistics
- Preserves phi-perm-GR correlations

---

## SLIDE 8: Model Training (Interactive Experiments)

**Experiment Framework (Dr. Pyrcz's Advice: "Try simplest things first")**

| Option | Choices | Purpose |
|--------|---------|---------|
| Model Type | Linear, Ridge, Random Forest | Baseline → Complex |
| Normalize | StandardScaler (ON by default) | Fair feature comparison |
| Sand Map | Include / Exclude / Smooth 3×3 | Handle deliberate noise |

**Experiment Comparison:**
- Each experiment saved as `solution_{experiment_name}.csv`
- Compare CV R² and prediction distributions
- Select best configuration for final submission

**Cross-Validation Results:**
- CV R²: [Value from training]
- Features used: ~80+ after engineering

---

## SLIDE 9: Uncertainty Quantification

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

## SLIDE 10: Results

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
- Good rock wells have higher predicted production

---

## SLIDE 11: Value Proposition

**Why Our Solution?**

| Traditional Decline Curves | Our ML Approach |
|---------------------------|-----------------|
| Needs production history | Works with well logs only |
| Point estimates | Uncertainty quantification |
| Manual parameter selection | Automated optimization |
| Single analog well | Learns from 71 wells |

**Novel Data Analytics:**
1. **MICE + CART Imputation** - Per SPE 218890, applied at depth level
2. **Rock Quality Features** - Industry expert-driven feature engineering
3. **Interactive Experiments** - Multiple model configurations compared
4. **StandardScaler Normalization** - Dr. Pyrcz's recommendation
5. **Spatial Integration** - Sand proportion map with noise handling

---

## SLIDE 12: Conclusion

**Summary:**
- Complete ML pipeline for oil production prediction
- Industry expert insights incorporated (rock quality focus)
- MICE imputation at depth level (academically correct)
- Interactive experimentation for model selection
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
- + Derived rock quality features (6)
- Total: ~80+ features

**Key References:**
- SPE 218890 (Abdulkhaleq et al. 2024) - MICE + CART
- Van Buuren (2018) - MICE before aggregation
- Hallam et al. (2022) - Multivariate imputation for well logs

**Compute Time:**
- Data loading & MICE: ~5 seconds
- Aggregation & feature engineering: ~2 seconds
- Model training (5-fold CV): ~10 seconds
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
