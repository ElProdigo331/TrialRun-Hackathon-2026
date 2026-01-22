# Energy AI Hackathon 2026 Presentation Outline

## SLIDE 1: Title
**Energy Usage Prediction for Hydraulic Fracturing Operations**

[TEAM NAME]

Team Members:
- [Member 1] - [Affiliation]
- [Member 2] - [Affiliation]
- [Member 3] - [Affiliation]
- [Member 4] - [Affiliation]

Energy AI Hackathon 2026 | January 23-25, 2026

---

## SLIDE 2: Problem Statement

**The Challenge**

Predict energy consumption during hydraulic fracturing ("fracking") operations for 50 test wells:
- Grid electricity (kWh)
- Diesel fuel (gallons)
- Compressed Natural Gas - CNG (MMBTU)

**Plus: Quantify Uncertainty**
- 100 probabilistic realizations per prediction
- Enables risk-aware decision making

**Why This Matters**
- Fuel logistics planning
- Cost optimization
- Environmental impact assessment

---

## SLIDE 3: Data Overview

**Training Data**
- [X] wells with historical energy usage
- Features: stages, clusters, stage times, fleet type, formation, temperature, etc.

**Key Insight: Fleet Types Determine Fuel Usage**

| Fleet Type | Uses Grid | Uses Diesel | Uses CNG |
|------------|-----------|-------------|----------|
| Grid       | Yes       | No          | No       |
| Diesel     | No        | Yes         | No       |
| Turbine    | No        | No          | Yes      |
| DGB        | No        | Yes         | Yes      |

This domain knowledge directly shaped our modeling approach.

---

## SLIDE 4: Our Approach - Separate Models

**Why Separate Models Per Fuel Type?**

Instead of one model predicting all fuels, we train specialized models:

1. **Grid Model** - Trained only on Grid-powered wells
2. **Diesel Model** - Trained on Diesel + DGB wells
3. **CNG Model** - Trained on Turbine + DGB wells

**Benefits:**
- No wasted predictions for unused fuel types
- Each model specializes in its fuel's patterns
- Better accuracy than single multi-output model

---

## SLIDE 5: Feature Engineering

**Domain-Informed Features**

| Feature | Formula | Rationale |
|---------|---------|-----------|
| Time_Overrun | Actual - Estimated Time | Delays affect fuel consumption |
| Total_Pumping_Time | Stages × Stage Time | Total operation duration |
| Clusters_per_Stage | Clusters / Stages | Fracturing intensity |

**Model Choice: Random Forest**
- Handles mixed feature types
- Robust to outliers
- Provides feature importance
- Proven in previous hackathons

---

## SLIDE 6: Uncertainty Quantification

**Method: Residual Bootstrapping**

1. Train model and compute residuals from cross-validation
2. For each test prediction, sample 100 residuals with replacement
3. Add sampled residuals to point estimate

**Result: 100 Realizations Per Prediction**

[INSERT: Histogram showing uncertainty distribution for sample well]

This provides operators with:
- Point estimate (most likely value)
- Uncertainty range (planning buffer)
- Full distribution for risk analysis

---

## SLIDE 7: Results - Model Performance

**Cross-Validation R² Scores**

| Model | CV R² | Interpretation |
|-------|-------|----------------|
| Grid  | [X.XX] | [Good/Moderate/Challenging] |
| Diesel | [X.XX] | [Good/Moderate/Challenging] |
| CNG | [X.XX] | [Good/Moderate/Challenging] |

[INSERT: Bar chart of CV scores]

---

## SLIDE 8: Results - Feature Importance

**Top Predictive Features**

[INSERT: Feature importance bar chart]

**Key Findings:**
- [Feature 1] most important for [fuel type]
- [Feature 2] drives [insight]
- [Domain interpretation of results]

---

## SLIDE 9: Solution Output

**Submission Format**

| Well | Fuel Type | Point Est. | R_1 ... R_100 |
|------|-----------|------------|---------------|
| Well_A | Grid | 45,230 | ... |
| Well_B | Diesel | 12,450 | ... |
| Well_C | DGB_Diesel | 8,200 | ... |
| Well_C | DGB_CNG | 156 | ... |

**Final Output:**
- 63 rows (50 wells, DGB wells have 2 rows)
- 100 uncertainty realizations each
- Ready for automated scoring

---

## SLIDE 10: Conclusions & Impact

**What We Built**
- Complete ML pipeline for energy prediction
- Separate models respecting domain logic
- Robust uncertainty quantification

**Real-World Value**
- **Cost Savings**: Avoid over-ordering fuel
- **Reliability**: Don't run short during operations
- **Planning**: Confidence intervals for logistics

**Future Improvements**
- Ensemble methods (XGBoost, Neural Networks)
- Spatial features if location data available
- Time-series patterns if temporal data added

---

## BACKUP SLIDES

### Backup 1: Data Cleaning

- Missing values: Median imputation (numeric), Mode (categorical)
- [X]% of data required imputation
- No rows dropped

### Backup 2: Hyperparameters

| Parameter | Value |
|-----------|-------|
| n_estimators | 100 |
| max_depth | 15 |
| random_state | 42 |

### Backup 3: Why Not Other Models?

- **Linear Regression**: Too simple, misses non-linear relationships
- **XGBoost**: Tried, similar performance, RF more interpretable
- **Neural Networks**: Overkill for dataset size, harder to explain

### Backup 4: Residual Analysis

[INSERT: Residual plot if available]

Residuals approximately normal, justifying bootstrapping approach.
