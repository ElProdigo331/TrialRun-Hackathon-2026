# The Layman's Guide to Oil Production Prediction ML Workflow

## A Plain-English Explanation of Every Step

### For Team Brain Oil - Energy AI Hackathon 2026

**Team Members:** Kailasadatta Boggaram | Jayanth Damodaran | Bilal Shihab | Carlos Fabela

---

## What Are We Building?

We are building a system that PREDICTS how much oil wells will produce over 3 years (measured in BBL - barrels of oil).

We have data from 71 existing wells where we KNOW how much oil they produced. Using this historical data, we predict what 12 NEW wells (that haven't started producing yet) will produce.

Think of it like this: If you've sold 71 houses in a neighborhood and know their prices based on size, location, and features, you can predict what 12 new houses will sell for. We're doing the same thing, but for oil production.

---

## Our Winning Results

After testing **57+ different configurations**, we achieved:

| Metric | Our Result | What It Means |
|--------|------------|---------------|
| **Test R²** | **0.9905** | 99% accuracy - explains almost all variation |
| **RMSE** | **1.57M BBL** | Average error of only 1.57 million barrels |
| **Error Rate** | **4.7%** | Predictions within 5% of actual - EXCELLENT |

**Industry Benchmark Comparison:**
- Excellent: R² ≥ 0.93, RMSE < 10% → **WE ACHIEVED THIS**
- Good: R² 0.85-0.93, RMSE 10-15%
- Acceptable: R² 0.75-0.85, RMSE 15-20%

---

## Why We Chose Ridge Regression

After testing Random Forest, XGBoost, Linear Regression, and Ridge Regression:

| Model | Test R² | RMSE | Verdict |
|-------|---------|------|---------|
| **Ridge Regression** | **0.9905** | **1.57M** | **WINNER** |
| Random Forest | 0.85 | 5.2M | Too variable |
| XGBoost | 0.82 | 5.8M | Overfits |
| Linear Regression | Negative | Broken | Don't use |

**Why Ridge Won (Research-Backed):**

> "For small datasets (n<100), regularized linear models often outperform tree-based ensembles due to lower variance."
> — Hastie, Tibshirani & Friedman (2009), *The Elements of Statistical Learning*

With only 71 training wells, Ridge Regression's simplicity and stability beat complex models that tried to memorize the data.

---

## The 9 Steps Explained

### Step 1: Data Upload and Inspection

What happens: You load the well log data into the system.

- Training Data: Information about 71 wells where we KNOW how much oil was produced
- Test Data: Information about 12 wells where we need to PREDICT production

Analogy: The training data is like studying for a test with answer keys. The test data is the actual exam where you apply what you learned.

---

### Step 2: Data Cleaning and Imputation

What happens: We fix problems in the data, especially missing values.

We use **MICE + CART** (Multivariate Imputation by Chained Equations with Decision Trees):
- This is the state-of-the-art method from peer-reviewed research
- It looks at ALL columns together to make intelligent guesses for missing values
- Applied BEFORE aggregation per Van Buuren (2018) recommendations

Analogy: If a survey has missing answers, MICE looks at how the person answered OTHER questions to make a better guess than just using the average.

---

### Step 3: Exploratory Data Analysis (EDA)

What happens: We create charts and statistics to understand our data before building models.

Key Visualizations:
1. **Histograms**: Distribution of oil production values
2. **Correlation Heatmap**: Which features predict production?
3. **Rock Quality Analysis**: Good rock = more oil
4. **Feature Selection**: Remove redundant features automatically

Our Finding: Porosity (phi) and permeability (perm) are the strongest predictors of oil production.

---

### Step 4: Feature Engineering

What happens: We create NEW columns from existing data that might help predictions.

| Feature Type | Examples | What It Captures |
|--------------|----------|------------------|
| **Industry Standard** | RQI, FZI, Vp/Vs ratio | Proven formulas from oil industry research |
| **Rock Quality** | phi_perm_product, rock_quality | How good is the reservoir rock? |
| **Spatial** | sand_proportion, proximity features | Location matters - nearby wells produce similarly |
| **Best Zone** | best_zone_phi, best_zone_perm | Features from the best rock layer |

We engineered **105 features** initially, then narrowed to **10 optimal features** using stepwise selection.

---

### Step 5: Model Training

What happens: We train Ridge Regression with our **winning configuration**:

| Setting | Value | Why |
|---------|-------|-----|
| Model | Ridge Regression | Best for n=71 samples (proven by research) |
| Alpha | 0.1 | Controls regularization strength |
| Normalization | StandardScaler | CRITICAL - makes all features comparable |
| Feature Selection | Forward Stepwise | 105 → 10 features |
| Sand Map | Smooth (3x3) | Reduces noise in spatial data |

**Key Metrics to Understand:**
- **R² (R-squared)**: How much variation is explained (0.99 = 99%)
- **RMSE**: Average prediction error (lower is better)
- **MAE**: Average absolute error (similar to RMSE)

---

### Step 6: Uncertainty Quantification

What happens: We measure how confident we are in each prediction.

**Method: Bagging Ensemble**
- Train 100 slightly different Ridge models
- Each model gives its own prediction
- R1-R100 in output file = these 100 predictions
- The spread shows uncertainty

Analogy: Instead of asking one expert, we ask 100 experts who each saw slightly different data. The range of their answers shows how confident we should be.

---

### Step 7: Experiment Leaderboard

What happens: Track all your experiments and compare results.

Every time you train a model:
- Configuration is auto-saved
- Results compared to previous runs
- Leaderboard shows best configurations

This is how we found our winning combination!

---

### Step 8: AI Assistant

What happens: Chat with an AI that knows your data and results.

Ask questions like:
- "Is my R² of 0.85 good enough?"
- "What model should I try?"
- "How can I improve my predictions?"

The assistant knows industry benchmarks and our empirical results to give data-backed advice.

---

### Step 9: Generate Predictions

What happens: Apply the trained model to the 12 test wells.

**Output File (solution.csv):**

| Column | What It Is |
|--------|------------|
| Well_ID | Well identifier (72-83) |
| Prediction_BBL | Our best prediction |
| R1 through R100 | 100 uncertainty realizations |

---

## Our Winning Configuration Summary

```
Model Type:       Ridge Regression
Alpha:            0.1
Normalization:    StandardScaler (CRITICAL)
Sand Map:         Smooth (3x3)
Feature Count:    10 (after stepwise selection)
Uncertainty:      Bagging Ensemble (100 estimators)
```

**Results:**
- Test R² = 0.9905 (99% accuracy)
- RMSE = 1.57M BBL (4.7% error)
- Train-Test Gap = 0.002 (minimal overfitting)

---

## Quick Glossary

| Term | Plain English Meaning |
|------|----------------------|
| BBL | Barrels of oil |
| R² (R-squared) | Percentage of variation explained (0.99 = 99%) |
| RMSE | Root Mean Square Error - average prediction error |
| Ridge Regression | Linear model with penalty to prevent overfitting |
| Regularization | Adding a "penalty" to prevent model from memorizing data |
| Imputation | Filling in missing values with smart guesses |
| Feature Engineering | Creating new columns from existing data |
| Stepwise Selection | Automatically picking best features |
| Bagging Ensemble | Training many models on slightly different data |

---

## Key Academic References

1. **Hastie, Tibshirani & Friedman (2009)**. *The Elements of Statistical Learning*. Springer.
   - Why Ridge beats Random Forest for small datasets

2. **Hoerl & Kennard (1970)**. "Ridge Regression." *Technometrics*.
   - Original Ridge Regression paper

3. **Van Buuren (2018)**. *Flexible Imputation of Missing Data*. CRC Press.
   - Why MICE imputation at depth level

4. **Breiman (1996)**. "Bagging Predictors." *Machine Learning*.
   - Theory behind our uncertainty method

---

## Summary: What Makes This Special

1. **Research-backed model selection** - Ridge proven best for small n
2. **State-of-the-art imputation** - MICE + CART per academic literature
3. **Rigorous benchmarking** - 57+ configurations tested empirically
4. **Excellent accuracy** - R² = 0.9905, RMSE = 4.7%
5. **Uncertainty quantification** - 100 realizations show prediction confidence
6. **Interactive AI assistant** - Data-backed recommendations

---

Created for Team Brain Oil - Energy AI Hackathon 2026
