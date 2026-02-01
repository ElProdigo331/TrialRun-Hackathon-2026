# Teammate Quick Start Guide

## Getting Started in 5 Minutes

### Step 1: Set Up Your Environment

**On Replit:**
1. Create a new Replit
2. Upload/import this project
3. Click "Run"
4. Open the Webview tab

**Locally:**
```bash
pip install streamlit pandas numpy scikit-learn matplotlib seaborn plotly
streamlit run app.py --server.port 5000
```

### Step 2: Run Through the Workflow

1. **Load Data** - Click "Load Data" (2025 data is pre-selected)
2. **Clean Data** - Go to Step 2, click "Apply Imputation"
3. **Feature Engineering** - Go to Step 4, click "Apply Feature Engineering"
4. **Train Models** - Go to Step 5, click "Train Models" (takes ~1 minute)
5. **Generate Predictions** - Go to Step 7, click "Generate Predictions"
6. **Download** - Click "Download Solution CSV"

## Understanding the Problem

We're predicting energy usage during hydraulic fracturing ("fracking") operations:

- **Grid (kWh)** - Electricity from the power grid
- **Diesel (gal)** - Diesel fuel consumption
- **CNG (MMBTU)** - Compressed natural gas

Each well has a "Fleet Type" that determines which energy sources it uses:
- **Grid wells** → Only use electricity
- **Diesel wells** → Only use diesel
- **Turbine wells** → Only use CNG
- **DGB wells** → Use BOTH diesel AND CNG (dual-fuel)

## The Solution File

The judges want a CSV with:
- Well name
- Fuel type label
- Point estimate (your best prediction)
- 100 "realizations" (uncertainty estimates)

DGB wells get TWO rows (one for diesel, one for CNG), so 50 test wells = 63 output rows.

## Key Machine Learning Concepts

**Random Forest** - An ensemble of decision trees that vote on predictions. Robust and handles messy data well.

**Cross-Validation** - Split data into folds, train on some, test on others. Gives reliable performance estimates.

**Residual Bootstrapping** - After predicting, look at the errors (residuals). Sample from these errors 100 times and add to predictions to show uncertainty.

## What You Can Explore

- **Step 3 (EDA)** - See charts of energy distributions and feature correlations
- **Step 5** - Adjust model settings (number of trees, max depth)
- **Step 6** - Understand the uncertainty in predictions

## When 2026 Hackathon Starts

1. Download the new training and test data files
2. Upload them in Step 1 instead of using 2025 data
3. Run through the same workflow
4. Download and submit!

## Questions?

The app is designed to be self-explanatory. Click through each step and read the descriptions. The "Quick Start" page (Step 8) has a summary of the entire process.
