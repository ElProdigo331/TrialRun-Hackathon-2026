# Energy AI Hackathon 2026 - Oil Production Prediction

## Overview
This project aims to predict the 3-year cumulative oil production in barrels (BBL) for 12 preproduction wells using machine learning techniques. The core challenge involves aggregating multi-row depth data for each well into a single representation suitable for predictive modeling. The project incorporates advanced data imputation, feature engineering, and robust model training with uncertainty quantification, all wrapped in an interactive Streamlit application. The ultimate goal is to provide accurate point predictions and 100 uncertainty realizations, aiding in strategic energy investment decisions and optimizing resource allocation.

## User Preferences
I want iterative development.
I prefer detailed explanations.
Ask before making major changes.

## System Architecture

### UI/UX Decisions
The application features a Streamlit interface designed for interactive exploration and model building. It includes a comprehensive AI Assistant providing industry benchmarks, session-aware guidance, and optimal setting recommendations based on extensive empirical testing. The interface also provides detailed visualization for data analysis, model evaluation, and uncertainty calibration.

### Technical Implementations
- **Data Preprocessing:** MICE (Multivariate Imputation by Chained Equations) with CART is applied at the depth level before aggregation to preserve data correlations. Aggregation involves calculating mean, standard deviation, min, max for numerical features, and facies distribution percentages per well.
- **Feature Engineering:** A rich set of 19 features across petrophysical, industry-standard, derived rock quality, analog well similarity, spatial proximity, and best zone categories are engineered to capture complex geological and production characteristics.
- **Model Training:** An interactive framework allows selection of various model types (Linear Regression, Ridge, Random Forest, XGBoost) with hyperparameter tuning via Optuna. Key metrics include R², MAE, RMSE, and OOB scores. SHAP values are used for model interpretability.
- **Uncertainty Quantification:** Two methods are implemented: Residual Bootstrap and Bagging Ensemble (using `sklearn.ensemble.BaggingRegressor` with 100 estimators) to generate 100 uncertainty realizations (R1-R100).
- **Experiment Tracking:** All experiment outputs, including model configurations and predictions, are auto-saved and tracked on an interactive leaderboard.
- **AI Assistant:** Provides expert guidance, industry benchmarks for performance evaluation, session context awareness, suggested questions, and data-backed optimal settings recommendations.
- **Scholarly Analysis:** A dedicated section provides peer-reviewed citations and methodological justifications for the approaches used.

### Feature Specifications
- **Target Variable:** 3-year cumulative oil production (BBL).
- **Input Data:** Well log data (petrophysical measurements at various depths), production history, and 2D sand proportion map.
- **Output:** `solution.csv` with `Well_ID`, `Prediction_BBL`, and 100 uncertainty realizations (`R1` to `R100`).

### System Design Choices
- **MICE Imputation:** Applied at the depth level to maintain feature relationships, adhering to established research.
- **Feature Normalization:** `StandardScaler` is critically applied to equalize feature scales and improve model performance.
- **Rock Quality Analysis:** Integrated based on industry expert insights, including RQI, FZI, and derived rock quality metrics.
- **Spatial Features:** Incorporates well coordinates (X, Y) and `sand_proportion` from a 2D map, along with proximity features, to account for geological heterogeneity.
- **Best Zone Features:** Extracts features from the "pay zone" (best rock quality depth) to preserve critical depth-level heterogeneity.
- **Model Selection Rationale:** Ridge Regression is identified as the optimal model for this small dataset (n=71) due to its lower variance, outperforming tree-based models.

## External Dependencies
- **streamlit:** For building the interactive web application.
- **pandas:** For data manipulation and analysis.
- **numpy:** For numerical operations.
- **scikit-learn:** For various machine learning algorithms, preprocessing, and model evaluation (e.g., `BaggingRegressor`, `StandardScaler`).
- **scipy:** Utilized for MICE imputation and smoothing functions.
- **matplotlib:** For static data visualizations.
- **seaborn:** For enhanced statistical data visualizations.
- **plotly:** For interactive plots.
- **optuna:** For hyperparameter optimization.
- **openai:** Integrated for the AI assistant functionality.
- **mlxtend:** Used for stepwise feature selection.