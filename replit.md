# Energy AI Hackathon 2026 Workflow Application

## Overview
This is a comprehensive Streamlit-based machine learning workflow application designed for the Energy AI Hackathon. It provides an end-to-end pipeline for predicting energy usage (Grid kWh, Diesel gal, CNG MMBTU) during hydraulic fracturing operations.

## Project Goal
Predict energy consumption for 50 wells with:
- Point estimates for each target (Grid, Diesel, CNG)
- 100 uncertainty realizations per prediction (probabilistic forecasting)

## Running the Application
```bash
streamlit run app.py --server.port 5000
```

## Project Structure
```
/
├── app.py                    # Main Streamlit application
├── data/                     # Data files
│   ├── HackathonData2025.csv # Training data (1,082 wells)
│   ├── testing.csv           # Test data (50 wells)
│   └── solution.csv          # Reference solution format
├── outputs/                  # Generated outputs
│   └── solution.csv          # Final predictions with realizations
├── attached_assets/          # Original uploaded files
├── data2025/                 # Extracted 2025 hackathon data
├── geostatspy/               # GeostatsPy library reference
├── pythondemos/              # Python numerical demos reference
├── geodatasets/              # Geo datasets reference
├── resources/                # Additional resources
└── .streamlit/config.toml    # Streamlit configuration
```

## Application Features

### 8-Step Workflow:
1. **Data Upload & Inspection** - Load training/test data, view statistics
2. **Data Cleaning & Imputation** - Handle missing values
3. **Exploratory Data Analysis** - Visualize distributions, correlations
4. **Feature Engineering** - Create derived features
5. **Model Training** - Train Random Forest models with cross-validation
6. **Uncertainty Quantification** - Residual bootstrapping analysis
7. **Generate Predictions** - Create solution file with 100 realizations
8. **Quick Start Guide** - Instructions for 2026 hackathon

## Modeling Strategy

### Fuel Type Segmentation:
| Fleet Type | Predicts | Reasoning |
|------------|----------|-----------|
| Grid | Grid (kWh) only | Electric-powered wells |
| Diesel | Diesel (gal) only | Diesel generator-powered wells |
| Turbine | CNG (MMBTU) only | Natural gas turbine-powered wells |
| DGB | BOTH Diesel AND CNG | Hybrid wells use both fuels |

### Key Features:
- Number of Stages, Number of Clusters
- Estimated/Actual Average Stage Time
- Ambient Temperature
- Frac Fleet, Fleet Type, Target Formation, Field Area
- Engineered: Time_Overrun, Total_Pumping_Time, Clusters_per_Stage

### Uncertainty Method:
Residual bootstrapping - sample 100 residuals with replacement and add to point estimates.

## For 2026 Hackathon Adaptation

1. Upload new training/test data files
2. Review any new columns or changed formats
3. Run through Steps 1-7
4. Download solution.csv

## Dependencies
- streamlit
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- plotly

## Solution File Format
| Column | Description |
|--------|-------------|
| Masked Well Name | Well identifier |
| Fuel Type | Grid, Diesel, or CNG |
| Fuel Value | Point estimate |
| R_1 through R_100 | 100 uncertainty realizations |
