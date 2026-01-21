# Energy AI Hackathon 2026 - ML Workflow Application

## Overview
This is a complete machine learning workflow application for the Energy AI Hackathon. It provides an interactive Streamlit interface to predict energy usage (Grid kWh, Diesel gal, CNG MMBTU) for hydraulic fracturing operations, including uncertainty quantification with 100 realizations per prediction.

## Quick Start

### Option 1: Run on Replit (Recommended)
1. Import this project to Replit
2. Click "Run" - the app will start automatically
3. Open the Webview to access the application

### Option 2: Run Locally
```bash
# Install dependencies
pip install streamlit pandas numpy scikit-learn matplotlib seaborn plotly

# Run the application
streamlit run app.py --server.port 5000
```

## How to Use the App

### Step 1: Load Data
- Check "Use 2025 training data" and "Use 2025 test data"
- Click the red "Load Data" button
- Review the data statistics and preview

### Step 2: Clean the Data
- Navigate to "2. Data Cleaning & Imputation" in the sidebar
- Click "Apply Imputation" to fill missing values
- Confirm success message appears

### Step 3: Explore the Data (Optional)
- Navigate to "3. Exploratory Data Analysis"
- Browse charts showing energy distributions, correlations, and patterns

### Step 4: Feature Engineering
- Navigate to "4. Feature Engineering"
- Click "Apply Feature Engineering"
- Creates: Time_Overrun, Total_Pumping_Time, Clusters_per_Stage

### Step 5: Train Models
- Navigate to "5. Model Training"
- Adjust settings if desired (defaults work well)
- Click "Train Models" and wait for completion
- Review performance metrics and feature importance

### Step 6: Generate Predictions
- Navigate to "7. Generate Predictions"
- Click "Generate Predictions"
- Download the solution CSV file

## Project Structure

```
/
├── app.py                    # Main Streamlit application
├── data/                     # Data files
│   ├── HackathonData2025.csv # Training data (1,082 wells)
│   ├── testing.csv           # Test data (50 wells)
│   └── solution.csv          # Reference solution format
├── outputs/                  # Generated predictions
│   └── solution.csv          # Your submission file
├── attached_assets/          # Original uploaded resources
├── .streamlit/
│   └── config.toml           # Streamlit configuration
├── replit.md                 # Project documentation
└── README.md                 # This file
```

## Data Description

### Training Data Features
| Feature | Description |
|---------|-------------|
| Well Name | Unique well identifier |
| # Stages | Number of fracturing stages |
| # Clusters | Perforation clusters per stage |
| Estimated Average Stage Time | Planned time per stage (minutes) |
| Actual Average Stage Time | Actual time per stage (minutes) |
| Frac Fleet | Equipment identifier |
| Fleet Type | Power source type (Grid, Diesel, Turbine, DGB) |
| Target Formation | Geological formation |
| Field Area | Geographic location |
| Ambient Temperature | Weather conditions (F) |
| Sand Provider | Proppant supplier |

### Target Variables
| Target | Description |
|--------|-------------|
| Grid | Electrical grid energy (kWh) |
| Diesel | Diesel fuel consumption (gal) |
| CNG | Compressed natural gas (MMBTU) |

### Fleet Types and Predictions
| Fleet Type | Predicts |
|------------|----------|
| Grid | Grid (kWh) only |
| Diesel | Diesel (gal) only |
| Turbine | CNG (MMBTU) only |
| DGB | Both Diesel AND CNG |

## Solution File Format

The output CSV follows hackathon submission requirements:

| Column | Description |
|--------|-------------|
| Masked Well Name | Well identifier |
| Fuel Type | Grid, Diesel, Turbine, DGB_Diesel, or DGB_CNG |
| Fuel Value | Point estimate prediction |
| R_1 through R_100 | 100 uncertainty realizations |

## Modeling Approach

1. **Random Forest Regressor** - Robust, handles mixed features
2. **Cross-Validation** - 5-fold CV for reliable performance estimation
3. **Residual Bootstrapping** - Sample residuals to generate 100 uncertainty realizations
4. **Separate Models per Target** - Grid, Diesel, and CNG each have dedicated models

## For 2026 Hackathon

When the new hackathon data arrives:
1. Upload new training/test files in Step 1
2. Check for any new columns or changed formats
3. Run through Steps 1-7
4. Download and submit solution.csv

## Dependencies

- streamlit
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- plotly

## Team Collaboration

Each teammate can:
1. Download this zip file
2. Extract and open in Replit or run locally
3. Practice with 2025 data
4. Be ready when 2026 data arrives

## Resources

Reference materials included in `attached_assets/`:
- GeostatsPy library examples
- Python numerical demos
- Geo datasets
- Course resources from Prof. Pyrcz
