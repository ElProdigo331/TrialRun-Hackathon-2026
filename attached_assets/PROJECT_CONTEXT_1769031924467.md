# Energy A.I. Hackathon 2025 - Complete Project Context

## Purpose of This Document
This document provides a complete record of all work performed on this project, the reasoning behind each decision, and the current state of the codebase. It is designed to be easily ingested by AI assistants (ChatGPT, Claude, etc.) to continue development or answer questions about the project.

---

## Project Goal
Build a machine learning workflow for the Energy A.I. Hackathon 2025 to predict energy usage during hydraulic fracturing (fracking) operations for 50 wells. 

**Prediction Targets:**
- Grid (kWh) - electrical grid energy
- Diesel (gal) - diesel fuel consumption
- CNG (MMBTU) - compressed natural gas consumption

**Deliverables:**
- Point estimates for each target
- 100 uncertainty realizations per prediction (for probabilistic forecasting)

---

## Data Overview

### Training Data: `data/HackathonData2025.csv`
- 1,082 wells with historical energy usage
- Contains features about well operations, equipment, and conditions

### Test Data: `data/testing.csv`
- 50 wells requiring predictions
- Same features as training data, but missing target values

### Key Features Used:
- `Number of Stages` - number of fracturing stages
- `Number of Clusters` - perforation clusters per stage
- `Estimated Avg Stage Time (min)` - time per stage
- `Ambient Temperature (F)` - weather conditions
- `Frac Fleet` - equipment identifier
- `Fleet Type` - type of power source (Grid, Diesel, Turbine, DGB)
- `Target Formation` - geological formation being fractured
- `Field/Area` - geographic location
- `Sand Provider` - proppant supplier

---

## Modeling Strategy

### Fuel Type Segmentation
Different well types use different energy sources, so we built separate models:

| Fleet Type | Predicts | Reasoning |
|------------|----------|-----------|
| Grid | Grid (kWh) only | Electric-powered wells |
| Diesel | Diesel (gal) only | Diesel generator-powered wells |
| Turbine | CNG (MMBTU) only | Natural gas turbine-powered wells |
| DGB (Dual-fuel Gas Blend) | BOTH Diesel AND CNG | Hybrid wells use both fuels |

### Why This Approach?
- Avoids predicting zero values for unused fuel types
- Each model specializes in its fuel type's patterns
- DGB wells require two output rows in the solution file

### Model Type
- **Random Forest Regressor** with hyperparameter tuning
- Cross-validation for robust performance estimation
- Chosen for its ability to handle mixed feature types and resistance to overfitting

---

## Uncertainty Quantification

### Method: Residual Bootstrapping
1. Train model and compute residuals on cross-validation folds
2. For each prediction, sample 100 residuals with replacement
3. Add sampled residuals to point estimate to create 100 realizations

### Why This Method?
- Captures model prediction uncertainty
- Computationally efficient
- Produces realistic spread based on actual model errors
- Required by hackathon submission format

---

## Project Structure

```
/
├── data/                          # Data files
│   ├── HackathonData2025.csv      # Training data (1,082 wells)
│   └── testing.csv                # Test data (50 wells)
│
├── src/                           # Modular Python code
│   ├── features.py                # Feature engineering functions
│   └── modeling.py                # Model training and prediction
│
├── notebooks/                     # Jupyter notebooks
│   └── main_workflow.ipynb        # Primary analysis notebook
│
├── outputs/                       # Generated outputs
│   └── solution.csv               # Final predictions with realizations
│
├── images/                        # Visualization assets
│
├── energy_hackathon_workflow.ipynb    # Original workflow notebook
├── Hackathon_ProjectTemplate.ipynb    # Hackathon template
├── Hackathon_PresentationTemplate.pptx
├── Data_Description2025.pdf
├── requirements.txt               # Python dependencies
├── README.md                      # Project documentation
└── replit.md                      # Replit configuration notes
```

### Why This Structure?
- **Modular code in `src/`**: Enables code reuse and testing
- **Notebooks in `notebooks/`**: Separates exploration from production code
- **Data in `data/`**: Clear data organization
- **Outputs in `outputs/`**: Easy to find generated files
- **Plug-and-play for 2026**: Drop in new data files and run

---

## Key Code Components

### `src/features.py`
Contains feature engineering functions:
- `create_features(df)` - Main feature transformation pipeline
- Handles categorical encoding
- Creates interaction features
- Manages missing values

### `src/modeling.py`
Contains modeling functions:
- `train_model(X, y)` - Trains Random Forest with cross-validation
- `predict_with_uncertainty(model, X, residuals, n_realizations=100)` - Generates point estimates and uncertainty realizations
- `generate_solution(test_df, models, residuals)` - Creates final submission file

---

## Solution File Format

The `solution.csv` follows hackathon submission requirements:

| Column | Description |
|--------|-------------|
| Well ID | Unique well identifier |
| Fuel Type | Grid, Diesel, or CNG |
| Point Estimate | Single best prediction |
| Real_1 through Real_100 | 100 uncertainty realizations |

**Note:** DGB wells have TWO rows (one for Diesel, one for CNG), resulting in 63 total rows for 50 wells.

---

## Development Environment

### Platform: Replit
- Python 3.11
- JupyterLab running on port 5000
- Connected to GitHub for version control

### Dependencies (requirements.txt):
```
numpy
pandas
scikit-learn
matplotlib
seaborn
jupyterlab
```

---

## GitHub Integration

### Repository
`github.com/ElProdigo331/UT-Energy-AI-Hackathon-2026`

### Workflow
1. Make changes in Replit
2. Open Git panel (Tools → Git)
3. Commit changes with descriptive message
4. Push to origin/main
5. Teammates can pull updates or import via Replit

### Team Collaboration
Teammates can create their own Replit workspace from the repo:
```
https://replit.com/github/ElProdigo331/UT-Energy-AI-Hackathon-2026
```

---

## Decisions and Rationale Log

| Decision | Rationale |
|----------|-----------|
| Separate models per fuel type | Avoids zero-value predictions, improves accuracy |
| Random Forest | Handles mixed features, robust to overfitting |
| Residual bootstrapping for uncertainty | Computationally efficient, realistic uncertainty |
| Modular `src/` structure | Enables 2026 hackathon reuse |
| GitHub integration | Team collaboration and version control |
| Cross-validation | Robust performance estimation |

---

## How to Run the Project

1. **Open JupyterLab** - Access via Replit webview (port 5000)
2. **Open `notebooks/main_workflow.ipynb`** or `energy_hackathon_workflow.ipynb`
3. **Run all cells** - Trains models and generates predictions
4. **Find output** - `outputs/solution.csv` contains final submission

---

## For 2026 Hackathon Adaptation

1. Replace `data/HackathonData2025.csv` with new training data
2. Replace `data/testing.csv` with new test data
3. Update feature lists in `src/features.py` if columns change
4. Run the workflow notebook
5. Submit `outputs/solution.csv`

---

## Current State (January 2026)

- All models trained and validated
- Solution file generated with 63 rows (50 wells, DGB wells have 2 rows)
- Uncertainty realizations computed via residual bootstrapping
- Project connected to GitHub for team access
- Ready for submission or continued refinement

---

## Contact/Attribution

- **Replit Workspace**: data2025
- **GitHub**: ElProdigo331/UT-Energy-AI-Hackathon-2026
- **Hackathon**: Energy A.I. Hackathon 2025

---

*This document was generated to provide complete context for AI assistants and team members working on this project.*
