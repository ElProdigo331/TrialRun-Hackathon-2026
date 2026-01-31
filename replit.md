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

1. **Data Loading, MICE & Aggregation** - Load data, apply MICE at depth level (before aggregation), then aggregate
2. **Data Quality Verification** - Verify MICE was applied correctly, check for remaining missing values
3. **Exploratory Data Analysis** - Visualize targets, features, correlations
4. **Feature Engineering** - phi_perm_product, rock_quality, impedance_ratio
5. **Model Training** - Random Forest with Optuna auto-tuning
6. **Generate Solution** - Point estimates + 100 realizations via residual bootstrapping
7. **AI Assistant** - Chat interface for ML guidance

**Note:** MICE is now applied at the depth level BEFORE aggregation per Van Buuren (2018) and Hallam et al. (2022). This preserves correlations and ensures all depth measurements contribute.

---

## Key Techniques (Host Recommended)

1. **MICE Imputation** - Multivariate Imputation by Chained Equations
   - Preserves correlations between features
   - Better than simple median imputation for 7-10% missing data

2. **Shapley Values** - Feature importance interpretation
   - Available in EDA section

3. **Correlation Analysis** - Identify top predictors

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
├── outputs/                  # Generated outputs
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

### Derived:
- **sand_proportion** - From 2D seismic map
- **phi_perm_product** - phi × log(perm)
- **rock_quality** - phi / GR
- **impedance_ratio** - AI / SI
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
- matplotlib
- seaborn
- plotly
- optuna
- openai (for AI assistant)
