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
3. **Exploratory Data Analysis** - 4 tabs: Target Analysis, Feature Analysis, Correlations, **Rock Quality Analysis**
4. **Feature Engineering** - 6 rock quality features (see below)
5. **Model Training** - Interactive model selection (Linear/Ridge/RF), normalization toggle, sand map handling
6. **Generate Solution** - Point estimates + 100 realizations, experiment tracking & comparison
7. **AI Assistant** - Chat interface for ML guidance

**Note:** MICE is now applied at the depth level BEFORE aggregation per Van Buuren (2018) and Hallam et al. (2022). This preserves correlations and ensures all depth measurements contribute.

---

## Industry Expert Insights (Incorporated)

From 4 industry heads at the hackathon:

| Insight | Implementation |
|---------|----------------|
| "Good rock = more oil" | Rock Quality Analysis tab, Good/Bad rock classification |
| "Time & location matter" | X, Y spatial correlations, sand_proportion map |
| "Format features correctly" | StandardScaler normalization, derived rock quality metrics |

---

## Key Techniques (Host Recommended)

1. **MICE Imputation** - Multivariate Imputation by Chained Equations
   - Uses DecisionTreeRegressor (CART) per SPE 218890
   - Applied at depth level before aggregation

2. **Feature Normalization** - StandardScaler (Dr. Pyrcz's advice)
   - Equalizes feature scales for fair comparison

3. **Rock Quality Analysis** - Industry expert recommendation
   - Good vs Bad rock classification
   - Production correlation by rock class

---

## Interactive Experiment Framework (Step 5)

| Option | Choices | Purpose |
|--------|---------|---------|
| **Model Type** | Linear Regression, Ridge, Random Forest | Try simple first (Dr. Pyrcz) |
| **Normalize Features** | Checkbox (default: ON) | StandardScaler |
| **Sand Map Handling** | Include, Exclude, Smooth (3x3) | Handle noisy sand map (Dinghan Wang) |
| **Experiment Name** | Auto-generated | Labels output files for comparison |

**Experiment outputs saved to:** `outputs/solution_{experiment_name}.csv`

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
├── outputs/                  # Generated outputs (experiment CSVs)
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

### Derived Rock Quality Features:
| Feature | Formula | Interpretation |
|---------|---------|----------------|
| **phi_perm_product** | phi × log(perm) | Flow productivity (higher = better) |
| **rock_quality** | phi / GR | Clean sand index (higher = cleaner) |
| **impedance_ratio** | AI / SI | Lithology contrast |
| **net_to_gross** | 1 - facies_5% - facies_6% | Sand vs shale ratio |
| **storage_capacity** | phi × depth_range | Total pore volume proxy |
| **flow_quality** | log(perm) / GR | Flow per unit shaliness |

### Analog Well Similarity (Nataly's Insight):
| Feature | Description |
|---------|-------------|
| **analog_similarity** | 1 / (1 + min_distance_to_good_producer) |
| **analog_production_proxy** | Weighted avg production of similar good wells |

Nataly (Hackathon Architect): "Look for correlation of known wells in good sand/rock that historically produced oil to the training wells."

### Spatial Proximity Features (User Insight):
| Feature | Description |
|---------|-------------|
| **spatial_proximity_score** | 1 / (1 + XY_distance_to_high_producer) |
| **spatial_production_estimate** | Weighted avg production based on XY proximity |
| **high_producers_nearby** | Count of high producers within 30 grid units |

User Insight: "Center region = low production, left/bottom = high production"

### Best Depth Features (Preserve Depth Variation):
Instead of just averaging rock properties, we capture the best intervals:

| Feature | Description |
|---------|-------------|
| **best_phi_value** | Highest porosity at any depth |
| **best_perm_value** | Highest permeability at any depth |
| **perm_at_best_phi** | Permeability at the highest-porosity depth |
| **phi_at_cleanest** | Porosity at the cleanest sand depth (lowest GR) |
| **pay_zone_fraction** | Fraction of depths with good rock (high φ, high k, low GR) |

Rationale: Some depths have very different rock features - the best intervals matter most for production.

### Spatial:
- **X, Y** - Well coordinates
- **sand_proportion** - From 2D seismic map
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
- scipy (for MICE, smoothing)
- matplotlib
- seaborn
- plotly
- optuna
- openai (for AI assistant)

---

## Scholarly Analysis Page

Available at sidebar option "Scholarly Analysis" - includes:
- 15+ peer-reviewed citations
- Methodology justification
- Downloadable HTML for print-to-PDF
