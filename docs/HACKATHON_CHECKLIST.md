# Energy AI Hackathon 2026 - Submission Checklist

## Judging Categories (Weight Assessment)

| Category | Weight | Description |
|----------|--------|-------------|
| **Technical Performance** | HIGH | Automated scoring of prediction accuracy |
| **Solution Design** | HIGH | Code quality, reproducibility, methodology |
| **Presentation** | HIGH | Slide deck + live presentation to judges |
| **Innovation** | MEDIUM | Creative approaches, domain integration |
| **Potential Impact** | LOW | Real-world applicability |

---

## Required Deliverables

### 1. solution.csv ✅ READY
- [x] 63 rows (50 wells, DGB wells have 2 rows)
- [x] Columns: Masked Well Name, Fuel Type, Fuel Value, R_1...R_100
- [x] 100 uncertainty realizations per prediction
- [x] Correct fuel type labels (Grid, Diesel, Turbine, DGB_Diesel, DGB_CNG)
- [ ] Regenerate with 2026 data when available

### 2. TeamName.ipynb ❌ NEEDS WORK
- [ ] Rename to actual team name
- [ ] Follow hackathon template structure
- [ ] Add markdown explanations for each section
- [ ] Include visualizations (actual vs predicted, feature importance)
- [ ] Add introduction and conclusion sections
- [ ] Document methodology decisions
- [ ] Ensure runs start-to-finish without errors

### 3. TeamName.pptx ❌ NOT STARTED
- [ ] Create using official template
- [ ] Slide 1: Title + ALL team member names
- [ ] Slide 2: Problem statement
- [ ] Slide 3: Dataset overview & challenges
- [ ] Slides 4-5: Methodology (separate models, uncertainty approach)
- [ ] Slides 6-7: Results (accuracy metrics, feature importance chart)
- [ ] Slide 8: Uncertainty interpretation
- [ ] Slide 9: Conclusions & real-world impact
- [ ] Backup slides for Q&A

### 4. README.md ✅ READY
- [x] Project description
- [x] Setup instructions
- [x] Usage guide
- [ ] Update to 2026 when data arrives
- [ ] Add team name

---

## Current Status vs Requirements

### ✅ COMPLETE
| Item | Status |
|------|--------|
| Prediction pipeline | Working with 2025 data |
| Solution file format | Matches requirements exactly |
| Uncertainty quantification | 100 realizations via residual bootstrapping |
| Separate fuel type models | Grid, Diesel, CNG models |
| Feature engineering | Time_Overrun, Total_Pumping_Time, Clusters_per_Stage |
| Cross-validation | 5-fold CV implemented |
| Modular code structure | Organized Streamlit app |
| Version control | GitHub integration ready |

### ⚠️ NEEDS WORK BEFORE HACKATHON
| Item | Priority | Notes |
|------|----------|-------|
| Jupyter notebook | HIGH | Convert Streamlit workflow to notebook format |
| Presentation slides | HIGH | Create compelling slide deck |
| Team name finalization | HIGH | Apply to all file names |
| Visualization exports | MEDIUM | Feature importance, prediction plots |
| Domain insights documentation | MEDIUM | Explain why decisions were made |

### ❌ WAIT FOR 2026 DATA
| Item | Notes |
|------|-------|
| Regenerate predictions | Run full pipeline with new data |
| Update results in notebook | Replace 2025 metrics with 2026 |
| Update presentation | Use actual 2026 results |
| Final accuracy metrics | Will be auto-scored by hackathon system |

---

## Pre-Submission Checklist (Final Day)

- [ ] Team name matches across all files
- [ ] solution.csv has exact required format (no index column)
- [ ] Notebook runs start-to-finish without errors
- [ ] All team member names on presentation title slide
- [ ] Files committed to hackathon GitHub repo
- [ ] Push before noon deadline
- [ ] Practice presentation timing

---

## Things We Can Prepare NOW

1. **Jupyter Notebook Template** - Structure with sections, ready to fill with 2026 results
2. **Presentation Outline** - Slides with methodology (won't change), placeholder for results
3. **Visualization Code** - Ready to generate plots once we have 2026 results
4. **Feature Importance Export** - Code to create publishable charts
5. **Domain Context Slides** - Explain fracking energy usage background

---

## Competition Timeline (Jan 23-25, 2026)

| Day | Focus |
|-----|-------|
| **Day 1 (Thu)** | Get 2026 data, run pipeline, verify results |
| **Day 2 (Fri)** | Iterate on models, finalize predictions, start slides |
| **Day 3 (Sat)** | Polish notebook, complete slides, practice presentation |
| **Deadline** | Submit by noon (typically day after presentations) |
