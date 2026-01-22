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

### 2. TeamName.ipynb ⚠️ TEMPLATE READY (Needs 2026 Data)
- [ ] **CRITICAL: Rename file to `<TeamName>.ipynb`** (exact team name from registration)
- [ ] **CRITICAL: Match Hackathon_ProjectTemplate.ipynb structure exactly**
- [x] Markdown explanations for each section
- [x] Visualizations code ready (actual vs predicted, feature importance)
- [x] Introduction and conclusion sections
- [x] Methodology decisions documented
- [ ] Fill in 2026 results when data arrives
- [ ] Test runs start-to-finish without errors

### 3. TeamName.pptx ⚠️ OUTLINE READY (Needs 2026 Data)
- [ ] **CRITICAL: Use official Hackathon_PresentationTemplate.pptx as base**
- [ ] **CRITICAL: Rename file to `<TeamName>.pptx`** (exact team name)
- [ ] Slide 1: Title + **ALL team member full names** (REQUIRED)
- [x] Slide 2: Problem statement (content ready in outline)
- [x] Slide 3: Dataset overview & challenges (content ready)
- [x] Slides 4-5: Methodology - separate models, uncertainty (content ready)
- [ ] Slides 6-7: Results (fill with 2026 accuracy metrics, charts)
- [x] Slide 8: Uncertainty interpretation (content ready)
- [x] Slide 9: Conclusions & real-world impact (content ready)
- [x] Backup slides outlined for Q&A

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

### File Naming (CRITICAL - Will Be Rejected If Wrong)
- [ ] Notebook renamed to `<TeamName>.ipynb` (exact registered team name)
- [ ] Presentation renamed to `<TeamName>.pptx` (exact registered team name)
- [ ] solution.csv named exactly `solution.csv` (no changes)

### Format Compliance
- [ ] solution.csv has exact required columns (no extra columns, no index)
- [ ] solution.csv has exactly 63 rows (or correct count for 2026 data)
- [ ] Notebook follows Hackathon_ProjectTemplate.ipynb structure
- [ ] Presentation uses Hackathon_PresentationTemplate.pptx styling

### Content Verification
- [ ] All team member FULL NAMES on presentation title slide
- [ ] Notebook runs start-to-finish without errors
- [ ] All 100 uncertainty realizations present in solution.csv

### Submission
- [ ] All files committed to hackathon GitHub repo (not personal repo)
- [ ] Push BEFORE noon deadline
- [ ] Verify files appear correctly on GitHub
- [ ] Practice presentation timing (~10 minutes)

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
