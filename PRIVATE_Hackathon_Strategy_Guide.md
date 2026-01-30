# ENERGY GLADIATORS - HACKATHON STRATEGY GUIDE

## [CONFIDENTIAL] FOR YOUR EYES ONLY - DO NOT SHARE OR INCLUDE IN SUBMISSION

---

## 1. COMPETITION TIMELINE

| When | What to Do |
|------|------------|
| Friday 5pm | Kickoff - receive 2026 dataset + GitHub repo access |
| Friday Night | Clone official repo, start fresh project, upload recipe zip as REFERENCE only |
| Saturday (Full Day) | MAIN BUILD DAY - rebuild pipeline fresh, use AI Assistant live, document everything |
| Saturday Night | Polish notebook and slides, transfer to official templates, practice presentation |
| Sunday AM | Final push to GitHub by NOON, triple-check file names, practice timing (about 10 min) |
| Sunday PM | Presentations to judges, be ready for Q&A |

---

## 2. STEP-BY-STEP EXECUTION PLAN

### Phase 1: Setup (Friday Night, about 30 min)
- Clone official hackathon GitHub repo (NOT your personal repo)
- Upload recipe zip as a REFERENCE - DO NOT copy-paste code directly
- Start a NEW app.py - rebuild the structure from scratch
- This creates commit history showing genuine development

### Phase 2: Data Exploration (Friday Night to Saturday AM)
- Upload NEW 2026 training/test data
- Use AI Assistant: "What patterns do you see in this data?"
- Document findings in notebook with markdown
- Create visualizations (histograms, box plots, correlation heatmap)

### Phase 3: Build and Adapt (Saturday)
- Rebuild pipeline using recipe as guide (9 steps)
- Ask AI Assistant: "What features should I engineer?"
- Ask AI Assistant: "Would a different model work better?"
- DOCUMENT these AI conversations - show judges you are exploring

### Phase 4: Polish (Saturday Night)
- Transfer work to official Jupyter notebook template
- Rename to EnergyGladiators.ipynb
- Add markdown explanations at EVERY step
- Create slides using official PowerPoint template
- ALL team member names on slide 1 (REQUIRED)

### Phase 5: Submission (Sunday AM)
- Generate solution.csv with 100 realizations
- VERIFY: Columns are Real_1 through Real_100 (not R_1)
- Push everything to official GitHub repo
- Final commit BEFORE noon deadline
- Push 30 minutes early for safety buffer

---

## 3. THE 9-STEP WORKFLOW

| Step | What It Does | Time Estimate |
|------|--------------|---------------|
| 1. Data Upload | Load training (1,082 wells) and test (50 wells) CSVs | 2 min |
| 2. Data Cleaning | Handle missing values (median for numbers, mode for categories) | 3 min |
| 3. EDA | Visualize distributions, correlations, patterns | 10 min |
| 4. Feature Engineering | Create Time_Overrun, Total_Pumping_Time, Clusters_per_Stage | 5 min |
| 5. Model Training | Train Random Forest for Grid, Diesel, CNG with 5-fold CV | 5 min |
| 6. Uncertainty | Residual bootstrapping for 100 realizations | 3 min |
| 7. Generate Predictions | Create solution.csv | 2 min |
| 8. Quick Start | Reference instructions | - |
| 9. AI Assistant | Ask questions, get help | Ongoing |

Total workflow time: About 30 minutes once familiar

---

## 4. CRITICAL DOMAIN KNOWLEDGE

### Fleet Type to Fuel Mapping

| Fleet Type | Uses Grid | Uses Diesel | Uses CNG |
|------------|-----------|-------------|----------|
| Grid       | YES       | NO          | NO       |
| Diesel     | NO        | YES         | NO       |
| Turbine    | NO        | NO          | YES      |
| DGB        | NO        | YES         | YES      |

Key Insight: DGB (Dual-fuel) wells generate TWO output rows (one Diesel, one CNG).
50 wells become 63 rows in solution.csv (13 DGB wells x 2 = 26, plus 37 single-fuel wells)

---

## 5. YOUR KILLER DIFFERENTIATOR

### Why Our Solution Beats Commercial Tools

"Commercial tools like Spotfire cost thousands per year and only show historical data. Our solution predicts future energy usage with uncertainty quantification - something even enterprise tools do not do. And our AI assistant means any engineer can adapt it to new problems without coding."

| Aspect | Commercial BI (Spotfire, Tableau) | Our Solution |
|--------|-----------------------------------|--------------|
| Approach | Descriptive (what happened) | PREDICTIVE (what will happen) |
| Output | Charts and dashboards | Point estimates + uncertainty ranges |
| Cost | $3,000-5,000/year license | Free (Python/Streamlit) |
| Adaptability | Fixed features | AI assistant adapts to any dataset |
| Skill Value | Vendor lock-in | Portable Python/ML skills |

The Key Differentiator:
- Spotfire = Reporting tool ("here is what your energy consumption was")
- Our solution = Intelligence tool ("here is what it will be, and how confident we are")

### Innovation Story for Judges

"We did not just build a model for the 2026 hackathon problem. We built a platform that can solve FUTURE problems. Ask it anything..."

Then do a LIVE DEMO - type a question like "What features would you suggest for this dataset?" and watch it respond.

---

## 6. AI ASSISTANT CAPABILITIES

The AI Assistant (Step 9) can now help with:

### Team Onboarding
- "Walk me through how this app works step by step"
- "What does Step 5 do?"
- "Why do DGB wells have two rows?"

### ML Guidance
- "What features should I engineer?"
- "Would XGBoost work better than Random Forest?"
- "Why is my model performing poorly?"

### Local Installation
- "How do I run this on my laptop?"
- Full step-by-step instructions for Windows/Mac/Linux

### Innovation Explanation
- "Why is this better than Spotfire?"
- "What makes this solution innovative?"

---

## 7. JUDGING CRITERIA - YOUR STRATEGY

| Criterion | Weight | Your Strategy |
|-----------|--------|---------------|
| Technical Accuracy | HIGH | Segmented models per fuel type (proven to work) |
| Reproducibility | 20 pts | random_state=42 everywhere, clean notebook, requirements.txt |
| Code Quality | 10 pts | Well-commented code, follow official template structure |
| Innovation | 15 pts | AI ASSISTANT - live, adaptive, industry-agnostic ML helper |
| Interpretability | 15 pts | Feature importance plots, domain-driven explanations |
| Presentation | HIGH | Live demo of AI Assistant answering questions |

---

## 8. CRITICAL COMPLIANCE CHECKLIST

| Requirement | Status | Action |
|-------------|--------|--------|
| Develop during hackathon | DONE | Rebuild fresh using recipe as guide |
| Use only 2026 official data | DONE | Train on new data only |
| solution.csv format | VERIFY! | Columns = Real_1 to Real_100 (not R_1) |
| Notebook = TeamName.ipynb | DONE | Rename to EnergyGladiators.ipynb |
| Slides = TeamName.pptx | DONE | All members on slide 1 |
| Push to official GitHub | DONE | Use THEIR repo, not yours |
| 100 realizations | DONE | Already implemented |
| Deadline: Noon Sunday | DONE | Push 30 min early for safety |

---

## 9. FINAL REMINDERS

### DO NOT:
- Copy-paste code from recipe (looks like pre-work)
- Use 2025 data for training (only 2026 official data)
- Forget team member names on slide 1
- Push to personal repo instead of official hackathon repo
- Include this PDF in any submission

### DO:
- Rebuild pipeline fresh - use recipe as reference only
- Use AI Assistant LIVE during hackathon (innovation demo)
- Document your thought process in notebook markdown
- Create commit history showing genuine development
- Practice your presentation (aim for about 10 minutes)
- Demo the AI Assistant answering a live question

---

## 10. RECIPE ZIP CONTENTS (Reference)

| File | Purpose |
|------|---------|
| app.py | Complete Streamlit app with 9-step workflow + AI Assistant |
| notebooks/EnergyGladiators.ipynb | Jupyter notebook template |
| presentations/EnergyGladiators.md | Slide outline with innovation story |
| requirements.txt | Python dependencies |
| data/*.csv | 2025 sample data (structure reference only) |
| docs/HACKATHON_CHECKLIST.md | Submission checklist |

---

## 11. QUICK REFERENCE - PRESENTATION TALKING POINTS

Opening:
"We are Team Energy Gladiators. We built an intelligent ML system that predicts energy consumption for hydraulic fracturing operations - with uncertainty quantification."

Innovation Moment:
"But here is what makes us different. Most teams built a model. We built a platform. Watch this..."
[Type into AI Assistant: "What features would you recommend for optimizing this model?"]

Closing:
"Commercial tools like Spotfire cost thousands and only tell you what happened. Our solution tells you what WILL happen, how confident we are, and can adapt to any future problem. Thank you."

---

Team Energy Gladiators - Good luck!
