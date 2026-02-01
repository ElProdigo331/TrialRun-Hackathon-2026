"""
Generate Brain_Oil.pptx PowerPoint presentation for Energy AI Hackathon 2026.
Following the hackathon template format.
"""

from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.enum.text import PP_ALIGN
import os

def add_title_slide(prs, title, subtitle, team_members):
    """Add title slide with team info."""
    slide_layout = prs.slide_layouts[6]  # Blank layout
    slide = prs.slides.add_slide(slide_layout)
    
    # Title
    title_box = slide.shapes.add_textbox(Inches(0.5), Inches(2), Inches(9), Inches(1))
    title_frame = title_box.text_frame
    title_para = title_frame.paragraphs[0]
    title_para.text = title
    title_para.font.size = Pt(44)
    title_para.font.bold = True
    title_para.alignment = PP_ALIGN.CENTER
    
    # Subtitle
    sub_box = slide.shapes.add_textbox(Inches(0.5), Inches(3.2), Inches(9), Inches(0.5))
    sub_frame = sub_box.text_frame
    sub_para = sub_frame.paragraphs[0]
    sub_para.text = subtitle
    sub_para.font.size = Pt(24)
    sub_para.alignment = PP_ALIGN.CENTER
    
    # Team members
    team_box = slide.shapes.add_textbox(Inches(0.5), Inches(4.2), Inches(9), Inches(1))
    team_frame = team_box.text_frame
    team_para = team_frame.paragraphs[0]
    team_para.text = f"Team Members: {team_members}"
    team_para.font.size = Pt(18)
    team_para.alignment = PP_ALIGN.CENTER
    
    # University
    uni_box = slide.shapes.add_textbox(Inches(0.5), Inches(5), Inches(9), Inches(0.5))
    uni_frame = uni_box.text_frame
    uni_para = uni_frame.paragraphs[0]
    uni_para.text = "The University of Texas at Austin"
    uni_para.font.size = Pt(16)
    uni_para.alignment = PP_ALIGN.CENTER
    
    # Date
    date_box = slide.shapes.add_textbox(Inches(0.5), Inches(5.5), Inches(9), Inches(0.5))
    date_frame = date_box.text_frame
    date_para = date_frame.paragraphs[0]
    date_para.text = "Energy AI Hackathon 2026"
    date_para.font.size = Pt(14)
    date_para.alignment = PP_ALIGN.CENTER

def add_content_slide(prs, title, bullet_points, subtitle=None):
    """Add a content slide with title and bullet points."""
    slide_layout = prs.slide_layouts[6]  # Blank layout
    slide = prs.slides.add_slide(slide_layout)
    
    # Title
    title_box = slide.shapes.add_textbox(Inches(0.5), Inches(0.3), Inches(9), Inches(0.8))
    title_frame = title_box.text_frame
    title_para = title_frame.paragraphs[0]
    title_para.text = title
    title_para.font.size = Pt(32)
    title_para.font.bold = True
    
    # Subtitle if provided
    start_y = 1.0
    if subtitle:
        sub_box = slide.shapes.add_textbox(Inches(0.5), Inches(1.0), Inches(9), Inches(0.5))
        sub_frame = sub_box.text_frame
        sub_para = sub_frame.paragraphs[0]
        sub_para.text = subtitle
        sub_para.font.size = Pt(18)
        sub_para.font.italic = True
        start_y = 1.5
    
    # Bullet points
    content_box = slide.shapes.add_textbox(Inches(0.5), Inches(start_y), Inches(9), Inches(5))
    content_frame = content_box.text_frame
    content_frame.word_wrap = True
    
    for i, point in enumerate(bullet_points):
        if i == 0:
            p = content_frame.paragraphs[0]
        else:
            p = content_frame.add_paragraph()
        p.text = f"• {point}"
        p.font.size = Pt(20)
        p.space_before = Pt(8)
        p.space_after = Pt(4)

def add_table_slide(prs, title, headers, rows, subtitle=None):
    """Add a slide with a table."""
    slide_layout = prs.slide_layouts[6]  # Blank layout
    slide = prs.slides.add_slide(slide_layout)
    
    # Title
    title_box = slide.shapes.add_textbox(Inches(0.5), Inches(0.3), Inches(9), Inches(0.8))
    title_frame = title_box.text_frame
    title_para = title_frame.paragraphs[0]
    title_para.text = title
    title_para.font.size = Pt(32)
    title_para.font.bold = True
    
    start_y = 1.0
    if subtitle:
        sub_box = slide.shapes.add_textbox(Inches(0.5), Inches(1.0), Inches(9), Inches(0.5))
        sub_frame = sub_box.text_frame
        sub_para = sub_frame.paragraphs[0]
        sub_para.text = subtitle
        sub_para.font.size = Pt(18)
        sub_para.font.italic = True
        start_y = 1.5
    
    # Table
    cols = len(headers)
    num_rows = len(rows) + 1  # +1 for header
    
    table = slide.shapes.add_table(num_rows, cols, Inches(0.5), Inches(start_y), 
                                    Inches(9), Inches(0.5 * num_rows)).table
    
    # Header row
    for i, header in enumerate(headers):
        cell = table.cell(0, i)
        cell.text = header
        cell.text_frame.paragraphs[0].font.bold = True
        cell.text_frame.paragraphs[0].font.size = Pt(14)
    
    # Data rows
    for row_idx, row_data in enumerate(rows):
        for col_idx, cell_data in enumerate(row_data):
            cell = table.cell(row_idx + 1, col_idx)
            cell.text = str(cell_data)
            cell.text_frame.paragraphs[0].font.size = Pt(12)

def create_presentation():
    """Create the complete hackathon presentation."""
    prs = Presentation()
    prs.slide_width = Inches(10)
    prs.slide_height = Inches(7.5)
    
    # Slide 1: Title
    add_title_slide(
        prs,
        "Brain_Oil",
        "Predicting 3-Year Cumulative Oil Production with Uncertainty",
        "Brain_Oil Team"
    )
    
    # Slide 2: Executive Summary
    add_content_slide(prs, "Executive Summary", [
        "Problem: Predict 3-year cumulative oil production for 12 preproduction wells using well logs with missing data and noisy seismic maps",
        "Approach: MICE+CART imputation at depth level, 19+ engineered features (RQI, FZI), Ridge Regression with two-stage feature selection",
        "Findings: Ridge Regression outperformed tree-based models (R²=0.9905 vs ~0.82-0.85) due to lower variance on small datasets (n=71)",
        "Recommendation: For small subsurface datasets, prioritize regularized linear models over tree ensembles; use Bagging for uncertainty"
    ])
    
    # Slide 3: Problem Statement
    add_content_slide(prs, "Problem Statement", [
        "Predict 3-year cumulative oil production (BBL) for 12 preproduction wells (IDs 72-83)",
        "Generate point estimates + 100 realizations for uncertainty quantification",
        "Training data: 71 production wells with ~21 depth measurements each",
        "Challenges: Missing well log data, noisy seismic-derived sand map, small dataset size"
    ])
    
    # Slide 4: Data & Preprocessing
    add_content_slide(prs, "Data & Preprocessing", [
        "Well log features: AI, SI, Vp, Vs, rho, K, G, phi, perm, GR, facies",
        "MICE + CART imputation at depth level (Van Buuren 2018, Hallam et al. 2022)",
        "Aggregation: Mean, std, min, max + best/worst zone features for depth heterogeneity",
        "Sand map: 3x3 smoothing to reduce deliberate noise (per architect guidance)",
        "Production history: Calculate 3-year cumulative oil from monthly data"
    ])
    
    # Slide 5: Feature Engineering
    add_content_slide(prs, "Feature Engineering (19+ Features)", [
        "RQI = 0.0314 × √(k/φ) - Reservoir Quality Index (Amaefule et al. 1993)",
        "FZI = RQI / (φ/(1-φ)) - Flow Zone Indicator for hydraulic units",
        "Vp/Vs ratio - Rock physics lithology and fluid indicator",
        "Best zone features: phi, perm, GR at depth with highest rock quality",
        "Analog similarity: Distance to top 25% producers in rock quality space",
        "Spatial proximity: Distance to high-production region centroid"
    ])
    
    # Slide 6: Feature Selection (Two-Stage)
    add_content_slide(prs, "Feature Selection (Two-Stage)", [
        "Stage 1: Correlation filter removes redundant features (r ≥ 0.98)",
        "Result: 105 → 61 features (removed 44 highly correlated)",
        "Stage 2: Stepwise forward selection with 5-fold CV (Miller 2002)",
        "Result: 61 → 10 optimal features",
        "Key insight: Dimensionality reduction improves generalization (Guyon & Elisseeff 2003)"
    ])
    
    # Slide 7: Model Selection
    add_table_slide(
        prs, 
        "Model Benchmarking Results",
        ["Model", "Test R²", "RMSE", "Overfitting Gap"],
        [
            ["Ridge Regression", "0.9905", "1.57M BBL", "0.002"],
            ["Random Forest", "~0.85", "5.2M BBL", "0.15"],
            ["XGBoost", "~0.82", "5.8M BBL", "0.18"],
            ["Linear Regression", "Overfit", "-", ">1.0"]
        ],
        subtitle="Ridge Regression significantly outperformed tree-based models"
    )
    
    # Slide 8: Why Ridge Won
    add_content_slide(prs, "Why Ridge Regression Won", [
        "Small dataset (n=71): Regularization prevents overfitting (Hastie et al. 2009)",
        "High-dimensional features: L2 penalty handles multicollinearity (Hoerl & Kennard 1970)",
        "Stability: CV R² std of only ±0.0456 indicates robust performance",
        "Simplicity: Linear model is more interpretable than tree ensembles",
        "Key lesson: 'For small datasets, simpler models often win' - Dr. Pyrcz"
    ])
    
    # Slide 9: Uncertainty Quantification
    add_content_slide(prs, "Uncertainty Quantification", [
        "Method: Bagging Ensemble with 100 Ridge estimators (Breiman 1996)",
        "Each estimator trained on bootstrap sample of training data",
        "Each estimator's prediction → one realization (R_1 to R_100)",
        "Captures model uncertainty, not just residual noise",
        "OOB R² validates ensemble quality without holdout data"
    ])
    
    # Slide 10: Final Results
    add_table_slide(
        prs,
        "Final Model Performance",
        ["Metric", "Value", "Industry Benchmark"],
        [
            ["Test R²", "0.9905", "Excellent (≥0.93)"],
            ["CV R² Mean", "0.9539 ± 0.0456", "Excellent"],
            ["Test RMSE", "1.57M BBL", "4.7% of mean (Excellent <10%)"],
            ["Features Used", "10", "Optimal via stepwise selection"]
        ],
        subtitle="Model achieves excellent accuracy with robust cross-validation"
    )
    
    # Slide 11: Key Innovations
    add_content_slide(prs, "Key Innovations", [
        "MICE imputation at depth level (before aggregation) - preserves correlations",
        "Two-stage feature selection: correlation filter + stepwise forward",
        "Best zone features capture depth heterogeneity that averages miss",
        "Systematic benchmarking: 57+ configurations tested to find optimal",
        "Bagging over residual bootstrap for uncertainty - captures model uncertainty"
    ])
    
    # Slide 12: References
    add_content_slide(prs, "Key References", [
        "Amaefule et al. (1993) - RQI/FZI for reservoir quality, SPE",
        "Breiman (1996) - Bagging predictors, Machine Learning",
        "Hastie, Tibshirani & Friedman (2009) - Elements of Statistical Learning",
        "Hoerl & Kennard (1970) - Ridge Regression, Technometrics",
        "Miller (2002) - Subset Selection in Regression, Chapman & Hall",
        "Van Buuren (2018) - Flexible Imputation of Missing Data, CRC Press"
    ])
    
    # Slide 13: Conclusions
    add_content_slide(prs, "Conclusions", [
        "Ridge Regression with stepwise selection achieved R²=0.9905 on test data",
        "Regularized linear models outperform tree ensembles on small datasets",
        "MICE imputation at depth level is critical for preserving correlations",
        "Feature engineering (RQI, FZI, best zone) captures reservoir quality",
        "Bagging provides robust uncertainty quantification (100 realizations)"
    ])
    
    # Slide 14: Thank You
    add_content_slide(prs, "Thank You!", [
        "Questions?",
        "",
        "Team Brain_Oil",
        "The University of Texas at Austin",
        "Energy AI Hackathon 2026"
    ])
    
    # Save presentation
    prs.save("Brain_Oil.pptx")
    print("Presentation saved to Brain_Oil.pptx")

if __name__ == "__main__":
    create_presentation()
