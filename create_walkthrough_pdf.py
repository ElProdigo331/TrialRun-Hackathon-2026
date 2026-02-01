"""
Generate Presentation_Walkthrough.pdf for Energy AI Hackathon 2026.
A presenter's guide matching the official 6-slide template format.
"""

from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from reportlab.lib.colors import HexColor, white

def create_walkthrough_pdf():
    """Create the presenter's walkthrough PDF."""
    doc = SimpleDocTemplate(
        "Presentation_Walkthrough.pdf",
        pagesize=letter,
        rightMargin=0.75*inch,
        leftMargin=0.75*inch,
        topMargin=0.75*inch,
        bottomMargin=0.75*inch
    )
    
    styles = getSampleStyleSheet()
    
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor=HexColor('#2c3e50'),
        alignment=1,
        spaceAfter=12
    )
    
    subtitle_style = ParagraphStyle(
        'CustomSubtitle',
        parent=styles['Normal'],
        fontSize=14,
        textColor=HexColor('#7f8c8d'),
        alignment=1,
        spaceAfter=6
    )
    
    section_style = ParagraphStyle(
        'SectionHeader',
        parent=styles['Heading2'],
        fontSize=18,
        textColor=HexColor('#2874a6'),
        spaceBefore=16,
        spaceAfter=8,
        borderPadding=5
    )
    
    subsection_style = ParagraphStyle(
        'SubsectionHeader',
        parent=styles['Heading3'],
        fontSize=14,
        textColor=HexColor('#1a5276'),
        spaceBefore=12,
        spaceAfter=6
    )
    
    body_style = ParagraphStyle(
        'BodyText',
        parent=styles['Normal'],
        fontSize=11,
        textColor=HexColor('#2c3e50'),
        spaceAfter=8,
        leading=14
    )
    
    talking_point_style = ParagraphStyle(
        'TalkingPoint',
        parent=styles['Normal'],
        fontSize=11,
        textColor=HexColor('#27ae60'),
        leftIndent=20,
        spaceAfter=6,
        leading=14
    )
    
    bullet_style = ParagraphStyle(
        'Bullet',
        parent=styles['Normal'],
        fontSize=11,
        textColor=HexColor('#2c3e50'),
        leftIndent=30,
        spaceAfter=4,
        leading=13
    )
    
    highlight_style = ParagraphStyle(
        'Highlight',
        parent=styles['Normal'],
        fontSize=11,
        textColor=HexColor('#c0392b'),
        backColor=HexColor('#fdebd0'),
        leftIndent=10,
        rightIndent=10,
        spaceBefore=8,
        spaceAfter=8,
        borderPadding=8
    )
    
    team_style = ParagraphStyle(
        'TeamStyle',
        parent=styles['Normal'],
        fontSize=12,
        textColor=HexColor('#2c3e50'),
        alignment=1,
        spaceAfter=4
    )
    
    story = []
    
    story.append(Paragraph("PRESENTATION WALKTHROUGH", title_style))
    story.append(Paragraph("Energy AI Hackathon 2026", subtitle_style))
    story.append(Spacer(1, 10))
    story.append(Paragraph("<b>Oil Production Prediction Using Machine Learning</b>", 
                          ParagraphStyle('BigSub', parent=subtitle_style, fontSize=16, textColor=HexColor('#2c3e50'))))
    story.append(Spacer(1, 30))
    
    story.append(Paragraph("<b>Team Brain Oil</b>", team_style))
    story.append(Paragraph("Kailasadatta Boggaram | Jayanth Damodaran | Bilal Shihab | Carlos Fabela", team_style))
    story.append(Spacer(1, 10))
    story.append(Paragraph("The University of Texas at Austin", 
                          ParagraphStyle('Uni', parent=team_style, fontSize=11, textColor=HexColor('#7f8c8d'))))
    story.append(Spacer(1, 30))
    
    story.append(Paragraph("<b>January 25, 2026</b>", 
                          ParagraphStyle('Date', parent=team_style, fontSize=12)))
    
    story.append(PageBreak())
    
    story.append(Paragraph("HOW TO USE THIS GUIDE", section_style))
    story.append(Paragraph(
        "This walkthrough follows the official 6-slide hackathon template. Each section corresponds to one slide. "
        "Use this as your speaking notes. You have less than 5 minutes to present.",
        body_style
    ))
    story.append(Spacer(1, 10))
    
    guide_data = [
        ['Symbol', 'Meaning'],
        ['SAY:', 'What to say out loud to the audience'],
        ['SHOW:', 'What to point to or emphasize'],
        ['KEY POINT:', 'The main takeaway for that slide']
    ]
    guide_table = Table(guide_data, colWidths=[1.5*inch, 5*inch])
    guide_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), HexColor('#2874a6')),
        ('TEXTCOLOR', (0, 0), (-1, 0), white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('GRID', (0, 0), (-1, -1), 0.5, HexColor('#bdc3c7')),
        ('BACKGROUND', (0, 1), (-1, -1), HexColor('#f8f9fa')),
        ('LEFTPADDING', (0, 0), (-1, -1), 10),
        ('TOPPADDING', (0, 0), (-1, -1), 8),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
    ]))
    story.append(guide_table)
    story.append(Spacer(1, 20))
    
    story.append(Paragraph("PRESENTATION TIMELINE (~5 minutes)", subsection_style))
    timeline_data = [
        ['Slide', 'Topic', 'Time'],
        ['1', 'Title & Team Introduction', '15 sec'],
        ['2', 'Executive Summary (4 questions)', '1 min'],
        ['3', 'Workflow Overview', '1 min'],
        ['4', 'Key Modeling Decisions', '1 min'],
        ['5', 'Results and Discussions', '1 min'],
        ['6', 'Feedback & Thank You', '45 sec']
    ]
    timeline_table = Table(timeline_data, colWidths=[0.8*inch, 3.5*inch, 1*inch])
    timeline_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), HexColor('#27ae60')),
        ('TEXTCOLOR', (0, 0), (-1, 0), white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('GRID', (0, 0), (-1, -1), 0.5, HexColor('#bdc3c7')),
        ('BACKGROUND', (0, 1), (-1, -1), HexColor('#f8f9fa')),
        ('TOPPADDING', (0, 0), (-1, -1), 8),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
    ]))
    story.append(timeline_table)
    
    story.append(PageBreak())
    
    story.append(Paragraph("SLIDE 1: TITLE", section_style))
    story.append(Paragraph("<b>SAY:</b> \"Good morning! We are Team Brain Oil, and we're presenting our machine learning solution for predicting oil production.\"", talking_point_style))
    story.append(Paragraph("<b>SHOW:</b> Point to team names on the slide.", talking_point_style))
    story.append(Paragraph("<b>KEY POINT:</b> Introduce yourselves confidently and quickly.", highlight_style))
    story.append(Spacer(1, 20))
    
    story.append(Paragraph("SLIDE 2: EXECUTIVE SUMMARY", section_style))
    story.append(Paragraph("<b>SAY:</b> \"Let me quickly answer the four key questions...\"", talking_point_style))
    story.append(Spacer(1, 5))
    story.append(Paragraph("• <b>The Problem:</b> Predict 3-year cumulative oil production for 12 new wells with 100 uncertainty realizations.", bullet_style))
    story.append(Paragraph("• <b>Our Solution:</b> Ridge Regression with alpha=1.0, StandardScaler normalization, stepwise feature selection, and Bagging for uncertainty.", bullet_style))
    story.append(Paragraph("• <b>What We Learned:</b> Simple models beat complex ones for small datasets. Porosity is the #1 predictor.", bullet_style))
    story.append(Paragraph("• <b>Our Results:</b> Test R² = 0.9905 (EXCELLENT), RMSE = 4.7% error.", bullet_style))
    story.append(Paragraph("<b>KEY POINT:</b> We achieved EXCELLENT results by matching model complexity to data size.", highlight_style))
    story.append(Spacer(1, 20))
    
    story.append(Paragraph("SLIDE 3: WORKFLOW OVERVIEW", section_style))
    story.append(Paragraph("<b>SAY:</b> \"Our workflow has 7 key steps...\"", talking_point_style))
    story.append(Spacer(1, 5))
    story.append(Paragraph("1. Data Loading - 71 training wells, 12 test wells, sand proportion map", bullet_style))
    story.append(Paragraph("2. MICE+CART Imputation - Fill missing values at depth level (Van Buuren 2018)", bullet_style))
    story.append(Paragraph("3. Aggregation - Multiple depth rows become one row per well (mean, std, min, max)", bullet_style))
    story.append(Paragraph("4. Feature Engineering - 105 features including RQI, FZI, spatial features", bullet_style))
    story.append(Paragraph("5. Feature Selection - Correlation filter then stepwise selection (105→10)", bullet_style))
    story.append(Paragraph("6. Model Training - Ridge Regression with StandardScaler", bullet_style))
    story.append(Paragraph("7. Uncertainty - Bagging Ensemble with 100 estimators", bullet_style))
    story.append(Paragraph("<b>KEY POINT:</b> Every step is research-backed with peer-reviewed citations.", highlight_style))
    
    story.append(PageBreak())
    
    story.append(Paragraph("SLIDE 4: KEY MODELING DECISIONS", section_style))
    story.append(Paragraph("<b>SAY:</b> \"Two key insights drove our success...\"", talking_point_style))
    story.append(Spacer(1, 5))
    story.append(Paragraph("<b>Why Ridge Regression?</b>", subsection_style))
    story.append(Paragraph("• With only 71 training wells, complex models overfit", bullet_style))
    story.append(Paragraph("• Hastie et al. (2009): Regularized linear models beat trees for small n", bullet_style))
    story.append(Paragraph("• Ridge R²=0.99 vs Random Forest R²=0.85 vs XGBoost R²=0.82", bullet_style))
    story.append(Spacer(1, 10))
    story.append(Paragraph("<b>Why Porosity (φ) Matters Most?</b>", subsection_style))
    story.append(Paragraph("• Porosity directly measures how much oil the rock can store", bullet_style))
    story.append(Paragraph("• It appears in the fundamental OOIP equation", bullet_style))
    story.append(Paragraph("• Domain experts always prioritize porosity over permeability", bullet_style))
    story.append(Spacer(1, 10))
    story.append(Paragraph("<b>KEY POINT:</b> Simple model + domain knowledge = winning combination.", highlight_style))
    story.append(Spacer(1, 20))
    
    story.append(Paragraph("SLIDE 5: RESULTS AND DISCUSSIONS", section_style))
    story.append(Paragraph("<b>SAY:</b> \"Our model achieved EXCELLENT performance by industry standards...\"", talking_point_style))
    story.append(Spacer(1, 5))
    
    results_data = [
        ['Metric', 'Our Result', 'Industry Benchmark'],
        ['Test R²', '0.9905', '≥0.93 = Excellent'],
        ['CV R²', '0.9539 ± 0.0456', 'Stable'],
        ['RMSE', '1.57M BBL (4.7%)', '<10% = Excellent']
    ]
    results_table = Table(results_data, colWidths=[1.5*inch, 2*inch, 2*inch])
    results_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), HexColor('#27ae60')),
        ('TEXTCOLOR', (0, 0), (-1, 0), white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('GRID', (0, 0), (-1, -1), 0.5, HexColor('#bdc3c7')),
        ('BACKGROUND', (0, 1), (-1, -1), HexColor('#f8f9fa')),
        ('TOPPADDING', (0, 0), (-1, -1), 8),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
    ]))
    story.append(results_table)
    story.append(Spacer(1, 10))
    story.append(Paragraph("• Predictions for all 12 wells (72-83) with 100 realizations each", bullet_style))
    story.append(Paragraph("• Production range: ~15M to ~47M BBL over 3 years", bullet_style))
    story.append(Paragraph("<b>KEY POINT:</b> R² = 0.99 means our model explains 99% of production variation.", highlight_style))
    
    story.append(PageBreak())
    
    story.append(Paragraph("SLIDE 6: FEEDBACK", section_style))
    story.append(Paragraph("<b>SAY:</b> \"To wrap up, here's what we learned and our feedback...\"", talking_point_style))
    story.append(Spacer(1, 5))
    story.append(Paragraph("<b>What We Learned:</b>", subsection_style))
    story.append(Paragraph("• Simple models win for small datasets", bullet_style))
    story.append(Paragraph("• Domain knowledge (porosity priority) improves results", bullet_style))
    story.append(Paragraph("• MICE+CART is the gold standard for imputation", bullet_style))
    story.append(Spacer(1, 10))
    story.append(Paragraph("<b>What We Liked:</b>", subsection_style))
    story.append(Paragraph("• Real-world petroleum engineering challenge", bullet_style))
    story.append(Paragraph("• Excellent workshop materials from Prof. Pyrcz and Prof. Foster", bullet_style))
    story.append(Spacer(1, 10))
    story.append(Paragraph("<b>SAY:</b> \"Thank you for your attention. We're happy to answer questions.\"", talking_point_style))
    story.append(Paragraph("<b>KEY POINT:</b> End confidently and invite questions.", highlight_style))
    
    story.append(PageBreak())
    
    story.append(Paragraph("QUICK REFERENCE CARD", section_style))
    config_data = [
        ['Setting', 'Value'],
        ['Model', 'Ridge Regression'],
        ['Alpha', '1.0'],
        ['Normalization', 'StandardScaler (CRITICAL)'],
        ['Sand Map', 'Smooth 3x3'],
        ['Feature Selection', 'Stepwise (105 → 10)'],
        ['Uncertainty', 'Bagging Ensemble (100)'],
        ['Test R²', '0.9905 (EXCELLENT)'],
        ['RMSE', '1.57M BBL (4.7%)']
    ]
    config_table = Table(config_data, colWidths=[2.5*inch, 3*inch])
    config_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), HexColor('#e74c3c')),
        ('TEXTCOLOR', (0, 0), (-1, 0), white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 11),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('GRID', (0, 0), (-1, -1), 0.5, HexColor('#bdc3c7')),
        ('BACKGROUND', (0, 1), (-1, -1), HexColor('#f8f9fa')),
        ('LEFTPADDING', (0, 0), (-1, -1), 10),
        ('TOPPADDING', (0, 0), (-1, -1), 8),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
    ]))
    story.append(config_table)
    story.append(Spacer(1, 20))
    
    story.append(Paragraph("ANTICIPATED Q&A", section_style))
    qa = [
        ("Q: Why not use Random Forest or XGBoost?", 
         "A: With only 71 training wells, complex models overfit. Ridge's R²=0.99 beat RF's 0.85 and XGB's 0.82."),
        ("Q: Why is porosity the most important feature?",
         "A: Porosity directly measures storage capacity. It's in the fundamental OOIP equation: OOIP = 7758 × A × h × φ × (1-Sw) / Bo."),
        ("Q: How did you handle missing data?",
         "A: MICE + CART imputation at the depth level before aggregation, per Van Buuren (2018) recommendations."),
        ("Q: How confident are you in these predictions?",
         "A: Very confident. R²=0.99 with only 4.7% error. The 100 realizations capture uncertainty via Bagging Ensemble.")
    ]
    
    for q, a in qa:
        story.append(Paragraph(f"<b>{q}</b>", body_style))
        story.append(Paragraph(a, bullet_style))
        story.append(Spacer(1, 8))
    
    doc.build(story)
    print("PDF created successfully: Presentation_Walkthrough.pdf")

if __name__ == "__main__":
    create_walkthrough_pdf()
