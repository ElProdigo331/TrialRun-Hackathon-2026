from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.colors import HexColor, black, white
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, KeepTogether
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY

def create_walkthrough_pdf():
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
        parent=styles['Title'],
        fontSize=24,
        spaceAfter=6,
        textColor=HexColor('#1a5276'),
        alignment=TA_CENTER
    )
    
    subtitle_style = ParagraphStyle(
        'Subtitle',
        parent=styles['Normal'],
        fontSize=14,
        spaceAfter=20,
        textColor=HexColor('#566573'),
        alignment=TA_CENTER
    )
    
    team_style = ParagraphStyle(
        'Team',
        parent=styles['Normal'],
        fontSize=12,
        spaceBefore=10,
        spaceAfter=30,
        textColor=HexColor('#2c3e50'),
        alignment=TA_CENTER,
        leading=18
    )
    
    section_style = ParagraphStyle(
        'Section',
        parent=styles['Heading1'],
        fontSize=16,
        spaceBefore=20,
        spaceAfter=10,
        textColor=HexColor('#1a5276'),
        borderPadding=5
    )
    
    subsection_style = ParagraphStyle(
        'Subsection',
        parent=styles['Heading2'],
        fontSize=13,
        spaceBefore=12,
        spaceAfter=6,
        textColor=HexColor('#2874a6')
    )
    
    body_style = ParagraphStyle(
        'CustomBody',
        parent=styles['Normal'],
        fontSize=11,
        spaceAfter=8,
        alignment=TA_JUSTIFY,
        leading=16
    )
    
    bullet_style = ParagraphStyle(
        'Bullet',
        parent=styles['Normal'],
        fontSize=11,
        leftIndent=20,
        spaceAfter=4,
        leading=15
    )
    
    talking_point_style = ParagraphStyle(
        'TalkingPoint',
        parent=styles['Normal'],
        fontSize=11,
        leftIndent=25,
        spaceAfter=3,
        textColor=HexColor('#27ae60'),
        leading=14
    )
    
    quote_style = ParagraphStyle(
        'Quote',
        parent=styles['Normal'],
        fontSize=10,
        leftIndent=30,
        rightIndent=30,
        spaceAfter=10,
        spaceBefore=10,
        textColor=HexColor('#7f8c8d'),
        alignment=TA_CENTER,
        fontName='Helvetica-Oblique',
        leading=14
    )
    
    highlight_style = ParagraphStyle(
        'Highlight',
        parent=styles['Normal'],
        fontSize=12,
        spaceBefore=10,
        spaceAfter=10,
        textColor=HexColor('#c0392b'),
        alignment=TA_CENTER,
        fontName='Helvetica-Bold'
    )
    
    story = []
    
    story.append(Paragraph("PRESENTATION WALKTHROUGH", title_style))
    story.append(Paragraph("Energy AI Hackathon 2026", subtitle_style))
    story.append(Spacer(1, 10))
    story.append(Paragraph("<b>Oil Production Prediction Using Machine Learning</b>", 
                          ParagraphStyle('BigSub', parent=subtitle_style, fontSize=16, textColor=HexColor('#2c3e50'))))
    story.append(Spacer(1, 30))
    
    story.append(Paragraph("<b>Team Brain Oil</b>", team_style))
    story.append(Paragraph("Kailasadatta Boggaram<br/>Jayanth Damodaran<br/>Bilal Shihab<br/>Carlos Fabela", team_style))
    story.append(Spacer(1, 20))
    story.append(Paragraph("The University of Texas at Austin", 
                          ParagraphStyle('Uni', parent=team_style, fontSize=11, textColor=HexColor('#7f8c8d'))))
    story.append(Spacer(1, 30))
    
    story.append(Paragraph("<b>February 1, 2026</b>", 
                          ParagraphStyle('Date', parent=team_style, fontSize=12)))
    
    story.append(PageBreak())
    
    story.append(Paragraph("HOW TO USE THIS GUIDE", section_style))
    story.append(Paragraph(
        "This walkthrough is designed to help you present our project confidently. Each section corresponds to slides "
        "in the PowerPoint and cells in the Jupyter notebook. Use it as your speaking notes during the presentation.",
        body_style
    ))
    story.append(Spacer(1, 10))
    
    guide_data = [
        ['Symbol', 'Meaning'],
        ['SAY:', 'What to say out loud to the audience'],
        ['SHOW:', 'What to point to or demonstrate'],
        ['KEY POINT:', 'The main takeaway for that section'],
        ['TRANSITION:', 'How to smoothly move to the next topic']
    ]
    guide_table = Table(guide_data, colWidths=[1.5*inch, 5*inch])
    guide_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), HexColor('#2874a6')),
        ('TEXTCOLOR', (0, 0), (-1, 0), white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('GRID', (0, 0), (-1, -1), 0.5, HexColor('#bdc3c7')),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('BACKGROUND', (0, 1), (-1, -1), HexColor('#f8f9fa')),
        ('LEFTPADDING', (0, 0), (-1, -1), 10),
        ('RIGHTPADDING', (0, 0), (-1, -1), 10),
        ('TOPPADDING', (0, 0), (-1, -1), 8),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
    ]))
    story.append(guide_table)
    story.append(Spacer(1, 20))
    
    story.append(Paragraph("PRESENTATION TIMELINE (10 minutes)", subsection_style))
    timeline_data = [
        ['Section', 'Time', 'Slides'],
        ['Introduction & Problem', '1 min', '1-2'],
        ['Data & Methodology', '3 min', '3-7'],
        ['Results & Why Ridge Won', '3 min', '8-11'],
        ['Predictions & Uncertainty', '2 min', '12-13'],
        ['Conclusion & Q&A', '1 min', '14-15']
    ]
    timeline_table = Table(timeline_data, colWidths=[3*inch, 1.2*inch, 1.2*inch])
    timeline_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), HexColor('#27ae60')),
        ('TEXTCOLOR', (0, 0), (-1, 0), white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('GRID', (0, 0), (-1, -1), 0.5, HexColor('#bdc3c7')),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('BACKGROUND', (0, 1), (-1, -1), HexColor('#f8f9fa')),
        ('TOPPADDING', (0, 0), (-1, -1), 8),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
    ]))
    story.append(timeline_table)
    
    story.append(PageBreak())
    
    story.append(Paragraph("PART 1: INTRODUCTION", section_style))
    story.append(Paragraph("Slides 1-2 | Notebook Cells 1-3", subtitle_style))
    story.append(Spacer(1, 10))
    
    story.append(Paragraph("Opening (Slide 1)", subsection_style))
    story.append(Paragraph("<b>SAY:</b> \"Good morning! We are Team Brain Oil, and today we'll show you how we predict "
                          "oil production for 12 new wells using machine learning.\"", talking_point_style))
    story.append(Paragraph("<b>SHOW:</b> Point to team names on the title slide.", talking_point_style))
    story.append(Spacer(1, 10))
    
    story.append(Paragraph("The Challenge (Slide 2)", subsection_style))
    story.append(Paragraph("<b>SAY:</b> \"The challenge is simple but important: Given well log data from 71 existing wells "
                          "where we know how much oil they produced, can we predict what 12 new wells will produce over 3 years?\"", talking_point_style))
    story.append(Spacer(1, 5))
    story.append(Paragraph("• We have 71 training wells with known production", bullet_style))
    story.append(Paragraph("• We need to predict for 12 test wells (IDs 72-83)", bullet_style))
    story.append(Paragraph("• Output: Point estimate + 100 uncertainty scenarios per well", bullet_style))
    story.append(Spacer(1, 10))
    
    story.append(Paragraph("<b>KEY POINT:</b> This is like predicting house prices from features - but for oil wells.", highlight_style))
    story.append(Spacer(1, 10))
    story.append(Paragraph("<b>TRANSITION:</b> \"Let me show you the data we worked with...\"", talking_point_style))
    
    story.append(PageBreak())
    
    story.append(Paragraph("PART 2: DATA & METHODOLOGY", section_style))
    story.append(Paragraph("Slides 3-7 | Notebook Cells 4-15", subtitle_style))
    story.append(Spacer(1, 10))
    
    story.append(Paragraph("Understanding the Data (Slide 3)", subsection_style))
    story.append(Paragraph("<b>SAY:</b> \"Each well has multiple measurements at different depths - about 21 readings per well. "
                          "These include porosity, permeability, and other rock properties.\"", talking_point_style))
    story.append(Spacer(1, 5))
    story.append(Paragraph("• <b>Porosity (phi)</b> - How much empty space in the rock (storage capacity)", bullet_style))
    story.append(Paragraph("• <b>Permeability (perm)</b> - How easily oil can flow through the rock", bullet_style))
    story.append(Paragraph("• <b>Gamma Ray (GR)</b> - Helps identify rock types (sand vs. shale)", bullet_style))
    story.append(Spacer(1, 10))
    
    story.append(Paragraph("Handling Missing Data (Slide 4)", subsection_style))
    story.append(Paragraph("<b>SAY:</b> \"Real-world data has gaps. Instead of throwing away incomplete records, we used a "
                          "smart technique called MICE - it fills in missing values by learning patterns from the data.\"", talking_point_style))
    story.append(Spacer(1, 5))
    story.append(Paragraph(
        "\"MICE + CART is the gold standard for missing data, recommended by Van Buuren in his 2018 textbook.\"",
        quote_style
    ))
    story.append(Spacer(1, 10))
    
    story.append(Paragraph("Aggregation: Multi-Row to Single-Row (Slide 5)", subsection_style))
    story.append(Paragraph("<b>SAY:</b> \"Here's the key insight - each well has ~21 depth readings, but we need ONE prediction per well. "
                          "So we aggregate: calculate the mean, standard deviation, minimum, and maximum of each property.\"", talking_point_style))
    story.append(Spacer(1, 5))
    story.append(Paragraph("<b>SHOW:</b> Diagram showing 21 rows becoming 1 row with summary statistics.", talking_point_style))
    story.append(Spacer(1, 10))
    
    story.append(Paragraph("Feature Engineering (Slide 6)", subsection_style))
    story.append(Paragraph("<b>SAY:</b> \"We created 19 industry-standard features that petroleum engineers actually use:\"", talking_point_style))
    story.append(Spacer(1, 5))
    story.append(Paragraph("• <b>RQI & FZI</b> - Rock quality indicators from Amaefule (1993)", bullet_style))
    story.append(Paragraph("• <b>Net-to-Gross</b> - Proportion of productive rock", bullet_style))
    story.append(Paragraph("• <b>Best Zone Features</b> - Properties at the highest-quality depth", bullet_style))
    story.append(Paragraph("• <b>Spatial Features</b> - Location and sand proportion from geology maps", bullet_style))
    story.append(Spacer(1, 10))
    
    story.append(Paragraph("<b>KEY POINT:</b> We started with 105 features and narrowed down to 10 optimal ones.", highlight_style))
    story.append(Spacer(1, 10))
    story.append(Paragraph("<b>TRANSITION:</b> \"Let me show you our spatial analysis...\"", talking_point_style))
    story.append(Spacer(1, 15))
    
    story.append(Paragraph("Spatial Analysis: Sand Heat Map (Slide 7)", subsection_style))
    story.append(Paragraph("<b>SAY:</b> \"This is our spatial analysis map. The background shows sand proportion - "
                          "yellow means more sand, darker means more shale. The dots are our wells.\"", talking_point_style))
    story.append(Spacer(1, 5))
    story.append(Paragraph("• <b>Circles</b> - Production wells (color = actual 3-year oil production)", bullet_style))
    story.append(Paragraph("• <b>X markers</b> - Pre-production wells (color = our predictions)", bullet_style))
    story.append(Paragraph("• <b>Yellow background</b> - High sand proportion (better reservoir quality)", bullet_style))
    story.append(Spacer(1, 10))
    story.append(Paragraph("<b>SAY:</b> \"Notice how the highest-producing wells (green/yellow dots) tend to be in "
                          "higher sand areas. This validates that our spatial features are capturing real geology.\"", talking_point_style))
    story.append(Spacer(1, 10))
    story.append(Paragraph("<b>KEY POINT:</b> Geology matters - high sand = high production. Our model learns this pattern.", highlight_style))
    story.append(Spacer(1, 10))
    story.append(Paragraph("<b>TRANSITION:</b> \"Now let's see how our model performed...\"", talking_point_style))
    
    story.append(PageBreak())
    
    story.append(Paragraph("PART 3: RESULTS & WHY RIDGE WON", section_style))
    story.append(Paragraph("Slides 8-11 | Notebook Cells 16-25", subtitle_style))
    story.append(Spacer(1, 10))
    
    story.append(Paragraph("Model Comparison (Slide 8)", subsection_style))
    story.append(Paragraph("<b>SAY:</b> \"We tested 57 different configurations across 4 model types. Here's what we found:\"", talking_point_style))
    story.append(Spacer(1, 10))
    
    model_data = [
        ['Model', 'Test R²', 'Verdict'],
        ['Ridge Regression', '0.9905', 'WINNER'],
        ['Random Forest', '0.85', 'Good'],
        ['XGBoost', '0.82', 'Acceptable'],
        ['Elastic Net', '0.88', 'Good']
    ]
    model_table = Table(model_data, colWidths=[2.5*inch, 1.5*inch, 1.5*inch])
    model_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), HexColor('#1a5276')),
        ('TEXTCOLOR', (0, 0), (-1, 0), white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 11),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('GRID', (0, 0), (-1, -1), 0.5, HexColor('#bdc3c7')),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('BACKGROUND', (0, 1), (-1, 1), HexColor('#d5f5e3')),
        ('FONTNAME', (0, 1), (-1, 1), 'Helvetica-Bold'),
        ('BACKGROUND', (0, 2), (-1, -1), HexColor('#f8f9fa')),
        ('TOPPADDING', (0, 0), (-1, -1), 8),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
    ]))
    story.append(model_table)
    story.append(Spacer(1, 15))
    
    story.append(Paragraph("Why Ridge Beat Complex Models (Slide 9)", subsection_style))
    story.append(Paragraph("<b>SAY:</b> \"You might ask - why did a simple model beat fancy ones like XGBoost? "
                          "The answer is in the data size.\"", talking_point_style))
    story.append(Spacer(1, 10))
    story.append(Paragraph(
        "\"For small datasets (n < 100), regularized linear models often outperform tree-based ensembles "
        "due to lower variance.\" — Hastie, Tibshirani & Friedman (2009), The Elements of Statistical Learning",
        quote_style
    ))
    story.append(Spacer(1, 10))
    story.append(Paragraph("<b>SAY:</b> \"With only 71 training wells, complex models overfit. Ridge keeps it simple and stable.\"", talking_point_style))
    story.append(Spacer(1, 10))
    
    story.append(Paragraph("Our Performance Metrics (Slide 10)", subsection_style))
    metrics_data = [
        ['Metric', 'Our Result', 'Industry Benchmark', 'Status'],
        ['Test R²', '0.9905', '≥ 0.93 = Excellent', 'EXCELLENT'],
        ['CV R²', '0.9539 ± 0.05', 'Stable', 'VERY STABLE'],
        ['RMSE', '1.57M BBL', '< 10% of mean', '4.7% - GREAT']
    ]
    metrics_table = Table(metrics_data, colWidths=[1.5*inch, 1.5*inch, 1.8*inch, 1.2*inch])
    metrics_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), HexColor('#27ae60')),
        ('TEXTCOLOR', (0, 0), (-1, 0), white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('GRID', (0, 0), (-1, -1), 0.5, HexColor('#bdc3c7')),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('BACKGROUND', (0, 1), (-1, -1), HexColor('#f8f9fa')),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
    ]))
    story.append(metrics_table)
    story.append(Spacer(1, 10))
    
    story.append(Paragraph("<b>SAY:</b> \"R² of 0.99 means our model explains 99% of the variation in oil production. "
                          "The error is only 4.7% of the average production - well within industry standards.\"", talking_point_style))
    story.append(Spacer(1, 10))
    
    story.append(Paragraph("Feature Importance (Slide 11)", subsection_style))
    story.append(Paragraph("<b>SAY:</b> \"The top predictor is porosity - phi_mean. This makes perfect sense because "
                          "porosity directly measures how much oil the rock can store.\"", talking_point_style))
    story.append(Spacer(1, 5))
    story.append(Paragraph("• More pore space = more oil storage = higher production", bullet_style))
    story.append(Paragraph("• This aligns with fundamental reservoir engineering principles", bullet_style))
    story.append(Spacer(1, 10))
    
    story.append(Paragraph("<b>KEY POINT:</b> Simple model + domain knowledge = best results.", highlight_style))
    story.append(Spacer(1, 10))
    story.append(Paragraph("<b>TRANSITION:</b> \"Let me show you our actual predictions...\"", talking_point_style))
    
    story.append(PageBreak())
    
    story.append(Paragraph("PART 4: PREDICTIONS & UNCERTAINTY", section_style))
    story.append(Paragraph("Slides 12-13 | Notebook Cells 26-32", subtitle_style))
    story.append(Spacer(1, 10))
    
    story.append(Paragraph("Final Predictions (Slide 12)", subsection_style))
    story.append(Paragraph("<b>SAY:</b> \"Here are our predictions for the 12 new wells. Production ranges from about "
                          "15 million to 47 million barrels over 3 years.\"", talking_point_style))
    story.append(Spacer(1, 10))
    story.append(Paragraph("<b>SHOW:</b> Point to the prediction chart showing all 12 wells.", talking_point_style))
    story.append(Spacer(1, 10))
    
    story.append(Paragraph("Uncertainty Quantification (Slide 13)", subsection_style))
    story.append(Paragraph("<b>SAY:</b> \"Predictions are only useful if we know how confident we are. We used a technique "
                          "called Bagging - training 100 different models and combining their predictions.\"", talking_point_style))
    story.append(Spacer(1, 5))
    story.append(Paragraph("• Each model sees a slightly different sample of the data", bullet_style))
    story.append(Paragraph("• This gives us 100 different predictions (R1-R100)", bullet_style))
    story.append(Paragraph("• The spread shows our uncertainty for each well", bullet_style))
    story.append(Spacer(1, 10))
    story.append(Paragraph(
        "\"Bagging reduces variance by averaging over bootstrap samples.\" — Breiman (1996)",
        quote_style
    ))
    story.append(Spacer(1, 10))
    
    story.append(Paragraph("<b>KEY POINT:</b> We don't just give a number - we give a range of possibilities.", highlight_style))
    story.append(Spacer(1, 10))
    story.append(Paragraph("<b>TRANSITION:</b> \"Let me wrap up with our key takeaways...\"", talking_point_style))
    
    story.append(PageBreak())
    
    story.append(Paragraph("PART 5: CONCLUSION", section_style))
    story.append(Paragraph("Slides 14-15 | Summary", subtitle_style))
    story.append(Spacer(1, 10))
    
    story.append(Paragraph("Key Achievements (Slide 14)", subsection_style))
    story.append(Paragraph("<b>SAY:</b> \"To summarize what we accomplished:\"", talking_point_style))
    story.append(Spacer(1, 10))
    story.append(Paragraph("1. <b>Excellent accuracy</b> - 99% R², 4.7% error rate", bullet_style))
    story.append(Paragraph("2. <b>Research-backed approach</b> - Every decision supported by peer-reviewed literature", bullet_style))
    story.append(Paragraph("3. <b>Uncertainty quantification</b> - 100 scenarios per prediction for risk assessment", bullet_style))
    story.append(Paragraph("4. <b>Interactive application</b> - Streamlit app for real-time experimentation", bullet_style))
    story.append(Spacer(1, 10))
    
    story.append(Paragraph("Why Our Approach Works (Slide 14 continued)", subsection_style))
    story.append(Paragraph("<b>SAY:</b> \"We succeeded because we matched the model complexity to the data size. "
                          "With 71 wells, simplicity wins over complexity.\"", talking_point_style))
    story.append(Spacer(1, 15))
    
    story.append(Paragraph("Thank You & Questions (Slide 15)", subsection_style))
    story.append(Paragraph("<b>SAY:</b> \"Thank you for your attention. We're happy to answer any questions about our "
                          "methodology, the domain science, or our interactive application.\"", talking_point_style))
    story.append(Spacer(1, 20))
    
    story.append(Paragraph("ANTICIPATED Q&A", section_style))
    story.append(Spacer(1, 10))
    
    story.append(Paragraph("Q: Why not use deep learning?", subsection_style))
    story.append(Paragraph("<b>A:</b> \"Deep learning needs thousands of examples to work well. With only 71 wells, "
                          "it would severely overfit. Our Ridge model is the right tool for this data size.\"", body_style))
    story.append(Spacer(1, 10))
    
    story.append(Paragraph("Q: How did you handle the spatial data?", subsection_style))
    story.append(Paragraph("<b>A:</b> \"We used a smoothed 3x3 sand proportion map. The smoothing reduces noise while "
                          "preserving the overall geological trends.\"", body_style))
    story.append(Spacer(1, 10))
    
    story.append(Paragraph("Q: What if the geology changes?", subsection_style))
    story.append(Paragraph("<b>A:</b> \"Our uncertainty ranges capture this. Wells in less-characterized areas will show "
                          "wider prediction intervals. The model learns from similar wells nearby.\"", body_style))
    story.append(Spacer(1, 10))
    
    story.append(Paragraph("Q: Can this scale to more wells?", subsection_style))
    story.append(Paragraph("<b>A:</b> \"Absolutely. And with more data, we could explore more complex models. "
                          "Our Streamlit app makes it easy to retrain with new data.\"", body_style))
    
    story.append(PageBreak())
    
    story.append(Paragraph("QUICK REFERENCE CARD", section_style))
    story.append(Spacer(1, 10))
    
    story.append(Paragraph("Key Numbers to Remember", subsection_style))
    numbers_data = [
        ['Item', 'Value'],
        ['Training Wells', '71'],
        ['Test Wells', '12 (IDs 72-83)'],
        ['Test R²', '0.9905 (99%)'],
        ['RMSE', '1.57M BBL (4.7%)'],
        ['Features Used', '10 (from 105)'],
        ['Uncertainty Scenarios', '100 per well'],
        ['Configurations Tested', '57+']
    ]
    numbers_table = Table(numbers_data, colWidths=[2.5*inch, 3*inch])
    numbers_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), HexColor('#8e44ad')),
        ('TEXTCOLOR', (0, 0), (-1, 0), white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 11),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('GRID', (0, 0), (-1, -1), 0.5, HexColor('#bdc3c7')),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('BACKGROUND', (0, 1), (-1, -1), HexColor('#f8f9fa')),
        ('LEFTPADDING', (0, 0), (-1, -1), 15),
        ('TOPPADDING', (0, 0), (-1, -1), 8),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
    ]))
    story.append(numbers_table)
    story.append(Spacer(1, 20))
    
    story.append(Paragraph("Academic References (If Asked)", subsection_style))
    story.append(Paragraph("• <b>Ridge Regression:</b> Hastie, Tibshirani & Friedman (2009)", bullet_style))
    story.append(Paragraph("• <b>Missing Data (MICE):</b> Van Buuren (2018)", bullet_style))
    story.append(Paragraph("• <b>Bagging:</b> Breiman (1996)", bullet_style))
    story.append(Paragraph("• <b>RQI/FZI:</b> Amaefule et al. (1993)", bullet_style))
    story.append(Spacer(1, 20))
    
    story.append(Paragraph("Winning Configuration", subsection_style))
    config_data = [
        ['Setting', 'Value'],
        ['Model', 'Ridge Regression'],
        ['Alpha', '0.1'],
        ['Normalization', 'StandardScaler (CRITICAL)'],
        ['Sand Map', 'Smooth 3x3'],
        ['Feature Selection', 'Stepwise (105 → 10)'],
        ['Uncertainty', 'Bagging Ensemble (100)']
    ]
    config_table = Table(config_data, colWidths=[2.5*inch, 3*inch])
    config_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), HexColor('#e74c3c')),
        ('TEXTCOLOR', (0, 0), (-1, 0), white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 11),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('GRID', (0, 0), (-1, -1), 0.5, HexColor('#bdc3c7')),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('BACKGROUND', (0, 1), (-1, -1), HexColor('#f8f9fa')),
        ('LEFTPADDING', (0, 0), (-1, -1), 15),
        ('TOPPADDING', (0, 0), (-1, -1), 8),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
    ]))
    story.append(config_table)
    story.append(Spacer(1, 30))
    
    story.append(Paragraph("Good luck with the presentation!", 
                          ParagraphStyle('GoodLuck', parent=highlight_style, textColor=HexColor('#27ae60'), fontSize=14)))
    story.append(Paragraph("Team Brain Oil - Energy AI Hackathon 2026", 
                          ParagraphStyle('Footer', parent=team_style, fontSize=10)))
    
    doc.build(story)
    print("PDF created successfully: Presentation_Walkthrough.pdf")

if __name__ == "__main__":
    create_walkthrough_pdf()
