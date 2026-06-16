# severity/pdf_generator.py
import os
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

def generate_pdf_report(filename, prediction, confidence, severity, risk, original_img_path, heatmap_img_path, logo_path, xai_report_text):
    """
    Generates a secure, structured medical diagnostic report chart including branded logo,
    dual visual evidence mapping, and human-interpretable clinical justification text.
    """
    doc = SimpleDocTemplate(
        filename,
        pagesize=letter,
        rightMargin=40, leftMargin=40,
        topMargin=40, bottomMargin=40
    )
    
    story = []
    styles = getSampleStyleSheet()
    
    primary_color = colors.HexColor("#000054")     
    secondary_color = colors.HexColor("#00c8ff")   
    text_dark = colors.HexColor("#111111")         
    
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=24,
        textColor=primary_color,
        spaceAfter=6
    )
    
    section_heading = ParagraphStyle(
        'SectionHeading',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=14,
        textColor=primary_color,
        spaceBefore=12,
        spaceAfter=8
    )
    
    body_style = ParagraphStyle(
        'BodyDark',
        parent=styles['BodyText'],
        fontName='Helvetica',
        fontSize=10.5,
        textColor=text_dark,
        leading=14
    )
    
    xai_box_style = ParagraphStyle(
        'XaiText',
        parent=styles['BodyText'],
        fontName='Helvetica-Oblique',
        fontSize=10,
        textColor=colors.HexColor("#333333"),
        leading=14
    )

    header_data = []
    if os.path.exists(logo_path):
        try:
            logo_img = Image(logo_path, width=50, height=50)
            header_data.append([logo_img, Paragraph("NeuroVision AI Diagnostic Report", title_style)])
        except Exception as e:
            header_data.append(["", Paragraph("NeuroVision AI Diagnostic Report", title_style)])
    else:
        header_data.append(["", Paragraph("NeuroVision AI Diagnostic Report", title_style)])
        
    header_table = Table(header_data, colWidths=[60, 470])
    header_table.setStyle(TableStyle([
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('LEFTPADDING', (1,0), (1,0), 0),
    ]))
    story.append(header_table)
    
    divider = Table([[""]], colWidths=[532], rowHeights=[3])
    divider.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (0,0), secondary_color),
        ('BOTTOMPADDING', (0,0), (-1,-1), 0),
        ('TOPPADDING', (0,0), (-1,-1), 0),
    ]))
    story.append(Spacer(1, 4))
    story.append(divider)
    story.append(Spacer(1, 15))

    story.append(Paragraph("Diagnostic Classification Summary", section_heading))
    
    risk_text_color = "#33cc33" 
    if risk == "Critical":
        risk_text_color = "#ff3333" 
    elif risk == "Moderate":
        risk_text_color = "#ffaa00" 
        
    summary_matrix = [
        [Paragraph("<b>Predicted Classification Class:</b>", body_style), Paragraph(prediction, body_style)],
        [Paragraph("<b>System Analysis Confidence:</b>", body_style), Paragraph(f"{confidence}%", body_style)],
        [Paragraph("<b>Calculated Progression Stage:</b>", body_style), Paragraph(severity, body_style)],
        [Paragraph("<b>Risk Evaluation Flag:</b>", body_style), Paragraph(f"<font color='{risk_text_color}'><b>{risk.upper()}</b></font>", body_style)],
    ]
    
    summary_table = Table(summary_matrix, colWidths=[200, 332])
    summary_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#f8f9fa")),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#e9ecef")),
        ('PADDING', (0,0), (-1,-1), 8),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
    ]))
    story.append(summary_table)
    story.append(Spacer(1, 15))

    story.append(Paragraph("Clinical Interpretability & Justification (XAI)", section_heading))
    xai_container_data = [[Paragraph(f"<b>Algorithmic Path Rationale:</b><br/>{xai_report_text}", xai_box_style)]]
    xai_table = Table(xai_container_data, colWidths=[532])
    xai_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (0,0), colors.HexColor("#f0faff")), 
        ('BOX', (0,0), (0,0), 1, colors.HexColor("#bee5eb")),
        ('LINELEFT', (0,0), (0,0), 4, secondary_color), 
        ('PADDING', (0,0), (-1,-1), 12),
    ]))
    story.append(xai_table)
    story.append(Spacer(1, 15))

    story.append(Paragraph("Visual Evidence Localization (MRI Scan vs Attention Map Overlay)", section_heading))
    image_row = []
    
    if os.path.exists(original_img_path):
        try:
            img_scan = Image(original_img_path, width=240, height=240)
            image_row.append(img_scan)
        except:
            image_row.append(Paragraph("[Error Loading Source MRI Image Frame]", body_style))
    else:
        image_row.append(Paragraph("[Source MRI Image File Not Extracted]", body_style))
        
    if os.path.exists(heatmap_img_path):
        try:
            img_heatmap = Image(heatmap_img_path, width=240, height=240)
            image_row.append(img_heatmap)
        except:
            image_row.append(Paragraph("[Error Loading Heatmap Frame Data]", body_style))
    else:
        image_row.append(Paragraph("[Attention Map Overlay Frame Missing]", body_style))

    image_table = Table([image_row], colWidths=[266, 266])
    image_table.setStyle(TableStyle([
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('BOTTOMPADDING', (0,0), (-1,-1), 10),
    ]))
    story.append(image_table)
    story.append(Spacer(1, 20))
    
    disclaimer_style = ParagraphStyle(
        'DisclaimerText',
        parent=styles['Normal'],
        fontName='Helvetica-Oblique',
        fontSize=8,
        textColor=colors.HexColor("#6c757d"),
        leading=11
    )
    disclaimer_text = (
        "Disclaimer: This document contains computed deep learning convolutional analytics "
        "intended purely as an interpretive auxiliary diagnostic aid. Absolute radiologist verification "
        "and multi-sequence clinical confirmation remain mandatory prerequisites prior to treatment roadmap initialization."
    )
    story.append(Paragraph(disclaimer_text, disclaimer_style))

    doc.build(story)
