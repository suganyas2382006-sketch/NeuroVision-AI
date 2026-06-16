import os
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors

def generate_pdf_report(filename, prediction, confidence, original_img_path, heatmap_img_path, logo_path):
    """
    Builds a structured medical PDF presentation sheet using ReportLab flowables.
    """
    doc = SimpleDocTemplate(filename, pagesize=letter, rightMargin=40, leftMargin=40, topMargin=40, bottomMargin=40)
    story = []
    styles = getSampleStyleSheet()
    
    title_style = ParagraphStyle(
        'DocTitle', parent=styles['Heading1'], fontName='Helvetica-Bold',
        fontSize=24, textColor=colors.HexColor('#000054'), spaceAfter=4
    )
    section_heading = ParagraphStyle(
        'SectionHeading', parent=styles['Heading2'], fontName='Helvetica-Bold',
        fontSize=15, textColor=colors.HexColor('#000054'), spaceBefore=12, spaceAfter=8
    )
    body_style = ParagraphStyle(
        'BodyText', parent=styles['Normal'], fontName='Helvetica', fontSize=11, leading=15, textColor=colors.HexColor('#333333')
    )

    header_data = []
    if os.path.exists(logo_path):
        img_logo = Image(logo_path, width=45, height=45)
        header_data.append([img_logo, Paragraph("NeuroVision AI Report<br/><font size=9 color='#666666'>Clinical Diagnostic Assistant Engine</font>", title_style)])
    else:
        header_data.append([Paragraph("NeuroVision AI Diagnostic Report", title_style)])
        
    header_table = Table(header_data, colWidths=[55, 455] if os.path.exists(logo_path) else [510])
    header_table.setStyle(TableStyle([('VALIGN', (0,0), (-1,-1), 'MIDDLE'), ('BOTTOMPADDING', (0,0), (-1,-1), 8)]))
    story.append(header_table)
    
    line_table = Table([[""]], colWidths=[530], rowHeights=[2])
    line_table.setStyle(TableStyle([('BACKGROUND', (0,0), (0,0), colors.HexColor('#00c8ff'))]))
    story.append(line_table)
    story.append(Spacer(1, 10))

    story.append(Paragraph("Diagnostic Classification Summary", section_heading))
    summary_text = f"<b>Predicted Classification Class:</b> {prediction}<br/><b>System Analysis Confidence:</b> {confidence}%"
    story.append(Paragraph(summary_text, body_style))
    story.append(Spacer(1, 15))

    story.append(Paragraph("Visual Evidence Localization (MRI Scan vs Attention Map Overlay)", section_heading))
    img_matrix_data = []
    if os.path.exists(original_img_path) and os.path.exists(heatmap_img_path):
        mri_flowable = Image(original_img_path, width=240, height=240)
        cam_flowable = Image(heatmap_img_path, width=240, height=240)
        img_matrix_data.append([mri_flowable, cam_flowable])
        img_matrix_data.append([Paragraph("Source MRI Structural Scan", body_style), Paragraph("Grad-CAM Activation Map Area", body_style)])
        
        image_table = Table(img_matrix_data, colWidths=[260, 260])
        image_table.setStyle(TableStyle([('ALIGN', (0,0), (-1,-1), 'CENTER'), ('VALIGN', (0,0), (-1,-1), 'MIDDLE'), ('TOPPADDING', (0,1), (-1,1), 4)]))
        story.append(image_table)
        
    story.append(Spacer(1, 20))
    disclaimer = "<i>Disclaimer: This document contains deep learning analytics intended purely as an interpretive diagnostic aid. Radiologist confirmation is required.</i>"
    story.append(Paragraph(disclaimer, ParagraphStyle('Disc', parent=body_style, fontSize=8.5, leading=11, textColor=colors.HexColor('#666666'))))

    doc.build(story)
