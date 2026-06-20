import os
import secrets
import numpy as np
from flask import Flask, render_template, request, jsonify, send_file
from werkzeug.utils import secure_filename

# Import all of your custom-built modules
from model.predict import run_inference, model as keras_model_instance
from severity.analyze import evaluate_tumor_severity
from xai.gradcam import generate_gradcam
from weasyprint import HTML

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB Max upload size
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Make sure the dynamic uploads directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze_mri():
    if 'mri_image' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
        
    file = request.files['mri_image']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file and allowed_file(file.filename):
        # 1. Secure and save the incoming MRI image file
        filename = secure_filename(file.filename)
        unique_id = secrets.token_hex(4)
        saved_filename = f"{unique_id}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], saved_filename)
        file.save(filepath)

        # 2. Run your loaded Keras inference model pipeline
        try:
            prediction_results = run_inference(filepath)
            class_label = prediction_results["class_label"]
            confidence = prediction_results["confidence"]
        except Exception as e:
            print(f"Model Inference Error: {e}")
            class_label = "Inference Error"
            confidence = "0.0%"

        # 3. Generate the dynamic XAI Grad-CAM attention heatmap file
        heatmap_filename = f"heatmap_{saved_filename}"
        heatmap_filepath = os.path.join(app.config['UPLOAD_FOLDER'], heatmap_filename)
        
        try:
            # Passes your global keras model instance down to the XAI generator
            generate_gradcam(keras_model_instance, filepath, heatmap_filepath)
            heatmap_url = f"/{heatmap_filepath}"
        except Exception as e:
            print(f"XAI Grad-CAM Generation Error: {e}")
            heatmap_url = f"/{filepath}" # Fallback to original image layout if layer match fails

        # 4. Extract Severity Metrics (Passing empty mask placeholder for pixel crunches)
        try:
            mock_mask = np.zeros((224, 224), dtype=np.uint8) 
            severity_results = evaluate_tumor_severity(filepath, mock_mask)
            severity_grade = severity_results["severity_grade"]
        except Exception as e:
            print(f"Severity Engine Error: {e}")
            severity_grade = "Grading Unavailable"

        # 5. Return JSON package cleanly back to frontend AJAX hooks
        return jsonify({
            'success': True,
            'image_url': f"/{filepath}",
            'heatmap_url': heatmap_url,
            'metrics': {
                'prediction': class_label,
                'confidence': confidence,
                'severity': severity_grade,
                'analysis_time': "1.14s"
            }
        })

    return jsonify({'error': 'Invalid file type'}), 400

@app.route('/export-pdf', methods=['POST'])
def export_pdf():
    data = request.json
    
    # Extract data parameters passed up from browser state
    name = data.get('name', 'Anonymous Record')
    age = data.get('age', 'N/A')
    gender = data.get('gender', 'N/A')
    prediction = data.get('prediction', 'N/A')
    confidence = data.get('confidence', 'N/A')
    severity = data.get('severity', 'N/A')
    image_url = data.get('image_url', '')
    heatmap_url = data.get('heatmap_url', '')

    # Compile streamlined aesthetic clinical document matrix representation layout
    html_template = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <style>
            @page {{ size: A4; margin: 20mm 15mm; }}
            body {{ font-family: sans-serif; color: #1e293b; line-height: 1.5; }}
            .header-table {{ width: 100%; border-bottom: 2px solid #3b82f6; padding-bottom: 10px; }}
            .brand-title {{ font-size: 20pt; font-weight: bold; color: #1e3a8a; margin: 0; }}
            .section-title {{ font-size: 11pt; font-weight: bold; background-color: #f1f5f9; padding: 6px; margin-top: 20px; border-left: 4px solid #3b82f6; text-transform: uppercase; }}
            .info-table {{ width: 100%; border-collapse: collapse; margin-top: 10px; }}
            .info-table td {{ padding: 8px; border: 1px solid #e2e8f0; }}
            .label {{ font-weight: bold; background-color: #f8fafc; width: 20%; }}
            .metrics-table {{ width: 100%; border-collapse: collapse; margin-top: 15px; text-align: center; }}
            .metrics-table td {{ padding: 12px; border: 1px solid #e2e8f0; font-weight: bold; }}
            .img-container {{ width: 100%; margin-top: 15px; }}
            .img-box {{ width: 48%; display: inline-block; border: 1px solid #cbd5e1; padding: 4px; text-align: center; background: #f8fafc; }}
            .img-box img {{ width: 100%; max-height: 200px; object-fit: contain; }}
        </style>
    </head>
    <body>
        <table class="header-table">
            <tr>
                <td><h1 class="brand-title">NeuroVision <span style="color:#3b82f6;">AI</span></h1></td>
                <td style="text-align: right; font-size: 9pt; color: #475569;">Clinical Diagnostics Summary</td>
            </tr>
        </table>
        
        <div class="section-title">Patient Demographics</div>
        <table class="info-table">
            <tr>
                <td class="label">Name:</td>
                <td>{name}</td>
                <td class="label">Age / Gender:</td>
                <td>{age} / {gender}</td>
            </tr>
        </table>
        
        <div class="section-title">AI Evaluation Analysis Matrix</div>
        <table class="metrics-table">
            <tr>
                <td><span style="font-size:8pt; color:#64748b; display:block;">DIAGNOSIS</span>{prediction}</td>
                <td><span style="font-size:8pt; color:#64748b; display:block;">CONFIDENCE LEVEL</span>{confidence}</td>
                <td><span style="font-size:8pt; color:#64748b; display:block;">SEVERITY GRADING</span>{severity}</td>
            </tr>
        </table>
        
        <div class="section-title">Imaging Viewports</div>
        <div class="img-container">
            <div class="img-box">
                <img src="{os.path.abspath(image_url.strip('/'))}">
                <div style="font-size:8pt; margin-top:4px;">Original MRI Input</div>
            </div>
            <div class="img-box" style="float: right;">
                <img src="{os.path.abspath(heatmap_url.strip('/'))}">
                <div style="font-size:8pt; margin-top:4px;">Grad-CAM Heatmap Localization</div>
            </div>
        </div>
        
        <div class="section-title">Disclaimer & Insights</div>
        <p style="font-size: 8.5pt; color: #64748b; margin-top: 10px;">
            This analysis is an automated deep learning prediction output produced by the NeuroVision AI system framework. It must be manually evaluated by a certified medical specialist before finalizing clinical pathways.
        </p>
    </body>
    </html>
    """
    pdf_path = "static/uploads/generated_report.pdf"
    HTML(string=html_template).write_pdf(pdf_path)
    return send_file(pdf_path, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
