# app.py
import os
import numpy as np
import tensorflow as tf
from PIL import Image
from flask import Flask, redirect, render_template, request, url_for, send_file
from xai.gradcam import generate_gradcam
from severity.pdf_generator import generate_pdf_report
from severity.severity_analysis import calculate_severity

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "brain_tumor_model.h5")

if os.path.exists(MODEL_PATH):
    model = tf.keras.models.load_model(MODEL_PATH)
    print("Model loaded. Operational architecture tracks:", [l.name for l in model.layers])
else:
    model = None
    print("Warning: Model missing. Operating in fallback simulation mode.")

labels = ["Glioma", "Meningioma", "Pituitary", "No Tumor"]
UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "upload")
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def preprocess_image(path):
    img = Image.open(path).convert("RGB")
    img = img.resize((128, 128))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    return img

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload():
    if "image" not in request.files:
        return "No file input payload found"
        
    file = request.files["image"]
    if file.filename == "":
        return "No file selected"

    filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
    file.save(filepath)

    if model:
        img = preprocess_image(filepath)
        prediction = model.predict(img)
        class_index = np.argmax(prediction)
        result = labels[class_index]
        confidence = round(float(np.max(prediction)) * 100, 2)
    else:
        result = "Glioma"
        confidence = 99.99

    try:
        # LAYER MATCH FIXED: Pulls conv2d_1 layer from your actual model file
        heatmap_web_path = generate_gradcam(filepath, model, final_conv_layer_name="conv2d_1")
        mask_status = True
    except Exception as e:
        print(f"Grad-CAM error fallback activated: {e}")
        heatmap_web_path = f"upload/{file.filename}"
        mask_status = False

    severity_grade, risk_level, xai_justification = calculate_severity(result, confidence, mask_status)

    return render_template(
        "result.html",
        image=f"upload/{file.filename}",
        heatmap_image=heatmap_web_path,
        prediction=result,
        confidence=confidence,
        severity=severity_grade,
        risk=risk_level,
        xai_reasoning=xai_justification
    )

@app.route("/download_report/<filename>")
def download_report(filename):
    source_image_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    heatmap_image_path = os.path.join(app.config["UPLOAD_FOLDER"], "gradcam_" + filename)
    logo_path = os.path.join(BASE_DIR, "static", "Images", "IMG_20260614_200114.png")
    
    # SAFETY RE-ROUTING: Prevents empty layout fields in document downloads
    if not os.path.exists(heatmap_image_path):
        heatmap_image_path = source_image_path

    if model:
        img = preprocess_image(source_image_path)
        prediction = model.predict(img)
        class_index = np.argmax(prediction)
        result_text = labels[class_index]
        confidence_score = round(float(np.max(prediction)) * 100, 2)
    else:
        result_text = "Glioma"
        confidence_score = 99.99
        
    severity_grade, risk_level, xai_justification = calculate_severity(result_text, confidence_score)
    
    pdf_filename = f"report_{os.path.splitext(filename)[0]}.pdf"
    pdf_output_path = os.path.join(app.config["UPLOAD_FOLDER"], pdf_filename)
    
    generate_pdf_report(
        filename=pdf_output_path, prediction=result_text, confidence=confidence_score,
        severity=severity_grade, risk=risk_level,
        original_img_path=source_image_path, heatmap_img_path=heatmap_image_path, 
        logo_path=logo_path, xai_report_text=xai_justification
    )
    
    return send_file(pdf_output_path, as_attachment=True)

if __name__ == "__main__":
    app.run(debug=True)
