import os
import numpy as np
import tensorflow as tf
from PIL import Image
from flask import Flask, redirect, render_template, request, url_for, send_file
from xai.gradcam import generate_gradcam
from severity.pdf_generator import generate_pdf_report

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "brain_tumor_model.h5")

if os.path.exists(MODEL_PATH):
    model = tf.keras.models.load_model(MODEL_PATH)
    print("Model loaded. Operational architecture tracks:", [l.name for l in model.layers])
else:
    raise FileNotFoundError(f"Missing core runtime network asset at {MODEL_PATH}")

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

    img = preprocess_image(filepath)
    prediction = model.predict(img)
    class_index = np.argmax(prediction)
    result = labels[class_index]
    confidence = float(np.max(prediction)) * 100

    try:
        # NOTE: Swap 'conv2d_last' with your model's exact last conv layer name string (e.g., 'conv2d_3')
        heatmap_web_path = generate_gradcam(filepath, model, final_conv_layer_name="conv2d_last")
    except Exception as e:
        print(f"Grad-CAM system exception: {e}")
        heatmap_web_path = f"upload/{file.filename}"

    return render_template(
        "result.html",
        image=f"upload/{file.filename}",
        heatmap_image=heatmap_web_path,
        prediction=result,
        confidence=round(confidence, 2)
    )


@app.route("/download_report/<filename>")
def download_report(filename):
    source_image_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    heatmap_image_path = os.path.join(app.config["UPLOAD_FOLDER"], "gradcam_" + filename)
    logo_path = os.path.join(BASE_DIR, "static", "images", "IMG_20260614_200114.png")
    
    img = preprocess_image(source_image_path)
    prediction = model.predict(img)
    class_index = np.argmax(prediction)
    result_text = labels[class_index]
    confidence_score = round(float(np.max(prediction)) * 100, 2)
    
    pdf_filename = f"report_{os.path.splitext(filename)[0]}.pdf"
    pdf_output_path = os.path.join(app.config["UPLOAD_FOLDER"], pdf_filename)
    
    generate_pdf_report(
        filename=pdf_output_path, prediction=result_text, confidence=confidence_score,
        original_img_path=source_image_path, heatmap_img_path=heatmap_image_path, logo_path=logo_path
    )
    
    return send_file(pdf_output_path, as_attachment=True)


if __name__ == "__main__":
    app.run(debug=True)
