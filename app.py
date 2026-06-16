import os
import numpy as np
import tensorflow as tf
from PIL import Image
from flask import Flask, redirect, render_template, request, url_for
from xai.gradcam import generate_gradcam

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "brain_tumor_model.h5")

if os.path.exists(MODEL_PATH):
    model = tf.keras.models.load_model(MODEL_PATH)
    # Debug helper line to see layer names inside your terminal console:
    print("Model Loaded. Available layers:", [l.name for l in model.layers])
else:
    raise FileNotFoundError(f"Critical Error: Model file missing at {MODEL_PATH}")

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

    # 1. Core Model Prediction
    img = preprocess_image(filepath)
    prediction = model.predict(img)
    class_index = np.argmax(prediction)
    result = labels[class_index]
    confidence = float(np.max(prediction)) * 100

    # 2. Real-Time Explainable AI (Grad-CAM) Execution
    # NOTE: Look at your terminal log when launching to verify your model's final conv layer name.
    # Replace 'conv2d_last' with that layer string (e.g., 'conv2d_3') if your architecture differs.
    try:
        heatmap_web_path = generate_gradcam(filepath, model, final_conv_layer_name="conv2d_last")
    except Exception as e:
        print(f"Grad-CAM generation failed, using fallback image. Error: {e}")
        heatmap_web_path = f"upload/{file.filename}"

    return render_template(
        "result.html",
        image=f"upload/{file.filename}",
        heatmap_image=heatmap_web_path,
        prediction=result,
        confidence=round(confidence, 2)
    )


if __name__ == "__main__":
    app.run(debug=True)
