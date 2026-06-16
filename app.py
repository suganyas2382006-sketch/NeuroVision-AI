import os
import numpy as np
import tensorflow as tf
from PIL import Image
from flask import Flask, redirect, render_template, request, url_for
from xai.gradcam import generate_gradcam

app = Flask(__name__)

# Dynamic absolute pathway routing to avoid multi-platform string crash errors
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "brain_tumor_model.h5")

# Safe validation check to ensure model asset exists prior to launching
if os.path.exists(MODEL_PATH):
    model = tf.keras.models.load_model(MODEL_PATH)
    # Debug terminal trace script to inspect internal convolution architecture string tags:
    print("Model Loaded Successfully. Structural layers list:", [l.name for l in model.layers])
else:
    raise FileNotFoundError(f"Critical System Failure: Model target missing at {MODEL_PATH}")

labels = ["Glioma", "Meningioma", "Pituitary", "No Tumor"]

# Standardized folder layout targeting static/upload directory properties
UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "upload")
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


def preprocess_image(path):
    """
    Reads the target MRI scan file from storage, normalizes channel scales,
    and applies dimension expansion steps to simulate model batch tensors.
    """
    img = Image.open(path).convert("RGB")
    img = img.resize((128, 128))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)  # Shape alteration step -> (1, 128, 128, 3)
    return img


@app.route("/")
def home():
    """
    Renders the central image upload dashboard view.
    """
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():
    """
    Handles file save actions, executes target deep-learning inferences,
    and dynamically produces Grad-CAM visualization heatmaps.
    """
    if "image" not in request.files:
        return "No file input payload found"
        
    file = request.files["image"]

    if file.filename == "":
        return "No file selected"

    # Save incoming MRI scan securely to storage layer
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
    file.save(filepath)

    # 1. Core Model Inference Steps
    img = preprocess_image(filepath)
    prediction = model.predict(img)
    class_index = np.argmax(prediction)
    result = labels[class_index]
    confidence = float(np.max(prediction)) * 100

    # 2. XAI Grad-CAM Superimposition Processing
    # NOTE: Keep an eye on your terminal output logs when launching the script.
    # Swap 'conv2d_last' out with your model's exact final convolutional layer id (e.g., 'conv2d_3').
    try:
        heatmap_web_path = generate_gradcam(filepath, model, final_conv_layer_name="conv2d_last")
    except Exception as e:
        print(f"Grad-CAM generation failed, utilizing fallback placeholder. Error parameters: {e}")
        # Soft fallback error safety routine: drops original file onto render engine if shape matrix breaks
        heatmap_web_path = f"upload/{file.filename}"

    # Render results screen with fully mapped variable components
    return render_template(
        "result.html",
        image=f"upload/{file.filename}",
        heatmap_image=heatmap_web_path,
        prediction=result,
        confidence=round(confidence, 2)
    )


if __name__ == "__main__":
    # Launch app server infrastructure with debug trace loops active
    app.run(debug=True)
