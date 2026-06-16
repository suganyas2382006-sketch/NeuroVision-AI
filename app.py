from flask import Flask, render_template, request, redirect, url_for
import os
import numpy as np
from PIL import Image
import tensorflow as tf

app = Flask(__name__)

# Load model
model = tf.keras.models.load_model("model/brain_tumor_model.h5")

labels = ["Glioma", "Meningioma", "Pituitary", "No Tumor"]

UPLOAD_FOLDER = "static/uploads"
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


@app.route("/upload", methods=["GET", "POST"])
def upload():
    if request.method == "POST":
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

        return render_template(
            "result.html",
            image=file.filename,
            prediction=result,
            confidence=round(confidence, 2)
        )

    return render_template("upload.html")


@app.route("/heatmap")
def heatmap():
    return render_template("heatmap.html")


@app.route("/xai")
def xai():
    return render_template("xai.html")


if __name__ == "__main__":
    app.run(debug=True)