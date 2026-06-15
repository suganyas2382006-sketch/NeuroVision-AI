from flask import Flask, render_template, request, redirect, url_for
import os
import json
from datetime import datetime

from severity.logic import get_severity
from xai.gradcam import generate_gradcam

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
HEATMAP_FOLDER = "static/heatmaps"
HISTORY_FILE = "instance/history.json"

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(HEATMAP_FOLDER, exist_ok=True)
os.makedirs("instance", exist_ok=True)


def save_history(entry):
    if not os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "w") as f:
            json.dump([], f)

    with open(HISTORY_FILE, "r") as f:
        data = json.load(f)

    data.append(entry)

    with open(HISTORY_FILE, "w") as f:
        json.dump(data, f, indent=4)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/upload")
def upload():
    return render_template("upload.html")


@app.route("/predict", methods=["POST"])
def predict():

    file = request.files["image"]

    if file.filename == "":
        return "No file selected"

    filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
    file.save(filepath)

    # -------------------------
    # FAKE MODEL PREDICTION (replace with ML model later)
    # -------------------------
    prediction = "Tumor Detected"
    confidence = 92.5

    severity = get_severity(confidence)

    # GradCAM (dummy for now)
    heatmap_path = generate_gradcam(filepath)

    # Save history
    save_history({
        "time": str(datetime.now()),
        "image": file.filename,
        "prediction": prediction,
        "confidence": confidence,
        "severity": severity
    })

    return render_template(
        "result.html",
        image=file.filename,
        prediction=prediction,
        confidence=confidence,
        severity=severity,
        heatmap=heatmap_path
    )


@app.route("/heatmap")
def heatmap():
    return render_template("heatmap.html")


@app.route("/xai")
def xai():
    return render_template("xai.html")


@app.route("/history")
def history():

    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r") as f:
            data = json.load(f)
    else:
        data = []

    return render_template("history.html", history=data)


if __name__ == "__main__":
    app.run(debug=True)