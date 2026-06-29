import os
import cv2
import numpy as np
import onnxruntime as ort

current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, 'brain_tumor_model.onnx')

# Fallback initialization check if conversion hasn't run yet
if not os.path.exists(model_path):
    print("[!] Warning: brain_tumor_model.onnx not found. Please convert your h5 model to ONNX format.")
    session = None
else:
    # Initialize the high-efficiency ONNX CPU runtime session
    session = ort.InferenceSession(model_path)
    input_name = session.get_inputs()[0].name

def run_inference(img_path):
    """
    Executes high-speed CPU forward-pass predictions using ONNX Runtime.
    """
    if session is None:
        return {"class_label": "Model Configuration Pending", "confidence": "0.0%"}

    img = cv2.imread(img_path)
    if img is None:
        raise ValueError("Could not read or parse the target image sequence array.")
        
    # Resize image to match standard input parameters (224x224)
    img_resized = cv2.resize(img, (224, 224))
    
    # ONNX requires explicit float32 arrays with normalizations scaled between [0, 1]
    img_tensor = np.expand_dims(img_resized, axis=0).astype(np.float32) / 255.0

    # Execute lightning-fast inference tracking pass
    raw_predictions = session.run(None, {input_name: img_tensor})[0]
    pred_index = np.argmax(raw_predictions[0])
    confidence_val = raw_predictions[0][pred_index]

    # Map output channels back to standard clinical diagnostics strings
    classes = ["Glioma", "Meningioma", "Pituitary", "No Tumor"]
    class_label = classes[pred_index] if pred_index < len(classes) else "Unknown Anomaly"

    return {
        "class_label": class_label,
        "confidence": f"{confidence_val * 100:.2f}%"
    }
