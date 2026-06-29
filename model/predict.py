import os
import cv2
import numpy as np
import tensorflow as tf

current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, 'brain_tumor_model.h5')

if os.path.exists(model_path):
    model = tf.keras.models.load_model(model_path)
else:
    model = None

def run_inference(img_path):
    if model is None:
        return {"class_label": "Model Weight Asset Missing", "confidence": "0.0%"}
        
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError("Could not read incoming image framework.")
        
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # MATCHES THE NEW 128x128 MODEL INPUT FORMAT
    img_resized = cv2.resize(img_rgb, (128, 128)) / 255.0
    input_tensor = np.expand_dims(img_resized, axis=0)
    
    predictions = model.predict(input_tensor, verbose=0)[0]
    pred_index = np.argmax(predictions)
    confidence_val = predictions[pred_index]
    
    # Class tracking order matching Keras flow_from_directory alphabetical sorting
    classes = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]
    
    return {
        "class_label": classes[pred_index],
        "confidence": f"{confidence_val * 100:.2f}%"
    }
