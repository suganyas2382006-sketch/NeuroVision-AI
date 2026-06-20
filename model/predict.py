import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image

BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, 'brain_tumor_model.h5')
LABELS_PATH = os.path.join(BASE_DIR, 'labels.txt')

model = tf.keras.models.load_model(MODEL_PATH)

with open(LABELS_PATH, 'r') as f:
    classes = [line.strip() for line in f.readlines()]

def run_inference(image_path):
    img = image.load_img(image_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0 

    predictions = model.predict(img_array)
    predicted_class_idx = np.argmax(predictions[0])
    
    confidence = predictions[0][predicted_class_idx] * 100
    predicted_label = classes[predicted_class_idx]

    return {
        "class_label": predicted_label,
        "confidence": f"{confidence:.1f}%"
    }
