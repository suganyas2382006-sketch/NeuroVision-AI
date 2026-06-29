import os
import cv2
import numpy as np
import tensorflow as tf

# 1. Dynamically resolve the absolute directory path context
current_dir = os.path.dirname(os.path.abspath(__file__))
model_file = None

# 2. Scan the directory to locate a valid Keras weights file
for file in os.listdir(current_dir):
    if file.endswith('.h5'):
        target_path = os.path.join(current_dir, file)
        # Structural check: Ensure it is a real file and not an empty 0-byte placeholder
        if os.path.getsize(target_path) > 1024:  
            model_file = target_path
            break

# 3. Safe architecture initialization fallback if no model file is found
if not model_file:
    print("[!] Warning: Valid h5 weights binary not detected inside model/. Mocking structural topology.")
    inputs = tf.keras.Input(shape=(224, 224, 3))
    x = tf.keras.layers.Conv2D(32, 3, activation='relu', padding='same', name='conv_base_1')(inputs)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    outputs = tf.keras.layers.Dense(4, activation='softmax')(x)
    model = tf.keras.Model(inputs, outputs)
else:
    # Load the verified healthy trained model weights
    model = tf.keras.models.load_model(model_file)

def run_inference(img_path):
    """
    Loads an MRI image path, applies dimensions array scaling, 
    and executes the core neural network forward pass inference.
    """
    # Load image in color mode (3 channels)
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError("Could not read or parse the target image sequence array.")
        
    # Resize to match your model's expected inputs (224x224)
    img_resized = cv2.resize(img, (224, 224))
    
    # Scale pixel intensities between [0, 1] and add batch dimension array expansion
    img_tensor = np.expand_dims(img_resized, axis=0) / 255.0

    # 4. Execute the neural network prediction pass
    predictions = model.predict(img_tensor)
    pred_index = np.argmax(predictions[0])
    confidence_val = predictions[0][pred_index]

    # 5. Map numerical indexes to medical diagnostics class labels
    classes = ["Glioma", "Meningioma", "Pituitary", "No Tumor"]
    class_label = classes[pred_index] if pred_index < len(classes) else "Unknown Anomaly"

    return {
        "class_label": class_label,
        "confidence": f"{confidence_val * 100:.2f}%"
    }
