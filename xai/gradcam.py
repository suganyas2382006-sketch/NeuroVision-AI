# xai/gradcam.py
import os
import numpy as np
import tensorflow as tf
cv2 = None

try:
    import cv2
except ImportError:
    pass  # Fallback manual pooling logic if OpenCV is not installed on mobile environment

def generate_gradcam(img_path, model, final_conv_layer_name="conv2d_1", intensity=0.4, res=128):
    """
    Computes Gradient-weighted Class Activation Mapping (Grad-CAM) to isolate 
    and visualize the spatial regions where the convolutional layers focus.
    """
    if model is None:
        raise ValueError("Grad-CAM Engine Error: Target TensorFlow model object is uninitialized.")

    # 1. Image Preprocessing Pipelines
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=(res, res))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_tensor = np.expand_dims(img_array, axis=0) / 255.0

    # 2. Extract Gradients via GradientTape Framework
    grad_model = tf.keras.models.Model(
        [model.inputs], 
        [model.get_layer(final_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_tensor)
        class_idx = np.argmax(predictions[0])
        loss = predictions[:, class_idx]

    # Calculate the gradients of the target class loss with respect to the feature map activations
    grads = tape.gradient(loss, conv_outputs)

    # 3. Compute Channel-Wise Weights (Global Average Pooling)
    guided_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # 4. Generate the Heatmap Matrix
    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(tf.multiply(guided_grads, conv_outputs), axis=-1)

    # Apply ReLU activation function (keep only positive influences)
    heatmap = np.maximum(heatmap, 0)
    
    # DYNAMIC NORMALIZATION FIX: Ensures lower confidence signals scale cleanly to bright colors
    max_val = np.max(heatmap)
    min_val = np.min(heatmap)
    if max_val - min_val > 0:
        heatmap = (heatmap - min_val) / (max_val - min_val)
    else:
        heatmap = np.zeros_like(heatmap)

    # 5. Build and Save the Color Image Output File
    output_dir = os.path.dirname(img_path)
    base_filename = os.path.basename(img_path)
    output_path = os.path.join(output_dir, "gradcam_" + base_filename)

    # Image formatting loop
    if cv2:
        img_raw = cv2.imread(img_path)
        img_raw = cv2.resize(img_raw, (res, res))
        
        # CHANNEL FIX: Force grayscale MRI matrices into 3 color channels
        if len(img_raw.shape) == 2 or img_raw.shape[2] == 1:
            img_raw = cv2.cvtColor(img_raw, cv2.COLOR_GRAY2BGR)

        heatmap_resized = cv2.resize(heatmap, (res, res))
        heatmap_uint8 = np.uint8(255 * heatmap_resized)
        
        # Color Overlay: Transform single-channel grayscale into a 3-channel Jet color spectrum
        color_map = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        
        # BLENDING FIX: Set original image weight to 0.6 so the heatmap can actually show through
        blended_img = cv2.addWeighted(img_raw, 0.6, color_map, intensity, 0)
        cv2.imwrite(output_path, blended_img)
    else:
        # Emergency Pure-Python fallback matrix if OpenCV is missing inside the container terminal
        from PIL import Image as PILImage
        from matplotlib import cm
        
        img_pil = img.convert("RGB")
        heatmap_pil = PILImage.fromarray(np.uint8(255 * heatmap)).resize((res, res))
        heatmap_rgba = np.array(cm.jet(np.array(heatmap_pil)))[:, :, :3] * 255
        
        # Manual fallback pixel blend matching the 0.6 / 0.4 weight scale
        blended_matrix = (np.array(img_pil).astype(float) * 0.6) + (heatmap_rgba * intensity)
        blended_img = PILImage.fromarray(np.uint8(np.clip(blended_matrix, 0, 255)))
        blended_img.save(output_path)

    # Returns the web-ready string relative path file path format used by the Flask Jinja engine template
    return f"upload/gradcam_{base_filename}"
