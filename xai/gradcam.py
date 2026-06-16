# xai/gradcam.py
import os
import numpy as np
import tensorflow as tf
cv2 = None

try:
    import cv2
except ImportError:
    pass  # Fallback manual pooling logic if OpenCV is not installed on mobile environment

def generate_gradcam(img_path, model, final_conv_layer_name="conv2d_1", intensity=0.5, res=128):
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
    # Create a functional sub-model mapping the original input layer straight to the target conv layer
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
    # Mean intensity of the gradients across each feature map channel
    guided_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # 4. Generate the Heatmap Matrix
    # Multiply feature maps by weight vectors, then average them across channels
    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(tf.multiply(guided_grads, conv_outputs), axis=-1)

    # Apply ReLU activation function (keep only positive influences) and normalize
    heatmap = np.maximum(heatmap, 0)
    max_val = np.max(heatmap)
    if max_val == 0:
        max_val = 1e-10  # Prevent math division-by-zero crashes
    heatmap /= max_val

    # 5. Build and Save the Color Image Output File
    output_dir = os.path.dirname(img_path)
    base_filename = os.path.basename(img_path)
    output_path = os.path.join(output_dir, "gradcam_" + base_filename)

    # Image formatting loop
    img_raw = cv2.imread(img_path) if cv2 else np.array(img)
    img_raw = cv2.resize(img_raw, (res, res)) if cv2 else img_raw

    # Resize the heatmap grid to match original dimensions
    if cv2:
        heatmap_resized = cv2.resize(heatmap, (res, res))
        heatmap_uint8 = np.uint8(255 * heatmap_resized)
        
        # Color Overlay: Transform single-channel grayscale into a 3-channel Jet color spectrum
        color_map = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        
        # Blend the heatmap directly on top of the original grayscale MRI scan image frame
        blended_img = cv2.addWeighted(img_raw, 1.0, color_map, intensity, 0)
        cv2.imwrite(output_path, blended_img)
    else:
        # Emergency Pure-Python fallback matrix if OpenCV is missing inside the container terminal
        from PIL import Image as PILImage
        from matplotlib import cm
        
        heatmap_pil = PILImage.fromarray(np.uint8(255 * heatmap)).resize((res, res))
        heatmap_rgba = np.array(cm.jet(np.array(heatmap_pil)))[:, :, :3] * 255
        
        blended_matrix = (np.array(img).astype(float) * (1 - intensity)) + (heatmap_rgba * intensity)
        blended_img = PILImage.fromarray(np.uint8(np.clip(blended_matrix, 0, 255)))
        blended_img.save(output_path)

    # Returns the web-ready string relative path file path format used by the Flask Jinja engine template
    return f"upload/gradcam_{base_filename}"
