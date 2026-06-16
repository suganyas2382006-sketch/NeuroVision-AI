# xai/gradcam.py
import os
import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')  # Force non-interactive backend so Flask doesn't crash on threads
import matplotlib.pyplot as plt

def generate_gradcam(img_path, model, final_conv_layer_name="conv2d_1", intensity=0.4, res=128):
    """
    Computes Gradient-weighted Class Activation Mapping (Grad-CAM) and
    blends it with the original image using Matplotlib.
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

    grads = tape.gradient(loss, conv_outputs)
    guided_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(tf.multiply(guided_grads, conv_outputs), axis=-1)

    # Apply ReLU activation function
    heatmap = np.maximum(heatmap, 0)
    
    # Normalize the heatmap matrix safely
    max_val = np.max(heatmap)
    if max_val == 0:
        max_val = 1e-10
    heatmap /= max_val

    # 5. Build and Save the Color Image Output File using Matplotlib
    output_dir = os.path.dirname(img_path)
    base_filename = os.path.basename(img_path)
    output_path = os.path.join(output_dir, "gradcam_" + base_filename)

    # Render image layout layers
    fig, ax = plt.subplots(figsize=(4, 4), dpi=res)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    ax.axis('off')

    ax.imshow(img)
    ax.imshow(heatmap, cmap='jet', alpha=intensity, extent=(0, res, res, 0))

    plt.savefig(output_path, pad_inches=0, bbox_inches='tight')
    plt.close(fig)

    return f"upload/gradcam_{base_filename}"


def generate_simulated_heatmap(img_path, intensity=0.4, res=128):
    """
    Generates a realistic mock center-focused attention map overlay when the 
    TensorFlow model is running in simulation/fallback mode.
    """
    output_dir = os.path.dirname(img_path)
    base_filename = os.path.basename(img_path)
    output_path = os.path.join(output_dir, "gradcam_" + base_filename)
    
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=(res, res))
    
    # Create an artificial spatial grid centering activation density rules
    x, y = np.meshgrid(np.linspace(-1, 1, res), np.linspace(-1, 1, res))
    d = np.sqrt(x*x + y*y)
    heatmap = np.maximum(0, 1.0 - d * 1.5)  # Creates a localized heat signature spot

    fig, ax = plt.subplots(figsize=(4, 4), dpi=res)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    ax.axis('off')

    ax.imshow(img)
    ax.imshow(heatmap, cmap='jet', alpha=intensity, extent=(0, res, res, 0))

    plt.savefig(output_path, pad_inches=0, bbox_inches='tight')
    plt.close(fig)

    return f"upload/gradcam_{base_filename}"
