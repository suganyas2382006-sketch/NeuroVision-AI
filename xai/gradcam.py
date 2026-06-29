import os
import cv2
import numpy as np
import tensorflow as tf

def generate_gradcam(model, img_path, output_path, layer_name=None):
    """
    Generates a Standalone Grad-CAM heatmap for a Keras model without blending.
    """
    # 1. Load and preprocess image
    img = cv2.imread(img_path)
    img_resized = cv2.resize(img, (224, 224))
    img_tensor = np.expand_dims(img_resized, axis=0) / 255.0

    # 2. Automatically locate the last convolutional layer
    if layer_name is None:
        for layer in reversed(model.layers):
            if isinstance(layer, (tf.keras.layers.Conv2D, tf.keras.layers.DepthwiseConv2D)) or 'conv' in layer.name.lower():
                layer_name = layer.name
                break
    
    if not layer_name:
        raise ValueError("Could not automatically locate a convolutional layer in the model.")

    # 3. Create sub-model
    grad_model = tf.keras.models.Model(
        inputs=[model.inputs],
        outputs=[model.get_layer(layer_name).output, model.output]
    )

    # 4. Compute gradients
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_tensor)
        pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)

    # 5. Global average pooling
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # 6. Weight the channels
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # 7. Apply ReLU and normalize safely to protect against zero-division errors
    heatmap = tf.maximum(heatmap, 0)
    max_val = tf.math.reduce_max(heatmap)
    
    if max_val > 0:
        heatmap = heatmap / max_val
        
    heatmap = heatmap.numpy()

    # 8. Upscale to match the original image resolution
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    
    # Apply standard medical jet-colormap directly to the isolated attention matrix
    standalone_heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    # Save the standalone color heatmap matrix cleanly (No blending with original 'img')
    cv2.imwrite(output_path, standalone_heatmap)
    return output_path
