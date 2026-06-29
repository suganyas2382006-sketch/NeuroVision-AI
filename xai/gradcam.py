import os
import cv2
import numpy as np
import tensorflow as tf

def generate_gradcam(model, img_path, output_path, layer_name=None):
    """
    Computes spatial gradient activations to isolate tumor features.
    """
    # 1. Preprocess the incoming image sequence structure
    img = cv2.imread(img_path)
    img_resized = cv2.resize(img, (224, 224))
    img_tensor = np.expand_dims(img_resized, axis=0) / 255.0

    # 2. Automatically locate the deepest convolutional node layer matrix
    if layer_name is None:
        for layer in reversed(model.layers):
            if isinstance(layer, (tf.keras.layers.Conv2D, tf.keras.layers.DepthwiseConv2D)) or 'conv' in layer.name.lower():
                layer_name = layer.name
                break
    
    if not layer_name:
        raise ValueError("Could not automatically resolve target convolutional tracking node layout.")

    # 3. Create a dual-output model tracking layer activations and predictions
    grad_model = tf.keras.models.Model(
        inputs=[model.inputs],
        outputs=[model.get_layer(layer_name).output, model.output]
    )

    # 4. Calculate functional gradients relative to the target class decision weight
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_tensor)
        pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    # Compute target layer gradients matching the model's analytical direction
    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # 5. Multiply spatial activation weights across the channel dimension grid
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # 6. Apply Rectified Linear Unit (ReLU) to isolate positive focus features
    heatmap = tf.maximum(heatmap, 0)
    max_val = tf.math.reduce_max(heatmap)
    
    # Safe numerical normalization array loop framework safeguard
    if max_val > 0:
        heatmap = heatmap / max_val
        
    heatmap = heatmap.numpy()

    # 7. Upscale the array bounds to match the original MRI scan scale boundaries
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    
    # 8. Render a high-contrast standalone spatial mapping tracking matrix
    standalone_color_map = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    # Write file out directly without blending or overlaying onto the pristine clinical image
    cv2.imwrite(output_path, standalone_color_map)
    return output_path
