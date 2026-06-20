import os
import cv2
import numpy as np
import tensorflow as tf

def generate_gradcam(model, img_path, output_path, layer_name=None):
    """
    Generates a Grad-CAM heatmap overlay for a Keras model.
    
    :param model: The loaded Keras .h5 model instance
    :param img_path: Path to the uploaded raw MRI scan
    :param output_path: Destination path to save the blended heatmap image
    :param layer_name: Name of the final convolutional layer (auto-detected if None)
    """
    # 1. Load and preprocess image to match model expectations
    img = cv2.imread(img_path)
    # Adjust target_size if your model doesn't use 224x224
    img_resized = cv2.resize(img, (224, 224))
    img_tensor = np.expand_dims(img_resized, axis=0) / 255.0

    # 2. Automatically locate the last convolutional layer if not provided
    if layer_name is None:
        for layer in reversed(model.layers):
            if isinstance(layer, (tf.keras.layers.Conv2D, tf.keras.layers.DepthwiseConv2D)) or 'conv' in layer.name.lower():
                layer_name = layer.name
                break
    
    if not layer_name:
        raise ValueError("Could not automatically locate a convolutional layer in the model.")

    # 3. Create a sub-model that maps the input image to the activations of the target conv layer
    # as well as the final model output prediction score
    grad_model = tf.keras.models.Model(
        inputs=[model.inputs],
        outputs=[model.get_layer(layer_name).output, model.output]
    )

    # 4. Compute the gradients of the top predicted class with respect to the conv layer activations
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_tensor)
        pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    # Gradients of the target class score w.r.t. the conv layer output feature map
    grads = tape.gradient(class_channel, conv_outputs)

    # 5. Global average pooling of the gradients to compute feature importance weights
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # 6. Weight the channels of the feature map activation by their importance weights
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # 7. Apply ReLU to isolate features that positively contribute to the target class decision
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    heatmap = heatmap.numpy()

    # 8. Upscale the heatmap to match the original MRI image resolution and blend
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    
    # Apply standard medical jet-colormap overlay
    color_heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    # Blend original image (65%) with the colorized attention map (35%)
    blended_img = cv2.addWeighted(img, 0.65, color_heatmap, 0.35, 0)

    # Save finalized visual frame output
    cv2.imwrite(output_path, blended_img)
    return output_path
