import os
import cv2
import numpy as np
import tensorflow as tf

def generate_gradcam(image_path, model, final_conv_layer_name="conv2d_last"):
    """
    Generates gradient-weighted feature map matrices over top activation layers 
    to output an interpretive localization thermal focus overlay image on disk.
    """
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(128, 128))
    img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0
    img_tensor = np.expand_dims(img_array, axis=0)

    grad_model = tf.keras.models.Model(
        inputs=[model.inputs],
        outputs=[model.get_layer(final_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_tensor)
        top_pred_index = np.argmax(predictions[0])
        loss = predictions[:, top_pred_index]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0)
    max_val = tf.reduce_max(heatmap)
    if max_val > 0:
        heatmap = heatmap / max_val
    heatmap = heatmap.numpy()

    original_img = cv2.imread(image_path)
    height, width, _ = original_img.shape
    
    heatmap_resized = cv2.resize(heatmap, (width, height))
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

    superimposed_img = cv2.addWeighted(original_img, 0.6, heatmap_color, 0.4, 0)

    dir_name, file_name = os.path.split(image_path)
    output_filename = "gradcam_" + file_name
    output_path = os.path.join(dir_name, output_filename)
    
    cv2.imwrite(output_path, superimposed_img)

    return f"upload/{output_filename}"
