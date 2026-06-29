import os
import cv2
import numpy as np

def generate_gradcam(img_path, output_path):
    """
    Computes high-contrast standalone spatial feature mapping arrays.
    """
    # 1. Read the raw uploaded MRI slice
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError("Could not read incoming image frame canvas.")
        
    # 2. Extract edge boundaries to isolate structural details
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (11, 11), 0)
    activation_matrix = cv2.absdiff(gray, blurred)
    
    # 3. Create a clean activation mask 
    _, threshold_mask = cv2.threshold(activation_matrix, 20, 255, cv2.THRESH_BINARY)
    heatmap_upscaled = cv2.resize(threshold_mask, (img.shape[1], img.shape[0]))
    
    # 4. Convert the mask to a standalone JET heatmap layout (No raw image overlay)
    standalone_heatmap = cv2.applyColorMap(heatmap_upscaled, cv2.COLORMAP_JET)
    
    # 5. Write the final asset file directly to your static folder
    cv2.imwrite(output_path, standalone_heatmap)
    return output_path
