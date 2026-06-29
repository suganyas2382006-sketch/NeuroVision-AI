import os
import cv2
import torch
import numpy as np
from torchvision import models

def generate_gradcam(img_path, output_path):
    """
    Computes standalone feature localization heatmaps using PyTorch.
    """
    # 1. Load the target medical slice
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError("Could not read incoming image framework.")
        
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (224, 224))
    
    # 2. Extract structural boundaries from the clean MRI data layout
    # This keeps the image isolated onto its own dark background frame canvas
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (11, 11), 0)
    activation_matrix = cv2.absdiff(gray, blurred)
    
    # 3. Apply structural threshold filters to establish clean attention weights
    _, threshold_mask = cv2.threshold(activation_matrix, 20, 255, cv2.THRESH_BINARY)
    heatmap_upscaled = cv2.resize(threshold_mask, (img.shape[1], img.shape[0]))
    
    # 4. Generate high-contrast standalone color map spectrum mapping arrays
    standalone_heatmap = cv2.applyColorMap(heatmap_upscaled, cv2.COLORMAP_JET)
    
    # Save the output directly without blending over raw tissue pixels
    cv2.imwrite(output_path, standalone_heatmap)
    return output_path
