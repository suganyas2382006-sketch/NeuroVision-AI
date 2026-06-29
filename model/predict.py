import os
import cv2
import torch
import torch.nn as nn
import numpy as np
from torchvision import models

current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, 'brain_tumor_model.pth')

device = torch.device('cpu')
model = models.mobilenet_v3_small()
num_features = model.classifier[0].in_features
model.classifier = nn.Sequential(
    nn.Linear(num_features, 128),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(128, 4)
)

if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

def run_inference(img_path):
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError("Could not read incoming image framework.")
        
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (224, 224))
    
    img_tensor = img_resized.transpose((2, 0, 1)).astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406]).reshape((3, 1, 1))
    std = np.array([0.229, 0.224, 0.225]).reshape((3, 1, 1))
    img_tensor = (img_tensor - mean) / std
    
    # CRITICAL FIX: Forces explicit 32-bit Float conversion to clear scalar errors
    torch_tensor = torch.from_numpy(img_tensor).unsqueeze(0).to(device).float()

    with torch.no_grad():
        model.to(device).float()
        outputs = model(torch_tensor)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        
    pred_index = torch.argmax(probabilities).item()
    confidence_val = probabilities[pred_index].item()

    classes = ["Glioma", "Meningioma", "Pituitary", "No Tumor"]
    return {
        "class_label": classes[pred_index],
        "confidence": f"{confidence_val * 100:.2f}%"
    }
