import os
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
import numpy as np

def load_real_dataset(dataset_path):
    print(f"[*] Scanning dataset directory: {dataset_path}")
    
    images = []
    labels = []
    
    # Precise folder name map matching your exact lowercase "notumor" structure
    categories_map = {
        "glioma": 0, "gliomatumor": 0, "glioma tumor": 0,
        "meningioma": 1, "meningiomatumor": 1, "meningioma tumor": 1,
        "pituitary": 2, "pituitarytumor": 2, "pituitary tumor": 2,
        "notumor": 3, "no tumor": 3, "no_tumor": 3
    }
    
    if not os.path.exists(dataset_path):
        raise ValueError(f"Base folder completely missing: {dataset_path}")

    for folder_name in os.listdir(dataset_path):
        folder_lower = folder_name.lower().strip()
        
        if folder_lower in categories_map:
            label_idx = categories_map[folder_lower]
            category_path = os.path.join(dataset_path, folder_name)
            
            print(f"[+] Found folder match: '{folder_name}' mapped to class {label_idx}")
            
            for file in os.listdir(category_path):
                if file.lower().endswith(('png', 'jpg', 'jpeg')):
                    img_path = os.path.join(category_path, file)
                    img = cv2.imread(img_path)
                    if img is None:
                        continue
                    
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img_resized = cv2.resize(img_rgb, (224, 224))
                    img_tensor = img_resized.transpose((2, 0, 1)).astype(np.float32) / 255.0
                    
                    images.append(img_tensor)
                    labels.append(label_idx)
                    
    if len(images) == 0:
        raise ValueError(f"Zero images loaded. Current contents of {dataset_path} are: {os.listdir(dataset_path)}")
        
    return np.array(images), np.array(labels)

def run_high_accuracy_training():
    print("[*] Compiling 90%+ Target Production Pipeline...")
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    train_path = os.path.join(base_dir, 'dataset', 'Training')
    
    try:
        X_train, y_train = load_real_dataset(train_path)
        print(f"[+] Successfully loaded {X_train.shape[0]} real MRI scans for training.")
    except Exception as e:
        print(f"[-] Data Loading Failure: {e}")
        return

    model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
    num_features = model.classifier[0].in_features
    model.classifier = nn.Sequential(
        nn.Linear(num_features, 128),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(128, 4)
    )
    
    for name, param in model.features.named_parameters():
        if int(name.split('.')[0]) > 4: 
            param.requires_grad = True
        else:
            param.requires_grad = False

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)

    # Force inputs into clean 32-bit float matrix arrays
    inputs = torch.from_numpy(X_train).float()
    targets = torch.from_numpy(y_train).long()
    
    print("[*] Initializing optimization epochs across real weights...")
    model.train()
    
    for epoch in range(5):
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        print(f"    -> Epoch {epoch+1}/5 completed. Loss: {loss.item():.4f}")

    output_dir = os.path.join(base_dir, 'model')
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, 'brain_tumor_model.pth')
    
    torch.save(model.state_dict(), save_path)
    print(f"[+] Real-world trained model saved to: {save_path}")

if __name__ == '__main__':
    run_high_accuracy_training()
