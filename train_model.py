import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# 1. Define an ultra-lightweight Mobile CNN architecture
class MicroTumorNetwork(nn.Module):
    def __init__(self):
        super(MicroTumorNetwork, self).__init__()
        # Tiny 3-layer feature extraction grid to keep size minimal
        self.features = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # 112x112
            
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)  # 56x56
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(16, 4) # 4 output classes
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)

def run_minimal_training():
    print("[*] Launching Micro-Training Pipeline...")
    
    # 2. Initialize our tiny network layout
    model = MicroTumorNetwork()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 3. Create simulated low-memory mock tensors
    # In live execution, load your gray/RGB images here
    print("[*] Generating minimal data matrix arrays...")
    mock_images = np.random.rand(4, 244, 224, 3).astype(np.float32)
    mock_labels = np.array([0, 1, 2, 3]) # One of each diagnostic type

    # Convert format to match PyTorch channel-first expectations: (B, C, H, W)
    inputs = torch.from_numpy(mock_images.transpose((0, 3, 1, 2)))
    targets = torch.from_numpy(mock_labels).long()

    # 4. Execute a rapid, low-overhead training pass
    model.train()
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    loss.backward()
    optimizer.step()
    
    print(f"[+] Optimization step complete. Loss: {loss.item():.4f}")

    # 5. Save the micro-binary asset configuration
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model')
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, 'brain_tumor_model.pth')
    
    torch.save(model.state_dict(), save_path)
    print(f"[+] Ultra-lightweight model compiled and saved to: {save_path}")
    print(f"[i] Final weights file size: ~{os.path.getsize(save_path) / 1024:.2f} KB (Extremely small!)")

if __name__ == '__main__':
    run_minimal_training()
