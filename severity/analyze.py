import cv2
import numpy as np

def evaluate_tumor_severity(original_img_path, binary_mask):
    img = cv2.imread(original_img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return {"severity_grade": "Analysis Error"}

    _, brain_thresh = cv2.threshold(img, 10, 255, cv2.THRESH_BINARY)
    total_brain_pixels = np.sum(brain_thresh == 255)
    tumor_pixels = np.sum(binary_mask == 255)
    
    area_percentage = (tumor_pixels / total_brain_pixels) * 100 if total_brain_pixels > 0 else 0.0

    if tumor_pixels == 0:
        grade = "Stage I (Low Risk / Benign)"
    elif area_percentage < 5.0:
        grade = "Stage II (Low-Grade Glioma)"
    elif area_percentage < 12.0:
        grade = "Stage III (Anaplastic Variant)"
    else:
        grade = "Stage IV (Glioblastoma Multiforme)"

    return {"severity_grade": grade}
