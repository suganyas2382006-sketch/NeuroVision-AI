import os

def generate_gradcam(image_path):

    # Dummy heatmap generation (replace with real Grad-CAM later)
    heatmap_path = image_path.replace("uploads", "heatmaps")

    try:
        os.system(f"cp {image_path} {heatmap_path}")
    except:
        pass

    return heatmap_path