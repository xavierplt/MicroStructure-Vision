import cv2
import numpy as np

def load_image(uploaded_file):
    """
    Loads an image from a StreamlitUploadedFile or a file path.
    Returns RGB image and Grayscale image.
    """
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image_bgr = cv2.imdecode(file_bytes, 1)
    if image_bgr is None:
        return None, None
    
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image_gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    
    return image_rgb, image_gray

def load_image_from_path(path):
    """
    Loads an image from a local path.
    Returns RGB image and Grayscale image.
    """
    image_bgr = cv2.imread(path)
    if image_bgr is None:
        return None, None
    
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image_gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    
    return image_rgb, image_gray

def overlay_mask(image_rgb, mask, color=(255, 0, 0), alpha=0.3):
    """
    Overlays a binary mask on an RGB image.
    """
    overlay = image_rgb.copy()
    overlay[mask > 0] = color
    return cv2.addWeighted(overlay, alpha, image_rgb, 1 - alpha, 0)

def overlay_labels(image_rgb, labels):
    """
    Overlays colored labels on an RGB image.
    Colorizes the labels using a jet colormap (or random colors).
    """
    # Create a colored label image
    # We map labels to 0-255 using int conversion for visualization if needed, 
    # but better to use a proper colormap for distinct grains.
    
    # Normalize labels to fit in 0-255 range for colormap application (excluding background 0)
    label_hue = np.uint8(179 * labels / np.max(labels))
    blank_ch = 255 * np.ones_like(label_hue)
    labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])
    labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2RGB)
    labeled_img[labels == 0] = 0

    return cv2.addWeighted(image_rgb, 0.7, labeled_img, 0.3, 0)
