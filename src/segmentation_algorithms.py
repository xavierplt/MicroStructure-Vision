import cv2
import numpy as np
from scipy import ndimage
from skimage.feature import peak_local_max
from skimage.segmentation import watershed

def preprocess_image(gray_image, use_clahe=True):
    """
    Applies preprocessing like CLAHE to enhance contrast.
    """
    if use_clahe:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        return clahe.apply(gray_image)
    return gray_image

def apply_otsu_segmentation(gray_image, kernel_size=3, invert=False):
    """
    Applies Otsu's thresholding followed by morphological closing and connected components.
    Added 'invert' option because grain boundaries are dark, grains are light.
    Usually we want to segment GRAINS (light).
    """
    # Preprocessing
    processed = preprocess_image(gray_image)
    
    # Blur to reduce noise
    blurred = cv2.GaussianBlur(processed, (5, 5), 0)
    
    # Otsu's Thresholding
    # If grains are light and boundaries are dark:
    # THRESH_BINARY: > threshold = 255 (Light/Grain), < threshold = 0 (Dark/Boundary)
    # THRESH_BINARY_INV: > threshold = 0, < threshold = 255
    
    threshold_type = cv2.THRESH_BINARY
    if invert:
        threshold_type = cv2.THRESH_BINARY_INV
        
    _, binary = cv2.threshold(blurred, 0, 255, threshold_type + cv2.THRESH_OTSU)
    
    # Morphological operations (Closing to fill gaps INSIDE grains)
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    # If binary is grains=255, closing removes black holes in white grains.
    closing = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    # Connected Components
    num_labels, labels = cv2.connectedComponents(closing)
    
    return labels, closing, num_labels

def apply_watershed_segmentation(gray_image, min_distance=10, invert=False):
    """
    Applies Marker-based Watershed segmentation.
    Distance transform -> marker extraction -> watershed flooding.
    """
    # Preprocessing
    processed = preprocess_image(gray_image)
    
    # Blur
    blurred = cv2.GaussianBlur(processed, (5, 5), 0)
    
    # Thresholding to define "Sure Foreground" vs "Sure Background"
    threshold_type = cv2.THRESH_BINARY
    if invert:
        threshold_type = cv2.THRESH_BINARY_INV
        
    _, binary = cv2.threshold(blurred, 0, 255, threshold_type + cv2.THRESH_OTSU)
    
    # Noise removal
    kernel = np.ones((3,3), np.uint8)
    opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)
    
    # SURE BACKGROUND area (dilate grain regions)
    sure_bg = cv2.dilate(opening, kernel, iterations=3)
    
    # DISTANCE TRANSFORM
    # Calculated on binary image where grains are white
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    
    # MARKERS (Local Maxima of Distance Transform)
    # Local max coords
    coords = peak_local_max(dist_transform, min_distance=min_distance, labels=opening)
    mask = np.zeros(dist_transform.shape, dtype=bool)
    mask[tuple(coords.T)] = True
    markers, _ = ndimage.label(mask)
    
    # Watershed
    # Needs markers (int32) and mask (image to flood usually gradient, or inverted distance)
    # Conventional Watershed: flooding from markers on a topographic map.
    # Scikit-image watershed: watershed(-dist, markers, mask=binary)
    # Flooding the inverted distance map (peaks become valleys), restricted to the binary mask.
    
    labels = watershed(-dist_transform, markers, mask=opening)
    
    return labels, dist_transform
