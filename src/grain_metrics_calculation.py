import numpy as np
import cv2

def calculate_grain_metrics(labels):
    """
    Calculates grain metrics based on labeled image.
    ASTM E112 G-number approximation: G = 3.322 * log10(Na) - 2.95
    where Na is the number of grains per mm^2.
    
    IMPORTANT: We need the physical scale of the image to calculate Na correctly.
    If scale is unknown, we report raw count and G-number assuming a standard 1mm^2 area (or user specific).
    Ref: ASTM E112 usually refers to 100x magnification.
    G = -3.32193 * log10(Mean Intercept Length) ...
    
    The formula provided in the PDF is: G = 3.322 * log10(Na) - 2.95
    Na = Number of grains per mm^2 at 1x magnification.
    """
    
    # Get unique labels (excluding 0 which is background)
    unique_labels, counts = np.unique(labels, return_counts=True)
    
    # Filter out background (label 0)
    if unique_labels[0] == 0:
        grain_areas = counts[1:]
        num_grains = len(unique_labels) - 1
    else:
        grain_areas = counts
        num_grains = len(unique_labels)
        
    return num_grains, grain_areas

def calculate_g_number(num_grains, image_area_mm2):
    """
    Calculates ASTM G-number.
    Na: Number of grains per mm^2.
    """
    if image_area_mm2 <= 0 or num_grains == 0:
        return 0
        
    Na = num_grains / image_area_mm2
    G = (3.322 * np.log10(Na)) - 2.95
    return G

def calculate_carbon_content(gray_image, labels):
    """
    Estimates carbon content based on dark pixel percentage.
    In steel, Pearlite (darker) contains carbon, Ferrite (lighter) is low carbon.
    Eutectoid steel (0.77% C) is 100% Pearlite.
    Ferrite is ~0% C.
    
    Simple estimation:
    % Carbon ~= (% Area Pearlite) * 0.77% (simplistic model)
    
    We identify "dark" regions.
    """
    # Threshold to find dark regions within the grains
    # Usually simple thresholding
    _, binary_dark = cv2.threshold(gray_image, 100, 255, cv2.THRESH_BINARY_INV) # Dark pixels become white (255)
    
    total_pixels = gray_image.size
    dark_pixels = np.count_nonzero(binary_dark)
    
    dark_ratio = dark_pixels / total_pixels
    
    # Rough empirical estimation (assuming dark = pearlite ~ 0.8% C max)
    carbon_percentage = dark_ratio * 0.77
    
    return carbon_percentage, dark_ratio
