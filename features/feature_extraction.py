"""
Feature extraction module for tomato grading.
This module extracts visual features from tomato images for classification.
"""

import cv2
import numpy as np
import os
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops
from skimage.color import rgb2hsv


def extract_color_features(image):
    """
    Extract color features from the image.
    
    Args:
        image: RGB image array
        
    Returns:
        features: list of color features
    """
    # Convert to different color spaces
    hsv_image = rgb2hsv(image)
    
    # Extract color statistics in different channels
    color_features = []
    
    # RGB channels
    for i in range(3):
        channel = image[:,:,i]
        color_features.extend([
            np.mean(channel),  # Mean
            np.std(channel),   # Standard deviation
            np.percentile(channel, 25),  # 1st quartile
            np.percentile(channel, 50),  # Median
            np.percentile(channel, 75),  # 3rd quartile
        ])
    
    # HSV channels
    for i in range(3):
        channel = hsv_image[:,:,i]
        color_features.extend([
            np.mean(channel),  # Mean
            np.std(channel),   # Standard deviation
            np.percentile(channel, 25),  # 1st quartile
            np.percentile(channel, 50),  # Median
            np.percentile(channel, 75),  # 3rd quartile
        ])
    
    return color_features


def extract_texture_features(image):
    """
    Extract texture features from the image using LBP and GLCM.
    
    Args:
        image: RGB image array
        
    Returns:
        features: list of texture features
    """
    # Convert to grayscale for texture analysis
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # Local Binary Pattern features
    radius = 3
    n_points = 8 * radius
    lbp = local_binary_pattern(gray, n_points, radius, method='uniform')
    lbp_hist, _ = np.histogram(lbp, bins=n_points+2, range=(0, n_points+2), density=True)
    
    # GLCM features (Gray-Level Co-occurrence Matrix)
    distances = [1]  # distance between pixel pairs
    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]  # angles to consider
    glcm = graycomatrix(gray, distances, angles, 256, symmetric=True, normed=True)
    
    # Calculate GLCM properties
    glcm_features = []
    glcm_properties = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation']
    
    for prop in glcm_properties:
        glcm_features.append(graycoprops(glcm, prop).mean())
    
    return list(lbp_hist) + glcm_features


def extract_shape_features(image):
    """
    Extract shape features from the image.
    
    Args:
        image: RGB image array
        
    Returns:
        features: list of shape features
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # Threshold to get binary image
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    
    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        # Return zeros if no contours found
        return [0] * 7
    
    # Get the largest contour (assuming it's the tomato)
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Calculate area and perimeter
    area = cv2.contourArea(largest_contour)
    perimeter = cv2.arcLength(largest_contour, True)
    
    # Calculate circularity: 4*pi*area/perimeter^2 (1 for perfect circle)
    circularity = (4 * np.pi * area) / (perimeter * perimeter) if perimeter > 0 else 0
    
    # Calculate bounding rectangle
    x, y, w, h = cv2.boundingRect(largest_contour)
    
    # Calculate aspect ratio
    aspect_ratio = float(w) / h if h > 0 else 0
    
    # Calculate equivalent diameter
    equi_diameter = np.sqrt(4 * area / np.pi)
    
    # Calculate extent: ratio of contour area to bounding rectangle area
    extent = float(area) / (w * h) if w * h > 0 else 0
    
    # Calculate convex hull and solidity
    hull = cv2.convexHull(largest_contour)
    hull_area = cv2.contourArea(hull)
    solidity = float(area) / hull_area if hull_area > 0 else 0
    
    return [area, perimeter, circularity, aspect_ratio, equi_diameter, extent, solidity]


def extract_features(image_path):
    """
    Extract all features from a tomato image.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        features: Combined list of features
    """
    # Read image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not open or find the image: {image_path}")
    
    # Convert from BGR to RGB (OpenCV loads images as BGR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Resize for consistency
    image = cv2.resize(image, (224, 224))
    
    # Extract features
    color_features = extract_color_features(image)
    texture_features = extract_texture_features(image)
    shape_features = extract_shape_features(image)
    
    # Combine all features
    all_features = color_features + texture_features + shape_features
    
    return all_features


def extract_features_batch(image_dir, label):
    """
    Extract features from all images in a directory with a specific label.
    
    Args:
        image_dir: Directory containing images
        label: Class label for the images
        
    Returns:
        features: List of feature vectors
        labels: List of labels
    """
    features = []
    labels = []
    
    # Get list of image files
    image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    for image_file in image_files:
        image_path = os.path.join(image_dir, image_file)
        try:
            feature_vector = extract_features(image_path)
            features.append(feature_vector)
            labels.append(label)
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
    
    return features, labels


if __name__ == "__main__":
    # Example usage
    image_path = "../dataset/ripe/ripe (1).jpg"
    features = extract_features(image_path)
    print(f"Extracted {len(features)} features from {image_path}")