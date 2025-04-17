"""
Tomato Grading AI - Main Application

This script provides a command-line interface for grading tomato images
using the trained machine learning models.
"""

import os
import sys
import cv2
import numpy as np
import argparse
import joblib
from features.feature_extraction import extract_features


def load_models():
    """Load the trained models and scaler"""
    try:
        # Load the scaler
        scaler = joblib.load('models/scaler.pkl')
        
        # Load SVM models
        svm_linear = joblib.load('models/svm_linear.pkl')
        svm_quadratic = joblib.load('models/svm_quadratic.pkl')
        
        return scaler, svm_linear, svm_quadratic
    
    except FileNotFoundError as e:
        print(f"Error loading models: {e}")
        print("Please run the model_training notebook first to train the models.")
        sys.exit(1)


def grade_tomato(image_path, scaler, model, model_name=""):
    """Grade a tomato image using the provided model"""
    try:
        # Extract features from the image
        features = extract_features(image_path)
        
        # Reshape features for scaling
        features_array = np.array(features).reshape(1, -1)
        
        # Scale features
        scaled_features = scaler.transform(features_array)
        
        # Predict class and probability
        prediction = model.predict(scaled_features)[0]
        probabilities = model.predict_proba(scaled_features)[0]
        
        # Map prediction to confidence
        class_indices = {'reject': 0, 'ripe': 1, 'unripe': 2}
        confidence = probabilities[class_indices[prediction]]
        
        # Print results
        if model_name:
            print(f"\n{model_name} Model Results:")
        print(f"Grade: {prediction.upper()}")
        print(f"Confidence: {confidence:.2%}")
        
        # Return prediction and confidence
        return prediction, confidence
        
    except Exception as e:
        print(f"Error grading image: {e}")
        return None, 0


def visualize_result(image_path, prediction):
    """Visualize the grading result"""
    try:
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError("Could not open or find the image")
        
        # Add prediction text to image
        text = f"Grade: {prediction.upper()}"
        
        # Define text color based on prediction
        if prediction == 'ripe':
            color = (0, 255, 0)  # Green
        elif prediction == 'unripe':
            color = (255, 255, 0)  # Yellow
        else:  # reject
            color = (0, 0, 255)  # Red
        
        # Add text
        cv2.putText(image, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        
        # Display image
        cv2.imshow('Tomato Grading Result', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
    except Exception as e:
        print(f"Error visualizing result: {e}")


def process_directory(directory, scaler, model):
    """Process all images in a directory"""
    # Check if the directory exists
    if not os.path.isdir(directory):
        print(f"Error: Directory '{directory}' does not exist.")
        return
    
    # Get image files
    image_files = [f for f in os.listdir(directory) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if not image_files:
        print(f"No image files found in directory: {directory}")
        return
    
    print(f"\nProcessing {len(image_files)} images in '{directory}'...")
    
    # Count categories
    grades = {'ripe': 0, 'unripe': 0, 'reject': 0}
    
    # Process each image
    for image_file in image_files:
        image_path = os.path.join(directory, image_file)
        try:
            prediction, _ = grade_tomato(image_path, scaler, model, "")
            if prediction:
                grades[prediction] += 1
        except Exception as e:
            print(f"Error processing {image_file}: {e}")
    
    # Print summary
    print("\nSummary:")
    total = sum(grades.values())
    for grade, count in grades.items():
        percentage = (count / total * 100) if total > 0 else 0
        print(f"{grade.upper()}: {count} ({percentage:.1f}%)")


def main():
    """Main function"""
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Tomato Grading AI')
    parser.add_argument('--image', help='Path to the image file to grade')
    parser.add_argument('--dir', help='Path to a directory of images to grade in batch')
    parser.add_argument('--model', choices=['linear', 'quadratic'], default='quadratic',
                      help='Model to use for grading (linear or quadratic SVM)')
    parser.add_argument('--visualize', action='store_true', help='Visualize the results')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Check if image or directory is provided
    if not args.image and not args.dir:
        parser.print_help()
        print("\nError: You must provide either an image file (--image) or a directory (--dir).")
        return
    
    print("Tomato Grading AI\n")
    
    # Load models
    scaler, svm_linear, svm_quadratic = load_models()
    
    # Select model
    model = svm_linear if args.model == 'linear' else svm_quadratic
    model_name = f"{args.model.capitalize()} SVM"
    
    # Process single image
    if args.image:
        if not os.path.isfile(args.image):
            print(f"Error: Image file '{args.image}' does not exist.")
            return
        
        print(f"Grading image: {args.image}")
        prediction, _ = grade_tomato(args.image, scaler, model, model_name)
        
        # Visualize if requested
        if args.visualize and prediction:
            visualize_result(args.image, prediction)
    
    # Process directory
    elif args.dir:
        process_directory(args.dir, scaler, model)


if __name__ == "__main__":
    main()