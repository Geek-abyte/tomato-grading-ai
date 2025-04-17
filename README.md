# Tomato Grading AI

This project implements an AI-based system for grading tomatoes into three categories:
- **Ripe**: Tomatoes that are ready for consumption
- **Unripe**: Tomatoes that need more time to ripen
- **Reject**: Tomatoes that are defective or unsuitable for consumption

## Features

- **Automated Grading**: Automatically grade tomatoes based on images
- **Multiple Feature Extraction**: Extracts color, texture, and shape features from tomato images
- **Machine Learning Models**: Uses Support Vector Machines (SVM) with different kernels
- **Batch Processing**: Process individual images or entire directories

## Project Structure

```
tomato-grading-ai/
│
├── dataset/
│   ├── ripe/      # Contains images of ripe tomatoes
│   ├── unripe/    # Contains images of unripe tomatoes
│   └── reject/    # Contains images of reject tomatoes
│
├── features/
│   └── feature_extraction.py  # Code for extracting features from tomato images
│
├── models/
│   ├── svm_linear.pkl      # Trained linear SVM model
│   ├── svm_quadratic.pkl   # Trained quadratic SVM model
│   └── scaler.pkl          # Feature scaler
│
├── notebooks/
│   └── model_training.ipynb  # Jupyter notebook for training models
│
├── main.py         # Main application script
├── requirements.txt  # Project dependencies
└── README.md       # Project documentation
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/tomato-grading-ai.git
cd tomato-grading-ai
```

2. Create a virtual environment (optional but recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Training the Models

1. Prepare your dataset by organizing tomato images into three folders:
   - `dataset/ripe/`: Images of ripe tomatoes
   - `dataset/unripe/`: Images of unripe tomatoes
   - `dataset/reject/`: Images of reject tomatoes

2. Open and run the Jupyter notebook:
```bash
jupyter notebook notebooks/model_training.ipynb
```

3. Follow the steps in the notebook to train the models.

### Using the Application

To grade a single tomato image:
```bash
python main.py --image path/to/tomato_image.jpg --visualize
```

To process a directory of tomato images:
```bash
python main.py --dir path/to/tomato_images/
```

To use the linear SVM model instead of the default quadratic model:
```bash
python main.py --image path/to/tomato_image.jpg --model linear
```

## Requirements

- Python 3.7+
- OpenCV
- NumPy
- scikit-learn
- scikit-image
- Matplotlib
- Pandas
- Seaborn
- Jupyter (for running the notebook)

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Dataset attribution (if applicable)
- References to relevant research papers or methodologies