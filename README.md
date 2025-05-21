## Image Classifier Project – Created for CS 639 AI and ML at Lewis and Clark College

Author: Igor Gromović

Course: CS 639 – Artificial Intelligence and Machine Learning

This project implements a hybrid image classification pipeline that leverages traditional image processing techniques along with deep learning feature extraction. The goal is to accurately classify landscape images from the Intel Image Classification Dataset using combined features from:

    HOG (Histogram of Oriented Gradients)

    ResNet-50 embeddings

    Grayscale luminosity statistics

Classification is performed using Logistic Regression with hyperparameter tuning via GridSearchCV.

## Dataset

Intel Image Classification Dataset

The dataset contains images categorized into 6 classes:

    buildings

    forest

    glacier

    mountain

    sea

    street

Each category has its own subfolder with .jpg files.

The dataset can be obtained from https://www.kaggle.com/datasets/puneet6060/intel-image-classification?resource=download

However the folders from the downloaded set were modified since the validation and train sets are created via the code.

## Feature Extraction

Three types of features are extracted from each image:
1. HOG Features

    Captures gradient orientations and edge structures.

    Converts images to grayscale before processing.

2. ResNet-50 Embeddings

    Uses pretrained ResNet-50 (without final classification layer).

    Extracts deep feature embeddings from the penultimate layer.

3. Luminosity Statistics

    Computes the mean pixel intensity of the grayscale image.

The three features are concatenated into a single feature vector for each image.

## Model Training & Evaluation

Training Process

    Dataset split: 80% training, 20% validation

    Model: Logistic Regression (best performance vs. SVM, XGBoost, Random Forest)

    Hyperparameter Tuning: Performed using GridSearchCV

    Final model trained with manually selected optimal parameters

Pipeline

Pipeline([
    ('scaler', StandardScaler()),
    ('logisticregression', LogisticRegression(solver='saga', max_iter=500, penalty='l2', C=1))
])

## Results

    Training and validation accuracies are reported.

    Confusion matrices are plotted for both datasets.

Example Accuracy
Metric	Value
Training Accuracy	~99%
Validation Accuracy	~96% (varies slightly with run)

## Runtime

    Feature Extraction: ~10–12 minutes (depends on machine specs)

    GridSearchCV: ~4–6 minutes

    Final Training: ~3–4 minutes

## Requirements

Install the required packages via pip:
```
pip install numpy matplotlib torch torchvision scikit-learn seaborn pillow scikit-image
```
## License

This project is part of an academic course and provided for educational purposes only.
