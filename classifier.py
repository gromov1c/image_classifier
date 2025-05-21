
# Image Classifier Project - CS 639 AI and ML
# Author: Igor Gromovic
# Runtime shouldn't be more than 20 minutes, and there won't be any messages displayed for the first few minutes.

import numpy as np
#import pandas as pd
import matplotlib.pyplot as plt
import torch
import seaborn as sns
from sklearn.decomposition import PCA
from PIL import Image
from glob import glob
from torchvision import models, transforms
from torch import nn
from skimage.feature import hog
#from skimage.color import rgb2gray
#from sklearn.pipeline import make_pipeline
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
#from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
#from sklearn.ensemble import RandomForestClassifier
#import xgboost as xgb
import time

# Path to Dataset
root_path = r'C:\\Users\\igrom\\Downloads\\Intel Training Dataset\\'

# split into subfolders based on class label
subfolders = sorted(glob(root_path + '*'))
label_names = [p.split('\\')[-1] for p in subfolders]
print(label_names)

# Resnet Preprocessing
preprocess_resnet = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Loading resnet
resnet50 = models.resnet50(pretrained=True)

# Modifying the model to extract embeddings from the penultimate layer
model_conv_features = nn.Sequential(*list(resnet50.children())[:-1])


# Putting the model in evaluation mode
model_conv_features.eval()

# Function to extract HOG features
def get_hog_features(image_path):
    image = Image.open(image_path).convert("L")  # Convert to grayscale
    image_np = np.array(image)
    features, hog_image = hog(image_np, pixels_per_cell=(16, 16), cells_per_block=(1, 1), visualize=True, feature_vector=True)
    return features

# Function to load an image, preprocess it for ResNet, and extract embeddings
def get_resnet_embedding(image_path):
    image = Image.open(image_path)
    image = preprocess_resnet(image)
    image = image.unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        embedding = model_conv_features(image).squeeze()  # Remove batch dimension
    return embedding.numpy()

# Function to extract luminosity features
def get_luminosity_features(image_path):
    image = Image.open(image_path).convert("L")
    image_np = np.array(image)
    luminosity_features = np.mean(image_np)
    return np.array([luminosity_features])

combined_features = []
labels = []

# Start timing
start_time = time.time()

# This should take 10-12 minutes to run, depending on the machine
for label, folder in enumerate(subfolders):
    image_files = sorted(glob(folder + '/*.jpg'))
    for image_file in image_files:
        hog_features = get_hog_features(image_file)
        resnet_embedding = get_resnet_embedding(image_file)
        luminosity_features = get_luminosity_features(image_file) 
        combined_feature = np.concatenate((hog_features, resnet_embedding, luminosity_features))  # Combine all features
        combined_features.append(combined_feature)
        labels.append(label)

combined_features = np.array(combined_features)
labels = np.array(labels)

# End timing
end_time = time.time()
duration = end_time - start_time
print(f"Feature extraction Runtime: {duration} seconds")

# Splitting data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(combined_features, labels, test_size=0.2, random_state=42)

# Gridsearch parameters for logistic regression
# I have also tried using XGBoost, Random Forest and SVC classifiers, however logistic regression seems to perform the best in my case.
clf = Pipeline([
    ('scaler', StandardScaler()),
    ('logisticregression', LogisticRegression(solver='saga', max_iter=10000, random_state=42))
])
param_grid = {
    # I'm including less parameters here for the sake of speed, however I've added a few more parameters 3 code blocks below which seem to improve the accuracy
    'logisticregression__solver': ['lbfgs', 'saga'],
    'logisticregression__max_iter': [10, 20, 50, 100],
}

# Start timing
start_time = time.time()

grid_search = GridSearchCV(clf, param_grid, cv=3, scoring='accuracy', verbose=2, n_jobs=-1)
grid_search.fit(X_train, y_train)

# Best parameters
print("Best parameters found: ", grid_search.best_params_)

# Best model
best_model = grid_search.best_estimator_

# Validation accuracy
val_accuracy = best_model.score(X_val, y_val)
print("Validation Accuracy of the best model: {:.2f}%".format(val_accuracy * 100))

# Training accuracy
train_accuracy = best_model.score(X_train, y_train)
print("Training Accuracy of the best model: {:.2f}%".format(train_accuracy * 100))

# End timing
end_time = time.time()
duration = end_time - start_time
print(f"GridSearchCV Runtime: {duration} seconds")

# Start timing
start_time = time.time()

# Training the classifier using different parameters, takes about 4 minutes on my machine
clf = LogisticRegression(solver='saga', max_iter=500, random_state=42, penalty='l2', C=1, n_jobs=-1)

clf.fit(X_train, y_train)

# End timing
end_time = time.time()
duration = end_time - start_time
print(f"Training using manually set parameters Runtime: {duration} seconds")

# Predictions and evaluation for validation data
predictions_val = clf.predict(X_val)
conf_matrix_val = confusion_matrix(y_val, predictions_val)
val_accuracy = accuracy_score(y_val, predictions_val)

# Predictions and evaluation for training data
predictions_train = clf.predict(X_train)
conf_matrix_train = confusion_matrix(y_train, predictions_train)
train_accuracy = accuracy_score(y_train, predictions_train)

# Display accuracies
print("Training Accuracy: {:.2f}%".format(train_accuracy * 100))
print("Validation Accuracy: {:.2f}%".format(val_accuracy * 100))

# Plotting confusion matrix for training data
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
sns.heatmap(conf_matrix_train, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.title("Training Data Confusion Matrix")
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.xticks(ticks=np.arange(len(label_names)) + 0.5, labels=label_names, rotation=45, ha="right")
plt.yticks(ticks=np.arange(len(label_names)) + 0.5, labels=label_names, rotation=0, ha="right")

# Plotting confusion matrix for validation data
plt.subplot(1, 2, 2) 
sns.heatmap(conf_matrix_val, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.title("Validation Data Confusion Matrix")
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.xticks(ticks=np.arange(len(label_names)) + 0.5, labels=label_names, rotation=45, ha="right")
plt.yticks(ticks=np.arange(len(label_names)) + 0.5, labels=label_names, rotation=0, ha="right")

plt.tight_layout()
plt.show()


