# AnemiaClassification using ML (Bagging + Random Forest)
This project applies machine learning techniques to classify anemia types using Complete Blood Count (CBC) data. It leverages a Bagging Classifier with a Linear Support Vector Classifier (SVC) and a Random Forest model to perform classification, evaluate model performance, and visualize results.

##Dataset
The data used in this project comes from the Kaggle dataset:
Anemia Types Classification

It includes various blood markers and diagnosis labels for different types of anemia.

## Features
Data loading and cleaning

Feature normalization

Hyperparameter tuning using GridSearchCV

Classification using:

Bagging with LinearSVC

RandomForestClassifier

Visualization of confusion matrices and feature importances

##Requirements
Make sure you have the following Python packages installed:

bash
Copy
Edit
pip install pandas numpy scikit-learn matplotlib
##Usage
Download the dataset from Kaggle and place diagnosed_cbc_data_v4.csv in the same directory.

Run the Python script.

## Steps Performed:
### 1. Preprocessing
Removes missing values.

Drops certain columns (LYMp, NEUTp, LYMn, NEUTn, and Diagnosis) from the feature set.

Normalizes all feature values (zero mean, unit variance).

### 2. Bagging with Linear SVC
Performs a grid search to tune the C parameter of LinearSVC within a BaggingClassifier.

Displays the weighted F1 scores from cross-validation.

Fits the model with the best C and prints the F1 score and confusion matrix.

### 3. Random Forest Classification
Performs a grid search to tune the max_depth parameter of a RandomForestClassifier.

Displays cross-validation scores.

Fits the model with the optimal depth.

Prints:

Feature importances as a bar chart

Model score

Out-of-Bag (OOB) score

Confusion matrix

## Example Output
F1 Scores from grid search

Confusion Matrix visualizations

Feature Importance bar chart for Random Forest

Scores including training and OOB accuracy

## Notes
Models are evaluated on training data only; consider using a proper train/test split or cross-validation on final models for real-world applications.

verbose=3 in RandomForestClassifier provides detailed training logs.
