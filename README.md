# AnemiaClassification using ML (Bagging + Random Forest)
This project applies machine learning techniques to classify anemia types using Complete Blood Count (CBC) data. It leverages a Bagging Classifier with a Linear Support Vector Classifier (SVC) and a Random Forest model to perform classification, evaluate model performance, and visualize results.

## Dataset
The data used in this project comes from the Kaggle dataset:
**[Anemia Types Classification](https://www.kaggle.com/datasets/ehababoelnaga/anemia-types-classification)**

It includes various blood markers and diagnosis labels for different types of anemia.

## Features
- Data loading and cleaning

- Feature normalization

- Hyperparameter tuning using GridSearchCV

- Classification using:

  - Bagging with LinearSVC

  - RandomForestClassifier

- Visualization of confusion matrices and feature importances

## Requirements
Make sure you have the following Python packages installed:

`pip install pandas numpy scikit-learn matplotlib`
## Usage
1. Download the dataset from Kaggle and place diagnosed_cbc_data_v4.csv in the same directory.

2. Run the Python script.

## Steps Performed:
### 1. Preprocessing
  - Removes missing values.

  - Drops certain columns (LYMp, NEUTp, LYMn, NEUTn, and Diagnosis) from the feature set.

  - Normalizes all feature values (zero mean, unit variance).

### 2. Bagging with Linear SVC
  - Performs a grid search to tune the C parameter of LinearSVC within a BaggingClassifier.

  - Displays the weighted F1 scores from cross-validation.

  - Fits the model with the best C and prints the F1 score and confusion matrix.

### 3. Random Forest Classification
  - Performs a grid search to tune the max_depth parameter of a RandomForestClassifier.

  - Displays cross-validation scores.

  - Fits the model with the optimal depth.

  - Prints:

  - Feature importances as a bar chart

  -  Model score

  - Out-of-Bag (OOB) score

  - Confusion matrix

## Results
  - F1 Scores from grid search
<img width="338" alt="image" src="https://github.com/user-attachments/assets/4292cbf1-209a-44ff-9a84-2438ef98e04e" />
    The bottom score is the score that the most optimal C value obtained on the test set
  - Bagging Classifier Confusion Matrix
<img width="607" alt="image" src="https://github.com/user-attachments/assets/e0ddc7fa-2684-4408-8b29-da940f0071d1" />
  - Feature Importance bar chart for Random Forest
<img width="800" alt="image" src="https://github.com/user-attachments/assets/8b370e11-f149-4251-bd13-0060639c8f28" />
  - Random Forest Scores
<img width="324" alt="image" src="https://github.com/user-attachments/assets/f784738b-c3a0-46c5-b41d-7419930181db" />
<img width="109" alt="image" src="https://github.com/user-attachments/assets/8f0ed33d-d1e6-4b77-945a-1080631561a9" />
  - Random Forest Confusion Matrix
<img width="596" alt="image" src="https://github.com/user-attachments/assets/dfb76425-1bd6-41d6-8c19-a936bae4a132" />

