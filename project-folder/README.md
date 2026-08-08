# Machine Learning Classification Model Comparison

## Problem Statement

The objective of this project is to implement and compare multiple machine learning classification models on the same dataset. The models are evaluated using standard classification performance metrics, and a Streamlit web application is developed to interactively view the performance of each model.

## Dataset Description

The Phishing Websites dataset from the UCI Machine Learning Repository is used for this project.

The dataset contains 11,055 instances and 31 columns. It includes 30 input features related to website characteristics and one target column named `Result`.

The target variable contains two classes:

- `-1`
- `1`

The dataset is used to train and evaluate machine learning models for binary classification.

An 80:20 train-test split is used. The test dataset contains 2,211 instances and is saved as `test_data.csv`.

## Machine Learning Models

The following classification models are implemented:

1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbors (kNN)
4. Gaussian Naive Bayes
5. Random Forest Classifier

## Model Performance Comparison

| ML Model | Accuracy | AUC | Precision | Recall | F1 Score | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9285 | 0.9808 | 0.9234 | 0.9504 | 0.9367 | 0.8551 |
| Decision Tree | 0.9711 | 0.9805 | 0.9702 | 0.9781 | 0.9741 | 0.9413 |
| kNN | 0.9480 | 0.9868 | 0.9486 | 0.9586 | 0.9535 | 0.8945 |
| Naive Bayes | 0.6006 | 0.9702 | 0.9971 | 0.2835 | 0.4415 | 0.3844 |
| Random Forest | 0.9742 | 0.9977 | 0.9696 | 0.9846 | 0.9770 | 0.9478 |

## Overall Winner

Random Forest provides the best overall performance on the selected dataset. It achieves the highest Accuracy, AUC, Recall, F1 Score, and MCC among the evaluated models.

## Streamlit Application

The Streamlit application allows the user to:

- Select a machine learning model from a dropdown.
- View the selected model's evaluation metrics.
- View the confusion matrix.
- Read a short description of the selected model.

## How to Run the Streamlit App

1. Install the required libraries:

   `pip install -r requirements.txt`

2. Run the Streamlit application:

   `streamlit run app.py`

3. The application will open in the web browser.

## Project Files

- `app.py` - Streamlit application
- `requirements.txt` - Required Python libraries
- `test_data.csv` - Test dataset used by the application
- `logistic_regression.pkl` - Trained Logistic Regression model
- `decision_tree.pkl` - Trained Decision Tree model
- `knn.pkl` - Trained kNN model
- `naive_bayes.pkl` - Trained Gaussian Naive Bayes model
- `random_forest.pkl` - Trained Random Forest model
- `scaler.pkl` - StandardScaler used for Logistic Regression and kNN
- `ML-Assignment2.ipynb` - Machine learning implementation notebook