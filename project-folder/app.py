from pathlib import Path
import streamlit as st

# Configure Streamlit page
st.set_page_config(
    page_title="ML Classification Model Comparison",
    layout="wide"
)

import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix
)

# Get current application directory
BASE_DIR = Path(__file__).resolve().parent


# App Title
st.title("Machine Learning Classification Model Comparison")


# Upload Test Dataset
uploaded_file = st.file_uploader(
    "Upload Test Dataset (CSV)",
    type=["csv"]
)


# Load Uploaded Test Dataset
if uploaded_file is not None:
    test_data = pd.read_csv(uploaded_file)
    st.success("Test dataset uploaded successfully.")
else:
    st.info("Please upload the test dataset (CSV) to view model results.")
    st.stop()


# Prepare Test Data
X_test = test_data.drop("Result", axis=1)
y_test = test_data["Result"]

st.write("Test Dataset Shape:", test_data.shape)


# Load Trained Models and Scaler
logistic_model = joblib.load(
    BASE_DIR / "logistic_regression.pkl"
)

decision_tree_model = joblib.load(
    BASE_DIR / "decision_tree.pkl"
)

knn_model = joblib.load(
    BASE_DIR / "knn.pkl"
)

naive_bayes_model = joblib.load(
    BASE_DIR / "naive_bayes.pkl"
)

random_forest_model = joblib.load(
    BASE_DIR / "random_forest.pkl"
)

scaler = joblib.load(
    BASE_DIR / "scaler.pkl"
)


# Model Selection Dropdown
model_name = st.selectbox(
    "Select Machine Learning Model",
    [
        "Logistic Regression",
        "Decision Tree",
        "kNN",
        "Naive Bayes",
        "Random Forest"
    ]
)


# Select Model and Prepare Test Data
if model_name == "Logistic Regression":
    selected_model = logistic_model
    X_input = scaler.transform(X_test)

elif model_name == "Decision Tree":
    selected_model = decision_tree_model
    X_input = X_test

elif model_name == "kNN":
    selected_model = knn_model
    X_input = scaler.transform(X_test)

elif model_name == "Naive Bayes":
    selected_model = naive_bayes_model
    X_input = X_test

else:
    selected_model = random_forest_model
    X_input = X_test


# Generate Predictions
y_pred = selected_model.predict(X_input)
y_prob = selected_model.predict_proba(X_input)[:, 1]


# Calculate Evaluation Metrics
accuracy = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_prob)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
mcc = matthews_corrcoef(y_test, y_pred)


# Display Selected Model Performance
st.subheader(f"{model_name} Performance")

performance_table = pd.DataFrame({
    "Accuracy": [accuracy],
    "AUC": [auc],
    "Precision": [precision],
    "Recall": [recall],
    "F1 Score": [f1],
    "MCC": [mcc]
})

performance_table = performance_table.round(4)

st.dataframe(
    performance_table,
    hide_index=True,
    use_container_width=True
)


# Display Confusion Matrix
st.subheader(f"{model_name} Confusion Matrix")

cm = confusion_matrix(y_test, y_pred)

fig, ax = plt.subplots(figsize=(4, 3))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["-1", "1"],
    yticklabels=["-1", "1"],
    ax=ax,
    cbar=False
)

ax.set_xlabel("Predicted Label")
ax.set_ylabel("Actual Label")

plt.tight_layout()


# Keep confusion matrix at a suitable size
graph_col, empty_col = st.columns([1, 2])

with graph_col:
    st.pyplot(fig, use_container_width=True)


# Model Descriptions
model_descriptions = {
    "Logistic Regression":
        "Logistic Regression is a classification algorithm that estimates "
        "the probability of a class using a logistic function.",

    "Decision Tree":
        "Decision Tree is a classification algorithm that makes predictions "
        "by splitting data into branches based on feature values.",

    "kNN":
        "k-Nearest Neighbors classifies a data point based on the classes "
        "of its nearest neighboring data points.",

    "Naive Bayes":
        "Naive Bayes is a probabilistic classification algorithm based on "
        "Bayes' theorem with an assumption of feature independence.",

    "Random Forest":
        "Random Forest is an ensemble classification algorithm that combines "
        "predictions from multiple decision trees."
}


# Display Model Description
st.subheader("Model Description")
st.write(model_descriptions[model_name])
