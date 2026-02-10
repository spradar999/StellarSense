

import streamlit as st
import pandas as pd
import numpy as np
import joblib

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, matthews_corrcoef,
    confusion_matrix, classification_report
)
import seaborn as sns
import matplotlib.pyplot as plt


# ------------------------
# Load models
# ------------------------
models = {
    "Logistic Regression": joblib.load("model/logistic_regression.pkl"),
    "Decision Tree": joblib.load("model/decision_tree.pkl"),
    "KNN": joblib.load("model/knn.pkl"),
    "Naive Bayes": joblib.load("model/naive_bayes.pkl"),
    "Random Forest": joblib.load("model/random_forest.pkl"),
    "XGBoost": joblib.load("model/xgboost.pkl")
}

scaler = joblib.load("model/scaler.pkl")
label_encoder = joblib.load("model/label_encoder.pkl")
import streamlit as st



st.title("🌌 StellarSense – Classifying Stars, Galaxies & Quasars")

st.subheader("📥 Download Sample Test Dataset")

with open("data/test_data.csv", "rb") as file:
    st.download_button(
        label="⬇️ Download Test CSV",
        data=file,
        file_name="test_data.csv",
        mime="text/csv"
    )

# ------------------------
# (a) Dataset Upload
# ------------------------
uploaded_file = st.file_uploader("Upload TEST CSV file", type=["csv"])

# ------------------------
# (b) Model Selection
# ------------------------
model_name = st.selectbox("Select Model", list(models.keys()))
model = models[model_name]

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("Uploaded Dataset")
    st.write(df.head())

    # Separate X and y
    X = df.drop(columns=["class","obj_ID","spec_obj_ID"], errors="ignore")
    y = df["class"]

    # Encode target
    y_true = label_encoder.transform(y)

    # Scale features
    X_scaled = scaler.transform(X)

    # ------------------------
    # Prediction
    # ------------------------
    y_pred = model.predict(X_scaled)
    y_prob = model.predict_proba(X_scaled)

    # ------------------------
    # (c) Evaluation Metrics
    # ------------------------
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="macro")
    rec = recall_score(y_true, y_pred, average="macro")
    f1 = f1_score(y_true, y_pred, average="macro")
    mcc = matthews_corrcoef(y_true, y_pred)
    auc = roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro")

    st.subheader("📊 Evaluation Metrics")
    metrics_df = pd.DataFrame({
        "Metric": ["Accuracy","AUC","Precision","Recall","F1","MCC"],
        "Value": [acc, auc, prec, rec, f1, mcc]
    })
    st.table(metrics_df)

    # ------------------------
    # (d) Confusion Matrix
    # ------------------------
    st.subheader("📉 Confusion Matrix")
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=label_encoder.classes_,
                yticklabels=label_encoder.classes_,
                ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)

    # ------------------------
    # Classification Report
    # ------------------------
    st.subheader("📑 Classification Report")
    report = classification_report(y_true, y_pred, target_names=label_encoder.classes_, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    st.dataframe(report_df)

