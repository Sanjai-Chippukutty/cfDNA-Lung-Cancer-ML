# Streamlit app will go here
# streamlit_app.py

import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load saved models
model_path = r"C:\Users\sanja\cfDNA_LungCancer_ML\models"
model = joblib.load(model_path + r"\random_forest_model.pkl")
imputer = joblib.load(model_path + r"\imputer.pkl")
scaler = joblib.load(model_path + r"\scaler.pkl")

st.set_page_config(page_title="Lung Cancer Detection", layout="wide")
st.title("🧬 Lung Cancer Prediction using cfDNA & miRNA")

st.markdown("""
This app predicts whether a patient sample indicates **Lung Cancer or Normal** based on **cfDNA methylation** and **miRNA expression** data.
""")

st.subheader("📥 Enter Feature Values")

# Load column names from your merged dataset
merged_df = pd.read_csv(r"C:\Users\sanja\cfDNA_LungCancer_ML\data\processed\merged_labeled_light.csv", index_col=0)
feature_columns = merged_df.drop(columns=['Label']).columns.tolist()

# Input fields dynamically
input_values = []
cols = st.columns(3)
for i, col in enumerate(feature_columns[:15]):  # limit to 15 features for demo
    val = cols[i % 3].number_input(f"{col}", value=0.0)
    input_values.append(val)

if st.button("🔍 Predict"):
    # Convert input to DataFrame
    input_df = pd.DataFrame([input_values], columns=feature_columns[:15])

    # Impute and scale
    input_imputed = imputer.transform(input_df)
    input_scaled = scaler.transform(input_imputed)

    # Predict
    prediction = model.predict(input_scaled)[0]
    prob = model.predict_proba(input_scaled)[0][prediction]

    label = "🟥 Lung Cancer" if prediction == 1 else "🟩 Normal"

    st.markdown(f"### 🧾 Prediction: **{label}**")
    st.markdown(f"Confidence: `{prob:.2%}`")

