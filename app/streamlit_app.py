
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Set page config
st.set_page_config(page_title="Lung Cancer Detection", page_icon="🧬", layout="centered")

# Title and description
st.title("🧬 Lung Cancer Detection Using cfDNA + miRNA")
st.markdown("Upload your cfDNA methylation + miRNA expression data to get lung cancer prediction using our ML model.")

# Load the model, imputer, and scaler
@st.cache_resource
def load_model():
    model = joblib.load("models/random_forest_model.pkl")
    imputer = joblib.load("models/imputer.pkl")
    scaler = joblib.load("models/scaler.pkl")
    return model, imputer, scaler

model, imputer, scaler = load_model()

# Define input features
features = ['gene1', 'gene2', 'gene3', 'miRNA_21', 'miRNA_34a']

# Section: Upload CSV
st.subheader("📂 Upload a CSV file with matching features")
uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

# Section: Manual Input
st.subheader("✍️ Or Enter Values Manually")

manual_input = {}
for feature in features:
    manual_input[feature] = st.number_input(feature, format="%.4f")

# Prediction logic
def predict(data):
    data_imputed = imputer.transform(data)
    data_scaled = scaler.transform(data_imputed)
    prediction = model.predict(data_scaled)
    probability = model.predict_proba(data_scaled)[0][1] * 100
    return prediction[0], probability

# Handle CSV prediction
if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
        if not all(f in df.columns for f in features):
            st.error(f"❌ CSV must contain these columns: {features}")
        else:
            pred, prob = predict(df[features])
            st.success(f"📊 Prediction: {'High probability of Lung Cancer' if pred == 1 else 'Low probability of Lung Cancer'}")
            st.info(f"Probability of Cancer: {prob:.2f}%")
    except Exception as e:
        st.error(f"⚠️ Error reading the file: {e}")

# Handle manual input prediction
if st.button("🔍 Predict Manually"):
    try:
        input_df = pd.DataFrame([manual_input])
        pred, prob = predict(input_df)
        st.success(f"📊 Manual Prediction Result: {'High probability of Lung Cancer' if pred == 1 else 'Low probability of Lung Cancer'}")
        st.info(f"Probability of Cancer: {prob:.2f}%")
    except Exception as e:
        st.error(f"⚠️ Error during manual prediction: {e}")

