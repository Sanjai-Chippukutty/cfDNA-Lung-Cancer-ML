import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ✅ Page config
st.set_page_config(page_title="Lung Cancer Detection Using cfDNA + miRNA", layout="centered")

# ✅ Title and instructions
st.title("🧬 Lung Cancer Detection Using cfDNA + miRNA")
st.markdown("Upload your cfDNA methylation + miRNA expression data to get lung cancer prediction using our ML model.")

@st.cache_resource
def load_model():
    model = joblib.load("models/random_forest_model.pkl")
    imputer = joblib.load("models/imputer.pkl")
    scaler = joblib.load("models/scaler.pkl")
    return model, imputer, scaler
    model, imputer, scaler = load_model()

# ✅ Define required features
REQUIRED_FEATURES = ['gene1', 'gene2', 'gene3', 'miRNA_21', 'miRNA_34a']

# ✅ Upload CSV section
st.header("📂 Upload a CSV file with matching features")
uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
        missing = [feat for feat in REQUIRED_FEATURES if feat not in df.columns]
        if missing:
            st.error(f"Missing required features: {missing}")
        else:
            input_data = df[REQUIRED_FEATURES]
            input_imputed = imputer.transform(input_data)
            input_scaled = scaler.transform(input_imputed)
            predictions = model.predict(input_scaled)
            probabilities = model.predict_proba(input_scaled)[:, 1]

            st.subheader("📊 Prediction Results")
            result_df = pd.DataFrame({
                "Prediction": ["High" if p == 1 else "Low" for p in predictions],
                "Probability (%)": [f"{prob * 100:.2f}" for prob in probabilities]
            })
            st.dataframe(result_df)

    except Exception as e:
        st.error(f"⚠️ Error reading the file: {e}")

# ✅ Manual input section
st.header("✍️ Or Enter Values Manually")

with st.form("manual_input_form"):
    manual_values = []
    for feature in REQUIRED_FEATURES:
        val = st.number_input(f"{feature}", min_value=0.0, step=0.01, format="%.4f")
        manual_values.append(val)
    submit = st.form_submit_button("Predict")

if submit:
    try:
        input_array = np.array(manual_values).reshape(1, -1)
        input_imputed = imputer.transform(input_array)
        input_scaled = scaler.transform(input_imputed)
        prediction = model.predict(input_scaled)[0]
        probability = model.predict_proba(input_scaled)[0][1]

        st.subheader("🔍 Manual Prediction Result")
        if prediction == 1:
            st.error("High probability of Lung Cancer.")
        else:
            st.success("Low probability of Lung Cancer.")
        st.write(f"**Probability of Cancer:** `{probability * 100:.2f}%`")

    except Exception as e:
        st.error(f"⚠️ Error during manual prediction: {e}")
