
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# Page config
st.set_page_config(
    page_title="Lung Cancer Detection App",
    layout="centered"
)

st.title("🧬 Lung Cancer Detection using cfDNA Methylation & miRNA Signatures")
st.markdown("Upload a CSV file _or_ manually enter data to predict lung cancer probability using a trained ML model.")

# Load model, imputer, scaler
try:
    base_model_dir = os.path.join(os.path.dirname(__file__), "..", "models")

    model = joblib.load(os.path.join(base_model_dir, "final_model.pkl"))
    imputer = joblib.load(os.path.join(base_model_dir, "imputer.pkl"))
    scaler = joblib.load(os.path.join(base_model_dir, "scaler.pkl"))

except Exception as e:
    st.error(f"❌ Error loading model files: {e}")
    st.stop()

# Load feature names
data_path = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "merged_labeled_light.csv")
try:
    merged_df = pd.read_csv(data_path, index_col=0)
    feature_columns = merged_df.drop(columns=["Label"]).columns.tolist()
except Exception:
    st.warning("⚠️ Dataset not found. Using default feature names.")
    feature_columns = ['gene1', 'gene2', 'gene3', 'miRNA_21', 'miRNA_34a']  # Customize based on your model

# --- Section 1: File Upload ---
st.header("📂 Upload CSV File")
uploaded_file = st.file_uploader("Upload a CSV file with correct feature columns", type=["csv"])

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)

        # Check feature compatibility
        missing = [f for f in feature_columns if f not in df.columns]
        if missing:
            st.error(f"❌ Missing required features: {missing}")
            st.stop()

        input_data = df[feature_columns]
        imputed = imputer.transform(input_data)
        scaled = scaler.transform(imputed)

        predictions = model.predict(scaled)
        probs = model.predict_proba(scaled)[:, 1]

        df["Prediction"] = predictions
        df["Cancer_Probability"] = probs

        st.subheader("✅ Batch Prediction Results")
        st.dataframe(df[["Prediction", "Cancer_Probability"]])

        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Download Results as CSV", data=csv, file_name="batch_predictions.csv")

    except Exception as e:
        st.error(f"❌ Error processing file: {e}")

# --- Section 2: Manual Input ---
st.header("📝 Or Manually Enter Values")

with st.form("manual_input_form"):
    user_input = []
    for feature in feature_columns:
        val = st.number_input(f"{feature}", min_value=0.0, max_value=1_000_000.0, step=0.01)
        user_input.append(val)

    predict_btn = st.form_submit_button("🔍 Predict from Manual Input")

if predict_btn:
    try:
        input_array = np.array(user_input).reshape(1, -1)
        input_imputed = imputer.transform(input_array)
        input_scaled = scaler.transform(input_imputed)

        prediction = model.predict(input_scaled)[0]
        prob = model.predict_proba(input_scaled)[0][1]

        st.subheader("✅ Manual Prediction Result")
        if prediction == 1:
            st.error(f"High risk of Lung Cancer predicted. 🔴")
        else:
            st.success(f"Low risk of Lung Cancer predicted. 🟢")
        st.write(f"**Predicted Probability:** `{prob * 100:.2f}%`")

    except Exception as e:
        st.error(f"❌ Error during manual prediction: {e}")

