import joblib
from pathlib import Path

# Set path to your existing model
old_model_path = Path("models/final_model.pkl")  # or "models/random_forest_model.pkl"

# Load the old model
try:
    model = joblib.load(old_model_path)
    print("✅ Old model loaded successfully.")
except Exception as e:
    print("❌ Failed to load model:", e)
    exit()

# Save (overwrite or create a new one) using current scikit-learn version
new_model_path = Path("models/final_model_resaved.pkl")
try:
    joblib.dump(model, new_model_path, protocol=4)  # Protocol 4 = cross-version compatible
    print(f"✅ Model re-saved to: {new_model_path}")
except Exception as e:
    print("❌ Failed to save model:", e)
