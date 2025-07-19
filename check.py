import joblib
import numpy as np
from pathlib import Path

model_path = Path(r"C:\Users\sanja\cfDNA-Lung-Cancer-ML\models")

imputer = joblib.load(model_path / "imputer.pkl")
scaler = joblib.load(model_path / "scaler.pkl")

print(type(imputer.statistics_), imputer.statistics_.dtype if hasattr(imputer, 'statistics_') else None)
print(type(scaler.mean_), scaler.mean_.dtype if hasattr(scaler, 'mean_') else None)
print(type(scaler.scale_), scaler.scale_.dtype if hasattr(scaler, 'scale_') else None)
