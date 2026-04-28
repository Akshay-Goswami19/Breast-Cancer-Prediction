import numpy as np
from tensorflow.keras.models import load_model
import joblib

# Load model
model = load_model("breast_cancer_model.h5")

# Load scaler
scaler = joblib.load("scaler.pkl")

# Example input (30 features)
sample = np.array([[ 
    14.5, 20.1, 90.2, 600.5, 0.1, 0.2, 0.15, 0.08, 0.18, 0.06,
    0.5, 1.2, 3.5, 40.0, 0.005, 0.02, 0.03, 0.01, 0.02, 0.003,
    16.0, 25.0, 100.0, 700.0, 0.14, 0.3, 0.4, 0.2, 0.3, 0.08
]])

# Step 1: Normalize
sample_scaled = scaler.transform(sample)

# Step 2: Predict
prediction = model.predict(sample_scaled)

# Step 3: Convert to result
if prediction[0][0] > 0.5:
    print("Malignant (Cancer)")
else:
    print("Benign (No Cancer)")

print("Probability:", prediction[0][0])