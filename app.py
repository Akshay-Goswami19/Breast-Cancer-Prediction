import streamlit as st
import numpy as np
import joblib
from tensorflow.keras.models import load_model

# ------------------ PAGE CONFIG ------------------
st.set_page_config(
    page_title="Breast Cancer Prediction",
    page_icon="🧬",
    layout="wide"
)

# ------------------ LOAD MODEL ------------------
model = load_model("breast_cancer_model.h5")
scaler = joblib.load("scaler.pkl")

# ------------------ TITLE ------------------
st.markdown(
    """
    <h1 style='text-align: center; color: #FF4B4B;'>
    🧬 Breast Cancer Prediction App
    </h1>
    <p style='text-align: center; font-size:18px;'>
    Enter patient details to predict whether the tumor is <b>Benign</b> or <b>Malignant</b>
    </p>
    """,
    unsafe_allow_html=True
)

st.write("---")

# ------------------ FEATURE NAMES ------------------
feature_names = [
    "Radius Mean", "Texture Mean", "Perimeter Mean", "Area Mean", "Smoothness Mean",
    "Compactness Mean", "Concavity Mean", "Concave Points Mean", "Symmetry Mean", "Fractal Dimension Mean",
    "Radius SE", "Texture SE", "Perimeter SE", "Area SE", "Smoothness SE",
    "Compactness SE", "Concavity SE", "Concave Points SE", "Symmetry SE", "Fractal Dimension SE",
    "Radius Worst", "Texture Worst", "Perimeter Worst", "Area Worst", "Smoothness Worst",
    "Compactness Worst", "Concavity Worst", "Concave Points Worst", "Symmetry Worst", "Fractal Dimension Worst"
]

# ------------------ NORMAL VALUES (approx reference) ------------------
normal_values = [
    14.0, 20.0, 90.0, 600.0, 0.1,
    0.2, 0.1, 0.05, 0.2, 0.06,
    0.5, 1.0, 3.0, 40.0, 0.005,
    0.02, 0.03, 0.01, 0.02, 0.003,
    16.0, 25.0, 100.0, 700.0, 0.14,
    0.3, 0.4, 0.2, 0.3, 0.08
]

# ------------------ INPUT UI ------------------
st.subheader("📝 Enter Patient Details")

cols = st.columns(5)
inputs = []

for i, (name, normal) in enumerate(zip(feature_names, normal_values)):
    col = cols[i % 5]
    
    with col:
        value = st.number_input(name, value=float(normal), key=name)
        st.caption(f"Normal: {normal}")
        st.markdown("<br>", unsafe_allow_html=True)
    inputs.append(value)

# Convert to numpy array
data = np.array([inputs])

# ------------------ PREDICTION ------------------
st.write("---")

if st.button("🔍 Predict"):
    try:
        data_scaled = scaler.transform(data)
        prediction = model.predict(data_scaled)[0][0]

        st.write("### 🧾 Prediction Result")

        if prediction > 0.5:
            st.error("⚠️ Malignant (Cancer Detected)")
        else:
            st.success("✅ Benign (No Cancer)")

        st.info(f"📊 Probability: {prediction:.4f}")

    except Exception as e:
        st.error(f"Error: {e}")

# ------------------ FOOTER ------------------
st.write("---")
st.markdown(
    "<p style='text-align:center;'>Built with ❤️ using Streamlit</p>",
    unsafe_allow_html=True
)