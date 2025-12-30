import streamlit as st
import numpy as np
import joblib

# Page config
st.set_page_config(page_title="Linear Regression App", page_icon="📘", layout="centered")

# Load model
model = joblib.load("linear_marks_model.pkl")

# Title
st.title("📊 Linear Regression Result Predictor")
st.write("Study Hour দিলে Result (100 এর মধ্যে) predict করবে")

st.divider()

# Input
study_hour = st.number_input(
    "📘 Study Hour লিখুন",
    min_value=0.0,
    max_value=24.0,
    step=0.5
)

# Prediction
if st.button("🔮 Predict Result"):
    input_data = np.array([[study_hour]])
    prediction = model.predict(input_data)

    st.success(f"✅ Predicted Result: **{prediction[0]:.2f} / 100**")
