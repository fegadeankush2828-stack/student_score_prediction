import streamlit as st
import pickle
import numpy as np


# Load trained model
model = pickle.load(open("model.pkl", "rb"))


# Page title
st.title("🎓 Student Exam Score Prediction")

st.write(
    "Predict exam score using study habits and previous performance"
)


# Input fields

hours_studied = st.number_input(
    "📘 Hours Studied",
    min_value=0.0,
    max_value=24.0,
    value=5.0
)


sleep_hours = st.number_input(
    "😴 Sleep Hours",
    min_value=0.0,
    max_value=24.0,
    value=7.0
)


attendance_percent = st.number_input(
    "📅 Attendance (%)",
    min_value=0.0,
    max_value=100.0,
    value=80.0
)


previous_scores = st.number_input(
    "📊 Previous Score",
    min_value=0,
    max_value=100,
    value=60
)

# Prediction button

if st.button("🔮 Predict Exam Score"):

    features = np.array(
        [[
            hours_studied,
            sleep_hours,
            attendance_percent,
            previous_scores
        ]]
    )


    prediction = model.predict(features)


    st.success(
        f"🎯 Predicted Exam Score: **{prediction[0]:.2f} / 100**"
    )


# Footer

st.caption(
    "Built with ❤️ using Streamlit and Machine Learning"
)
