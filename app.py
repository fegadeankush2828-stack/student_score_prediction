import streamlit as st
import pickle
import numpy as np
import pandas as pd

# --- Page Configuration ---
st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom Styling ---
st.markdown("""
    <style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #4CAF50;
        margin-bottom: 20px;
    }
    .metric-value {
        font-size: 36px;
        font-weight: bold;
        color: #1E3A8A;
    }
    .metric-label {
        font-size: 16px;
        color: #555555;
    }
    </style>
""", unsafe_allow_html=True)

# --- Load the Pickle Model Safely ---
@st.cache_resource
def load_model():
    try:
        with open("model.pkl", "rb") as file:
            model = pickle.load(file)
        return model
    except FileNotFoundError:
        st.error("⚠️ `model.pkl` file not found! Please make sure it is in the same directory as this script.")
        return None
    except Exception as e:
        st.error(f"⚠️ Error loading model: {e}")
        return None

model = load_model()

# --- App Layout & Design ---
st.title("🎓 Student Exam Score Predictor")
st.markdown("Predict a student's final exam score based on study habits, attendance, and past academic performance using machine learning.")
st.write("---")

if model is not None:
    # Creating structured columns for user inputs vs output display
    col1, col2 = st.columns([1, 1.2], gap="large")

    with col1:
        st.subheader("📝 Input Student Metrics")
        st.markdown("Adjust the sliders below to enter the student's metrics.")
        
        # Inputs based directly on your model's feature names
        hours_studied = st.slider(
            "📚 Hours Studied (per day)", 
            min_value=0.0, 
            max_value=12.0, 
            value=6.0, 
            step=0.5,
            help="Total hours spent studying per day."
        )
        
        sleep_hours = st.slider(
            "😴 Sleep Hours (per day)", 
            min_value=3.0, 
            max_value=10.0, 
            value=7.0, 
            step=0.5,
            help="Average daily sleep duration."
        )
        
        attendance_percent = st.slider(
            "🏫 Attendance Rate (%)", 
            min_value=0.0, 
            max_value=100.0, 
            value=85.0, 
            step=1.0,
            help="Classroom attendance percentage."
        )
        
        previous_scores = st.slider(
            "📊 Previous Exam Score (0-100)", 
            min_value=0, 
            max_value=100, 
            value=75, 
            step=1,
            help="Historical average performance in past assessments."
        )

    with col2:
        st.subheader("🔮 Prediction Results")
        
        # Match features exactly with the model's feature_names_in_
        features = pd.DataFrame([[hours_studied, sleep_hours, attendance_percent, previous_scores]], 
                                columns=['hours_studied', 'sleep_hours', 'attendance_percent', 'previous_scores'])
        
        # Predict
        try:
            prediction = model.predict(features)[0]
            # Clip predictions so they make sense on a 0-100 scale if needed
            final_score = np.clip(round(prediction, 2), 0.0, 100.0)
            
            # Display Prediction in a visually appealing card
            st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Predicted Final Exam Score</div>
                    <div class="metric-value">{final_score} / 100</div>
                </div>
            """, unsafe_allow_html=True)
            
            # Interactive Insights
            st.markdown("### 💡 Quick Summary & Insights")
            if final_score >= 85:
                st.success("🌟 **Excellent Prospects!** The model predicts an outstanding result. Keep maintaining these habits!")
            elif final_score >= 50:
                st.info("👍 **Solid Path.** A passing and respectable score is predicted. Minor adjustments to study time or attendance could push this even higher.")
            else:
                st.warning("⚠️ **Warning Zone.** The prediction falls below average. Increasing study sessions and attending classes consistently are recommended.")
                
        except Exception as e:
            st.error(f"Error predicting with inputs: {e}")

else:
    st.info("💡 Once you place your `model.pkl` in the directory, restart the application to load the interface.")
