import streamlit as st
import pandas as pd
import joblib
import os

# Set page config
st.set_page_config(page_title="No-Show Predictor", layout="wide")

# Load model
@st.cache_resource
def load_model():
    return joblib.load("no_show_model.pkl")

model = load_model()

# Title
st.title("Medical Appointment No-Show Predictor")
st.markdown("Upload appointment data and predict who is likely to miss their medical appointment.")

# File upload
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file:
    # Read data
    data = pd.read_csv(uploaded_file)

    st.subheader("Uploaded Data Preview")
    st.dataframe(data.head(), use_container_width=True)

    # Preprocessing to match model format
    try:
        data['ScheduledDay'] = pd.to_datetime(data['ScheduledDay'], errors='coerce')
        data['AppointmentDay'] = pd.to_datetime(data['AppointmentDay'], errors='coerce')
        data['waiting_days'] = (data['AppointmentDay'] - data['ScheduledDay']).dt.days
        data['Gender'] = data['Gender'].map({'F': 0, 'M': 1})
        data['scheduled_dayofweek'] = data['ScheduledDay'].dt.dayofweek
        data['appointment_dayofweek'] = data['AppointmentDay'].dt.dayofweek
        data['same_day'] = (data['waiting_days'] == 0).astype(int)
        data['age_group'] = pd.cut(data['Age'], bins=[0, 12, 18, 35, 60, 100], labels=[0, 1, 2, 3, 4])

        # Drop unused
        data.drop(columns=['PatientId', 'AppointmentID', 'ScheduledDay', 'AppointmentDay'], inplace=True)

        # Fill missing neighborhood or age_group
        data = pd.get_dummies(data, columns=['Neighbourhood', 'age_group'], drop_first=True)

        # Align columns with model's training data
        expected_cols = model.get_booster().feature_names
        for col in expected_cols:
            if col not in data.columns:
                data[col] = 0
        data = data[expected_cols]

        # Predict
        st.subheader("Prediction Results")
        predictions = model.predict(data)
        data['No-Show Probability'] = model.predict_proba(data)[:, 1]
        data['Prediction'] = predictions
        data['Prediction Label'] = data['Prediction'].map({0: "✅ Will Show", 1: "❌ No-Show Likely"})

        # Show results
        st.dataframe(data[['No-Show Probability', 'Prediction Label']], use_container_width=True)

        # Download
        csv = data.to_csv(index=False).encode('utf-8')
        st.download_button("⬇Download Full Prediction CSV", csv, "no_show_predictions.csv", "text/csv")

    except Exception as e:
        st.error(f"Error processing file: {e}")

else:
    st.info("Please upload a CSV file to get started.")
