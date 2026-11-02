import gradio as gr
import joblib
import numpy as np
import pandas as pd
import os

# Load trained model
model = joblib.load('my_model.pkl')

def predict(gender,age,hypertension,heart_disease,smoking_history,bmi,HbA1c_level,blood_glucose_level):

    # Create input dataframe with correct feature order
    input_data = pd.DataFrame([[
        gender,age,hypertension,heart_disease,smoking_history,bmi,HbA1c_level,blood_glucose_level
    ]], columns=[
        gender,age,hypertension,heart_disease,smoking_history,bmi,HbA1c_level,blood_glucose_level
    ])

    # Make prediction directly using original input features
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0]

    if prediction == 1:
        return f"⚠️Diabetes Detected\nConfidence: {probability[1]*100:.2f}%"
    else:
        return f"✅ No Diabetes Detected\nConfidence: {probability[0]*100:.2f}%"

# Build the dashboard
demo = gr.Interface(
    fn=predict,
    inputs=[
        gr.Radio(choices=[0, 1], label="Gender(0=Female, 1=Male)", value=1),
        gr.Number(label="Age (years)", value=50),
        gr.Slider(0, 1, step=1, label="Hypertension", value=1),
        gr.Radio(choices=[0,1],label="Heart Disease",value=1),
        gr.radio(choices=[current,ever,former,never,no info,not_current],label="Smoking History", value=1),
        gr.Slider(10,70, step=1, label="BMI", value=1),
        gr.slider(3, 9, label="HbA1c_level", value=0.1),
        gr.slider(80, 300, label="blood_glucose_level", value=0)
    ],
    outputs=gr.Textbox(label="Prediction Result", lines=3),
    title="Diabetes Prediction Dashboard",
    description="Enter patient details and click Submit to predict diabetes."
)


# Launch for Render (uses PORT environment variable)
# Launch for Render (uses PORT environment variable)
demo.launch(
    share=True,
    server_name="0.0.0.0",
    server_port=int(os.environ.get("PORT", 7860))
)
