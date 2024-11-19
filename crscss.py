import streamlit as st
import numpy as np
import joblib

# Load the pre-trained model and label encoder
model = joblib.load('crop_prediction_model.pkl')
label_encoder = joblib.load('le.pkl') 

# Set the title and style the page
st.markdown("""
    <style>
        .title {
            text-align: center;
            font-size: 40px;
            color: #4CAF50;
            font-family: 'Arial', sans-serif;
            padding: 20px;
        }
        .description {
            text-align: center;
            font-size: 20px;
            color: #555555;
            font-family: 'Arial', sans-serif;
            padding: 10px;
        }
        .input-section {
            font-size: 18px;
            color: #333;
            font-family: 'Arial', sans-serif;
            margin-top: 30px;
        }
        .prediction-result {
            font-size: 24px;
            color: #4CAF50;
            font-weight: bold;
            text-align: center;
            padding: 20px;
        }
        .button {
            display: flex;
            justify-content: center;
            margin-top: 20px;
        }
    </style>
""", unsafe_allow_html=True)

# Title of the app
st.markdown('<div class="title">Crop Prediction App</div>', unsafe_allow_html=True)

# Description of the app
st.markdown('<div class="description">This app predicts the best crop to grow based on soil and environmental conditions. Enter the required values and click "Predict" to see the crop category.</div>', unsafe_allow_html=True)

# Collecting user input for the features
N = st.number_input('Nitrogen content (N) in the soil (kg/ha)', min_value=0, max_value=500, step=1)
P = st.number_input('Phosphorus content (P) in the soil (kg/ha)', min_value=0, max_value=500, step=1)
K = st.number_input('Potassium content (K) in the soil (kg/ha)', min_value=0, max_value=500, step=1)
temperature = st.number_input('Temperature (°C)', min_value=0, max_value=50, step=1)
humidity = st.number_input('Humidity (%)', min_value=0, max_value=100, step=1)
ph = st.number_input('Soil pH', min_value=0.0, max_value=14.0, step=0.1)
rainfall = st.number_input('Rainfall (mm)', min_value=0, max_value=500, step=1)

# Styling for the input section
st.markdown('<div class="input-section">Enter the environmental conditions and soil data below:</div>', unsafe_allow_html=True)

# Prepare the input data in the same format as used for training the model
input_data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])

# Button to trigger prediction
if st.button('Predict Crop', key='predict_button'):
    # Make a prediction using the trained model
    prediction = model.predict(input_data)
    
    # If label encoding was used during training, decode the prediction to human-readable labels
    predicted_crop = label_encoder.inverse_transform(prediction)  # Decode the category
    
    # Show the prediction result with styling
    st.markdown(f'<div class="prediction-result">The predicted crop category is: {predicted_crop[0]}</div>', unsafe_allow_html=True)
