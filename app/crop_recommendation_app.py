import streamlit as st
import joblib
import numpy as np
from utils import get_weather

# Load the trained model & scaler
model = joblib.load("models/crop_recommendation_model.pkl")
scaler = joblib.load("models/scaler.pkl")

st.title("🌾 AI Crop Recommendation System")
st.write("Get the best crops to plant based on your city's weather!")

# User input
city = st.text_input("Enter your city (e.g., Lagos)")

if st.button("Recommend Crops"):
    weather = get_weather(city)
    if weather:
        st.write(f"🌡 Temperature: {weather['temperature']}°C")
        st.write(f"💧 Humidity: {weather['humidity']}%")
        st.write(f"🌧 Rainfall: {weather['rainfall']} mm")

        # Prepare input for model
        features = np.array([[weather["temperature"], weather["humidity"], weather["rainfall"]]])
        features_scaled = scaler.transform(features)

        # Predict crop
        crop = model.predict(features_scaled)[0]
        st.success(f"✅ Recommended Crop: **{crop}**")
    else:
        st.error("⚠ Unable to fetch weather data. Please try again.")
