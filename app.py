import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

# Load the trained CNN model
model = load_model('final_model.h5')

# Function to preprocess the uploaded image
def preprocess_image(image):
    resized_image = image.resize((150, 150))  # Resize the image to match model input size
    normalized_image = np.array(resized_image) / 255.0  # Normalize pixel values
    preprocessed_image = np.expand_dims(normalized_image, axis=0)  # Add batch dimension
    return preprocessed_image

# Streamlit UI
st.write("""
    # Weather Classifier App
    \nUses Convolutional Neural Network with 90% Accuracy
""")
st.text("Upload an image (rainy, sunny, cloudy, or sunset) to predict its weather condition.")

# Upload image
uploaded_image = st.file_uploader("Choose an image (JPEG, PNG) to classify: ", type=["jpg", "png", "jpeg"])

if uploaded_image is not None:
    # Display the uploaded image
    image = Image.open(uploaded_image)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Preprocess the image
    preprocessed_image = preprocess_image(image)

    # Make prediction
    prediction = model.predict(preprocessed_image)

    # Define weather categories
    weather_categories = ['Cloudy', 'Rainy', 'Sunny', 'Sunset']

    # Map prediction index to category
    max_index = np.argmax(prediction)
    predicted_weather = weather_categories[max_index]

    # Display prediction result
    st.write("Predicted Weather Condition:", predicted_weather)
    st.write("Prediction Confidence:", prediction[0][max_index])
