import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Function to load the trained model
@st.cache(allow_output_mutation=True)
def load_model():
    model = tf.keras.models.load_model('final_model.h5')
    return model

# Function to preprocess the uploaded image
def preprocess_image(image):
    size = (150, 150)
    resized_image = image.resize(size)  # Resize the image to match model input size
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

    # Load the model
    model = load_model()

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
