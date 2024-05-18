import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
import os

def load_model_safely(model_path):
    try:
        st.write(f"Attempting to load the model from: {model_path}")
        model = load_model(model_path)
        st.write("Model loaded successfully!")
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Ensure the file path is correct
model_path = 'final_model.h5'
if not os.path.exists(model_path):
    st.error(f"The model file {model_path} does not exist.")
    st.stop()

# Load the CNN model
model = load_model_safely(model_path)

if model is None:
    st.error("Failed to load the model. Please check the logs for more details.")
    st.stop()

# Function to preprocess the image
def preprocess_image(image):
    resized_image = image.resize((150, 150))  # Adjust the size according to your model's input size
    normalized_image = np.array(resized_image) / 255.0
    preprocessed_image = np.expand_dims(normalized_image, axis=0)
    return preprocessed_image

# Streamlit UI
st.image("weather.jpg", width=400)
st.write("""
    # Model Deployment on the Cloud
    \nAn Application of Convolutional Neural Network in Weather ['cloudy', 'rainy', 'shine', 'sunset'] 
    Prediction with an Accuracy Rate of 90%.
""")
st.text("Using the Weather Dataset to predict from an uploaded image.")

# Upload image
uploaded_image = st.file_uploader("Choose an image that can be classified as rainy, sunset, shine, or cloudy: ", type=["jpg", "png", "jpeg"])

if uploaded_image is not None:
    # Display the uploaded image
    image = Image.open(uploaded_image)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Preprocess the image
    preprocessed_image = preprocess_image(image)

    prediction = model.predict(preprocessed_image)

    # Define categories
    categories = ['cloudy', 'rainy', 'shine', 'sunset']

    # Create a dictionary to map index to categories
    category_mapping = {i: category for i, category in enumerate(categories)}

    # Display prediction
    max_index = np.argmax(prediction)
    predicted_category = category_mapping[max_index]
    st.write("Prediction Category:", predicted_category)
    st.write("Prediction Probability:", prediction[0][max_index])
