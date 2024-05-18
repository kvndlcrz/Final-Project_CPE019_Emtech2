import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np

# Define the load_model function
@st.cache(allow_output_mutation=True)
def load_model():
    model_path = 'model.h5'  # Adjust the path to your model file
    model = tf.keras.models.load_model(model_path, compile=False)
    return model

# Load the model
model = load_model()

# Streamlit app title
st.write("""
# Multi-class Weather Classification System
""")

# File uploader
file = st.file_uploader("Choose a weather photo from your computer", type=["jpg", "png"])

# Function to preprocess and predict
def import_and_predict(image_data, model):
    size = (64, 64)  # Adjust size if your model expects a different input size
    image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
    img = np.asarray(image)
    img = img / 255.0  # Normalize the image if the model expects normalized input
    img_reshape = img[np.newaxis, ...]
    prediction = model.predict(img_reshape)
    return prediction

# Process and display prediction
if file is None:
    st.text("Please upload an image file")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    prediction = import_and_predict(image, model)
    
    class_names = ['Cloudy', 'Rain', 'Shine', 'Sunrise']
    predicted_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction)  # Confidence of the prediction

    st.success(f"Prediction: {predicted_class} with {confidence:.2f} confidence")
