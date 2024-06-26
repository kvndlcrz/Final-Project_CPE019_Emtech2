import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load the pre-trained weather classification model
def load_model():
    try:
        st.write("Loading the pre-trained weather classification model...")
        model = tf.keras.models.load_model('final_model.h5')
        st.success("Model loaded successfully!")
        return model
    except Exception as e:
        st.error(f"Error loading the model: {e}")
        return None

# Preprocess the uploaded image
def preprocess_image(image):
    size = (150, 150)
    resized_image = image.resize(size)  # Resize the image to match the model input size
    normalized_image = np.array(resized_image) / 255.0  # Normalize pixel values
    preprocessed_image = np.expand_dims(normalized_image, axis=0)  # Add batch dimension
    return preprocessed_image

# UI setup
st.write("""
### <span style='color:yellow'>Weather Vision:</span> <span style='color:white'>Predicting Weather Conditions from Image</span>
<div style="text-align: center;">Predict the weather condition from uploaded images. Possible conditions: Cloudy, Rainy, Shine, Sunrise</div>
""", unsafe_allow_html=True)

# GitHub link
st.write(""" <div style="text-align: center;"><br>
    \nhttps://github.com/kvndlcrz/Final-Project_CPE019_Emtech2.git
 <br><br>""", unsafe_allow_html=True)

# Upload image
uploaded_image = st.file_uploader(
    label=" Choose an image (jpg, png, jpeg) to classify:",
    type=["jpg", "png", "jpeg"]
)

if uploaded_image is not None:
    # Display the uploaded image
    image = Image.open(uploaded_image)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Load the model
    model = load_model()

    if model is not None:
        # Preprocess the image
        preprocessed_image = preprocess_image(image)

        # Make prediction
        prediction = model.predict(preprocessed_image)

        # Define weather categories
        weather_conditions = ['Cloudy', 'Rainy', 'Shine', 'Sunrise']

        # Determine the predicted weather condition
        predicted_condition = weather_conditions[np.argmax(prediction)]

        # Display the prediction
        st.write("Predicted Weather Condition:", predicted_condition)
    else:
        st.write("Failed to load the model.")
else:
    st.write("Please upload an image to get a prediction.")
