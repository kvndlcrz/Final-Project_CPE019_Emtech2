import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load the pre-trained weather classification model
@st.cache(allow_output_mutation=True)
def load_model():
    model = tf.keras.models.load_model('final_model.h5')
    return model

# Preprocess the uploaded image
def preprocess_image(image):
    resized_image = image.resize((150, 150))  # Resize image to match model input size
    normalized_image = np.array(resized_image) / 255.0  # Normalize pixel values
    preprocessed_image = np.expand_dims(normalized_image, axis=0)  # Add batch dimension
    return preprocessed_image

# Streamlit UI
st.write("""
    # Weather Classifier App
    \nPredicts weather condition from uploaded images
""")

st.text("Upload an image (rainy, sunny, cloudy, or sunset) to predict its weather condition.")

# Upload image
uploaded_image = st.file_uploader("Choose an image (JPEG, PNG) to classify: ", type=["jpg", "png", "jpeg"])

if uploaded_image is not None:
    # Display uploaded image
    image = Image.open(uploaded_image)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Load the model
    model = load_model()

    # Preprocess image
    preprocessed_image = preprocess_image(image)

    # Make prediction
    prediction = model.predict(preprocessed_image)

    # Define weather categories
    weather_conditions = ['Cloudy', 'Rainy', 'Sunny', 'Sunset']

    # Determine predicted weather condition
    predicted_condition = weather_conditions[np.argmax(prediction)]

    # Display prediction
    st.write("Predicted Weather Condition:", predicted_condition)
