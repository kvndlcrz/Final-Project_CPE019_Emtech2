import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np

# Define a function to load the model
@st.cache(allow_output_mutation=True)
def load_model():
    try:
        # Load the model
        model = tf.keras.models.load_model('model.h5')
        
        # Loop through the layers and replace invalid characters in the names
        for layer in model.layers:
            layer.name = layer.name.replace('/', '_')
        
        return model
    except Exception as e:
        st.error(f"Error loading the model: {e}")
        return None

# Function to preprocess the image and make predictions
def import_and_predict(image_data, model):
    size = (64, 64)  # Adjust size if your model expects a different input size
    image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
    img = np.asarray(image)
    img = img / 255.0  # Normalize the image if the model expects normalized input
    img_reshape = img[np.newaxis, ...]
    prediction = model.predict(img_reshape)
    return prediction

# Load the model
model = load_model()

# If the model is successfully loaded, continue with the rest of the code
if model is not None:
    st.write("""
    # Multi-class Weather Classification System
    """)

    file = st.file_uploader("Choose a weather photo from your computer", type=["jpg", "png"])

    # If a file is uploaded, display the image and make predictions
    if file is not None:
        image = Image.open(file)
        st.image(image, use_column_width=True)
        prediction = import_and_predict(image, model)
        
        # Class names for your model
        class_names = ['Cloudy', 'Rain', 'Shine', 'Sunrise']
        
        # Get the predicted class and confidence
        predicted_class = class_names[np.argmax(prediction)]
        confidence = np.max(prediction)  # Confidence of the prediction

        st.success(f"Prediction: {predicted_class} with {confidence:.2f} confidence")
