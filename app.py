import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import os

class WeatherClassifier:
    def __init__(self, model_path='final_model.h5'):
        self.model_path = model_path
        self.model = self.load_model()

    @st.cache(allow_output_mutation=True)
    def load_model(self):
        try:
            # Get the absolute path to the model
            model_path = os.path.join(os.path.dirname(__file__), self.model_path)
            st.write(f"Attempting to load the model from: {model_path}")

            # Load the model with custom objects if needed
            custom_objects = {
                # 'CustomLayer': CustomLayer,  # Example of a custom layer
                # 'custom_activation': custom_activation,  # Example of a custom activation function
            }

            model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
            st.write("Model loaded successfully!")
            return model
        except Exception as e:
            st.error(f"Error loading model: {e}")
            return None

    def predict(self, image_data):
        if self.model is None:
            st.error("Failed to load the model. Please check the logs for more details.")
            return None
        
        size = (64, 64)
        image = Image.open(image_data)
        image = image.resize(size)
        img_array = np.array(image)
        img_array = img_array / 255.0  # Normalize the image
        img_array = np.expand_dims(img_array, axis=0)
        prediction = self.model.predict(img_array)
        return prediction

# Streamlit UI
st.write("# Weather Classification System")

# File uploader
file = st.file_uploader("Choose a weather image from your computer", type=["jpg", "png"])

# Create Weather Classifier object
weather_classifier = WeatherClassifier()

# Main logic
if file is None:
    st.text("Please upload an image file")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    prediction = weather_classifier.predict(file)
    
    # Weather class names
    class_names = ['Cloudy', 'Rain', 'Shine', 'Sunrise']  
    
    # Output the result
    if prediction is not None:
        result = class_names[np.argmax(prediction)]
        st.success(f"Predicted Weather: {result}")
