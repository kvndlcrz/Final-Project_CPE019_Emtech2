import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

class WeatherClassifier:
    model_path = 'final_model.h5'

    @staticmethod
    @st.cache(allow_output_mutation=True)
    def load_model(model_path):
        try:
            st.write("Attempting to load the model...")
            model = tf.keras.models.load_model(model_path)
            st.write("Model loaded successfully!")
            return model
        except Exception as e:
            st.error(f"Error loading model: {e}")
            return None

    @staticmethod
    def preprocess_image(image):
        size = (150, 150)
        resized_image = image.resize(size)  # Resize image to match model input size
        normalized_image = np.array(resized_image) / 255.0  # Normalize pixel values
        preprocessed_image = np.expand_dims(normalized_image, axis=0)  # Add batch dimension
        return preprocessed_image

    @staticmethod
    def predict(model, image_data):
        if model is None:
            st.error("Failed to load the model. Please check the logs for more details.")
            return None
        
        try:
            image = Image.open(image_data)
            preprocessed_image = WeatherClassifier.preprocess_image(image)
            prediction = model.predict(preprocessed_image)
            return prediction
        except Exception as e:
            st.error(f"Error predicting image: {e}")
            return None

# Streamlit UI
st.write("""# Weather Classifier App""")

# File uploader
file = st.file_uploader("Choose a weather image from your computer", type=["jpg", "png", "jpeg"])

# Load the model
model = WeatherClassifier.load_model(WeatherClassifier.model_path)

# Main logic
if file is None:
    st.text("Please upload an image file")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    prediction = WeatherClassifier.predict(model, file)
    
    # Weather class names
    weather_conditions = ['Cloudy', 'Rainy', 'Sunny', 'Sunset']  
    
    # Output the result
    if prediction is not None:
        predicted_condition = weather_conditions[np.argmax(prediction)]
        st.success("Predicted Weather Condition: {}".format(predicted_condition))
