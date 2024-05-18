import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np

# Set page config
st.set_page_config(page_title="Multi-class Weather Classification", layout="wide")

# Title and student details
st.title("Multi-class Weather Classification")
st.markdown("""
Name:
- Kevin Roi A. Sumaya
- Daniela D. Rabang

Course/Section: CPE019/CPE32S5

Date Submitted: May 17, 2024
""")

# Load the trained model
@st.cache(allow_output_mutation=True)
def load_model():
    model = tf.keras.models.load_model('model.h5')  # Adjust the model file path
    return model

# Define the class names for weather categories
class_names = ['Cloudy', 'Rain', 'Shine', 'Sunrise']  # Adjust as per your dataset

model = load_model()

# Streamlit app
st.title("Weather Classification")
st.write("Upload an image to classify the type of weather.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

def import_and_predict(image_data, model):
    size = (64, 64)  # Adjust size if your model expects a different input size
    image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
    img = np.asarray(image)
    img = img / 255.0  # Normalize the image if the model expects normalized input
    img_reshape = img[np.newaxis, ...]
    prediction = model.predict(img_reshape)
    return prediction

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    st.write("")
    st.write("Classifying...")

    prediction = import_and_predict(image, model)
    predicted_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction)

    st.write(f"Prediction: {predicted_class}")
    st.write(f"Confidence: {confidence:.2f}")

# Displaying example images for each weather category
st.write("## Example Images by Category")
example_images = {
    'Cloudy':
    'Rain': 
    'Shine': 
    'Sunrise': 
}
for label, path in example_images.items():
    image = Image.open(path)
    st.image(image, caption=f'Example of {label}', use_column_width=True)
