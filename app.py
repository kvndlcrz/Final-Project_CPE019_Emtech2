import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

# Load the CNN model
model = load_model('cnn_model.h5')

# Function to preprocess the image
def preprocess_image(image):
    resized_image = image.resize((150, 150))  # Adjust the size according to your model's input size
    normalized_image = np.array(resized_image) / 255.0
    preprocessed_image = np.expand_dims(normalized_image, axis=0)
    return preprocessed_image

st.write("Cañesa, Anjela T. | Llave, Kenneth Gabriel | CPE019 | CPE32S5")
st.image("weather.jpg", width=400)
st.write("""
        # Model Deployment on the Cloud
        \n An Application of Convolutional Neural Network in Weather ['cloudy', 'rainy', 'sunshine', 'sunset'] 
        Prediction with an Accuracy Rate of 82%.
        """)
st.text("Using the Weather Dataset to predict from an uploaded image.")

# Upload image
uploaded_image = st.file_uploader("Choose an image that can be classified as rainy, sunset, sun shine, or cloudy: ", type=["jpg", "png", "jpeg"])

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
