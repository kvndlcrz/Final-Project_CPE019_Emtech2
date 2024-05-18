import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# Function to load the model
@st.cache(allow_output_mutation=True)
def load_model():
    try:
        st.write("Attempting to load the model...")
        model = tf.keras.models.load_model('best.h5')
        st.write("Model loaded successfully!")
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Attempt to load the model
model = load_model()

# If model loading fails, stop execution
if model is None:
    st.error("Failed to load the model. Please check the logs for more details.")
    st.stop()

# Streamlit UI
st.write("""# Weather Classification System""")

# File uploader
file = st.file_uploader("Choose a weather image from your computer", type=["jpg", "png"])

# Function to preprocess and predict
def import_and_predict(image_data, model):
    size = (64, 64)
    image = Image.open(image_data)
    image = image.resize(size)
    img_array = np.array(image)
    img_array = img_array / 255.0  # Normalize the image
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)
    return prediction

# Main logic
if file is None:
    st.text("Please upload an image file")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    prediction = import_and_predict(file, model)
    
    # Weather class names
    class_names = ['Cloud', 'Rain', 'Shine', 'Sunrise']  
    
    # Output the result
    result = class_names[np.argmax(prediction)]
    st.success("Predicted Weather: {}".format(result))
