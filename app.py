import streamlit as st
import tensorflow as tf

@st.cache(allow_output_mutation=True)
def load_model():
    try:
        model = tf.keras.models.load_model('model.h5')
        return model
    except Exception as e:
        st.error(f"Error loading the model: {e}")
        return None

model = load_model()

if model is not None:
    st.write("""
    # Multi-class Weather Classification System
    """)

    file = st.file_uploader("Choose a weather photo from your computer", type=["jpg", "png"])

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
