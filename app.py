import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import os

# Hide TensorFlow warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

st.set_page_config(page_title="Medical Image Classification", layout="centered")

st.title("Explainable AI for Medical Image Classification")
st.write("Upload a Chest X-ray image to detect Pneumonia")

# Cache model so it loads only once
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("cnn_medical_model.h5")
    return model

model = load_model()

# Image preprocessing
def preprocess_image(image):
    img = np.array(image)
    img = cv2.resize(img, (224, 224))
    
    if img.shape[-1] == 4:  # remove alpha channel if exists
        img = img[:, :, :3]

    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    
    return img


uploaded_file = st.file_uploader(
    "Upload Chest X-ray Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    try:
        processed_image = preprocess_image(image)

        prediction = model.predict(processed_image)

        prob = float(prediction[0][0])

        if prob > 0.5:
            st.error(f"Pneumonia Detected (Confidence: {prob:.2f})")
        else:
            st.success(f"Normal (Confidence: {1-prob:.2f})")

    except Exception as e:
        st.error("Error during prediction")
        st.text(str(e))


st.markdown("---")
st.write("Built with Streamlit and TensorFlow")