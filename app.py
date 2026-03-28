import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt

# Load model
model = tf.keras.models.load_model("cnn_medical_model.h5")

st.title("Explainable AI for Medical Image Classification")

uploaded_file = st.file_uploader("Upload Chest X-ray Image", type=["jpg","jpeg","png"])

if uploaded_file is not None:

    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes,1)

    st.image(img, caption="Uploaded Image", width="stretch")

    img_resized = cv2.resize(img,(224,224))
    img_norm = img_resized/255.0
    img_input = np.expand_dims(img_norm, axis=0)

    prediction = model.predict(img_input)[0][0]

    if prediction > 0.5:
        label = "PNEUMONIA"
        confidence = prediction
    else:
        label = "NORMAL"
        confidence = 1 - prediction

    st.subheader("Prediction: " + label)
    st.write("Confidence:", round(float(confidence),3))

    # Occlusion Sensitivity
    heatmap = np.zeros((224,224))
    patch_size = 20

    for y in range(0,224,patch_size):
        for x in range(0,224,patch_size):

            occluded = img_norm.copy()
            occluded[y:y+patch_size, x:x+patch_size] = 0

            occluded_input = np.expand_dims(occluded,axis=0)

            pred = model.predict(occluded_input)[0][0]

            heatmap[y:y+patch_size, x:x+patch_size] = pred

    heatmap = (heatmap - heatmap.min())/(heatmap.max()-heatmap.min())

    heatmap = np.uint8(255*heatmap)
    heatmap = cv2.applyColorMap(heatmap,cv2.COLORMAP_JET)

    superimposed = cv2.addWeighted(img_resized,0.6,heatmap,0.4,0)

    st.subheader("Explainable AI Heatmap")
    st.image(superimposed)