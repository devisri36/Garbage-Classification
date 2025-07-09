
import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from PIL import Image

# Set title
st.title("🗑️ Garbage Classification using MobileNetV2")
st.markdown("Upload an image of garbage to classify it into: **Cardboard, Glass, Metal, Paper, Plastic, Trash**.")

# Load model
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("garbageClassifierModel.keras")
    return model

model = load_model()
class_labels = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']
img_size = (124, 124)

# File uploader
uploaded_file = st.file_uploader("Choose a garbage image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display image
    img = Image.open(uploaded_file)
    st.image(img, caption='Uploaded Image', use_container_width=True)

    # Preprocess
    img = img.resize(img_size)
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    prediction = model.predict(img_array)
    predicted_class = class_labels[np.argmax(prediction)]
    confidence = np.max(prediction)

    # Display result
    #st.write("Raw Prediction:", prediction)

    st.success(f"🧾 Predicted Class: **{predicted_class}**")
    #st.info(f"Confidence: **{confidence*100:.2f}%**")
