import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import gdown  # Install via: pip install gdown
import os

MODEL_URL = "https://drive.google.com/uc?id=1-0W38GF3yidHN719ORBgFWObBZwrNJwo"  # Replace with actual file ID
MODEL_PATH = "modelvgg_img.h5"

# Download the model if not already present
if not os.path.exists(MODEL_PATH):
    with st.spinner("Downloading model..."):
        gdown.download(MODEL_URL, MODEL_PATH, quiet=False)

# Load the model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()

CLASS_NAMES = ["BCC (Basal Cell Carcinoma)", "SCC (Squamous Cell Carcinoma)", "Melanoma"]

# Function to preprocess the image
def preprocess_image(image):
    image = image.resize((224, 224))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

st.title("Skin Cancer Classification App")
st.write("Upload an image to classify it as BCC, SCC, or Melanoma.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)
    
    st.write("Classifying...")

    processed_image = preprocess_image(image)
    predictions = model.predict(processed_image)
    predicted_class = np.argmax(predictions)
    confidence = np.max(predictions)

    st.write(f"### Prediction: {CLASS_NAMES[predicted_class]}")
    st.write(f"**Confidence:** {confidence:.2f}")
