import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import gdown  # Install via: pip install gdown
import os

# Model URL & Path
MODEL_URL = "https://drive.google.com/uc?id=1-0W38GF3yidHN719ORBgFWObBZwrNJwo"
MODEL_PATH = "modelvgg_img.h5"

# Download the model if not present
if not os.path.exists(MODEL_PATH):
    with st.spinner("Downloading model..."):
        gdown.download(MODEL_URL, MODEL_PATH, quiet=False)

# Load the model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()

CLASS_NAMES = ["BCC (Basal Cell Carcinoma)", "SCC (Squamous Cell Carcinoma)", "Melanoma"]

# Function to preprocess image
def preprocess_image(image):
    image = image.resize((224, 224))  # Resize to match model input
    image = np.array(image) / 255.0   # Normalize pixel values
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# Streamlit UI
st.title("Skin Cancer Classification")
st.write("Upload an image and click to predict.")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Add a classify button
    if st.button("Classify"):
        st.write("Classifying... 🔄")
        
        # Preprocess & Predict
        processed_image = preprocess_image(image)
        predictions = model.predict(processed_image)
        predicted_class = np.argmax(predictions)
        confidence = np.max(predictions) * 100  # Convert to percentage

        # Display prediction result
        st.success(f"### Prediction: {CLASS_NAMES[predicted_class]}")

        # Format confidence percentage with color
        if confidence >= 65:
            st.markdown(f'<div style="background-color:#DFF2BF;padding:10px;border-radius:5px;">'
                        f'<strong>Confidence:</strong> {confidence:.2f}%</div>', 
                        unsafe_allow_html=True)
        else:
            st.markdown(f'<div style="background-color:#FFBABA;padding:10px;border-radius:5px;">'
                        f'<strong>Confidence:</strong> {confidence:.2f}%</div>', 
                        unsafe_allow_html=True)
