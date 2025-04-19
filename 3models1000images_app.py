import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import gdown

# Google Drive file IDs for the models
MODEL_URLS = {
    "VGG16": "https://drive.google.com/uc?id=1FOXTOm0yVvgBTKQ1C6z8AxhaDwTz3dRe",
    "InceptionV3": "https://drive.google.com/uc?id=1j1XVxqYXTNgy_z10Uh6nRXSspdFg4ndP",
    "ResNet101": "https://drive.google.com/uc?id=1E1ixPtqw0BaAt78jT-5-u_9TbI7OmxH7"
}

MODEL_FILES = {
    "VGG16": "modelvgg_1000img.h5",
    "InceptionV3": "modelInception_img.keras",
    "ResNet101": "CNNresnet101_img.keras"
}

CLASS_NAMES = ["BCC (Basal Cell Carcinoma)", "SCC (Squamous Cell Carcinoma)", "Melanoma"]

# Function to download model if not present
def download_model(model_name):
    model_path = MODEL_FILES[model_name]
    if not os.path.exists(model_path):
        with st.spinner(f"Downloading {model_name} model..."):
            gdown.download(MODEL_URLS[model_name], model_path, quiet=False)
    return model_path

# Load selected model
@st.cache_resource
def load_model(model_name):
    model_path = download_model(model_name)
    return tf.keras.models.load_model(model_path)

# Preprocess image
def preprocess_image(image):
    image = image.resize((224, 224))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

# Streamlit UI
st.set_page_config(page_title="Skin Cancer Classification", layout="centered")
st.title("🧬 Skin Cancer Classification")
st.markdown("Upload an image and choose a model to classify the type of skin cancer.")

# Select model
model_choice = st.selectbox("Select Model", list(MODEL_FILES.keys()))
model = load_model(model_choice)

# Upload image
uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    if st.button("Predict"):
        st.write("⏳ Predicting...")
        processed_image = preprocess_image(image)
        predictions = model.predict(processed_image)
        predicted_class = np.argmax(predictions)
        confidence = np.max(predictions) * 100

        st.success(f"### Prediction: {CLASS_NAMES[predicted_class]}")
        
        # Confidence bar
        color = "#DFF2BF" if confidence >= 65 else "#FFBABA"
        st.markdown(f'''
            <div style="background-color:{color};padding:10px;border-radius:5px;">
                <strong>Confidence:</strong> {confidence:.2f}%
            </div>
        ''', unsafe_allow_html=True)
