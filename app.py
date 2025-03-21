import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image

# Load the trained model
@st.cache_resource
def load_trained_model():
    model_path = "modelvgg_img.h5"  # Change to .keras if needed
    model = load_model(model_path)
    return model

model = load_trained_model()

# Define class labels
CLASS_LABELS = ["BCC (Basal Cell Carcinoma)", "SCC (Squamous Cell Carcinoma)", "Melanoma"]

st.title("Skin Cancer Classification App")

st.write("Upload an image to classify it as **BCC, SCC, or Melanoma**.")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display uploaded image
    image_disp = Image.open(uploaded_file)
    st.image(image_disp, caption="Uploaded Image", use_container_width=True)

    # Preprocess the image
    def preprocess_image(img):
        img = img.resize((224, 224))  # Resize to match model input
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)  # Expand dimensions for model
        img_array = img_array / 255.0  # Normalize pixel values
        return img_array

    # Predict when button is pressed
    if st.button("Classify"):
        preprocessed_img = preprocess_image(image_disp)
        prediction = model.predict(preprocessed_img)
        predicted_class = np.argmax(prediction, axis=1)[0]
        confidence = np.max(prediction) * 100

        st.success(f"**Prediction:** {CLASS_LABELS[predicted_class]}")
        st.info(f"**Confidence:** {confidence:.2f}%")
