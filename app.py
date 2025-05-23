import streamlit as st
import torch
from PIL import Image
import numpy as np
import cv2
from ultralytics import YOLO

# Set page config
st.set_page_config(page_title="Sedan & SUV Detector", layout="centered")

# Load the model
@st.cache_resource
def load_model():
    model = YOLO("vehicle_classifier_app/best.pt")
    return model

model = load_model()

# Class names for display
class_names = ['SUV', 'Sedan']

# App Header
st.title("🚘 Sedan & SUV Detector")
st.markdown("Upload an image to detect and classify **Sedans** and **SUVs** using a YOLO-based model.")

# File uploader
uploaded_file = st.file_uploader("📸 Upload a Vehicle Image", type=["jpg", "jpeg", "png"])

# Process the uploaded file
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="🖼️ Uploaded Image", use_container_width=True)

    with st.spinner("Detecting vehicles..."):
        results = model(image)

    # Plot detection results
    st.subheader("📍 Detection Results:")
    res_plotted = results[0].plot()  # YOLO built-in plot with boxes/labels
    st.image(res_plotted, caption="🧭 Detected Vehicles", use_container_width=True)

    # Identify top-1 confident detection
    st.subheader("🔍 Most Confident Detection:")

    top_result = None
    top_conf = 0

    for r in results:
        for box in r.boxes:
            conf = float(box.conf)
            if conf > top_conf:
                top_conf = conf
                top_result = box

    if top_result:
        cls_id = int(top_result.cls)
        label = class_names[cls_id] if cls_id < len(class_names) else f"Class {cls_id}"
        st.success(f"**{label}** detected with **{top_conf:.2f}** confidence.")
    else:
        st.warning("⚠️ No confident detection found. Try another image.")

else:
    st.info("👈 Please upload an image to get started.")
