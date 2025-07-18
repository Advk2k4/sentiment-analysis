import sys
import os
import cv2
import streamlit as st
import numpy as np
from PIL import Image

# Add backend path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from backend.predict import predict_emotion
from nlp.sentiment import analyze_sentiment

st.set_page_config(page_title="Real-Time Multimodal Sentiment Analyzer", layout="centered")
st.title("🧠 Real-Time Sentiment Analyzer")
st.write("This app uses your webcam or image upload to analyze facial emotions and text sentiment.")

use_camera = st.checkbox("Enable Webcam")

uploaded_image = st.file_uploader("📤 Upload an Image for Emotion Detection", type=["jpg", "jpeg", "png"])

use_text = st.checkbox("Analyze Text Sentiment")

# Handle uploaded image
if uploaded_image is not None:
    image = Image.open(uploaded_image)
    image_np = np.array(image.convert('RGB'))
    st.image(image_np, caption="Uploaded Image", use_container_width=True)

    results = predict_emotion(image_np)
    for label, conf in results:
        if label == "No face detected":
            st.warning("😕 No face detected. Try a front-facing, well-lit photo.")
        else:
            st.success(f"Detected Emotion: **{label}** (Confidence: {conf:.2f})")


# Handle webcam
if use_camera:
    run = st.checkbox("Start Camera")
    FRAME_WINDOW = st.image([])
    camera = cv2.VideoCapture(0)

    while run:
        _, frame = camera.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = predict_emotion(frame)
        for label, conf in results:
            st.write(f"Detected Emotion: **{label}** (Confidence: {conf:.2f})")
        FRAME_WINDOW.image(frame)
    else:
        st.write("Stopped.")

# Handle text sentiment
if use_text:
    text_input = st.text_input("Type something:")
    if text_input:
        sentiment, score = analyze_sentiment(text_input)
        st.write(f"Text Sentiment: **{sentiment}** (Score: {score:.2f})")
