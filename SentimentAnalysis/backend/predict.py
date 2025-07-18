import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load the model once
model = load_model('models/emotion_model.h5')

# Labels matching model output
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Load face detector once
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def predict_emotion(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    faces = face_cascade.detectMultiScale(gray, 1.1, 5)
    results = []

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y + h, x:x + w]
        roi_resized = cv2.resize(roi_gray, (64, 64))  # MATCH MODEL INPUT
        roi_normalized = roi_resized.astype("float32") / 255.0
        roi = np.expand_dims(roi_normalized, axis=(0, -1))  # shape: (1, 64, 64, 1)

        prediction = model.predict(roi, verbose=0)[0]
        max_index = np.argmax(prediction)
        predicted_label = emotion_labels[max_index]
        confidence = prediction[max_index]

        results.append((predicted_label, confidence))

    return results

