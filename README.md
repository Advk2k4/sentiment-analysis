# 🧠 Real-Time Multimodal Sentiment Analyzer

A real-time sentiment analysis application that combines facial expression recognition and text sentiment analysis into a single, easy-to-use web interface. Built using deep learning (CNN) for facial emotion detection, NLP for sentiment classification, and Streamlit for the UI.

---

## 🔍 Overview

This app captures live video from your webcam or allows image upload to detect human emotions like **Happy**, **Sad**, **Angry**, **Neutral**, etc., using a trained deep learning model. It also supports text-based sentiment analysis using the VADER NLP model.

---

## 🎯 Features

- 🖼️ **Facial Emotion Detection**
  - Upload an image or use your webcam.
  - Detects emotions in real-time using OpenCV + CNN.
  - Displays predicted emotion and confidence score.

- 💬 **Text Sentiment Analysis**
  - Enter any sentence or phrase.
  - Classifies as **Positive**, **Negative**, or **Neutral**.

- ⚡ **Streamlit UI**
  - Clean and interactive user interface.
  - Runs entirely in the browser with no external dependencies.

---

## 🧠 Emotions Detected

- Angry
- Disgust
- Fear
- Happy
- Sad
- Surprise
- Neutral

---

## 🗂️ Project Structure

```
SentimentAnalysis/
├── backend/
│   └── predict.py                # Loads model and performs emotion detection
├── frontend/
│   └── app.py                    # Streamlit-based web app
├── models/
│   └── emotion_model.h5          # Trained CNN model for facial emotion
├── nlp/
│   └── sentiment.py              # VADER-based text sentiment analyzer
├── requirements.txt              # Required Python packages
└── README.md                     # Project overview and documentation
```

---

## 🚀 Getting Started

### 1. Clone the Repo

```bash
git clone https://github.com/yourusername/RealTime_Multimodal_Sentiment_Analyzer.git
cd RealTime_Multimodal_Sentiment_Analyzer
```

### 2. Create a Virtual Environment

```bash
python -m venv sentimentenv
source sentimentenv/bin/activate  # On Windows use: sentimentenv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

> **Note:** TensorFlow 2.13+ is required. If you're on Apple Silicon (M1/M2), ensure you install the correct version via `tensorflow-macos`.

### 4. Run the App

```bash
cd frontend
streamlit run app.py
```

---

## 🛠️ Tech Stack

- Python
- Streamlit
- OpenCV
- TensorFlow / Keras
- NumPy, Pandas
- VADER Sentiment (NLTK)

---

## 📊 Dataset Used

- **FER-2013**: Facial Expression Recognition dataset used to train the CNN.
- **VADER**: Rule-based sentiment analysis for text (from NLTK).

---

## 📌 TODO / Improvements

- Add multi-face support.
- Integrate audio-based sentiment recognition.
- Save analysis history/logs for export.

---

## 🤝 Contribution Guide

1. Fork the repository.
2. Create a new branch: `git checkout -b feature-name`
3. Make your changes and commit: `git commit -m "Add new feature"`
4. Push to the branch: `git push origin feature-name`
5. Open a pull request!

---

## 📄 License

This project is licensed under the MIT License.

---

## 🙋‍♂️ Author

**Aadvik Mishra**  
Computer Engineering @ UMass Amherst  
[LinkedIn](https://www.linkedin.com/in/aadvik-mishra-2a8981252) | [GitHub](https://github.com/Advk2k4)
