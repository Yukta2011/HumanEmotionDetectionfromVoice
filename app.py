# app.py
import streamlit as st
import soundfile as sf
import librosa
import joblib
import numpy as np
import tempfile

st.title("🎙️ Human Emotion Detection from Voice")

# Load the trained model
model = joblib.load("emotion_model.pkl")  # Ensure this file exists in the same directory

# Define feature extraction function
def extract_features(file_path):
    X, sample_rate = librosa.load(file_path, res_type='kaiser_fast')
    mfcc = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
    chroma = np.mean(librosa.feature.chroma_stft(y=X, sr=sample_rate).T, axis=0)
    mel = np.mean(librosa.feature.melspectrogram(y=X, sr=sample_rate).T, axis=0)
    return np.hstack([mfcc, chroma, mel])

# Predict emotion
def predict_emotion(audio_file):
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(audio_file.read())
        features = extract_features(tmp.name)
        prediction = model.predict([features])
        return prediction[0]

# Upload audio
uploaded_file = st.file_uploader("Upload or record your voice (WAV only)", type=["wav"])

if uploaded_file is not None:
    st.audio(uploaded_file, format='audio/wav')
    emotion = predict_emotion(uploaded_file)
    st.success(f" Predicted Emotion: **{emotion.upper()}**")

