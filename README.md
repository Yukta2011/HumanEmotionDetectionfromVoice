# HumanEmotionDetectionfromVoice
 Human Emotion Detection from Voice

This project uses machine learning to detect human emotions from voice recordings. By extracting features such as MFCC, Chroma, and Mel spectrograms from audio files, the model classifies emotions like **happy**, **sad**, **angry**, and more. Built with **Python**, **Librosa**, **Scikit-learn**, and **Streamlit** for real-time predictions via a web interface.


 Features

-  Load and extract features from voice recordings
-  Train machine learning classifiers (SVM, Random Forest)
-  Display prediction results in a user-friendly web app
-  Real-time voice recording and prediction
-  Support for RAVDESS dataset


 Emotions Detected

- Neutral
- Calm
- Happy
- Sad
- Angry
- Fearful
- Disgust
- Surprised



Tech Stack

| Layer        | Tools Used                                                                           |
|--------------|--------------------------------------------------------------------------------------|
| Language     | Python                                                                               |
| Audio Processing | Librosa, Soundfile                                                               |
| ML Libraries | Scikit-learn                                                                         |
| GUI          | Streamlit                                                                            |
| Data         | [RAVDESS Dataset](https://www.kaggle.com/uwrfkaggler/ravdess-emotional-speech-audio) |
| Others       | NumPy, Pandas, Pickle                                                                |


 Project Structure
 HumanEmotionDetectionfromVoice/
│
├── dataset/ # RAVDESS dataset
├── extractdata.py # Feature extraction script
├── train_model.py # Train and save ML model
├── model.pkl # Trained model
├── app.py # Streamlit web application
├── requirements.txt # Required Python packages
└── README.md # Project documentation

---
 How to Run

 1. Clone the Repository

```bash
git clone https://github.com/Yukta2011/HumanEmotionDetectionfromVoice.git
cd HumanEmotionDetectionfromVoice

2. Install Dependencies
pip install -r requirements.txt
or
pip install --user -r requirements.txt

3. Extract Features
dataset/
├── Actor_01/
│   ├── 03-01-01-01-01-01-01.wav
│   └── ...
├── Actor_02/
│   └── ...

run
python extractdata.py

4. Train the Model
python train_model.py

5. Run the Web App
streamlit run app.py

 Model Training Details
>Features: MFCC, Chroma, Mel Spectrogram
>Model used: Support Vector Machine / Random Forest
>Data Split: Train-Test Split (e.g., 80-20)
>Evaluation Metrics: Accuracy, Confusion Matrix, Classification Report

Sample UI
Record voice via browser 🎙
See predicted emotion 
Visualize session emotion trends

Dependencies
numpy
pandas
librosa
scikit-learn
soundfile
streamlit

Yukta Walanju
Aspiring AI/ML Developer & Data Scientist
Mumbai, India
 yuktawalanju16@gmail.com








