import os
import librosa
import numpy as np
import pickle

# Feature extraction function
def extract_features(file_path):
    try:
        x, sample_rate = librosa.load(file_path, res_type='kaiser_fast')

        mfcc = librosa.feature.mfcc(y=x, sr=sample_rate, n_mfcc=40)
        mfcc = np.mean(mfcc.T, axis=0)

        chroma = librosa.feature.chroma_stft(y=x, sr=sample_rate)
        chroma = np.mean(chroma.T, axis=0)

        mel = librosa.feature.melspectrogram(y=x, sr=sample_rate)
        mel = np.mean(mel.T, axis=0)

        # Standardize mel length
        mel = mel[:128] if len(mel) > 128 else np.pad(mel, (0, 128 - len(mel)))

        return np.hstack([mfcc, chroma, mel])  # Final shape: (180,)
    
    except Exception as e:
        print(f"[ERROR] Failed to extract from {file_path}: {e}")
        return np.zeros(180)

# Emotion labels from filename
emotions = {
    '01': 'neutral',
    '02': 'calm',
    '03': 'happy',
    '04': 'sad',
    '05': 'angry',
    '06': 'fearful',
    '07': 'disgust',
    '08': 'surprised'
}

def get_label(filename):
    try:
        part = filename.split("-")[2]
        return emotions.get(part, "unknown")
    except IndexError:
        return "unknown"

# Path to dataset
dataset_path = "C:\\Users\\asus\\Desktop\\HumanEmotionDetectionfromVoice.py\\dataset"

features, labels = [], []

# Walk through all subfolders and files
for root, dirs, files in os.walk(dataset_path):
    for file in files:
        if file.endswith(".wav"):
            file_path = os.path.join(root, file)
            label = get_label(file)

            if label == "unknown":
                print(f"[SKIP] Unknown label in file: {file}")
                continue

            feature = extract_features(file_path)
            features.append(feature)
            labels.append(label)

            print(f"[INFO] {file} => {label} | Feature shape: {feature.shape}")

# Verify features
print(f"\n[INFO] All features 180-length:", all(len(f) == 180 for f in features))
print(f"[INFO] Total valid samples: {len(features)}")

# Save features and labels
with open("features_labels.pkl", "wb") as f:
    pickle.dump((features, labels), f)

print(f"\n[INFO] Feature extraction and saving complete.")

