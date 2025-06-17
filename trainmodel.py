import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# Load features and labels
with open("features_labels.pkl", "rb") as f:
    features, labels = pickle.load(f)

print(f"Loaded {len(features)} samples.")

# Confirm all are same length
lengths = [len(f) for f in features]
if len(set(lengths)) != 1:
    raise ValueError("Inconsistent feature lengths detected. Check extractdata.py!")

# Split
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Train
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

# Save model
joblib.dump(model, "emotion_model.pkl")
print("Model saved as emotion_model.pkl")


