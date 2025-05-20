import os
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# Dummy example data (replace with real MFCC data and labels)
import numpy as np
X = np.random.rand(100, 40 * 174)
y_gender = np.random.randint(0, 2, 100)
y_age = np.random.randint(0, 3, 100)
y_emotion = np.random.randint(0, 8, 100)

os.makedirs("saved_models", exist_ok=True)

# Gender model
gender_scaler = StandardScaler()
X_gender_scaled = gender_scaler.fit_transform(X)
gender_model = RandomForestClassifier()
gender_model.fit(X_gender_scaled, y_gender)
joblib.dump(gender_model, 'saved_models/gender_model.joblib')
joblib.dump(gender_scaler, 'saved_models/gender_scaler.joblib')

# Age model
age_scaler = StandardScaler()
X_age_scaled = age_scaler.fit_transform(X)
age_model = RandomForestClassifier()
age_model.fit(X_age_scaled, y_age)
joblib.dump(age_model, 'saved_models/age_model.joblib')
joblib.dump(age_scaler, 'saved_models/age_scaler.joblib')

# Emotion model
emotion_scaler = StandardScaler()
X_emotion_scaled = emotion_scaler.fit_transform(X)
emotion_model = RandomForestClassifier()
emotion_model.fit(X_emotion_scaled, y_emotion)
joblib.dump(emotion_model, 'saved_models/emotion_model.joblib')
joblib.dump(emotion_scaler, 'saved_models/emotion_scaler.joblib')

print("✅ All models trained and saved to 'saved_models/'")
