import tkinter as tk
from tkinter import filedialog
from tkinter import ttk
import joblib
import librosa
import numpy as np
import os
import warnings

warnings.filterwarnings('ignore', category=FutureWarning)

MAX_PAD_LEN = 174
N_MFCC = 40
SAMPLE_RATE = 22050

MODEL_DIR = "saved_models"
GENDER_MODEL_PATH = os.path.join(MODEL_DIR, 'gender_model.joblib')
GENDER_SCALER_PATH = os.path.join(MODEL_DIR, 'gender_scaler.joblib')
AGE_MODEL_PATH = os.path.join(MODEL_DIR, 'age_model.joblib')
AGE_SCALER_PATH = os.path.join(MODEL_DIR, 'age_scaler.joblib')
EMOTION_MODEL_PATH = os.path.join(MODEL_DIR, 'emotion_model.joblib')
EMOTION_SCALER_PATH = os.path.join(MODEL_DIR, 'emotion_scaler.joblib')

gender_map_gui = {0: 'Male', 1: 'Female'}

age_map_gui = {0: 'Youth (0-18)', 1: 'Adult (19-60)', 2: 'Senior (61+)'}
SENIOR_AGE_LABEL = 'Senior (61+)'

emotion_map_gui = {0: 'Neutral', 1: 'Calm', 2: 'Happy', 3: 'Sad', 4: 'Angry', 5: 'Fearful', 6: 'Disgust', 7: 'Surprised'}
try:
    print("Loading models and scalers...")
    gender_model = joblib.load(GENDER_MODEL_PATH)
    gender_scaler = joblib.load(GENDER_SCALER_PATH)
    age_model = joblib.load(AGE_MODEL_PATH)
    age_scaler = joblib.load(AGE_SCALER_PATH)
    emotion_model = joblib.load(EMOTION_MODEL_PATH)
    emotion_scaler = joblib.load(EMOTION_SCALER_PATH)
    print("Models and scalers loaded successfully.")
except FileNotFoundError as e:
    print(f"Error loading models/scalers: {e}")
    print("Please ensure 'train_models.py' was run successfully and model files exist in the 'saved_models' directory.")
    exit()
except Exception as e:
    print(f"An unexpected error occurred during loading: {e}")
    exit()
def extract_features(file_path, n_mfcc=N_MFCC, max_pad_len=MAX_PAD_LEN, sr=SAMPLE_RATE):
    try:
        audio, sample_rate = librosa.load(file_path, res_type='kaiser_fast', sr=sr, duration=10)
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=n_mfcc)

        pad_width = max_pad_len - mfccs.shape[1]
        if pad_width > 0:
            mfccs_padded = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')
        else:
            mfccs_padded = mfccs[:, :max_pad_len]

        features_flat = mfccs_padded.flatten()
        return features_flat

    except Exception as e:
        print(f"Error processing {file_path} for feature extraction: {e}")
        return None
def predict_audio(file_path):
    status_label.config(text="Processing audio...")
    root.update_idletasks()

    features = extract_features(file_path)
    if features is None:
        status_label.config(text="Error: Could not extract features from audio.")
        return

    try:
        features_2d = features.reshape(1, -1)

        scaled_features_gender = gender_scaler.transform(features_2d)
        gender_pred_encoded = gender_model.predict(scaled_features_gender)[0]
        gender_pred_label = gender_map_gui.get(gender_pred_encoded, f"Unknown Gender ({gender_pred_encoded})")

        print(f"Predicted Gender Code: {gender_pred_encoded}, Label: {gender_pred_label}")

        if gender_pred_label == 'Female':
            status_label.config(text="Upload male voice.")
            return

        scaled_features_age = age_scaler.transform(features_2d)
        age_pred_encoded = age_model.predict(scaled_features_age)[0]
        age_pred_label = age_map_gui.get(age_pred_encoded, f"Unknown Age Group ({age_pred_encoded})")

        print(f"Predicted Age Code: {age_pred_encoded}, Label: {age_pred_label}")

        result_text = f"Gender: {gender_pred_label}\n"
        result_text += f"Predicted Age Group: {age_pred_label}\n"

        is_senior = (age_pred_label == SENIOR_AGE_LABEL)
        print(f"Is Senior: {is_senior}")

        if is_senior:
            scaled_features_emotion = emotion_scaler.transform(features_2d)
            emotion_pred_encoded = emotion_model.predict(scaled_features_emotion)[0]
            emotion_pred_label = emotion_map_gui.get(emotion_pred_encoded, f"Unknown Emotion ({emotion_pred_encoded})")

            print(f"Predicted Emotion Code: {emotion_pred_encoded}, Label: {emotion_pred_label}")

            result_text += "Status: Senior Citizen\n"
            result_text += f"Predicted Emotion: {emotion_pred_label}"
        else:
             result_text += "Status: Not Senior Citizen"

        status_label.config(text=result_text)

    except Exception as e:
        status_label.config(text=f"An error occurred during prediction: {e}")
        print(f"Prediction error details: {e}")
def upload_action():
    try:
        file_path = filedialog.askopenfilename(
            title="Select Audio File",
            filetypes=(("Audio Files", "*.wav *.mp3"), ("All files", "*.*"))
        )
        if file_path:
            print(f"File selected: {file_path}")
            if not os.path.exists(file_path):
                 status_label.config(text="Error: Selected file not found.")
                 return
            predict_audio(file_path)
        else:
            print("File selection cancelled.")
    except Exception as e:
        status_label.config(text=f"Error during file selection: {e}")
        print(f"File dialog error: {e}")
if __name__ == "__main__":
    root = tk.Tk()
    root.title("Age & Emotion Detector (Male Voices)")
    root.geometry("450x300")

    style = ttk.Style()
    try:
        style.theme_use('clam')
    except tk.TclError:
        print("ttk 'clam' theme not available, using default.")

    main_frame = ttk.Frame(root, padding="20")
    main_frame.pack(expand=True, fill=tk.BOTH)

    title_label = ttk.Label(main_frame, text="Voice Analysis", font=("Helvetica", 16, "bold"))
    title_label.pack(pady=(0, 15))

    upload_button = ttk.Button(main_frame, text="Upload Audio File (.wav, .mp3)", command=upload_action, width=30)
    upload_button.pack(pady=10)

    status_label = ttk.Label(
        main_frame,
        text="Upload an audio file to begin analysis.",
        justify=tk.LEFT,
        wraplength=400,
        padding=(10, 10),
        relief=tk.SUNKEN,
        borderwidth=1,
        anchor='nw'
        )
    status_label.pack(pady=10, fill=tk.BOTH, expand=True)


    print("Starting GUI...")
    root.mainloop()
    print("GUI closed.")
