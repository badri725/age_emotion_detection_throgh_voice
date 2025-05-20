# Age and Emotion Detection from Male Voice

This project implements a machine learning pipeline to analyze voice recordings. It performs the following tasks:

1.  **Gender Detection:** Identifies the gender of the speaker.
2.  **Input Filtering:** Accepts only **male** voices for further processing. Female voices are rejected.
3.  **Age Group Estimation:** Predicts the age group (e.g., Youth, Adult, Senior) for accepted male voices.
4.  **Conditional Emotion Detection:** If the predicted age group for a male voice is "Senior" (defined as > 60 years old), the system also predicts the speaker's emotion (e.g., Happy, Sad, Angry, Neutral).

The project includes scripts for training the necessary models and a graphical user interface (GUI) built with Tkinter for easy interaction.

## Features

*   Loads audio files (`.wav`, `.mp3`).
*   Extracts MFCC features using `librosa`.
*   Trains separate models (using `scikit-learn`) for Gender, Age Group, and Emotion classification.
*   Provides evaluation metrics (Accuracy, Precision, Recall, Confusion Matrix) during training.
*   Requires models to achieve at least 70% accuracy on the test set (checked during training).
*   Saves trained models and scalers using `joblib`.
*   A user-friendly GUI (`tkinter`) to:
    *   Upload audio files.
    *   Display processing status and results based on the defined logic.
    *   Reject non-male voices explicitly.


