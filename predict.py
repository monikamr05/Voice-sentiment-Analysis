"""
Prediction utilities for uploaded audio files.
"""

import os
import pickle
import numpy as np
import librosa
import librosa.display
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import (
    MODEL_PATH,
    NORMALIZER_PATH,
    LABEL_ENCODER_PATH,
    SENTIMENT_MAP,
    SENTIMENT_COLORS,
    SAMPLE_RATE,
    DURATION_SEC,
)
from audio_features import extract_from_file, normalize_features, load_audio_file


_model = None
_normalizer = None
_label_encoder = None


def _get_keras():
    try:
        from tensorflow import keras
        return keras
    except ImportError as e:
        raise ImportError(
            "TensorFlow is required for prediction. Use Python 3.9–3.12 and "
            "run: pip install tensorflow"
        ) from e


def load_model():
    """Load trained model and preprocessing artifacts (cached)."""
    global _model, _normalizer, _label_encoder
    keras = _get_keras()
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"Model not found at {MODEL_PATH}. Run: python train_model.py"
            )
        _model = keras.models.load_model(MODEL_PATH)
    if _normalizer is None:
        if not os.path.exists(NORMALIZER_PATH):
            raise FileNotFoundError("Normalizer not found. Train the model first.")
        data = np.load(NORMALIZER_PATH)
        _normalizer = (data["mean"], data["std"])
    if _label_encoder is None:
        with open(LABEL_ENCODER_PATH, "rb") as f:
            _label_encoder = pickle.load(f)
    return _model, _normalizer, _label_encoder


def predict_emotion(filepath):
    """
    Predict emotion and sentiment from audio file.
    Returns dict with emotion, sentiment, confidence, probabilities.
    """
    model, (mean, std), le = load_model()

    features = extract_from_file(filepath)
    X = np.expand_dims(features, axis=0).astype(np.float32)
    X_norm, _, _ = normalize_features(X, mean=mean, std=std)

    probs = model.predict(X_norm, verbose=0)[0]
    idx = int(np.argmax(probs))
    emotion = le.inverse_transform([idx])[0]
    confidence = float(probs[idx] * 100)
    sentiment = SENTIMENT_MAP.get(emotion, "Neutral")

    prob_dict = {
        le.inverse_transform([i])[0]: round(float(probs[i]) * 100, 2)
        for i in range(len(probs))
    }

    return {
        "emotion": emotion,
        "sentiment": sentiment,
        "confidence": round(confidence, 2),
        "probabilities": prob_dict,
        "sentiment_color": SENTIMENT_COLORS.get(sentiment, "#6c757d"),
    }


def generate_waveform_plot(filepath, output_path):
    """Save waveform image for display."""
    y, sr = load_audio_file(filepath)
    fig, ax = plt.subplots(figsize=(10, 3))
    librosa.display.waveshow(y, sr=sr, ax=ax, color="#0d6efd")
    ax.set_title("Waveform")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    plt.tight_layout()
    plt.savefig(output_path, dpi=100, bbox_inches="tight", facecolor="white")
    plt.close()


def generate_spectrogram_plot(filepath, output_path):
    """Save mel spectrogram image for display."""
    y, sr = load_audio_file(filepath)
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    S_db = librosa.power_to_db(S, ref=np.max)
    fig, ax = plt.subplots(figsize=(10, 4))
    img = librosa.display.specshow(S_db, sr=sr, x_axis="time", y_axis="mel", ax=ax)
    fig.colorbar(img, ax=ax, format="%+2.0f dB")
    ax.set_title("Mel Spectrogram")
    plt.tight_layout()
    plt.savefig(output_path, dpi=100, bbox_inches="tight", facecolor="white")
    plt.close()
