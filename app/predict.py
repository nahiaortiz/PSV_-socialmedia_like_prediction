import numpy as np
import joblib
import re
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet import preprocess_input
from datetime import datetime
import os

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")

IMG_SIZE   = 224
LABEL_COLS = ["CROWD", "CELEBRATION", "GRAPHIC", "PLAYER", "ACTION"]
THRESHOLD  = 0.5
 
# ── Load models once at startup ───────────────────────────────────────────────
efficientnet = load_model(os.path.join(MODELS_DIR, "efficient-model.keras"))
random_forest = joblib.load(os.path.join(MODELS_DIR, "random_forest_likes.pkl"))


def preprocess_image(image_bytes: bytes) -> np.ndarray:
    image = tf.image.decode_image(image_bytes, channels=3, expand_animations=False)
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
    image = preprocess_input(image)
    return np.expand_dims(image.numpy(), axis=0)  # shape: (1, 224, 224, 3)


def extract_caption_features(text: str) -> dict:
    if not text:
        text = ""
    return {
        "caption_length":     len(text),
        "word_count":         len(text.split()),
        "hashtag_count":      len(re.findall(r"#\w+", text)),
        "mention_count":      len(re.findall(r"@\w+", text)),
        "has_emoji":          int(bool(re.search(r"[^\w\s,.\-!?']", text))),
        "exclamation_count":  text.count("!"),
    }


def extract_time_features(timestamp: datetime) -> dict:
    return {
        "hour":        timestamp.hour,
        "day_of_week": timestamp.weekday(),
        "month":       timestamp.month,
        "is_weekend":  int(timestamp.weekday() in [5, 6]),
    }


def predict(image_bytes: bytes, caption: str, timestamp: datetime) -> dict:
    # 1. EfficientNet → image labels
    image_array  = preprocess_image(image_bytes)
    label_probs  = efficientnet.predict(image_array)[0]        # shape: (5,)
    label_binary = (label_probs >= THRESHOLD).astype(int)
    labels_dict  = dict(zip(LABEL_COLS, label_binary.tolist()))

    # 2. Build feature vector for Random Forest
    caption_feats = extract_caption_features(caption)
    time_feats    = extract_time_features(timestamp)

    feature_vector = (
        label_binary.tolist() +
        list(caption_feats.values()) +
        list(time_feats.values())
    )

    # 3. Random Forest → likes prediction
    predicted_likes = random_forest.predict([feature_vector])[0]

    return {
        "labels":          labels_dict,
        "label_probs":     dict(zip(LABEL_COLS, label_probs.tolist())),
        "predicted_likes": round(predicted_likes),
    }