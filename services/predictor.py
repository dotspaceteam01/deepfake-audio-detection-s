import os
import joblib
import numpy as np

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

PROJECT_ROOT = os.path.dirname(
    BASE_DIR
)

MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "svm_model_2.pkl"
)

SCALER_PATH = os.path.join(
    PROJECT_ROOT,
    "scaler_2.pkl"
)

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)


def predict_audio(mfcc_features):

    scaled_features = scaler.transform(
        np.array(mfcc_features).reshape(1, -1)
    )

    prediction = model.predict(
        scaled_features
    )[0]

    probabilities = model.predict_proba(
        scaled_features
    )[0]

    genuine_probability = float(
        probabilities[0] * 100
    )

    deepfake_probability = float(
        probabilities[1] * 100
    )

    confidence = max(
        genuine_probability,
        deepfake_probability
    )

    return (
        prediction,
        confidence,
        genuine_probability,
        deepfake_probability
    )