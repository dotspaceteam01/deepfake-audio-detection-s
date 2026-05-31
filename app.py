from fastapi import FastAPI
from fastapi import UploadFile
from fastapi import File
from fastapi import HTTPException
from fastapi.middleware.cors import CORSMiddleware

import os
import uuid

from utils.feature_extractor import (
    extract_audio_features
)

from services.predictor import (
    predict_audio
)

app = FastAPI(
    title="Deepfake Audio Detection API",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def health():
    return {
        "status": "running"
    }


@app.post("/predict")
async def predict(
    audio_file: UploadFile = File(...)
):

    if not audio_file.filename.endswith(".wav"):
        raise HTTPException(
            status_code=400,
            detail="Only WAV files are allowed"
        )

    os.makedirs(
        "uploads",
        exist_ok=True
    )

    unique_filename = (
        f"{uuid.uuid4()}.wav"
    )

    temp_path = os.path.join(
        "uploads",
        unique_filename
    )

    try:

        with open(
            temp_path,
            "wb"
        ) as buffer:

            buffer.write(
                await audio_file.read()
            )

        features = extract_audio_features(
            temp_path
        )

        (
    prediction,
    confidence,
    genuine_probability,
    deepfake_probability
) = predict_audio(
    features["mfcc_features"]
)

        result = (
            "genuine"
            if prediction == 0
            else "deepfake"
        )

        mfcc_dict = {}

        for i, value in enumerate(
            features["mfcc_features"]
        ):
            mfcc_dict[
                f"mfcc_{i+1}"
            ] = round(
                float(value),
                4
            )

        return {
    "prediction": result,

    "confidence": round(
        confidence,
        2
    ),

    "probabilities": {
        "genuine": round(
            genuine_probability,
            2
        ),
        "deepfake": round(
            deepfake_probability,
            2
        )
    },

    "audio_info": {
                "duration":
                    features["duration"],

                "sample_rate":
                    features["sample_rate"]
            },

            "features": {

                "rms_energy":
                    features["rms_energy"],

                "zero_crossing_rate":
                    features[
                        "zero_crossing_rate"
                    ],

                "spectral_centroid":
                    features[
                        "spectral_centroid"
                    ],

                **mfcc_dict
            }
        }

    finally:

        if os.path.exists(
            temp_path
        ):
            os.remove(
                temp_path
            )