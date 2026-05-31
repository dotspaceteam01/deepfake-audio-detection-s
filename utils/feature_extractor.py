import librosa
import numpy as np


def extract_audio_features(audio_path):

    y, sr = librosa.load(
        audio_path,
        sr=None
    )

    duration = len(y) / sr

    rms = float(
        np.mean(
            librosa.feature.rms(y=y)
        )
    )

    zcr = float(
        np.mean(
            librosa.feature.zero_crossing_rate(y)
        )
    )

    spectral_centroid = float(
        np.mean(
            librosa.feature.spectral_centroid(
                y=y,
                sr=sr
            )
        )
    )

    mfcc = librosa.feature.mfcc(
        y=y,
        sr=sr,
        n_mfcc=13
    )

    mfcc_mean = np.mean(
        mfcc.T,
        axis=0
    )

    return {
        "sample_rate": sr,
        "duration": round(duration, 2),
        "rms_energy": round(rms, 6),
        "zero_crossing_rate": round(zcr, 6),
        "spectral_centroid": round(
            spectral_centroid,
            2
        ),
        "mfcc_features": mfcc_mean.tolist()
    }