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

    delta = librosa.feature.delta(
        mfcc
    )

    delta2 = librosa.feature.delta(
        mfcc,
        order=2
    )

    mfcc_mean = np.mean(
        mfcc,
        axis=1
    )

    delta_mean = np.mean(
        delta,
        axis=1
    )

    delta2_mean = np.mean(
        delta2,
        axis=1
    )

    combined_features = np.concatenate([
        mfcc_mean,
        delta_mean,
        delta2_mean,
        [rms],
        [zcr],
        [spectral_centroid]
    ])

    return {
        "sample_rate": sr,
        "duration": round(duration, 2),
        "rms_energy": round(rms, 6),
        "zero_crossing_rate": round(zcr, 6),
        "spectral_centroid": round(
            spectral_centroid,
            2
        ),
        "mfcc_features": combined_features.tolist()
    }