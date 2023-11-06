import json
import pickle
import numpy as np
from typing import Any

import librosa
import soundfile as sf


def load_wav(file_path, offset=0.0, duration=None, sr=None, mono=True):
    audio, sampling_rate = librosa.core.load(
        file_path,
        offset=offset,
        duration=duration,
        sr=sr,
        mono=mono,
    )
    return audio, sampling_rate


def save_wav(file_path, audio, sampling_rate, subtype="PCM_24"):
    sf.write(
        file=file_path,
        data=audio,
        samplerate=sampling_rate,
        subtype=subtype,
    )
    print(f"wav chunk saved at {file_path}")


def pickle_dump(path: str, data: Any) -> None:
    with open(path, "wb") as f:
        pickle.dump(data, f)


def pickle_load(path: str) -> Any:
    with open(path, "rb") as f:
        return pickle.load(f)


def npy_save(path: str, data: np.ndarray) -> None:
    with open(path, "wb") as f:
        np.save(f, data)


def npy_load(path: str) -> np.ndarray:
    with open(path, "rb") as f:
        return np.load(f)


def save_json(p, data, default=None) -> None:
    with open(p, 'w') as outfile:
        json.dump(data, outfile, indent=4, default=default)


def load_json(p) -> Any:
    with open(p) as json_file:
        return json.load(json_file)
