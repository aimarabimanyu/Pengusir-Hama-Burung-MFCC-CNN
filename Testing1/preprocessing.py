import os
from re import L
import librosa
import json
import math
import warnings
import csv

warnings.filterwarnings("ignore")


DATA_PATH = "./Testing1/TestAudio/capunchin4.wav"
JSON_PATH = "./Testing1/test.json"
SAMPLE_RATE = 22050
SOUND_DURATION = 3  # *detik
SAMPLES_PER_SOUND = SAMPLE_RATE * SOUND_DURATION


def save_mfcc(
    json_path, num_mfcc=13, n_fft=2048, hop_length=512, num_segments=4
):

    # buat dictionary untuk simpan atribut mapping, labels, dan MFCCs
    dataset = {"mfcc": []}

    samples_per_segment = int(SAMPLES_PER_SOUND / num_segments)
    num_mfcc_vectors_per_segment = math.ceil(samples_per_segment / hop_length)

    # muat file audio dan potong untuk memastikan konsistensi panjang di antara file yang berbeda
    signal, sample_rate = librosa.load(DATA_PATH)

    # seleksi kondisi jika audio kurang dari 3 detik
    if len(signal) >= SAMPLES_PER_SOUND:

        # memastikan konsistensi panjang audio
        signal = signal[:SAMPLES_PER_SOUND]

        # extract MFCCs
        MFCCs = librosa.feature.mfcc(
            signal,
            sample_rate,
            n_mfcc=num_mfcc,
            n_fft=n_fft,
            hop_length=hop_length,
        )

        # simpan data ke dictionary
        dataset["mfcc"].append(MFCCs.T.tolist())

    # simpan MFCC menjadi json
    with open(json_path, "w") as fp:
        json.dump(dataset, fp, indent=4)

if __name__ == "__main__":
    save_mfcc(JSON_PATH, num_segments=10)
