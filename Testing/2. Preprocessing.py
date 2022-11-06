import os
from re import L
import librosa
import json
import math
import warnings
import csv

warnings.filterwarnings("ignore")


DATA_PATH = "./AudioTesting/pipit1.wav"
JSON_PATH = "./Testing/test.json"
SAMPLE_RATE = 22050
SOUND_DURATION = 3  # measured in seconds
SAMPLES_PER_SOUND = SAMPLE_RATE * SOUND_DURATION


def save_mfcc(
    json_path, num_mfcc=13, n_fft=2048, hop_length=512, num_segments=4
):

    # dictionary to store mapping, labels, and MFCCs
    dataset = {"mfcc": []}

    samples_per_segment = int(SAMPLES_PER_SOUND / num_segments)
    num_mfcc_vectors_per_segment = math.ceil(samples_per_segment / hop_length)

    # load audio file and slice it to ensure length consistency among different files
    signal, sample_rate = librosa.load(DATA_PATH)

    # drop audio files with less than pre-decided number of samples
    if len(signal) >= SAMPLES_PER_SOUND:

        # ensure consistency of the length of the signal
        signal = signal[:SAMPLES_PER_SOUND]

        # extract MFCCs
        MFCCs = librosa.feature.mfcc(
            signal,
            sample_rate,
            n_mfcc=num_mfcc,
            n_fft=n_fft,
            hop_length=hop_length,
        )

        # store data for analysed track
        dataset["mfcc"].append(MFCCs.T.tolist())

    # save MFCCs to json file
    with open(json_path, "w") as fp:
        json.dump(dataset, fp, indent=4)

if __name__ == "__main__":
    save_mfcc(JSON_PATH, num_segments=10)
