import os
from re import L
import librosa
import json
import math
import warnings
import csv
import time


warnings.filterwarnings("ignore")


SAMPLE_RATE = 22050
SOUND_DURATION = 3  # measured in seconds
SAMPLES_PER_SOUND = SAMPLE_RATE * SOUND_DURATION


def save_mfcc(
    data_path, json_path, num_mfcc=13, n_fft=2048, hop_length=512, num_segments=4
):

     # buat dictionary untuk simpan atribut mapping, labels, dan MFCCs
    dataset = {"mfcc": []}

    samples_per_segment = int(SAMPLES_PER_SOUND / num_segments)
    num_mfcc_vectors_per_segment = math.ceil(samples_per_segment / hop_length)

    # muat file audio dan potong untuk memastikan konsistensi panjang di antara file yang berbeda
    signal, sample_rate = librosa.load(data_path)

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

def main():
    i = 0
    while True:
        i+=1
        data_path = './Record/audio{}.wav'.format(i)
        json_path = './ExtAudio/test{}.json'.format(i)
        isExist = './Record/audio{}.wav'.format(i+1)
        if os.path.exists(isExist) == False:
            time.sleep(6)
        save_mfcc(data_path, json_path, num_segments=10)
        time.sleep(3)
