import os
import librosa
import json
import math
import warnings
import csv

warnings.filterwarnings("ignore")
DATASET_PATH = "Dataset"
JSON_PATH = "./data.json"
CSV_PATH = "./data.csv"
SAMPLE_RATE = 22050
SOUND_DURATION = 3  # *detik
SAMPLES_PER_SOUND = SAMPLE_RATE * SOUND_DURATION


def save_mfcc(
    dataset_path, json_path, num_mfcc=13, n_fft=2048, hop_length=512, num_segments=4
):

    # buat dictionary untuk simpan atribut mapping, labels, dan MFCCs
    dataset = {"mapping": [], "labels": [], "mfcc": [], "files": []}

    samples_per_segment = int(SAMPLES_PER_SOUND / num_segments)
    num_mfcc_vectors_per_segment = math.ceil(samples_per_segment / hop_length)

    # looping sub-folder dataset
    for i, (dirpath, dirnames, filenames) in enumerate(os.walk(dataset_path)):

        # cek agar memproses sub-folder dataset
        if dirpath is not dataset_path:

            # simpan nama sub-folder pada key mapping di dictionary
            semantic_label = dirpath.split("/")[-1]
            dataset["mapping"].append(semantic_label)
            print("\nProcessing: {}".format(semantic_label))

            # process semua audio di sub-folder
            for f in filenames:
                file_path = os.path.join(dirpath, f)

                # muat file audio dan potong untuk memastikan konsistensi panjang di antara file yang berbeda
                signal, sample_rate = librosa.load(file_path)

                # seleksi kondisi jika audio kurang dari 3 detik
                if len(signal) >= SAMPLES_PER_SOUND:

                    # memastikan konsistensi panjang audio
                    signal = signal[:SAMPLES_PER_SOUND]

                    # extract MFCC
                    MFCCs = librosa.feature.mfcc(
                        signal,
                        sample_rate,
                        n_mfcc=num_mfcc,
                        n_fft=n_fft,
                        hop_length=hop_length,
                    )

                    # simpan data ke dictionary
                    dataset["mfcc"].append(MFCCs.T.tolist())
                    dataset["labels"].append(i - 1)
                    dataset["files"].append(file_path)
                    print("{}: {}".format(file_path, i - 1))

    # simpan MFCC menjadi file json
    with open(json_path, "w") as fp:
        json.dump(dataset, fp, indent=4)

    # simpan MFCC menjadi file csv
    j = len(dataset["labels"])
    i = 0
    temp = []

    while i < j:
        temp1 = []
        temp1.extend([dataset["labels"][i]])
        temp1.extend([dataset["files"][i]])
        temp1.extend([dataset["mfcc"][i]])
        temp.append(temp1)
        i += 1

    with open("data.csv", "w") as file:
        writer = csv.writer(file)

        writer.writerows(map(lambda x: [x], temp))


if __name__ == "__main__":
    save_mfcc(DATASET_PATH, JSON_PATH, num_segments=10)
