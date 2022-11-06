import os
from re import L
import librosa
import json
import math
import warnings
import csv

warnings.filterwarnings("ignore")
DATASET_PATH = "Dataset"
JSON_PATH = "./Preprocessing/data.json"
SAMPLE_RATE = 22050
SOUND_DURATION = 3  # measured in seconds
SAMPLES_PER_SOUND = SAMPLE_RATE * SOUND_DURATION


def save_mfcc(
    dataset_path, json_path, num_mfcc=13, n_fft=2048, hop_length=512, num_segments=4
):

    # dictionary to store mapping, labels, and MFCCs
    dataset = {"mapping": [], "labels": [], "mfcc": [], "files": []}

    samples_per_segment = int(SAMPLES_PER_SOUND / num_segments)
    num_mfcc_vectors_per_segment = math.ceil(samples_per_segment / hop_length)

    # loop through all genre sub-folder
    for i, (dirpath, dirnames, filenames) in enumerate(os.walk(dataset_path)):

        # ensure we're processing a genre sub-folder level
        if dirpath is not dataset_path:

            # save genre label (i.e., sub-folder name) in the mapping
            semantic_label = dirpath.split("/")[-1]
            dataset["mapping"].append(semantic_label)
            print("\nProcessing: {}".format(semantic_label))

            # process all audio files in genre sub-dir
            for f in filenames:
                file_path = os.path.join(dirpath, f)

                # load audio file and slice it to ensure length consistency among different files
                signal, sample_rate = librosa.load(file_path)

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
                    dataset["labels"].append(i - 1)
                    dataset["files"].append(file_path)
                    print("{}: {}".format(file_path, i - 1))

    # save MFCCs to json file
    with open(json_path, "w") as fp:
        json.dump(dataset, fp, indent=4)

    # save MFCCs to csv file
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
