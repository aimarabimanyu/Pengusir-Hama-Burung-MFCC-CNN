import json
import numpy as np
from tensorflow import keras
from keras.models import model_from_json
import time
import os
import warnings


warnings.filterwarnings("ignore")


def main():
    i = 0
    while True:
        i+=1
        data_path = './ExtAudio/test{}.json'.format(i)
        audio_path = './Record/audio{}.wav'.format(i)
        isExist = './ExtAudio/test{}.json'.format(i+1)
        if os.path.exists(isExist) == False:
            time.sleep(6)
        
        # ambil model.json
        json_file = open("D:\Kuliah\Semester 5\Pemrosesan Suara\Tugas\Project Pemrosesan Suara\model.json", 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        model = model_from_json(loaded_model_json)

        # ambil weights ke model
        model.load_weights("D:\Kuliah\Semester 5\Pemrosesan Suara\Tugas\Project Pemrosesan Suara\model.h5")
        print("Loaded model from disk")

        # ambil data json dan tambah satu dimensi ke data
        with open(data_path, "r") as ft:
            datatest = json.load(ft)

        A = np.array(datatest["mfcc"])
                
        A = A[..., np.newaxis]
                
        # lakukan prediksi
        predictionTest = model.predict(A)

        # ambil nilai index
        predicted_indexTest = np.argmax(predictionTest, axis=1)
                
        print("Predicted label: {}".format(predicted_indexTest))
        time.sleep(3)
        os.remove(data_path)
        os.remove(audio_path)