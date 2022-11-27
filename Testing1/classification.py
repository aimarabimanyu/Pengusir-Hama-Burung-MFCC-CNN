import json
import numpy as np
from tensorflow import keras
from keras.models import model_from_json

TEST_PATH = "./Testing1/test.json"

# ambil model.json
json_file = open("..\Project Pemrosesan Suara\model.json", 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)

# ambil weights ke model
model.load_weights("..\Project Pemrosesan Suara\model.h5")
print("Loaded model from disk")

# ambil data json dan tambah satu dimensi ke data
with open(TEST_PATH, "r") as ft:
    datatest = json.load(ft)

A = np.array(datatest["mfcc"])
    
A = A[..., np.newaxis]
    
# lakukan prediksi
predictionTest = model.predict(A)

# ambil nilai index
predicted_indexTest = np.argmax(predictionTest, axis=1)
    
print("Predicted label: {}".format(predicted_indexTest))