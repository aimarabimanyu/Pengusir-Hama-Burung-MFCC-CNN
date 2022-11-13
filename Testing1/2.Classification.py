import json
import numpy as np
from tensorflow import keras
from keras.models import model_from_json

TEST_PATH = "./Testing1/test.json"

# load json and create model
json_file = open("./model.json", 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)

# load weights into new model
model.load_weights("./model.h5")
print("Loaded model from disk")

with open(TEST_PATH, "r") as ft:
    datatest = json.load(ft)

A = np.array(datatest["mfcc"])
    
A = A[..., np.newaxis]
    
# perform prediction
predictionTest = model.predict(A)

# get index with max value
predicted_indexTest = np.argmax(predictionTest, axis=1)
    
print("Predicted label: {}".format(predicted_indexTest))