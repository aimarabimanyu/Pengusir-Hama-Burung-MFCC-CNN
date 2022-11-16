import json
import numpy as np
from tensorflow import keras
from keras.models import model_from_json
import time
import os


i = 0
while True:
    i+=1
    data_path = './ExtAudio/test{}.json'.format(i)
    audio_path = './Record/audio{}.wav'.format(i)
    isExist = './ExtAudio/test{}.json'.format(i+1)
    if os.path.exists(isExist) == False:
        time.sleep(6)
    
    # load json and create model
    json_file = open("./model.json", 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)

    # load weights into new model
    model.load_weights("./model.h5")
    print("Loaded model from disk")

    with open(data_path, "r") as ft:
        datatest = json.load(ft)

    A = np.array(datatest["mfcc"])
            
    A = A[..., np.newaxis]
            
    # perform prediction
    predictionTest = model.predict(A)

    # get index with max value
    predicted_indexTest = np.argmax(predictionTest, axis=1)
            
    print("Predicted label: {}".format(predicted_indexTest))
    time.sleep(3)
    os.remove(data_path)
    os.remove(audio_path)