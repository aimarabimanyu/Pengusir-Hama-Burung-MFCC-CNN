import threading
import input_audio
import preprocessing
import classification
import time

# buat thread

t1 = threading.Thread(target=input_audio.main)
t2 = threading.Thread(target=preprocessing.main)
t3 = threading.Thread(target=classification.main)

t1.start()
time.sleep(7)
t2.start()
time.sleep(7)
t3.start()
