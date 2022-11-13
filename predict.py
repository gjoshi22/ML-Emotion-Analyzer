import keras
import numpy as np
from tensorflow.python.keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import matplotlib

#Get data from directory
t = 'train'
v = 'test'
tdata = ImageDataGenerator(rescale=1./255)
vdata = ImageDataGenerator(rescale=1./255)

data_train = tdata.flow_from_directory(t,target_size=(48,48),batch_size=64,color_mode="grayscale",class_mode='categorical')
data_val = vdata.flow_from_directory(v,target_size=(48,48),batch_size=64,color_mode="grayscale",class_mode='categorical')

#Load model
model = load_model('cnn.h5')

#Print model summary
model.summary()

#Make a prediction
prediction = model.predict(data_val[5][0])
matplotlib.image.imsave('prediction.png', data_val[5][0])
print(prediction[0])
max_index = int(np.argmax(prediction[0]))
emotions = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}
print(emotions[max_index])
