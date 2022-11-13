from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D
from keras.optimizers import Adam
from keras.layers import MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

#Get data from directory
t = 'train'
v = 'test'
tdata = ImageDataGenerator(rescale=1./255)
vdata = ImageDataGenerator(rescale=1./255)

data_train = tdata.flow_from_directory(t, target_size=(48,48), batch_size=64, color_mode="grayscale", class_mode='categorical')
data_val = vdata.flow_from_directory(v, target_size=(48,48), batch_size=64, color_mode="grayscale", class_mode='categorical')

#create model
cnn = Sequential()

cnn.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48,48,1)))
cnn.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
cnn.add(MaxPooling2D(pool_size=(2, 2)))
cnn.add(Dropout(0.25))

cnn.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
cnn.add(MaxPooling2D(pool_size=(2, 2)))
cnn.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
cnn.add(MaxPooling2D(pool_size=(2, 2)))
cnn.add(Dropout(0.25))

cnn.add(Flatten())
cnn.add(Dense(1024, activation='relu'))
cnn.add(Dropout(0.5))
cnn.add(Dense(7, activation='softmax'))

#Train the model
cnn.compile(loss='categorical_crossentropy',optimizer=Adam(lr=0.0001, decay=1e-6),metrics=['accuracy'])
history = cnn.fit(data_train,steps_per_epoch=28709 // 64,epochs=50,validation_data=data_val,validation_steps=7178 // 64)
cnn.save_weights('cnn-weights.h5')
cnn.save('cnn.h5')

#Plot accuracy vs epoch for train set and validation set
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('cnn.png')

