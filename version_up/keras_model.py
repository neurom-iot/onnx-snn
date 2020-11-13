import tensorflow as tf
import keras
from keras.callbacks import ModelCheckpoint
from keras import Input
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten

'''
    step 3.
    To convert the SNN-format model(nengo-dl) to DNN-format model(ONNX),
    a temporary Keras model should be created.
    This code perform that model is just trained epoch 1 with random weights. 
'''

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

img_rows = 28
img_cols = 28

input_shape = (img_rows, img_cols, 1)
x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.

print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)

modelSavePath = './model_convolution.h5'
modelCheckPoint = ModelCheckpoint(modelSavePath)

model = Sequential()
model.add(Conv2D(32, kernel_size=3, input_shape=(28, 28, 1), activation='relu'))
model.add(Conv2D(64, kernel_size=3, strides=2, activation='relu'))
model.add(Conv2D(128, kernel_size=3, strides=2, activation='relu'))
model.add(Flatten())
model.add(Dense(784, activation='relu'))
model.add(Dense(10, activation='softmax'))

# model.add(Conv2D(
#          filters=32,
#          kernel_size=(3, 3),
#          strides=(1, 1),
#          padding='same',
#          activation='relu',
#          input_shape=(28, 28, 1)))
# model.add(Flatten())
# model.add(Dense(units=10, activation='relu'))

model.summary()

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train,
          epochs=1,
          batch_size=64,
          verbose=1,
          callbacks=[modelCheckPoint])

