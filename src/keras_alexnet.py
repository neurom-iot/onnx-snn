from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten, Activation, AveragePooling2D, BatchNormalization
from keras.callbacks import ModelCheckpoint, EarlyStopping
import os
import numpy as np
np.random.seed(1000)

##Alexnet based keras

# MODEL_SAVE_FOLDER_PATH = './model/'

# if not os.path.exists(MODEL_SAVE_FOLDER_PATH):
#   os.mkdir(MODEL_SAVE_FOLDER_PATH)

# model_path = MODEL_SAVE_FOLDER_PATH + 'mnist-' + '{epoch:02d}-{val_loss:.4f}.h5'

# cb_checkpoint = ModelCheckpoint(filepath = model_path, monitor = 'val_loss',
#                                 verbose = 1, save_best_only = True)

# cb_early_stopping = EarlyStopping(monitor = 'val_loss', patience = 10)

classes = 10

(X_train, Y_train), (X_validation, Y_validation) = mnist.load_data()

X_train = X_train.reshape(X_train.shape[0], 28, 28, 1).astype('float32') / 255
X_validation = X_validation.reshape(X_validation.shape[0], 28, 28, 1).astype('float32') / 255

X_train.resize(X_train.shape[0], 224, 224, 1)
X_validation.resize(X_validation.shape[0], 224, 224, 1)

Y_train = np_utils.to_categorical(Y_train, 10)
Y_validation = np_utils.to_categorical(Y_validation, 10)

model = Sequential()

#block 1
model.add(Conv2D(input_shape = (224, 224, 1), filters=96, kernel_size = (11, 11), strides = (4, 4), padding = 'valid', name = 'block1_conv'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size = (2, 2), strides=(2, 2), padding = 'valid', name = 'block1_maxpool'))

#block 2
model.add(Conv2D(filters = 256, kernel_size=(11, 11), strides = (1, 1), padding = 'valid', name = 'block2_conv'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size = (2, 2), strides = (2, 2), padding = 'valid', name = 'block2_maxpool'))

#block 3
model.add(Conv2D(filters = 384, kernel_size = (3, 3), strides = (1, 1), padding = 'valid', name = 'block3_conv'))
model.add(BatchNormalization())
model.add(Activation('relu'))
#block 4
model.add(Conv2D(filters = 384, kernel_size = (3, 3), strides = (1, 1), padding = 'valid', name = 'block4_conv'))
model.add(BatchNormalization())
model.add(Activation('relu'))
#block 5
model.add(Conv2D(filters = 256, kernel_size = (3, 3), strides = (1, 1), padding = 'valid', name = 'block5_conv'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size = (2, 2), strides = (2, 2), padding = 'valid', name = 'block3_maxpool'))

model.add(Flatten(name = 'flatten'))

model.add(Dense(4096, name = 'fullconnected1'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.4))

model.add(Dense(4096, name = 'fullconnected2'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.4))

model.add(Dense(1000, name = 'fullconnected3'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.4))

model.add(Dense(classes, activation = 'softmax', name = 'predictions'))

model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

history = model.fit(X_train, Y_train,
                    validation_data = (X_validation, Y_validation),
                    epochs = 10, batch_size = 100, verbose = 0)
                    # callbacks = [cb_checkpoint, cb_early_stopping])

test_loss, test_acc = model.evaluate(X_validation, Y_validation)
print('test_acc: ', test_acc)
model.save("../model/dnn/alexnet.h5")