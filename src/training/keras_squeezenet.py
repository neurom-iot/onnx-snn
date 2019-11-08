from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Model
from keras.layers import Input, Dense, Convolution2D, MaxPooling2D, Dropout, Flatten, Activation, AveragePooling2D, BatchNormalization, concatenate, GlobalAveragePooling2D
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras import backend as K
import os
import time
import numpy as np
np.random.seed(1000)
K.tensorflow_backend.set_image_dim_ordering('tf')

MODEL_SAVE_FOLDER_PATH = '../../model/dnn/squeezenet/'

if not os.path.exists(MODEL_SAVE_FOLDER_PATH):
  os.mkdir(MODEL_SAVE_FOLDER_PATH)

model_path = MODEL_SAVE_FOLDER_PATH + 'mnist-' + '{epoch:02d}-{val_loss:.4f}.h5'

cb_checkpoint = ModelCheckpoint(filepath=model_path, monitor = 'val_loss',
                                verbose=1, save_best_only = True)

cb_early_stopping = EarlyStopping(monitor='val_loss', patience = 10)

classes = 10

(X_train, Y_train), (X_validation, Y_validation) = mnist.load_data()

X_train = X_train.reshape(X_train.shape[0], 28, 28, 1).astype('float32') / 255
X_validation = X_validation.reshape(X_validation.shape[0], 28, 28, 1).astype('float32') / 255

X_train.resize(X_train.shape[0], 224, 224, 1)
X_validation.resize(X_validation.shape[0], 224, 224, 1)

Y_train = np_utils.to_categorical(Y_train, 10)
Y_validation = np_utils.to_categorical(Y_validation, 10)

def fire_module(seq, fire_id, squeeze = 16, expand = 64):     #The meaning of expansion means that it extends as much as the number of filters.
    block_id = 'block' + str(fire_id) + '_'
    
    sqz = Convolution2D(filters = squeeze, kernel_size = (1, 1), padding = 'valid', name = block_id + 'squeeze')(seq)
    sqz = BatchNormalization()(sqz)
    sqz = Activation('relu')(sqz)

    left = Convolution2D(filters = expand, kernel_size = (1, 1), padding = 'valid', name = block_id + 'expand1')(sqz)
    left = BatchNormalization()(left)
    left = Activation('relu')(left)

    right = Convolution2D(filters = expand, kernel_size = (3, 3), padding = 'same', name = block_id + 'expand2')(sqz)
    right = BatchNormalization()(right)
    right = Activation('relu')(right)

    output = concatenate([left, right], axis = 3, name = block_id + 'concat')
    return output

inputs = Input(shape = (224, 224, 1))
x = Convolution2D(filters = 64, kernel_size = (3, 3), strides = (2, 2), padding = 'valid', name = 'block1_conv1')(inputs)
x = Activation('relu', name = 'relu_conv1')(x)
x = MaxPooling2D(pool_size = (3, 3), strides = (2, 2), name = 'block1_pool')(x)

x = fire_module(x, fire_id = 2, squeeze = 16, expand = 64)
x = fire_module(x, fire_id = 3, squeeze = 16, expand = 64)
X_train = MaxPooling2D(pool_size = (3, 3), strides = (2, 2), name = 'block2_pool')(x)
 
x = fire_module(x, fire_id = 4, squeeze = 32, expand = 128)
x = fire_module(x, fire_id = 5, squeeze = 32, expand = 128)
x = MaxPooling2D(pool_size = (3, 3), strides = (2, 2), name = 'block3_pool')(x)
 
x = fire_module(x, fire_id = 6, squeeze = 48, expand = 192)
x = fire_module(x, fire_id = 7, squeeze = 48, expand = 192)
x = fire_module(x, fire_id = 8, squeeze = 64, expand = 256)
x = fire_module(x, fire_id = 9, squeeze = 64, expand = 256)
x = Convolution2D(filters = classes, kernel_size = (1, 1), padding = 'valid', name = 'block4_conv')(x)
x = Activation('relu')(x)
x = BatchNormalization()(x)

x = GlobalAveragePooling2D()(x)

x = Activation('softmax',  name = 'predictions')(x)
model = Model(inputs = inputs, outputs = x)

model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

start_time = time.time()

model.fit(X_train, Y_train, validation_data = (X_validation, Y_validation), epochs = 100, batch_size = 32, verbose = 0, callbacks = [cb_checkpoint, cb_early_stopping])

test_loss, test_acc = model.evaluate(X_validation, Y_validation)
print('test_acc: ', test_acc)
print('run time :', round(time.time() - start_time, 3))
model.save(MODEL_SAVE_FOLDER_PATH + "/squeezenet.h5")