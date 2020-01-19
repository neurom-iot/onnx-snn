from keras.datasets import mnist
from keras.utils import np_utils, multi_gpu_model
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten, Activation, AveragePooling2D, BatchNormalization
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras import backend as K
from PIL import Image
from keras.callbacks import TensorBoard
import os
import time
import numpy as np
import pickle
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# The GPU id to use, usually either "0" or "1";
# os.environ["CUDA_VISIBLE_DEVICES"]="1"

np.random.seed(1000)
K.tensorflow_backend.set_image_dim_ordering('tf')
##Vgg16 based keras

def img_resize(value, img_array):
    print("--images are being resizing--")
    result = np.zeros((len(img_array), value[0], value[1]))

    for index in range(len(img_array)):

        img = Image.fromarray(img_array[index], 'L')

        img = img.resize((value[0], value[1]))

        img = np.array(img)

        result[index] = img

    print("--image resize complete--")
    return result

MODEL_SAVE_FOLDER_PATH = '../../model/dnn/lenet-1/'

if not os.path.exists(MODEL_SAVE_FOLDER_PATH):
  os.mkdir(MODEL_SAVE_FOLDER_PATH)

model_path = MODEL_SAVE_FOLDER_PATH + 'mnist-' + '{epoch:02d}-{val_loss:.4f}.h5'

cb_checkpoint = ModelCheckpoint(filepath = model_path, monitor = 'val_loss',
                                verbose = 1, save_best_only = True)

cb_early_stopping = EarlyStopping(monitor = 'val_loss', patience = 10)

classes = 10
input_shape = (28, 28)

(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

X_train = img_resize(input_shape, X_train)
X_test = img_resize(input_shape, X_test)

X_train = np.expand_dims(X_train, axis = -1)
X_test = np.expand_dims(X_test, axis = -1)

Y_train = np_utils.to_categorical(Y_train, 10)
Y_test = np_utils.to_categorical(Y_test, 10)

model = Sequential()

model.add(Conv2D(input_shape = (input_shape[0], input_shape[1], 1), filters = 4, kernel_size = (5, 5), strides=(1, 1), padding = 'same', name = 'conv1'))
model.add(Activation('tanh'))
model.add(AveragePooling2D(pool_size = (2, 2), strides = (2, 2), name = 'avgpool1'))

model.add(Conv2D(filters = 12, kernel_size = (5, 5), strides=(1, 1), padding = 'valid', name = 'conv2'))
model.add(Activation('tanh'))
model.add(AveragePooling2D(pool_size = (2, 2), strides = (2, 2), name = 'avgpool2'))

model.add(Flatten(name = 'flatten'))

model.add(Dense(classes, activation = 'softmax',  name = 'predictions'))
tensorboard = TensorBoard(log_dir='lenetlogs/{}'.format(time.time()), histogram_freq=0, write_graph=True, write_images=True)
model.compile(loss = 'categorical_crossentropy', optimizer = 'SGD', metrics = ['accuracy'])

start_time = time.time()
history = model.fit(X_train, Y_train,
                    validation_data = (X_test, Y_test),
                    epochs = 100, batch_size = 200, verbose = 1,
                    callbacks = [tensorboard])
                    # callbacks = [cb_checkpoint, cb_early_stopping])

with open(MODEL_SAVE_FOLDER_PATH + 'history.pickle', 'wb') as file_pi:
        pickle.dump(history.history, file_pi)

test_loss, test_acc = model.evaluate(X_test, Y_test)
print('prediction_acc: ', test_acc)
print('prediction_loss: ', test_loss)
print('run time :', round(time.time()-start_time, 3))
model.save(MODEL_SAVE_FOLDER_PATH + "lenet-1_tanh.h5")