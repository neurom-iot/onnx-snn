import tensorflow as tf
from keras.applications.resnet50 import ResNet50, decode_predictions
resnet = ResNet50()
resnet.save("../model/dnn/resnet.h5")
resnet.summary()