import os
import onnx
import keras2onnx
import onnxruntime
from keras.models import load_model

def modelToOnnx(target_path, result_path):
    MODEL_SAVE_FOLDER_PATH = '../model/model2onnx/'

    if not os.path.exists(MODEL_SAVE_FOLDER_PATH):
        os.mkdir(MODEL_SAVE_FOLDER_PATH)
    model = load_model(target_path)

    # convert to onnx model
    onnx_model = keras2onnx.convert_keras(model, model.name)

    # runtime prediction
    content = onnx_model.SerializeToString()
    temp_model_file = MODEL_SAVE_FOLDER_PATH + result_path
    onnx.save_model(onnx_model, temp_model_file)

if __name__ == "__main__":
    modelToOnnx("../model/dnn/vgg16.h5", "vgg162onnx.onnx")