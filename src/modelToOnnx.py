import onnx
import keras2onnx
import onnxruntime
from keras.models import load_model

def modelToOnnx(target_path, result_path):
    model = load_model(target_path)

    # convert to onnx model
    onnx_model = keras2onnx.convert_keras(model, model.name)

    # runtime prediction
    content = onnx_model.SerializeToString()
    temp_model_file = result_path
    onnx.save_model(onnx_model, temp_model_file)

if __name__ == "__main__":
    modelToOnnx("../model/dnn/dnn_model(keras).h5", "../model/model2onnx/dnn2onnx.onnx")