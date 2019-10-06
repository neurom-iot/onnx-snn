import onnx
import keras2onnx
import onnxruntime
from keras.models import load_model

def modelToOnnx():
    model = load_model("model.h5")

    # convert to onnx model
    onnx_model = keras2onnx.convert_keras(model, model.name)

    # runtime prediction
    content = onnx_model.SerializeToString()
    temp_model_file = 'model.onnx'
    onnx.save_model(onnx_model, temp_model_file)

if __name__ == "__main__":
    modelToOnnx()