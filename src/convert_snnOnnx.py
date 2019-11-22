import os
import onnx

class convert_snnOnnx:
    def __init__(self):
        return

    def run(self, onnx_path, result_path, neuron_type):
        MODEL_SAVE_FOLDER_PATH = '../model/onnx2snn/'

        if not os.path.exists(MODEL_SAVE_FOLDER_PATH):
            os.mkdir(MODEL_SAVE_FOLDER_PATH)
        sn_index = 1
        onnx_model = onnx.load(onnx_path)
        node_len = len(onnx_model.graph.node)
        for index in range(node_len):
            op_type = onnx_model.graph.node[index].op_type.lower()
            if op_type == "relu" or op_type == "sigmoid" or op_type == "tanh":
                onnx_model.graph.node[index].op_type = neuron_type
                onnx_model.graph.node[index].name = neuron_type + "_" + str(sn_index)
                sn_index = sn_index + 1
        onnx.save(onnx_model, result_path)

if __name__ == "__main__":
    onnx_file_path = "../model/model2onnx/vgg162onnx.onnx"
    result_file_path = "../model/onnx2snn/vgg162snn.onnx"
    cso = convert_snnOnnx()
    cso.run(onnx_file_path, result_file_path, "LIF")