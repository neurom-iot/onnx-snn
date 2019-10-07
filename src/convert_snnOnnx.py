import onnx
# onnx->sn-onnx
# sn: Spiking Neuron

def convert_snnOnnx(onnx_path):
    sn_index = 1
    conv_index = 1
    maxpool_index = 1
    onnx_model = onnx.load(onnx_path)
    node_len = len(onnx_model.graph.node)
    for index in range(node_len):
        op_type = onnx_model.graph.node[index].op_type.lower()
        if op_type == "relu" or op_type == "sigmoid" or op_type == "tanh":
            convert_name = "Sn"
            onnx_model.graph.node[index].op_type = convert_name
            onnx_model.graph.node[index].name = convert_name + "_" + str(sn_index)
            sn_index = sn_index + 1
        elif op_type == "conv":
            convert_name = "Sn_Conv2D"
            onnx_model.graph.node[index].op_type = convert_name
            onnx_model.graph.node[index].name = convert_name + "_" + str(conv_index)
            conv_index = conv_index + 1
        elif op_type == "maxpool":
            convert_name = "Sn_MaxPool2D"
            onnx_model.graph.node[index].op_type = convert_name
            onnx_model.graph.node[index].name = convert_name + "_" + str(maxpool_index)
            maxpool_index = maxpool_index + 1

    onnx.save(onnx_model, 'snn.onnx')

if __name__ == "__main__":
    convert_snnOnnx('cnnmodel.onnx')
