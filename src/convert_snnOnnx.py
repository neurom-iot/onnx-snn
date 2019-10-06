import onnx
# onnx->sn-onnx
# sn: Spiking Neuron

def convert_snnOnnx(onnx_path):
    onnx_model = onnx.load(onnx_path)
    sn_name = "Sn"
    node_len = len(onnx_model.graph.node)
    for index in range(node_len):
        op_type = onnx_model.graph.node[index].op_type.lower()
        if op_type == "relu" or op_type == "sigmoid" or op_type == "tanh":
            onnx_model.graph.node[index].name = sn_name
            onnx_model.graph.node[index].op_type = sn_name
            onnx_model.graph.node[index].output[0] = onnx_model.graph.node[index].output[0].lower().replace(op_type, sn_name)
            onnx_model.graph.node[index + 1].input[0] = onnx_model.graph.node[index + 1].input[0].lower().replace(op_type, sn_name)
    onnx.save(onnx_model, 'ssn.onnx')

if __name__ == "__main__":
    convert_snnOnnx('cnnmodel.onnx')
