import re
import onnx
import numpy as np
from onnx import numpy_helper

class onnxToNengo:
    def __init__(self, onnx_path):
        self.nengoCode = ""
        self.onnx_model = onnx.load(onnx_path)
        # self.printNode()
        self.setNetwork()

    def setNetwork(self):
        #import code
        self.nengoCode += "import nengo\n"
        self.nengoCode += "import nengo_dl\n"
        self.nengoCode += "import numpy as np\n"

        #Network code
        self.neuron_type = self.getNeuronType()
        self.nengoCode += "with nengo.Network() as net:\n"
        node_len = len(self.onnx_model.graph.node)
        input_info = np.array(self.onnx_model.graph.input[0].type.tensor_type.shape.dim)
        for index in range(node_len):
            node_info = self.onnx_model.graph.node[index]
            op_type = node_info.op_type.lower()
            if op_type == "conv":
                self.nengoCode += self.genConv(node_info)

            elif op_type == "maxpool":
                self.nengoCode += self.genMaxpool(node_info)

            elif op_type == "reshape":
                regex = re.compile("flatten")
                if regex.findall(node_info.name):
                    self.nengoCode += self.genFlatten(node_info)

            elif op_type == "matmul":
                self.nengoCode += self.genMatmul(node_info)

            elif op_type == "softmax":
                self.nengoCode += self.genSoftmax(node_info)

    def genConv(self, node_info):
        inputinfo = self.getInputDataInfo(node_info)
        print(inputinfo)
        for index in range(len(node_info.attribute)):
            if node_info.attribute[index].name == "kernel_shape":
                kernel_shape = np.array(node_info.attribute[index].ints)
            elif node_info.attribute[index].name == "strides":
                stride = node_info.attribute[index].ints
        code = ""
        code += "conv\n"
        return code

    def genMaxpool(self, node_info):
        for index in range(len(node_info.attribute)):
            if node_info.attribute[index].name == "kernel_shape":
                kernel_shape = np.array(node_info.attribute[index].ints)
            elif node_info.attribute[index].name == "strides":
                stride = node_info.attribute[index].ints
        code = ""
        code += "maxpool\n"
        return code

    def genFlatten(self, node_info):
        code = ""
        code += "flatten\n"
        return code

    def genMatmul(self, node_info):
        inputinfo = self.getInputDataInfo(node_info)
        print(inputinfo)
        code = ""
        code += "matmul\n"
        return code
    
    def genSoftmax(self, node_info):
        code = ""
        code += "softmax\n"
        return code
    
    def getNeuronType(self):
        node_len = len(self.onnx_model.graph.node)
        for index in range(node_len):
            node_info = self.onnx_model.graph.node[index]
            op_type = node_info.op_type.lower()
            if op_type == "lif":
                code = "LIF"
        return code

    def makefile(self, result_path):
        file = open(result_path, 'w', encoding="utf-8")
        file.write(self.getNengoCode())
        file.close()

    def getNengoCode(self):
        return self.nengoCode

    def getInputDataInfo(self, node_info):
        regex = re.compile("W|W\d*")
        for m in range(len(self.onnx_model.graph.initializer)):
            weight_name = self.onnx_model.graph.initializer[m].name
            for n in range(len(node_info.input)):
                if node_info.input[n] == weight_name and regex.findall(weight_name):
                    shape = self.onnx_model.graph.initializer[m].dims
                    return shape
        return None

    def printNode(self):
        print(self.onnx_model.graph.node)

class convert_snnOnnx:
    def __init__(self):
        return

    def convert_snnOnnx(self, onnx_path, result_path, neuron_kind):
        sn_index = 1
        onnx_model = onnx.load(onnx_path)
        node_len = len(onnx_model.graph.node)
        for index in range(node_len):
            op_type = onnx_model.graph.node[index].op_type.lower()
            if op_type == "relu" or op_type == "sigmoid" or op_type == "tanh":
                onnx_model.graph.node[index].op_type = neuron_kind
                onnx_model.graph.node[index].name = neuron_kind + "_" + str(sn_index)
                sn_index = sn_index + 1
        onnx.save(onnx_model, result_path)
        
if __name__ == "__main__":
    onnx_file_path = "dnn2onnx.onnx"
    result_file_path = "dnn2snn.onnx"
    cso = convert_snnOnnx()
    cso.convert_snnOnnx(onnx_file_path, result_file_path, "LIF")
    otn = onnxToNengo(result_file_path)
    otn.makefile("nengo_code.py")