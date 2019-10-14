import re
import onnx
import numpy as np
from onnx import numpy_helper

class onnxToNengo:
    def __init__(self, onnx_path):
        self.nengoCode = ""
        self.neuron_type = "LIF"    #default Neuron Type = LIF
        self.onnx_model = onnx.load(onnx_path)
        # self.printNode()
        self.setNetwork()

    def setNetwork(self):
        #init code generating
        self.nengoCode += self.genInit(self.neuron_type)

        #layer code generating
        onnx_model_graph = self.onnx_model.graph
        onnx_model_graph_node = onnx_model_graph.node
        node_len = len(onnx_model_graph.node)
        input_info = np.array(onnx_model_graph.input[0].type.tensor_type.shape.dim)
        self.nengoCode += self.genInput(input_info)

        for index in range(node_len):
            node_info = onnx_model_graph_node[index]
            op_type = node_info.op_type.lower()
            if op_type == "conv":
                self.nengoCode += self.genConv(index, onnx_model_graph_node)

            elif op_type == "maxpool":
                self.nengoCode += self.genMaxpool(node_info)

            elif op_type == "reshape":
                regex = re.compile("flatten")
                if regex.findall(node_info.name):
                    self.nengoCode += self.genFlatten(node_info)

            elif op_type == "matmul":
                self.nengoCode += self.genMatmul(index, onnx_model_graph_node)

    def genInit(self, neuron_type):
        code = ""
        code += "import nengo\n"
        code += "import numpy as np\n\n"
        code += "with nengo.Network() as net:\n"
        code += "\tnet.config[nengo.Ensemble].max_rates = nengo.dists.Choice([100])\n"
        code += "\tnet.config[nengo.Ensemble].intercepts = nengo.dists.Choice([0])\n"
        code += "\tneuron_type = nengo." + str(neuron_type) + "(amplitude=0.01)\n"
        code += "\tnengo_dl.configure_settings(trainable=False)\n"
        code += "\t\n"
        return code
    
    def genInput(self, input_info):
        length = len(input_info)
        dim_array = []
        for index in range(1, length):
            dim_array.append(re.findall('\d+', str(input_info[index]))[0])
        
        code = ""
        code += "\tinp = nengo.Node([0]"
        for index in range(len(dim_array)):
            code += " * " + str(dim_array[index])
        code += ")\n"
        return code

    def genConv(self, index, onnx_model_graph_node):
        node_info = onnx_model_graph_node[index]
        neuron_type = self.getNeuronType(index, onnx_model_graph_node)
        input_info = self.getLayer2LayerInputDataInfo(node_info)
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

    def genMatmul(self, index, onnx_model_graph_node):
        node_info = onnx_model_graph_node[index]
        neuron_type = self.getNeuronType(index, onnx_model_graph_node)
        input_info = self.getLayer2LayerInputDataInfo(node_info)
        code = ""
        code += "matmul\n"
        return code
    
    def getNeuronType(self, node_index, onnx_model_graph_node):
        node_len = len(onnx_model_graph_node)
        for index in range(node_index, node_len):
            node_info = onnx_model_graph_node[index]
            op_type = node_info.op_type.lower()
            if op_type == "lif":
                neuron_type = "lif"
                return neuron_type
            elif op_type == "softmax":
                neuron_type = "softmax"
                return neuron_type
        return None

    def makefile(self, result_path):
        file = open(result_path, 'w', encoding="utf-8")
        file.write(self.getNengoCode())
        file.close()

    def getNengoCode(self):
        return self.nengoCode

    def getLayer2LayerInputDataInfo(self, node_info):
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

    def convert_snnOnnx(self, onnx_path, result_path, neuron_type):
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
    onnx_file_path = "../model/model2onnx/cnn2onnx.onnx"
    result_file_path = "../model/onnx2snn/cnn2snn.onnx"
    cso = convert_snnOnnx()
    cso.convert_snnOnnx(onnx_file_path, result_file_path, "LIF")
    otn = onnxToNengo(result_file_path)
    otn.makefile("../result/nengo_code.py")