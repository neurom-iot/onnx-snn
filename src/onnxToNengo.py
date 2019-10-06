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

            elif op_type == "matmul":
                self.nengoCode += self.genMatmul(node_info)

            elif op_type == "softmax":
                self.nengoCode += self.genSoftmax(node_info)

    def genConv(self, node_info):
        code = ""
        code += "conv\n"
        return code

    def genMaxpool(self, node_info):
        code = ""
        code += "maxpool\n"
        return code

    def genMatmul(self, node_info):
        code = ""
        code += "matmul\n"
        return code
    
    def genSoftmax(self, node_info):
        code = ""
        code += "softmax\n"
        return code

    def makefile(self, result_path):
        file = open(result_path, 'w', encoding="utf-8")
        file.write(self.getNengoCode())
        file.close()

    def getNengoCode(self):
        return self.nengoCode

    def printNode(self):
        print(self.onnx_model.graph.node)
        
if __name__ == "__main__":
    otn = onnxToNengo("ssn.onnx")
    otn.makefile("nengo_code.py")



