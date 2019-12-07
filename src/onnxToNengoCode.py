import os
import re
import onnx
import numpy as np

class toNengoCode:
    def __init__(self, onnx_path, amplitude = 0.01, batch_size = 200, steps = 30, epochs = 10, learning_rate = 0.001, device = "cpu"):
        self.nengoCode = ""
        self.neuron_type = "LIF"                    #default Neuron Type = LIF
        self.amplitude = amplitude                  #default amplitude = 0.01
        self.batch_size = batch_size                #default batch_size = 200
        self.steps = steps                          #default steps = 50
        self.epochs = epochs                        #default epochs = 10
        self.learning_rate = learning_rate          #default learning_rate = 0.01
        self.device = device                        #default device = gpu
        self.onnx_model = onnx.load(onnx_path)
        
        # self.print_node()
        self.set_network()

    def set_network(self):
        #init code generating
        self.nengoCode += self.gen_init()

        #layer code generating
        onnx_model_graph = self.onnx_model.graph
        node_len = len(onnx_model_graph.node)
        input_shape = np.array(onnx_model_graph.input[0].type.tensor_type.shape.dim)
        temp = []
        regex = re.compile(":.*\d*")
        for index in range(1, len(input_shape)):
            data = regex.findall(str(input_shape[index]))[0]
            temp.append(int(data[2:len(data)]))
        input_shape = temp
        code, output_shape = self.gen_input(input_shape)
        self.nengoCode += code
        for index in range(node_len):
            node_info = onnx_model_graph.node[index]
            op_type = node_info.op_type.lower()
            if op_type == "conv":
                code, output_shape = self.convert_conv2d(output_shape, index, onnx_model_graph)
                self.nengoCode += code
            # elif op_type == "pad":
            #     self.convert_zeropad2d(output_shape, index, onnx_model_graph)
            elif op_type == "batchnormalization":
                code, output_shape = self.convert_batchnormalization(output_shape, node_info)
                self.nengoCode += code
            elif op_type == "maxpool":
                code, output_shape = self.convert_maxpool2d(output_shape, node_info)
                self.nengoCode += code
            elif op_type == "averagepool":
                code, output_shape = self.convert_avgpool2d(output_shape, node_info)
                self.nengoCode += code
            elif op_type == "reshape":
                regex = re.compile("flatten")
                if regex.findall(node_info.name):
                    code, output_shape= self.convert_flatten(output_shape)
                    self.nengoCode += code
            elif op_type == "matmul":
                code, output_shape = self.convert_dense(output_shape, index, onnx_model_graph)
                self.nengoCode += code
        self.nengoCode += self.gen_probe()
        self.nengoCode += self.gen_train(self.batch_size, self.steps, self.epochs, self.learning_rate, self.device)

    def gen_init(self):
        code = ""
        code += "#onnx to nengo convert code\n"
        code += "#training the network using a rate-based approximation\n"
        code += "import nengo\n"
        code += "import nengo_dl\n"
        code += "import numpy as np\n"
        code += "import tensorflow as tf\n\n"
        code += self.gen_dataset()
        code += self.gen_classification()
        code += "with nengo.Network(seed=1000) as net:\n"
        code += "\tnet.config[nengo.Ensemble].max_rates = nengo.dists.Choice([100])\n"
        code += "\tnet.config[nengo.Ensemble].intercepts = nengo.dists.Choice([0])\n"
        code += "\tdefault_neuron_type = nengo." + str(self.neuron_type) + "(amplitude=" + str(self.amplitude) + ")\n"
        code += "\tnengo_dl.configure_settings(trainable=False)\n"
        code += "\n"
        return code

    def gen_dataset(self):
        code = ""
        code += "import gzip\n"
        code += "import pickle\n"
        code += "from urllib.request import urlretrieve\n"
        code += "import zipfile\n\n"
        code += "urlretrieve(\"http://deeplearning.net/data/mnist/mnist.pkl.gz\", \"mnist.pkl.gz\")\n"
        code += "with gzip.open(\"mnist.pkl.gz\") as f:\n"
        code += "\ttrain_data, _, test_data = pickle.load(f, encoding=\"latin1\")\n"
        code += "train_data = list(train_data)\n"
        code += "test_data = list(test_data)\n\n"
        code += "for data in (train_data, test_data):\n"
        code += "\tone_hot = np.zeros((data[0].shape[0], 10))\n"
        code += "\tone_hot[np.arange(data[0].shape[0]), data[1]] = 1\n"
        code += "\tdata[1] = one_hot\n\n"
        return code

    def gen_input(self, input_shape):
        length = len(input_shape)
        dim_array = []
        for index in range(0, length):
            dim_array.append(re.findall('\d+', str(input_shape[index]))[0])
        code = ""
        code += "\tinp = nengo.Node([0]"
        for index in range(len(dim_array)):
            code += " * " + str(dim_array[index])
        code += ")\n"
        code += "\tx = inp\n\n"
        output_shape = input_shape
        return code, output_shape

    def convert_conv2d(self, input_shape, index, onnx_model_graph):
        onnx_model_graph_node = onnx_model_graph.node
        node_info = onnx_model_graph_node[index]
        neuron_type = self.get_neuronType(index, onnx_model_graph_node)
        filters = self.get_filterNum(node_info, onnx_model_graph)
        for index in range(len(node_info.attribute)):
            if node_info.attribute[index].name == "kernel_shape":
                kernel_size = node_info.attribute[index].ints[0]
            elif node_info.attribute[index].name == "strides":
                strides = node_info.attribute[index].ints[0]
            elif node_info.attribute[index].name == "auto_pad":
                padding = node_info.attribute[index].s.decode('ascii').lower()
                if padding != "valid":
                    padding = "same"
        if padding == "same":
            output_shape = [input_shape[0], input_shape[1], filters]
        else:
            output_shape = [int((input_shape[0] - kernel_size) / strides + 1), int((input_shape[1] - kernel_size) / strides + 1), filters]
        code = ""
        code += "\tx = nengo_dl.tensor_layer(x, tf.layers.conv2d, shape_in=(" + str(input_shape[0]) + ", " + str(input_shape[1]) + ", " + str(input_shape[2]) + "), filters=" + str(filters) + ", kernel_size=" + str(kernel_size) + ", padding=\"" + padding + "\")\n"
        if neuron_type == "lif" :
            code += "\tx = nengo_dl.tensor_layer(x, nengo.LIF(amplitude=" + str(self.amplitude) + "))\n\n"
        elif neuron_type == "lifrate":
            code += "\tx = nengo_dl.tensor_layer(x, nengo.LIFRate)(amplitude=" + str(self.amplitude) + "))\n\n"
        elif neuron_type == "adaptivelif":
            code += "\tx = nengo_dl.tensor_layer(x, nengo.AdaptiveLIF(amplitude=" + str(self.amplitude) + "))\n\n"
        elif neuron_type == "adaptivelifrate":
            code += "\tx = nengo_dl.tensor_layer(x, nengo.AdaptiveLIFRate(amplitude=" + str(self.amplitude) + "))\n\n"
        elif neuron_type == "izhikevich":
            code += "\tx = nengo_dl.tensor_layer(x, nengo.Izhikevich(amplitude=" + str(self.amplitude) + "))\n\n"
        elif neuron_type == "softlifrate":
            code += "\tx = nengo_dl.tensor_layer(x, nengo_dl.neurons.SoftLIFRate())\n\n"
        elif neuron_type == None:   #default neuron_type = LIF
            code += "\tx = nengo_dl.tensor_layer(x, default_neuron_type)\n\n"
        return code, output_shape

    # def convert_zeropad2d(self, input_shape, index, onnx_model_graph):
    #     onnx_model_graph_node = onnx_model_graph.node
    #     node_info = onnx_model_graph_node[index]
    #     pad_pads_name = node_info.input[1]
    #     pad_value_name = node_info.input[2]
    #     for m in range(len(onnx_model_graph.initializer)):
    #         name = onnx_model_graph.initializer[m].name
    #         for n in range(len(node_info.input)):
    #             if node_info.input[n] == name and name == pad_pads_name:
    #                 pad_pads = onnx_model_graph.initializer[m].int64_data
    #             elif node_info.input[n] == name and name == pad_value_name:
    #                 pad_value = int(onnx_model_graph.initializer[m].float_data[0])
    #     data = []
    #     for m in range(len(pad_pads)):
    #         if pad_pads[m] != 0:
    #             data.append(pad_pads[m])
    #     pad_pads = data
    #     if len(pad_pads) == 4:
    #         return
    #     elif len(pad_pads) == 2:
    #         return
    #     elif len(pad_pads) == 1:
    #         return

    def convert_batchnormalization(self, input_shape, node_info):
        for index in range(len(node_info.attribute)):
            if node_info.attribute[index].name == "momentum":
                momentum = round(node_info.attribute[index].f, 4)
                if momentum == 0:
                    momentum = 0.99
            elif node_info.attribute[index].name == "epsilon":
                epsilon = round(node_info.attribute[index].f, 4)
                if epsilon == 0:
                    epsilon = 0.001
        code = ""
        code += "\tx = nengo_dl.tensor_layer(x, tf.layers.batch_normalization, shape_in=(" + str(input_shape[0]) + ", " + str(input_shape[1]) + ", " + str(input_shape[2]) + "), momentum=" + str(momentum) + ", epsilon=" + str(epsilon) + ")\n\n"
        output_shape = input_shape
        return code, output_shape

    def convert_maxpool2d(self, input_shape, node_info):
        for index in range(len(node_info.attribute)):
            if node_info.attribute[index].name == "kernel_shape":
                pool_size = node_info.attribute[index].ints[0]
            elif node_info.attribute[index].name == "strides":
                strides = node_info.attribute[index].ints[0]
        output_shape = [int(input_shape[0]/strides), int(input_shape[1]/strides), input_shape[2]]
        code = ""
        code += "\tx = nengo_dl.tensor_layer(x, tf.layers.max_pooling2d, shape_in=(" + str(input_shape[0]) + ", " + str(input_shape[1]) + ", " + str(input_shape[2]) + "), pool_size=" + str(pool_size) + ", strides=" + str(strides) + ")\n\n"
        return code, output_shape

    def convert_avgpool2d(self, input_shape, node_info):
        for index in range(len(node_info.attribute)):
            if node_info.attribute[index].name == "kernel_shape":
                pool_size = node_info.attribute[index].ints[0]
            elif node_info.attribute[index].name == "strides":
                strides = node_info.attribute[index].ints[0]
        output_shape = [int(input_shape[0]/strides), int(input_shape[1]/strides), input_shape[2]]
        code = ""
        code += "\tx = nengo_dl.tensor_layer(x, tf.layers.average_pooling2d, shape_in=(" + str(input_shape[0]) + ", " + str(input_shape[1]) + ", " + str(input_shape[2]) + "), pool_size=" + str(pool_size) + ", strides=" + str(strides) + ")\n\n"
        return code, output_shape

    def convert_flatten(self, input_shape):
        code = ""
        code += "\tx = nengo_dl.tensor_layer(x, tf.layers.flatten)\n\n"
        output_shape = 1
        for index in range(len(input_shape)):
            output_shape *= input_shape[index]
        output_shape = [output_shape, 1]
        return code, output_shape

    def convert_dense(self, input_shape, index, onnx_model_graph):
        onnx_model_graph_node = onnx_model_graph.node
        node_info = onnx_model_graph_node[index]
        dense_num = self.get_dense_num(node_info, onnx_model_graph)
        neuron_type = self.get_neuronType(index, onnx_model_graph_node)
        code = ""
        code += "\tx = nengo_dl.tensor_layer(x, tf.layers.dense, units=" + str(dense_num) + ")\n"
        if neuron_type != "softmax":
            if neuron_type == "lif" :
                code += "\tx = nengo_dl.tensor_layer(x, nengo.LIF(amplitude=" + str(self.amplitude) + "))\n"
            elif neuron_type == "lifrate":
                code += "\tx = nengo_dl.tensor_layer(x, nengo.LIFRate)(amplitude=" + str(self.amplitude) + "))\n"
            elif neuron_type == "adaptivelif":
                code += "\tx = nengo_dl.tensor_layer(x, nengo.AdaptiveLIF(amplitude=" + str(self.amplitude) + "))\n"
            elif neuron_type == "adaptivelifrate":
                code += "\tx = nengo_dl.tensor_layer(x, nengo.AdaptiveLIFRate(amplitude=" + str(self.amplitude) + "))\n"
            elif neuron_type == "izhikevich":
                code += "\tx = nengo_dl.tensor_layer(x, nengo.Izhikevich(amplitude=" + str(self.amplitude) + "))\n"
            elif neuron_type == "softlifrate":
                code += "\tx = nengo_dl.tensor_layer(x, nengo_dl.neurons.SoftLIFRate(amplitude=" + str(self.amplitude) + "))\n"
            elif neuron_type == None:   #default neuron_type = LIF
                code += "\tx = nengo_dl.tensor_layer(x, default_neuron_type)\n"
        code += "\n"
        output_shape = [dense_num, 1]
        return code, output_shape
    
    def gen_probe(self, synapse = 0.01):
        code = ""
        code += "\tout_p = nengo.Probe(x)\n"
        code += "\tout_p_filt = nengo.Probe(x, synapse=" + str(synapse) + ")\n\n"
        return code

    def gen_classification(self):
        code = ""
        code += "def objective(outputs, targets):\n"
        code += "\treturn tf.nn.softmax_cross_entropy_with_logits_v2(logits=outputs, labels=targets)\n\n"
        code += "def classification_error(outputs, targets):\n"
        code += "\treturn 100 * tf.reduce_mean(tf.cast(tf.not_equal(tf.argmax(outputs[:, -1], axis=-1), tf.argmax(targets[:, -1], axis=-1)), tf.float32))\n\n"
        return code

    def gen_train(self, batch_size, steps, epochs, learning_rate, device):
        code = ""
        code += "minibatch_size = " + str(batch_size) + "\n"
        code += "sim = nengo_dl.Simulator(net, minibatch_size=minibatch_size" + ", device = \"/" + str(device) + ":0\")\n\n"
        code += "train_data = {inp: train_data[0][:, None, :], out_p: train_data[1][:, None, :]}\n\n"
        code += "n_steps = " + str(steps) + "\n"
        code += "test_data = {inp: np.tile(test_data[0][:minibatch_size*2, None, :], (1, n_steps, 1)), out_p_filt: np.tile(test_data[1][:minibatch_size*2, None, :], (1, n_steps, 1))}\n\n"
        code += "print(\"error before training: %.2f%%\" % sim.loss(test_data, {out_p_filt: classification_error}))\n"
        code += "opt = tf.train.RMSPropOptimizer(learning_rate=" + str(learning_rate) + ")\n"
        code += "sim.train(train_data, opt, objective={out_p: objective}, n_epochs=" + str(epochs) + ")\n"
        code += "sim.save_params(\"./model_params\")\n"
        code += "print(\"error after training: %.2f%%\" % sim.loss(test_data, {out_p_filt: classification_error}))\n\n"
        code += "sim.close()"
        return code

    def get_neuronType(self, node_index, onnx_model_graph_node):
        node_len = len(onnx_model_graph_node)
        for index in range(node_index, node_len):
            node_info = onnx_model_graph_node[index]
            op_type = node_info.op_type.lower()
            if op_type == "lif" or op_type == "lifrate" or op_type == "adaptivelif" or op_type == "adaptivelifrate" or op_type == "izhikevich" or op_type == "softmax" or op_type == "softlifrate":
                return op_type
        return None

    def makefile(self, result_path):
        file = open(result_path, 'w', encoding="utf-8")
        file.write(self.get_nengoCode())
        file.close()

    def get_nengoCode(self):
        return self.nengoCode

    def get_filterNum(self, node_info, onnx_model_graph):
        weight_name = node_info.input[1]
        for m in range(len(onnx_model_graph.initializer)):
            name = onnx_model_graph.initializer[m].name
            for n in range(len(node_info.input)):
                if node_info.input[n] == name and name == weight_name:
                    shape = onnx_model_graph.initializer[m].dims
                    return shape[0]
        return None

    def get_dense_num(self, node_info, onnx_model_graph):
        weight_name = node_info.input[1]
        for m in range(len(onnx_model_graph.initializer)):
            name = onnx_model_graph.initializer[m].name
            for n in range(len(node_info.input)):
                if node_info.input[n] == name and name == weight_name:
                    shape = onnx_model_graph.initializer[m].dims
                    return shape[1]
        return None

    def print_node(self):
        print(self.onnx_model.graph.node)
        
if __name__ == "__main__":
    otn = toNengoCode("../model/onnx2snn/lenet2snn.onnx")
    otn.makefile("../result/nengo_code.py")