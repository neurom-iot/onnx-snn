#onnx to nengo convert codeimport nengo
import nengo_dl
import numpy as np

with nengo.Network() as net:
	net.config[nengo.Ensemble].max_rates = nengo.dists.Choice([100])
	net.config[nengo.Ensemble].intercepts = nengo.dists.Choice([0])
	neuron_type = nengo.LIF(amplitude=0.01)
	nengo_dl.configure_settings(trainable=False)
	
	inp = nengo.Node([0] * 28 * 1)
	x = nengo_dl.tensor_layer(inp, tf.layers.conv2d, shape_in=(28, 28, 1), filters=32, kernel_size=3)
	x = nengo_dl.tensor_layer(x, neuron_type)
	x = nengo_dl.tensor_layer(inp, tf.layers.conv2d, shape_in=(26, 26, 32), filters=64, kernel_size=3)
	x = nengo_dl.tensor_layer(x, neuron_type)
	x = nengo_dl.tensor_layer(x, tf.layers.average_pooling2d, shape_in=(24, 24, 64), pool_size=2, strides=2)
flatten
matmul
matmul
