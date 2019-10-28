#onnx to nengo convert code
#training the network using a rate-based approximation
import nengo
import nengo_dl
import numpy as np
import tensorflow as tf

import gzip
import pickle
from urllib.request import urlretrieve
import zipfile

urlretrieve("http://deeplearning.net/data/mnist/mnist.pkl.gz", "mnist.pkl.gz")
with gzip.open("mnist.pkl.gz") as f:
	train_data, _, test_data = pickle.load(f, encoding="latin1")
train_data = list(train_data)
test_data = list(test_data)

for data in (train_data, test_data):
	one_hot = np.zeros((data[0].shape[0], 10))
	one_hot[np.arange(data[0].shape[0]), data[1]] = 1
	data[1] = one_hot

def objective(outputs, targets):
	return tf.nn.softmax_cross_entropy_with_logits_v2(logits=outputs, labels=targets)

def classification_error(outputs, targets):
	return 100 * tf.reduce_mean(tf.cast(tf.not_equal(tf.argmax(outputs[:, -1], axis=-1), tf.argmax(targets[:, -1], axis=-1)), tf.float32))

with nengo.Network() as net:
	net.config[nengo.Ensemble].max_rates = nengo.dists.Choice([100])
	net.config[nengo.Ensemble].intercepts = nengo.dists.Choice([0])
	default_neuron_type = nengo.LIF(amplitude=0.01)
	nengo_dl.configure_settings(trainable=False)

	inp = nengo.Node([0] * 28 * 28 * 1)
	x = inp

	x = nengo_dl.tensor_layer(x, tf.layers.conv2d, shape_in=(28, 28, 1), filters=32, kernel_size=3, padding="same")
	x = nengo_dl.tensor_layer(x, nengo.LIF(amplitude=0.01))

	x = nengo_dl.tensor_layer(x, tf.layers.conv2d, shape_in=(28, 28, 32), filters=64, kernel_size=3, padding="same")
	x = nengo_dl.tensor_layer(x, nengo.LIF(amplitude=0.01))

	x = nengo_dl.tensor_layer(x, tf.layers.average_pooling2d, shape_in=(28, 28, 64), pool_size=2, strides=2)

	x = nengo_dl.tensor_layer(x, tf.layers.flatten)

	x = nengo_dl.tensor_layer(x, tf.layers.dense, units=128)

	x = nengo_dl.tensor_layer(x, tf.layers.dense, units=10)

	out_p = nengo.Probe(x)
	out_p_filt = nengo.Probe(x, synapse=0.01)

	minibatch_size = 200
	sim = nengo_dl.Simulator(net, minibatch_size=200, device = "/cpu:0")

	train_data = {inp: train_data[0][:, None, :], out_p: train_data[1][:, None, :]}

	n_steps = 30
	test_data = {inp: np.tile(test_data[0][:minibatch_size*2, None, :], (1, n_steps, 1)), out_p_filt: np.tile(test_data[1][:minibatch_size*2, None, :], (1, n_steps, 1))}

print("error before training: %.2f%%" % sim.loss(test_data, {out_p_filt: classification_error}))
opt = tf.train.RMSPropOptimizer(learning_rate=0.001)
sim.train(train_data, opt, objective={out_p: objective}, n_epochs=10)
sim.save_params("./mnist_params")
print("error after training: %.2f%%" % sim.loss(test_data, {out_p_filt: classification_error}))

sim.close()