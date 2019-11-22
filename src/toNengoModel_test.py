import onnxToNengoModel
import numpy as np
import gzip
import pickle
from urllib.request import urlretrieve
import zipfile
import nengo
import nengo_dl
import tensorflow as tf

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

onnx_path = "../model/onnx2snn/cnn2snn.onnx"
otn = onnxToNengoModel.toNengoModel(onnx_path)
model = otn.get_model()
inp = otn.get_inputProbe()
pre_layer = otn.get_endLayer()

with model:
    out_p = nengo.Probe(pre_layer)
    out_p_filt = nengo.Probe(pre_layer, synapse = 0.01)

minibatch_size = 200
sim = nengo_dl.Simulator(model, minibatch_size = minibatch_size, device = "/cpu:0")

train_data = {inp: train_data[0][:, None, :], out_p: train_data[1][:, None, :]}

n_steps = 30
test_data = {inp: np.tile(test_data[0][:minibatch_size*2, None, :], (1, n_steps, 1)), out_p_filt: np.tile(test_data[1][:minibatch_size*2, None, :], (1, n_steps, 1))}

print("error before training: %.2f%%" % sim.loss(test_data, {out_p_filt: classification_error}))
opt = tf.train.RMSPropOptimizer(learning_rate=0.001)
sim.train(train_data, opt, objective={out_p: objective}, n_epochs=10)
sim.save_params("./model_params")
print("error after training: %.2f%%" % sim.loss(test_data, {out_p_filt: classification_error}))
sim.close()