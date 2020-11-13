import onnx
import numpy as np

'''
    step 6.
    Weights_writer.py perform reconverting DNN to SNN.
    For more information, please see the captions below
'''

ONNX_FILE_PATH = './model/adjusted_model_convolution.onnx'

onnx_model = onnx.load_model(ONNX_FILE_PATH)

onnx_weights = onnx_model.graph.initializer

print('** Checking ONNX weights dims... **')
for i in onnx_weights:
    print(i.dims)
print('** Done.... **')

#===========================================#
# ------ Layer shape information ------
#   SNN                       ONNX
# 34832,
# 3, 3, 1, 32               784, 10
# 32,                       10
# 3, 3, 32, 64              3200, 784
# 64,                       784
# 3, 3, 64, 128             128, 64, 3, 3
# 128,                      128
# 3200, 784                 64, 32, 3, 3
# 784,                      64
# 784, 10                   32, 1, 3, 3
# 10                        32
# -------------------------------------
# Caution - Float_data in ONNX model is saved by form of enumeration without distinguishing dimensions.
# So, reconverting reverse-direction must be considered of original dimension shapes.

first_npz = onnx_weights[8].float_data
second_npz = onnx_weights[9].float_data
third_npz = onnx_weights[6].float_data
fourth_npz = onnx_weights[7].float_data
fifth_npz = onnx_weights[4].float_data
sixth_npz = onnx_weights[5].float_data
seventh_npz = onnx_weights[2].float_data
eighth_npz = onnx_weights[3].float_data
nineth_npz = onnx_weights[0].float_data
tenth_npz = onnx_weights[1].float_data

print(type(first_npz), len(first_npz))

first_npz = np.array(first_npz)
first_npz = np.reshape(first_npz, [32, 1, 3, 3])
first_npz = np.reshape(first_npz, [3, 3, 1, 32])
print('numpy : ', first_npz.shape, type(first_npz))

second_npz = np.array(second_npz)
second_npz = np.reshape(second_npz, [32])
print('numpy : ', second_npz.shape, type(second_npz))

third_npz = np.array(third_npz)
third_npz = np.reshape(third_npz, [64, 32, 3, 3])
third_npz = np.reshape(third_npz, [3, 3, 32, 64])
print('numpy : ', third_npz.shape, type(third_npz))

fourth_npz = np.array(fourth_npz)
fourth_npz = np.reshape(fourth_npz, [64])
print('numpy : ', fourth_npz.shape, type(fourth_npz))

fifth_npz = np.array(fifth_npz)
fifth_npz = np.reshape(fifth_npz, [128, 64, 3, 3])
fifth_npz = np.reshape(fifth_npz, [3, 3, 64, 128])
print('numpy : ', fifth_npz.shape, type(fifth_npz))

sixth_npz = np.array(sixth_npz)
sixth_npz = np.reshape(sixth_npz, [128])
print('numpy :', sixth_npz.shape, type(sixth_npz))

seventh_npz = np.array(seventh_npz)
seventh_npz = np.reshape(seventh_npz, [3200, 784])
print('numpy :', seventh_npz.shape, type(seventh_npz))

eighth_npz = np.array(eighth_npz)
eighth_npz = np.reshape(eighth_npz, [784])
print('numpy :', eighth_npz.shape, type(eighth_npz))

nineth_npz = np.array(nineth_npz)
nineth_npz = np.reshape(nineth_npz, [784, 10])
print('numpy :', nineth_npz.shape, type(nineth_npz))

tenth_npz = np.array(tenth_npz)
tenth_npz = np.reshape(tenth_npz, [10])
print('numpy :', tenth_npz.shape, type(tenth_npz))

graph = np.load('./nengo-snn_convolutiontest.npz')

batch_input_graph = graph.get('arr_0')
print(batch_input_graph.shape)

np.savez('nengo-snn_convolutiontest.npz',
         batch_input_graph, first_npz, second_npz, third_npz, fourth_npz, fifth_npz, sixth_npz, seventh_npz, eighth_npz, nineth_npz, tenth_npz)

saved_npz = np.load('nengo-snn_convolutiontest.npz')

for i in saved_npz.keys():
    print(i)

layer1 = saved_npz.get('arr_0')
print(layer1)
print(layer1.shape)