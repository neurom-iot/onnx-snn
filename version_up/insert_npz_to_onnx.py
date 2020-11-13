# import numpy as np
# import onnx
# from onnx import numpy_helper
# from onnx import helper

'''
    step 2.
    This code is required to config SNN's weights information, also it does not perform
    other functions because it just checks information with print() but it is very important step.
'''

import numpy as np
import onnx
from onnx import numpy_helper
from onnx import helper

x = np.load('./model/nengo-snn_convolutiontest.npz')

for k in x.iterkeys():
    print(k)

arr0 = x.get('arr_0')
arr1 = x.get('arr_1')
arr2 = x.get('arr_2')
arr3 = x.get('arr_3')
arr4 = x.get('arr_4')
arr5 = x.get('arr_5')
arr6 = x.get('arr_6')
arr7 = x.get('arr_7')
arr8 = x.get('arr_8')
arr9 = x.get('arr_9')
arr10 = x.get('arr_10')

print('layer 0 : ', len(arr0), arr0.shape, type(arr0))
print('layer 1 : ', len(arr1), arr1.shape, type(arr1))
print('layer 2 : ', len(arr2), arr2.shape, type(arr2))
print('layer 3 : ', len(arr3), arr3.shape, type(arr3))
print('layer 4 : ', len(arr4), arr4.shape, type(arr4))
print('layer 5 : ', len(arr5), arr5.shape, type(arr5))
print('layer 6 : ', len(arr6), arr6.shape, type(arr6))
print('layer 7 : ', len(arr7), arr7.shape, type(arr7))
print('layer 8 : ', len(arr8), arr8.shape, type(arr8))
print('layer 9 : ', len(arr9), arr9.shape, type(arr9))
print('layer 10 : ', len(arr10), arr10.shape, type(arr10))

k = onnx.load('./model/model_convolution.onnx')

layers = k.graph.initializer

for i in layers:
    print(i.dims)

npz_first_layer = arr3

print(arr3[0])
print(len(arr3[0][0]))

count = 0
for data in npz_first_layer:
    for indi in data:
        for more in indi:
            count += 1
            for deep in more:
                pass
print('count : ', count)
print('shape : ', npz_first_layer.shape)

bias = arr2
print(bias)

# print(arr0[0])
# print('-----------')
# print(arr1[0])
# print('-----------')
# print(arr2[0])
# print('-----------')
# print(arr3[0])
# print('-----------')
# print(arr4[0])
# print('-----------')
# print(arr5)

#
# import tensorlayer
# tensorlayer.files.save_npz()
# model = onnx.load('temp_onnx_model.onnx')
#
# with open('onnx_graph.txt', 'w') as f:
#     f.write(str(model.graph))
#
# print(onnx.helper.printable_graph(model.graph))
#
#
# onnx.checker.check_model(model)
# print('the model is checked!')
#
# weights = model.graph.initializer
# print(weights)
#
# w1 = numpy_helper.to_array(weights[0])
# print('layer 1 : ', w1)
#
# x = np.load('mnist_params.npz')
# for k in x.iterkeys():
#     print(k)
#
# layer_1 = x.get('arr_0')
# layer_2 = x.get('arr_1')
# layer_3 = x.get('arr_2')
# layer_4 = x.get('arr_3')
# layer_5 = x.get('arr_4')
# layer_6 = x.get('arr_5')
# layer_7 = x.get('arr_6')
# layer_8 = x.get('arr_7')
# layer_9 = x.get('arr_8')
#
# # print('layer 1 : ', layer_1, len(layer_1), type(layer_1))
# # print('layer 2 : ', layer_2, len(layer_2), type(layer_2))
# # print('layer 3 : ', layer_3, len(layer_3), type(layer_3))
# # print('layer 4 : ', layer_4, len(layer_4), type(layer_4))
# # print('layer 5 : ', layer_5, len(layer_5), type(layer_5))
# # print('layer 6 : ', layer_6, len(layer_6), type(layer_6))
# # print('layer 7 : ', layer_7, len(layer_7), type(layer_7))
# # print('layer 8 : ', layer_8, len(layer_8), type(layer_8))
# # print('layer 9 : ', layer_9, len(layer_9), type(layer_9))
#
# print('layer 1 : ', len(layer_1), type(layer_1))
# print('layer 2 : ', len(layer_2), type(layer_2))
# print('layer 3 : ', len(layer_3), type(layer_3))
# print('layer 4 : ', len(layer_4), type(layer_4))
# print('layer 5 : ', len(layer_5), type(layer_5))
# print('layer 6 : ', len(layer_6), type(layer_6))
# print('layer 7 : ', len(layer_7), type(layer_7))
# print('layer 8 : ', len(layer_8), type(layer_8))
# print('layer 9 : ', len(layer_9), type(layer_9))
#
# for x in layer_2:
#     print('x : ', x)
#
#     for y in x:
#         print('y : ', y)
#
#         for z in y:
#             print('z : ', z)
#
#
# npz_counter = 0
# onnx_counter = 0
#
# graph_info = model.graph.initializer
#
# print(graph_info)
#
# for g in graph_info:
#     if str(g).__contains__('float_data'):
#         onnx_counter += 1
#
# print(onnx_counter)