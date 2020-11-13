# SNN Convolution Layer Version #
# Convolution Layer was trained by nengo_dl.simulator,
# Unfortunately, ONNX is not supporting SNN activation function, this temporary converting program uses DNN activation function named Relu.
# However, It will be enable to that the final version will be not using DNN activation function such as Relu, but will be supporting SNN activation function in ONNX.

# Current status of task : It supports fully connected layer and convolution layer.
# To do : Support for other operations(BatchNormalization, Maxpool, and so on..)
#         Reform the code for semi-automate about conversion task with refactoring.

# ! Temporary version is not automatically converting SNN model to ONNX model.
import numpy as np
import onnx

'''
    step 5.
    Weights_writer.py perform converting SNN to DNN.
    For more information, please see the captions below
'''

ONNX_MODEL_PATH = './model/model_convolution.onnx'
NEW_ONNX_MODEL_PATH = './model/adjusted_model_convolution.onnx'

snn_weights = np.load('./nengo-snn_convolutiontest.npz')

def insert_snn_weight_2_onnx_model_convolution(snn_layer, onnx_layer, channel_dims):
    '''
    this method insert snn weigths to onnx model, which supports convolution neural network.

    For example, [3, 3, 1, 32] is means that's convolution layer
    so snn_layer and onnx_layer must be consisted of [None, None, None, None]

    :param snn_layer: snn layer about convolution layer
    :param onnx_layer: onnx layer about convolution layer
    :param channel_dims: image's color channel -> gray = 1, RGB = 3, etc...
    '''

    print('===========================================================')
    print('                 SNN shape and ONNX shape')
    print('!! Users must check that the two layer are the same shape !!')
    print('SNN shape  : ', snn_layer.shape)
    print('ONNX shape : ', onnx_layer.dims)

    # This paramether is very important, This parameter count each layer's weight number.
    # If they are not the same number, it is means the weights are inserted in the wrong layer.
    # Thus raising the error, and the programmer must recheck shape both of onnx and snn.
    onnx_counter = 0
    snn_counter = 0

    # Remind that convolution is consist of four dimension.
    one_snn_weight = snn_layer[channel_dims-1][0]
    one_snn_weight = one_snn_weight[0][0]
    print('Before :      snn layer [1] : ', one_snn_weight)
    print('Before : onnx.float_data[1] : ', onnx_layer.float_data[5])

    for i in range(0, channel_dims):
        # ONNX's convolution has the channel ordering set by reverse, it should be carefully.
        one_channel_weights = snn_layer[i]

        print('Shape : ', one_channel_weights.shape)

        for o in one_channel_weights:
            for filters in o:
                for individual in filters:
                    onnx_layer.float_data[onnx_counter] = individual
                    onnx_counter += 1
                    snn_counter += 1

    print('Onnx weights count : ', onnx_counter)
    print('Snn weigths count :', snn_counter)

    if not onnx_counter == snn_counter:
        raise Exception('Please check weights parameter count in onnx and snn. Are they corresponding layer to each other?')

    print('After :  onnx.float_data[1] : ', onnx_layer.float_data[0])
    print('Inserting convolution layer is done..')
    print('===========================================================')

def insert_snn_weight_2_onnx_model_fully_connected(snn_layer, onnx_layer):
    '''
    this method insert snn weigths to onnx model, which supports fully connected layer.

    For example, [128, 10] flattened to 1280, so snn_layer and onnx_layer must be consisted of [None, None]
    :param snn_layer: snn layer about fully connected layer
    :param onnx_layer: onnx layer about fully connected layer

    '''
    print('===========================================================')
    print('                 SNN shape and ONNX shape')
    print('!! Users must check that the two layer are the same shape !!')
    print('SNN shape  : ', snn_layer.shape)
    print('ONNX shape : ', onnx_layer.dims)

    count = 0

    print('Before :      snn layer [1] : ', snn_layer[0][1])
    print('Before : onnx.float_data[1] : ', onnx_layer.float_data[1])

    for i in range(0, len(snn_layer)):
        weights = snn_layer[i]

        for weight in weights:
            onnx_layer.float_data[count] = weight
            count += 1

    print('After :  onnx.float_data[1] : ', onnx_layer.float_data[1])
    print('Inserting fully connected layer is done..')
    print('===========================================================')


def insert_snn_weight_2_onnx_model_biases_channels(snn_layer, onnx_layer):
    '''
        this method insert snn weigths to onnx model, which supports bias or channel.

        For example, snn layer (10,) corresponds onnx layer [10,] , so snn_layer and onnx_layer must be consisted of [None,]
        :param snn_layer: snn layer about bias or channel
        :param onnx_layer: onnx layer about bias or channel

    '''
    print('===========================================================')
    print('                 SNN shape and ONNX shape')
    print('!! Users must check that the two layer are the same shape !!')
    print('SNN shape  : ', snn_layer.shape)
    print('ONNX shape : ', onnx_layer.dims)

    print('Before :      snn layer [1] : ', snn_layer[1])
    print('Before : onnx.float_data[1] : ', onnx_layer.float_data[1])
    for i in range(0, len(snn_layer)):
        weight = snn_layer[i]

        onnx_layer.float_data[i] = weight

    print('After :  onnx.float_data[1] : ', onnx_layer.float_data[1])
    print('Inserting bias or channel layer is done..')
    print('===========================================================')


# ----------- snn weight components display -----------
for k in snn_weights.keys():
    print(k)
# -----------------------------------------------------

# ------ each layer weight information extraction ------
layer0 = snn_weights.get('arr_0')
layer1 = snn_weights.get('arr_1')
layer2 = snn_weights.get('arr_2')
layer3 = snn_weights.get('arr_3')
layer4 = snn_weights.get('arr_4')
layer5 = snn_weights.get('arr_5')
layer6 = snn_weights.get('arr_6')
layer7 = snn_weights.get('arr_7')
layer8 = snn_weights.get('arr_8')
layer9 = snn_weights.get('arr_9')
layer10 = snn_weights.get('arr_10')

# # It is very important to check numpy arrays shape
print('layer 0 : ', len(layer0), layer1.shape, type(layer0))
print('layer 1 : ', len(layer1), layer2.shape, type(layer1))
print('layer 2 : ', len(layer2), layer3.shape, type(layer2))
print('layer 3 : ', len(layer3), layer4.shape, type(layer3))
print('layer 4 : ', len(layer4), layer5.shape, type(layer4))
print('layer 5 : ', len(layer5), layer6.shape, type(layer5))
print('layer 6 : ', len(layer6), layer7.shape, type(layer6))
print('layer 7 : ', len(layer7), layer8.shape, type(layer7))
print('layer 8 : ', len(layer8), layer9.shape, type(layer8))
print('layer 9 : ', len(layer9), layer10.shape, type(layer9))
print('layer 10 : ', len(layer10), layer10.shape, type(layer10))

# onnx model load
onnx_model = onnx.load_model(ONNX_MODEL_PATH)

# onnx_graph information extraction
onnx_weights = onnx_model.graph.initializer

# Also, Checking ONNX model's weights shape is very important,
# this is because they have some different shape both.
# -------------------------------------
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

print('** Checking ONNX weights dims... **')
for i in onnx_weights:
    print(i.dims)
print('** Done.... **')

# !! It is possible to directly modify float_data in onnx_model.graph.initializer though python code

## Modifying ONNX model's first layer
# This corresponds to the last of second layer of SNN
snn_first_layer = layer9
onnx_first_layer = onnx_weights[0]
insert_snn_weight_2_onnx_model_fully_connected(snn_first_layer, onnx_first_layer)

# 2
snn_second_layer = layer10
onnx_second_layer = onnx_weights[1]
insert_snn_weight_2_onnx_model_biases_channels(snn_second_layer, onnx_second_layer)

# 3
snn_third_layer = layer7
onnx_third_layer = onnx_weights[2]
insert_snn_weight_2_onnx_model_fully_connected(snn_third_layer, onnx_third_layer)

# 4
snn_fourth_layer = layer8
onnx_fourth_layer = onnx_weights[3]
insert_snn_weight_2_onnx_model_biases_channels(snn_fourth_layer, onnx_fourth_layer)

# 5
snn_fifth_layer = layer5
onnx_fifth_layer = onnx_weights[4]
insert_snn_weight_2_onnx_model_convolution(snn_fifth_layer, onnx_fifth_layer, 3)

# # 6
snn_sixth_layer = layer6
onnx_sixth_layer = onnx_weights[5]
insert_snn_weight_2_onnx_model_biases_channels(snn_sixth_layer, onnx_sixth_layer)

# # 7
snn_seventh_layer = layer3
onnx_seventh_layer = onnx_weights[6]
insert_snn_weight_2_onnx_model_convolution(snn_seventh_layer, onnx_seventh_layer, 3)

# # 8
snn_eighth_layer = layer4
onnx_eighth_layer = onnx_weights[7]
insert_snn_weight_2_onnx_model_biases_channels(snn_eighth_layer, onnx_eighth_layer)

# # 9
snn_nineth_layer = layer1
onnx_nineth_layer = onnx_weights[8]
insert_snn_weight_2_onnx_model_convolution(snn_nineth_layer, onnx_nineth_layer, 3)

# 10
snn_tenth_layer = layer2
onnx_tenth_layer = onnx_weights[9]
insert_snn_weight_2_onnx_model_biases_channels(snn_tenth_layer, onnx_tenth_layer)

# ONNX model save.
onnx.save_model(onnx_model, NEW_ONNX_MODEL_PATH)