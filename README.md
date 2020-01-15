# ONNX for SNN(Spiking Neural Networks)

## What is ONNX-SNN
* ONNX is an open format to represent deep learning models.
   * ONNX protocol - https://github.com/onnx/onnx/blob/master/onnx/onnx.proto
* ONNX-SNN is an extension for Spiking Neural Networks.
* Its main purpose is to translate DNN model to SNN model.

## Necessary Libraries
* Tensorflow 1.10.0
* Keras 2.2.2
* Nengo 2.8.0
* NengoDL 2.2.0
* ONNX 1.6.0
* Numpy 1.14.5
* Protobuf 3.6.0
* Cudatoolkit 9.0
* Cudnn 7.6.4

## Features
* ONNX-SNN
    * DNN's neuron -> spike neuron
    * Neuron activation (Sigmoid, Relu, tanh, etc.) -> Spike (LIF, softLIF, etc.)
* NengoDL with ONNX-SNN
    * Build Deep Spiking neural networks with ONNX-SNN
    * Training weights with NengoDL (Rate neuron)
    * Writing the trained model to ONNX-SNN
    
## source
* convert_snnOnnx.py
    * ONNX -> ONNX-SNN
* onnxToNengoCode.py
    * ONNX-SNN -> Nengo Code
* onnxToNengoModel.py
    * ONNX-SNN -> Nengo Model

## Progress
* Converting ONNX of DNN to ONNX-SNN
* Reading ONNX-SNN and building Nengodl code

## To do list
* Training, prediction target data --> Mnist
---
* convert_conv2d - padding(=border_mode) problem --> O
* convert_pool2d - kind deal(Max, Average) --> O
* convert_flatten function --> O
* convert_matmul function --> O
* convert_batchnormalization function --> O
* softmax --> O
* training, simulation --> O
* Run neuron type(LIF, LIFRate, AdaptiveLIF, AdaptiveLIFRate, Izhikevich) --> O
* -->11/13/2019
---
* nengo_dl support only Sequential network(ex vggnet, alexnet)
---
* Apply to different models(vgg16, vgg19, alexnet) --> O
* nengo_dl model -> onnx-snn -> onnx -> keras model --> X
* onnx model, weight -> onnx-snn -> nengo_dl model --> X
---
