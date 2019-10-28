# ONNX for SNN(Spiking Neural Networks)

## What is ONNX-SNN
* ONNX is an open format to represent deep learning models.
   * ONNX protocol - https://github.com/onnx/onnx/blob/master/onnx/onnx.proto
* ONNX-SNN is an extension for Spiking Neural Networks.
* Its main purpose is to translate DNN model to SNN model.

## Features
* ONNX-SNN
    * DNN's neuron -> spike neuron
    * Neuron activation (Sigmoid, Relu, tanh, etc.) -> Spike (LIF, softLIF, etc.)
* NengoDL with ONNX-SNN
    * Build Deep Spiking neural networks with ONNX-SNN
    * Training weights with NengoDL (Rate neuron)
    * Writing the trained model to ONNX-SNN

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
* softmax --> O
* training, simulation --> O
* -->10/29/2019
---
* Apply to different models --> X
* nengo_dl model --> onnx-snn --> onnx --> X
* onnx model, weight --> onnx-snn --> nengo_dl model --> X
---