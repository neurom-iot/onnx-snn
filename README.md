# ONNX for SNN(Spiking Neural Networks)

## What is ONNX-SNN
* ONNX is an open format to represent deep learning models.
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
