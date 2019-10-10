import onnx_pb2

modelProto = onnx_pb2.ModelProto()
with open('single_relu.onnx', 'rb') as f:
    modelProto.ParseFromString(f.read())

# print(modelProto)

print('ir_version: {}'.format(modelProto.ir_version))
print('graph.name: {}'.format(modelProto.graph.name))
print(modelProto.graph.node)
print(modelProto.graph.input)
print(modelProto.graph.output)
