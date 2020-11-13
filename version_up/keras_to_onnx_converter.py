import onnx
import onnxruntime as rt
import keras2onnx
import tensorflow as tf

'''
    step 4.
    The model in step 3 is converted ONNX format.
    
    Why do we have to perform 3 and 4 steps : 
        To reduce opportunity costs as must as possible by using already commercialized modules such as keras2onnx.   
'''

keras_model = tf.keras.models.load_model('model_convolution.h5')

onnx_model = keras2onnx.convert_keras(keras_model, 'model_convolution.onnx')
onnx_file = open('model_convolution.onnx', 'wb')
onnx_file.write(onnx_model.SerializeToString())
onnx_file.close()
