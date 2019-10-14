from keras.datasets import mnist
from keras import models
from keras import layers
from keras.utils import to_categorical
from keras.models import load_model
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten
# from keras.applications.resnet50 import ResNet50

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# 이미지 데이터 준비하기 (모델에 맞는 크기로 바꾸고 0과 1사이로 스케일링)
train_images = train_images.reshape((60000, 28 * 28))
train_images = train_images.astype('float32') / 255
test_images = test_images.reshape((10000, 28 * 28))
test_images = test_images.astype('float32') / 255

# 레이블을 범주형으로 인코딩
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

# 모델 정의하기 (여기에서는 Sequential 클래스 사용)
model = models.Sequential()
model.add(layers.Dense(512, activation='relu', input_shape=(28 * 28,)))
model.add(layers.Dense(10, activation='softmax'))

model.compile(optimizer='rmsprop',
                loss='categorical_crossentropy',
                metrics=['accuracy'])

# # fit() 메서드로 모델 훈련 시키기
# model.fit(train_images, train_labels, epochs=5, batch_size=128)

# # 테스트 데이터로 정확도 측정하기
# test_loss, test_acc = model.evaluate(test_images, test_labels)

# print('test_acc: ', test_acc)

# 모델 저장
model.save("../model/dnn/dense_model(keras).h5")