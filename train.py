from keras import datasets
from keras import utils

from model import MnasNet

input_shape = (224, 224)
batch_size = 2048
epochs = 100

model = MnasNet(input_shape=input_shape+(3,), pooling='avg')
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

(x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()
y_train = utils.to_categorical(y_train, nb_classes)
y_test = utils.to_categorical(y_test, nb_classes)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          validation_data=(x_test, y_test),
          shuffle=True)
