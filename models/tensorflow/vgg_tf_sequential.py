"""
    VGG11 model written in TensorFlow Keras
"""
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Conv2D, Dense, AveragePooling2D, Flatten, BatchNormalization, \
    DepthwiseConv2D, Dropout, MaxPool2D


def VGG11_Sequential():
    model = Sequential()

    layer1 = Sequential()
    layer1.add(Conv2D(64, kernel_size=3, padding='same', activation='relu'))
    layer1.add(MaxPool2D(pool_size=2, strides=2))

    layer2 = Sequential()
    layer2.add(Conv2D(128, kernel_size=3, padding='same', activation='relu'))
    layer2.add(MaxPool2D(pool_size=2, strides=2))

    layer3 = Sequential()
    layer3.add(Conv2D(256, kernel_size=3, padding='same', activation='relu'))
    layer3.add(Conv2D(256, kernel_size=3, padding='same', activation='relu'))
    layer3.add(MaxPool2D(pool_size=2, strides=2))

    layer4 = Sequential()
    layer4.add(Conv2D(512, kernel_size=3, padding='same', activation='relu'))
    layer4.add(Conv2D(512, kernel_size=3, padding='same', activation='relu'))
    layer4.add(MaxPool2D(pool_size=2, strides=2))

    layer5 = Sequential()
    layer5.add(Conv2D(512, kernel_size=3, padding='same', activation='relu'))
    layer5.add(Conv2D(512, kernel_size=3, padding='same', activation='relu'))
    layer5.add(MaxPool2D(pool_size=2, strides=2))

    layer6 = Sequential()
    layer6.add(Flatten())
    layer6.add(Dense(512, activation='relu'))
    layer6.add(Dropout(0.5))

    layer6.add(Dense(512, activation='relu'))
    layer6.add(Dropout(0.5))

    layer6.add(Dense(10))

    model.add(layer1)
    model.add(layer2)
    model.add(layer3)
    model.add(layer4)
    model.add(layer5)
    model.add(layer6)

    return model

model = VGG11_Sequential()
model.build((None, 32, 32, 3))
model.summary()