"""
    VGG11 model written in TensorFlow Keras
"""
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Conv2D, Dense, AveragePooling2D, Flatten, BatchNormalization, \
    DepthwiseConv2D, Dropout, MaxPooling2D


def VGG11_Sequential():
    model = Sequential()

    #layer1 = Sequential()
    model.add(Conv2D(64, kernel_size=(3,3), padding='same', input_shape=(32, 32, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

    #layer2 = Sequential()
    model.add(Conv2D(128, kernel_size=(3,3), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

    #layer3 = Sequential()
    model.add(Conv2D(256, kernel_size=(3,3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(256, kernel_size=(3,3), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

    #layer4 = Sequential()
    model.add(Conv2D(512, kernel_size=(3,3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(512, kernel_size=(3,3), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

    #layer5 = Sequential()
    model.add(Conv2D(512, kernel_size=(3,3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(512, kernel_size=(3,3), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

    #layer6 = Sequential()
    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))

    model.add(Dense(512))
    model.add(Activation('relu'))


    model.add(Dense(10))
    model.add(Activation('softmax'))

    '''
    model.add(layer1)
    model.add(layer2)
    model.add(layer3)
    model.add(layer4)
    model.add(layer5)
    model.add(layer6)
    '''
    return model

#model = VGG11_Sequential()
#model.build((None, 32, 32, 3))
#model.summary()