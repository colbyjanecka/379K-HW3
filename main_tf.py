import tensorflow as tf
import argparse
import time
from models.tensorflow import vgg_tf_sequential, mobilenet_tf, vgg_tf
import os
import sys
import numpy as np
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
#tf.debugging.set_log_device_placement(True)
#print(tf.config.list_physical_devices('GPU'), "ffff")
conv_arch = ((1, 64), (1, 128), (2, 256), (2, 512), (2, 512))
# Argument parser
parser = argparse.ArgumentParser(description='EE379K HW3 - Starter TensorFlow code')
# Define the mini-batch size, here the size is 128 images per batch
parser.add_argument('--batch_size', type=int, default=128, help='Number of samples per mini-batch')
# Define the number of epochs for training
parser.add_argument('--epochs', type=int, default=100, help='Number of epoch to train')
args = parser.parse_args()

epochs = args.epochs
batch_size = args.batch_size

random_seed = 1
tf.random.set_seed(random_seed)

# TODO: Insert your model here
model = vgg_tf_sequential.VGG11_Sequential()
nonsequential_model = vgg_tf.vgg_tf(conv_arch)



# TODO: Load the training and testing datasets
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.cifar10.load_data()
# TODO: Convert the datasets to contain only float values
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
# TODO: Normalize the datasets
X_train = X_train/255.0
X_test = X_test/255.0
print(np.min(X_test))
# TODO: Encode the labels into one-hot format
y_train = tf.keras.utils.to_categorical(y_train)
y_test = tf.keras.utils.to_categorical(y_test)


print(X_train.shape, y_train.shape)
#print(X_train[0])
print(y_train)
# TODO: Configures the model for training using compile method
model.compile(tf.keras.optimizers.Adam(learning_rate=0.0001), loss="categorical_crossentropy", metrics=['accuracy'])
# TODO: Train the model using fit method

start_time = time.time()
print("Started at : ", start_time)
"""
model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1)

print("TIMER VALUE (TRAINING): ", time.time() - start_time)

"""

model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs, batch_size=batch_size, verbose=1)

print("TIMER VALUE (TRAINING and VALIDATION): ", time.time() - start_time)

# TODO: Save the weights of the model in .ckpt format
model.save_weights("vgg_tf_trained/vgg_tf_trained")

