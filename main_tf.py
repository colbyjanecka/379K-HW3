import tensorflow as tf
import argparse
import time
from models.tensorflow import vgg_tf_sequential

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
model.build((None, 32, 32, 3))
model.summary()

# TODO: Load the training and testing datasets
df_train, df_test = tf.keras.datasets.cifar10.load_data()
X_train, y_train = df_train
X_test, y_test = df_test
# TODO: Convert the datasets to contain only float values
X_test = tf.convert_to_tensor(X_test, dtype=float)
X_train = tf.convert_to_tensor(X_train, dtype=float)
# TODO: Normalize the datasets
X_train = tf.linalg.normalize(X_train)
X_test = tf.linalg.normalize(X_test)
# TODO: Encode the labels into one-hot format
y_test = tf.one_hot(y_test, 10)
y_train = tf.one_hot(y_train, 10)
y_train = tf.squeeze(y_train)
y_test = tf.squeeze(y_test)
# TODO: Configures the model for training using compile method
loss = "categorical_crossentropy"
model.compile(tf.keras.optimizers.Adam(), loss, tf.keras.metrics.categorical_accuracy)
# TODO: Train the model using fit method
#print(X_train[0].shape, y_train.shape)
start_time = time.time()
print("Started at : ", start_time)

model.fit(X_train[0], y_train, epochs=epochs, batch_size=batch_size)

print("TIMER VALUE (TRAINING): ", time.time() - start_time)

"""

model.fit(X_train[0], y_train, validation_data=(X_test, y_test), epochs=epochs, batch_size=batch_size)
"""

# TODO: Save the weights of the model in .ckpt format
model.save_weights("vgg_tf_trained/vgg_tf_trained")
