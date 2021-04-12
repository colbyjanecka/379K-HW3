import tensorflow as tf
import argparse
import time
from models.tensorflow import vgg_tf_sequential

model = vgg_tf_sequential.VGG11_Sequential()

model.load_weights("vgg_tf_trained/vgg_tf_trained")

model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs, batch_size=batch_size, verbose=1)
