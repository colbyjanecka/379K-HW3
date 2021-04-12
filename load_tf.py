import tensorflow as tf
import argparse
import time
from models.tensorflow import vgg_tf_sequential

model = vgg_tf_sequential.VGG11_Sequential()

model.load_weights("HW3_files/HW3_files/vgg_tf_trained/vgg_tf_trained.ckpt")