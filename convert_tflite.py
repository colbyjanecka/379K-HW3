import tflite_runtime
import tensorflow as tf
from models.tensorflow import vgg_tf_sequential, mobilenet_tf


def export_tf_to_tflite(model, outputPath):

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    print("converted " + outputPath)

    with open('models/tflite/' + outputPath + '.tflite', 'wb') as f:
        f.write(tflite_model)


mbnv1_model = mobilenet_tf.MobileNetv1()
print("created model")
mbnv1_model.load_weights("mbnv1_tf.ckpt")
print("loaded weights")
export_tf_to_tflite(mbnv1_model, "mbnv1")

vgg_model = vgg_tf_sequential.VGG11_Sequential()
print("created model")
vgg_model.load_weights("vgg_tf_trained/vgg_tf_trained")
print("loaded weights")
export_tf_to_tflite(vgg_model, "vgg")