import torch
from models.pytorch import mobilenet_pt
from models.pytorch import vgg_pt
import onnx
import onnxruntime
from models.tensorflow import vgg_tf_sequential, mobilenet_tf
import keras2onnx

def export_pt_to_onnx(model, outputFileName):
    x = torch.randn(1, 3, 32, 32, requires_grad=True)
    torch_out = model(x)
    outputPath = "models/onnx/" + outputFileName + ".onnx"
    torch.onnx.export(model, x, outputPath,  export_params=True,  opset_version=10)

def export_tf_to_onnx(model, outputPath):
    onnx_model = keras2onnx.convert_keras(model, model.name)
    outputPath = "models/onnx/" + outputPath + ".onnx"
    keras2onnx.save_model(onnx_model, outputPath)
    return

model = mobilenet_pt.MobileNetv1()
model.load_state_dict(torch.load("mbnv1_pt.pt", map_location=torch.device('cpu')))
export_pt_to_onnx(model, "mbnv1_pt")

model2 = vgg_pt.VGG_pt()
model2.load_state_dict(torch.load("vgg_pt_trained/vgg_pt_trained.pt", map_location=torch.device('cpu')))
export_pt_to_onnx(model2, "vgg_pt")

mbnv1_model = mobilenet_tf.MobileNetv1()
print("created model")
mbnv1_model.load_weights("mbnv1_tf.ckpt")
print("loaded weights")
export_tf_to_onnx(mbnv1_model, "mbnv1_tf")

vgg_model = vgg_tf_sequential.VGG11_Sequential()
print("created model")
vgg_model.load_weights("vgg_tf_trained/vgg_tf_trained")
print("loaded weights")
export_tf_to_onnx(vgg_model, "vgg_tf")
