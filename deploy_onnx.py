import numpy as np
import onnxruntime
from tqdm import tqdm
import os
from PIL import Image
import argparse

# TODO: add argument parser
parser = argparse.ArgumentParser(description='EE379K HW3 - ONNX Deployment code')
# TODO: add one argument for selecting PyTorch or TensorFlow option of the code
parser.add_argument('--framework', type=str, help='Framework to use. Can be PyTorch or Tensorflow (pt or tf respectively)')
# TODO: add one argument for selecting VGG or MobileNet-v1 models
parser.add_argument('--model', type=str, help='Model to use. Can be VGG or MobileNet-v1 (vgg or mbnv1 respectively)')
# TODO: Modify the rest of the code to use those arguments correspondingly
args = parser.parse_args()
framework = str(args.framework)
model = str(args.model)

onnx_model_name = model + "_" + framework + ".onnx" 

# Create Inference session using ONNX runtime
sess = onnxruntime.InferenceSession("models/onnx/" + onnx_model_name)

# Get the input name for the ONNX model
input_name = sess.get_inputs()[0].name
print("Input name  :", input_name)

# Get the shape of the input
input_shape = sess.get_inputs()[0].shape
print("Input shape :", input_shape)

# Mean and standard deviation used for PyTorch models
mean = np.array((0.4914, 0.4822, 0.4465))
std = np.array((0.2023, 0.1994, 0.2010))

# Label names for CIFAR10 Dataset
label_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

total = 0
total_correct = 0

# The test_deployment folder contains all 10.000 images from the testing dataset of CIFAR10 in .png format
for filename in tqdm(os.listdir("test_deployment")):
    # Take each image, one by one, and make inference
    with Image.open(os.path.join("test_deployment", filename)).resize((32, 32)) as img:
        print("Image shape:", np.float32(img).shape)

        if framework == 'pt':
        # For PyTorch models ONLY: normalize image
            input_image = (np.float32(img) / 255. - mean) / std
        # For PyTorch models ONLY: Add the Batch axis in the data Tensor (C, H, W)
            input_image = np.expand_dims(np.float32(input_image), axis=0)

        if framework == 'tf':
        # For TensorFlow models ONLY: Add the Batch axis in the data Tensor (H, W, C)
            input_image = np.expand_dims(np.float32(img), axis=0)
            print("Image shape after expanding size:", input_image.shape)

        if framework == 'pt':
        # For PyTorch models ONLY: change the order from (B, H, W, C) to (B, C, H, W)
            input_image = input_image.transpose([0, 3, 1, 2])

        # Run inference and get the prediction for the input image
        pred_onnx = sess.run(None, {input_name: input_image})[0]

        # Find the prediction with the highest probability
        top_prediction = np.argmax(pred_onnx[0])

        # Get the label of the predicted class
        pred_class = label_names[top_prediction]

        true_label = filename.split('_')[1].split('.')[0]

        if pred_class == true_label:
            total_correct = total_correct + 1
        total = total + 1

print(total_correct/total*100)

    

