import numpy as np
import onnxruntime
from tqdm import tqdm
import os
from PIL import Image

# TODO: add argument parser

# TODO: add one argument for selecting PyTorch or TensorFlow option of the code

# TODO: add one argument for selecting VGG or MobileNet-v1 models

# TODO: Modify the rest of the code to use those arguments correspondingly

onnx_model_name = "" # TODO: insert ONNX model name

# Create Inference session using ONNX runtime
sess = onnxruntime.InferenceSession(onnx_model_name)

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

# The test_deployment folder contains all 10.000 images from the testing dataset of CIFAR10 in .png format
for filename in tqdm(os.listdir("HW3_files/test_deployment")):
    # Take each image, one by one, and make inference
    with Image.open(os.path.join("HW3_files/test_deployment", filename)).resize((32, 32)) as img:
        print("Image shape:", np.float32(img).shape)

        # For PyTorch models ONLY: normalize image
        # input_image = (np.float32(img) / 255. - mean) / std
        # For PyTorch models ONLY: Add the Batch axis in the data Tensor (C, H, W)
        # input_image = np.expand_dims(np.float32(input_image), axis=0)

        # For TensorFlow models ONLY: Add the Batch axis in the data Tensor (H, W, C)
        input_image = np.expand_dims(np.float32(img), axis=0)
        print("Image shape after expanding size:", input_image.shape)

        # For PyTorch models ONLY: change the order from (B, H, W, C) to (B, C, H, W)
        # input_image = input_image.transpose([0, 3, 1, 2])

        # Run inference and get the prediction for the input image
        pred_onnx = sess.run(None, {input_name: input_image})[0]

        # Find the prediction with the highest probability
        top_prediction = np.argmax(pred_onnx[0])

        # Get the label of the predicted class
        pred_class = label_names[top_prediction]
