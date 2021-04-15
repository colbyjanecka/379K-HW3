import numpy as np
import onnxruntime
from tqdm import tqdm
import os
import time
from PIL import Image
import argparse
import psutil
import telnetlib as tel
import gpiozero
from measurement import getTelnetPower

dict(psutil.virtual_memory()._asdict())
pre_inference_memory = psutil.virtual_memory().used/1000000
print('current memory usage (pre-inference):', psutil.virtual_memory().used/1000000, 'MB')
time.sleep(1)

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

total_time = 0

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

# create a text file to log the results
out_fname = '/home/pi/pi_log_' + model + "_" + framework + '.txt'
header = "time W temp"
header = "\t".join( header.split(' ') )
out_file = open(out_fname, 'w')
out_file.write(header)
out_file.write("\n")
SP2_tel = tel.Telnet("192.168.4.1")
total_power = 0

# The test_deployment folder contains all 10.000 images from the testing dataset of CIFAR10 in .png format
for filename in tqdm(os.listdir("test_deployment")):
    # Take each image, one by one, and make inference
    with Image.open(os.path.join("test_deployment", filename)).resize((32, 32)) as img:
        #print("Image shape:", np.float32(img).shape)

        if framework == 'pt':
        # For PyTorch models ONLY: normalize image
            input_image = (np.float32(img) / 255. - mean) / std
        # For PyTorch models ONLY: Add the Batch axis in the data Tensor (C, H, W)
            input_image = np.expand_dims(np.float32(input_image), axis=0)

        if framework == 'tf':
        # For TensorFlow models ONLY: Add the Batch axis in the data Tensor (H, W, C)
            input_image = np.expand_dims(np.float32(img), axis=0)
            #print("Image shape after expanding size:", input_image.shape)

        if framework == 'pt':
        # For PyTorch models ONLY: change the order from (B, H, W, C) to (B, C, H, W)
            input_image = input_image.transpose([0, 3, 1, 2])

        # Get start time before inference
        start_time = time.time()

        # Run inference and get the prediction for the input image
        pred_onnx = sess.run(None, {input_name: input_image})[0]

        # Add inference time to total time
        total_time = total_time + (time.time() - start_time)

        # Find the prediction with the highest probability
        top_prediction = np.argmax(pred_onnx[0])

        # Get the label of the predicted class
        pred_class = label_names[top_prediction]

        true_label = filename.split('_')[1].split('.')[0]


    # after inference, save the statistics for cpu usage and power consumption
    last_time = time.time()#time_stamp
    total_power = getTelnetPower(SP2_tel, total_power)
    
    cpu_temp = gpiozero.CPUTemperature().temperature
    
    time_stamp = last_time
    fmt_str = "{}\t"*3
    out_ln = fmt_str.format(time_stamp, total_power, cpu_temp)    
    out_file.write(out_ln)
    out_file.write("\n")

print(total_correct/total*100)
print("Total Inference Time: ", total_time)
print('inference memory usage:', cumulative_memory/count)

    

