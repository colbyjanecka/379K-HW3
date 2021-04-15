from PIL import Image
import numpy as np
import os
from tqdm import tqdm
import tflite_runtime.interpreter as tflite
import argparse
import time
import psutil
import telnetlib as tel
from measurement import getTelnetPower, getTemps

dict(psutil.virtual_memory()._asdict())
pre_inference_memory = psutil.virtual_memory().used/1000000
print('current memory usage (pre-inference):', psutil.virtual_memory().used/1000000, 'MB')
time.sleep(1)

# TODO: add argument parser
parser = argparse.ArgumentParser(description='EE379K HW3 - TFLite Deployment code')
# TODO: add one argument for selecting VGG or MobileNet-v1 models
parser.add_argument('--model', type=str, help='Model to use. Can be VGG or MobileNet-v1 (vgg or mbnv1 respectively)')
# TODO: Modify the rest of the code to use the arguments correspondingly
args = parser.parse_args()
model = str(args.model)

tflite_model_name = model + ".tflite"  # TODO: insert TensorFlow Lite model name

# Get the interpreter for TensorFlow Lite model
interpreter = tflite.Interpreter(model_path="models/tflite/" + tflite_model_name)

# Very important: allocate tensor memory
interpreter.allocate_tensors()

# Get the position for inserting the input Tensor
input_details = interpreter.get_input_details()
# Get the position for collecting the output prediction
output_details = interpreter.get_output_details()

# Label names for CIFAR10 Dataset
label_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

total = 0
total_correct = 0
count = 0
cumulative_memory = 0
total_time = 0

# create a text file to log the results
out_fname = '/home/odroid/mc1_log_' + model + '_tflite.txt'
header = "time W temp4 temp5 temp6 temp7"
header = "\t".join( header.split(' ') )
out_file = open(out_fname, 'w')
out_file.write(header)
out_file.write("\n")
SP2_tel = tel.Telnet("192.168.4.1")
total_power = 0
true_start = time.time()

for filename in tqdm(os.listdir("test_deployment")):
    with Image.open(os.path.join("test_deployment", filename)).resize((32, 32)) as img:
        input_image = np.expand_dims(np.float32(img), axis=0)

        # Set the input tensor as the image
        interpreter.set_tensor(input_details[0]['index'], input_image)

        # Get start time before inference
        start_time = time.time()

        # Run the actual inference
        interpreter.invoke()

        # Add inference time to total time
        total_time = total_time + (time.time() - start_time)

        # Get the output tensor
        pred_tflite = interpreter.get_tensor(output_details[0]['index'])

        # Find the prediction with the highest probability
        top_prediction = np.argmax(pred_tflite[0])

        # Get the label of the predicted class
        pred_class = label_names[top_prediction]

        # get true label
        true_label = filename.split('_')[1].split('.')[0]

        # increment total correct if needed
        if pred_class == true_label:
            total_correct = total_correct + 1
        total = total + 1

        dict(psutil.virtual_memory()._asdict())
        cumulative_memory = cumulative_memory + (psutil.virtual_memory().used/1000000 - pre_inference_memory)
        count = count + 1

    # after inference, save the statistics for cpu usage and power consumption
    last_time = time.time()#time_stamp
    total_power = getTelnetPower(SP2_tel, total_power)
    temps = getTemps()
    time_stamp = last_time
    fmt_str = "{}\t"*6
    out_ln = fmt_str.format(time_stamp, total_power, \
			temps[0], temps[1], temps[2], temps[3])    
    out_file.write(out_ln)
    out_file.write("\n")

#Loop to keep gathering data for a total of 20 min
while ((time.time() - true_start) < 1200):  
    # after inference, save the statistics for cpu usage and power consumption
    last_time = time.time()#time_stamp
    total_power = getTelnetPower(SP2_tel, total_power)
    temps = getTemps()
    time_stamp = last_time
    fmt_str = "{}\t"*6
    out_ln = fmt_str.format(time_stamp, total_power, \
			temps[0], temps[1], temps[2], temps[3])    
    out_file.write(out_ln)
    out_file.write("\n")

    elapsed = time.time() - last_time
    DELAY = 0.63
    time.sleep(max(0, DELAY - elapsed))

print("Accuracy: ", str(total_correct/total*100))
print("Total Inference Time: ", total_time)
print('inference memory usage:', cumulative_memory/count)