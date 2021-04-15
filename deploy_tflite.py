from PIL import Image
import numpy as np
import os
from tqdm import tqdm
import tflite_runtime.interpreter as tflite
import argparse
import time
import psutil

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

print("Accuracy", str(total_correct/total*100))
print("Total Inference Time: ", total_time)
print('inference memory usage:', cumulative_memory/count)