import torch
import torch.nn as nn
from torchsummary import summary
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import argparse
import time
from models.pytorch import vgg_pt

model = vgg_pt.VGG_pt()
model.load_state_dict(torch.load("HW3_files/HW3_files/vgg_pt_trained.pt", map_location=torch.device('cpu')))


print(model.eval())