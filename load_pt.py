import torch
import torch.nn as nn
from torchsummary import summary
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import argparse
import time
from models.pytorch import vgg_pt

model = vgg_pt.VGG_pt()
model.load_state_dict(torch.load("vgg_pt_trained/vgg_pt_trained.pt"))


print(model.eval())