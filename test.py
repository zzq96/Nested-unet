import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchsummary
import numpy as np
import os

resnet = models.resnet50()
torchsummary.summary(resnet, (3, 600, 600), device="cpu")
print(resnet)