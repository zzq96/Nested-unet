import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import numpy as np
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
model = models.resnet101()
model.to("cuda")
print(torch.cuda.device_count())