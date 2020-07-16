import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import numpy as np
import os

os.environ["CUDA_VISIBLE_DEVICES"] = '1, 0'
print(torch.cuda.device_count())