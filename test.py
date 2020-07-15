import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

input5 = torch.ones(3, 3, 7,7)
input5 = F.interpolate(input=input5, size= (10, 10), mode='bilinear', align_corners=True)
print(input5)