import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import numpy as np
import os
from PIL import Image

img_dir = './SegmentationClassAug'
palette = Image.open('Datasets/VOC2012/SegmentationClass/2007_000032.png').getpalette()
for filename in os.listdir(img_dir):
    if 'png' not in filename:
        continue
    img = Image.open(os.path.join(img_dir, filename))

    img.putpalette(palette)
    img.save(os.path.join(img_dir, filename))