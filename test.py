from torch.utils.tensorboard import SummaryWriter  
import numpy as np  
from torchvision import transforms
import matplotlib.pyplot as plt
from torchvision.transforms import functional as F
from PIL import Image

img = Image.open("test_imgs/2007_000033.jpg")#RGB模式
target = Image.open("test_imgs/2007_000033.png")#P模式8位调色板模式, 像素值已经是对应类别的编号了
img = F.hflip(img)
target = F.hflip(target)
img.save("1.jpg")
target.save("2.png")