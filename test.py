from torch.utils.tensorboard import SummaryWriter  
import numpy as np  
from torchvision import transforms

img_batch = np.zeros((16, 3, 100, 100))  
for i in range(16):  
    img_batch[i, 0] = np.arange(0, 10000).reshape(100, 100) / 10000 / 16 * i  
    img_batch[i, 1] = (1 - np.arange(0, 10000).reshape(100, 100) / 10000) / 16 * i  

writer = SummaryWriter()  
writer.add_images('my_image_batch', img_batch, 0)  
writer.close()