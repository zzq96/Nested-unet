import os
import torch
from torch import nn
import random
import numpy as np
import PIL
from PIL import Image
import numbers
import torchvision
from torchvision.datasets.vision import VisionDataset
from torchvision.transforms import functional as F

def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed) # cpu
    torch.cuda.manual_seed_all(seed)  # gpu
    torch.backends.cudnn.deterministic = True  # consistent results on the cpu and gpu

def str2bool(v):
    if v.lower() in ['true', 1]:
        return True
    elif v.lower() in ['false', 0]:
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def load_checkpoint(model, checkpoint_PATH, epoch = None,  best_iou = None, optimizer = None):

    model_CKPT = torch.load(checkpoint_PATH)
    if epoch is not None:
        epoch = model_CKPT['epoch']
    model.load_state_dict(model_CKPT['state_dict'])
    print('loading checkpoint: state_dict!')
    if epoch is not None:
        best_iou = model_CKPT['best_iou']
        print('loading checkpoint: best_iou!')
    if epoch is not None:
        optimizer.load_state_dict(model_CKPT['optimizer'])
        print('loading checkpoint: optimizer!')
    # if epoch is not None:
    #     scheduler.load_state_dict(model_CKPT['scheduler'])
    #     print('loading checkpoint: scheduler!')

    return model, epoch, best_iou, optimizer


#现在只实现了横向拼接
def PILImageConcat(imgs):
    for i in range(1, len(imgs)):
        if imgs[0].size != imgs[1].size or imgs[0].mode != imgs[1].mode:
            raise RuntimeError("拼接图像模式or长宽不匹配!")

    # 单幅图像尺寸
    width, height = imgs[0].size

    # 创建空白长图
    # result = Image.new(ims[0].mode, (width, height * len(ims)))
    result = Image.new(imgs[0].mode, (width * len(imgs), height))

    # 拼接图片
    for i, img in enumerate(imgs):
        result.paste(img, box=(i * width,0 ))

    return result

def init_weights(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        # nn.init.kaiming_uniform_(m.weight.data, nonlinearity='relu')
        nn.init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()
        
def get_upsampling_weight(in_channels, out_channels, kernel_size):
    """造一个双线性插值卷积核"""
    """通过测试"""
    factor = (kernel_size + 1)//2 # 采样因子
    if kernel_size % 2 == 1:
        center = factor - 1 #采样点
    else:
        center = factor - 0.5
    
    og = np.ogrid[:kernel_size, :kernel_size]
    filt = (1 - abs(og[0] - center) / factor) * (1 - abs(og[1] - center) / factor) #根据像素点离采样点的远近分配权重
    weight = np.zeros((in_channels, out_channels, kernel_size, kernel_size), dtype=np.float64)
    weight[range(in_channels), range(out_channels), :, :] = filt
    return torch.from_numpy(weight)

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count