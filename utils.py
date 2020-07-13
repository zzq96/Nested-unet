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
def str2bool(v):
    if v.lower() in ['true', 1]:
        return True
    elif v.lower() in ['false', 0]:
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def load_checkpoint(epoch, model, best_iou, optimizer, scheduler, checkpoint_PATH):

    model_CKPT = torch.load(checkpoint_PATH)
    epoch = model_CKPT['epoch']
    model.load_state_dict(model_CKPT['state_dict'])
    print('loading checkpoint!')
    best_iou = model_CKPT['best_iou']
    optimizer.load_state_dict(model_CKPT['optimizer'])
    scheduler.load_state_dict(model_CKPT['scheduler'])

    return epoch, model, best_iou, optimizer, scheduler

class VOCSegmentation(VisionDataset):
    """`Pascal VOC <http://host.robots.ox.ac.uk/pascal/VOC/>`_ Segmentation Dataset.

    Args:
        root (string): Root directory of the VOC Dataset.
        year (string, optional): The dataset year, supports years 2007 to 2012.
        image_set (string, optional): Select the image_set to use, ``train``, ``trainval`` or ``val``
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        transforms (callable, optional): A function/transform that takes input sample and its target as entry
            and returns a transformed version.
    """

    def __init__(self,
                 root,
                 year='2012',
                 image_set='train', crop_size = (224, 224), use=1):
        """
        use:int, 用1/use的数据集
        """
        #super(VOCSegmentation, self).__init__(root, transforms, transform, target_transform)
        if not os.path.isdir(root):
            raise RuntimeError(root + 'is not a dir')
        self.crop_size = crop_size
        if crop_size:
            self.crop = MultiRandomCrop(crop_size)
        self.To_Tensor = torchvision.transforms.ToTensor()
        self.year = year
        self.image_set = image_set
        self.root = root
        voc_root = os.path.join(self.root, "VOC" + year)
        image_dir = os.path.join(voc_root, 'JPEGImages')
        mask_dir = os.path.join(voc_root, 'SegmentationClass')
        print(voc_root)

        if not os.path.isdir(voc_root):
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

        splits_dir = os.path.join(voc_root, 'ImageSets/Segmentation')

        split_f = os.path.join(splits_dir, image_set.rstrip('\n') + '.txt')

        with open(os.path.join(split_f), "r") as f:
            file_names = [x.strip() for x in f.readlines()]

        self.images = [os.path.join(image_dir, x + ".jpg") for x in file_names]
        self.masks = [os.path.join(mask_dir, x + ".png") for x in file_names]


        self.images = self._filter(self.images)
        self.masks = self._filter(self.masks)

        length = len(self.images)
        self.images = self.images[:length//use]
        self.masks = self.masks[:length//use]

        assert (len(self.images) == len(self.masks))
    
    def _filter(self, images):
        return [im for im in images if (Image.open(im).size[0]>=self.crop_size[1] and \
            Image.open(im).size[1]>=self.crop_size[0])]


    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is the image segmentation.
        """

        img = Image.open(self.images[index])#RGB模式
        target = Image.open(self.masks[index])#P模式8位调色板模式, 像素值已经是对应类别的编号了
        if 'crop' in dir(self):
            if img.size[1] < self.crop_size[0] or img.size[0] < self.crop_size[1]:#PIL图像的size长宽是反的！
                print("resize")
                img, target= img.resize((self.crop_size[1],self.crop_size[0])), target.resize((self.crop_size[1],self.crop_size[0]))
            else:
                img, target = self.crop(img, target)

        # img, target = self.To_Tensor(img), self.To_Tensor(target)
        if random.random() < 0.5:
            img = F.hflip(img)
            target = F.hflip(target)

        img, target = self.To_Tensor(img), torch.from_numpy(np.array(target))
        target[target==255] = 0

        return torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(img), target.long()

    def __len__(self):
        return len(self.images)

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

class MultiRandomCrop(object):
    """Crop the given PIL Image at a random location.

    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made.
        padding (int or sequence, optional): Optional padding on each border
            of the image. Default is None, i.e no padding. If a sequence of length
            4 is provided, it is used to pad left, top, right, bottom borders
            respectively. If a sequence of length 2 is provided, it is used to
            pad left/right, top/bottom borders, respectively.
        pad_if_needed (boolean): It will pad the image if smaller than the
            desired size to avoid raising an exception. Since cropping is done
            after padding, the padding seems to be done at a random offset.
        fill: Pixel fill value for constant fill. Default is 0. If a tuple of
            length 3, it is used to fill R, G, B channels respectively.
            This value is only used when the padding_mode is constant
        padding_mode: Type of padding. Should be: constant, edge, reflect or symmetric. Default is constant.

             - constant: pads with a constant value, this value is specified with fill

             - edge: pads with the last value on the edge of the image

             - reflect: pads with reflection of image (without repeating the last value on the edge)

                padding [1, 2, 3, 4] with 2 elements on both sides in reflect mode
                will result in [3, 2, 1, 2, 3, 4, 3, 2]

             - symmetric: pads with reflection of image (repeating the last value on the edge)

                padding [1, 2, 3, 4] with 2 elements on both sides in symmetric mode
                will result in [2, 1, 1, 2, 3, 4, 4, 3]

    """

    def __init__(self, size, padding=None, pad_if_needed=False, fill=0, padding_mode='constant'):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.padding = padding
        self.pad_if_needed = pad_if_needed
        self.fill = fill
        self.padding_mode = padding_mode

    @staticmethod
    def get_params(img, output_size):
        """Get parameters for ``crop`` for a random crop.

        Args:
            img (PIL Image): Image to be cropped.
            output_size (tuple): Expected output size of the crop.

        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for random crop.
        """
        w, h = img.size
        th, tw = output_size
        if w == tw and h == th:
            return 0, 0, h, w

        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)
        return i, j, th, tw

    def __call__(self, *imgs):
        """
        Args:
            img (PIL Image): Image to be cropped.

        Returns:
            PIL Image: Cropped image.
        """

        result = []
        i, j, h, w = self.get_params(imgs[0], self.size)
        for img in imgs:
            if self.padding is not None:
                img = F.pad(img, self.padding, self.fill, self.padding_mode)

            # pad the width if needed
            if self.pad_if_needed and img.size[0] < self.size[1]:
                img = F.pad(img, (self.size[1] - img.size[0], 0), self.fill, self.padding_mode)
            # pad the height if needed
            if self.pad_if_needed and img.size[1] < self.size[0]:
                img = F.pad(img, (0, self.size[0] - img.size[1]), self.fill, self.padding_mode)
            result.append(F.crop(img, i, j, h, w))
        return result

    def __repr__(self):
        return self.__class__.__name__ + '(size={0}, padding={1})'.format(self.size, self.padding)

def load_data_VOCSegmentation(year='2012', batch_size = 62, crop_size=None, root='../../Datasets/VOC', num_workers=4, use=1):

    print('year=%s, batch_size=%d'%(year, batch_size))
    voc_train = VOCSegmentation(root=root, year=year, image_set='train', crop_size=crop_size, use=use)
    print('已读取train, 共有%s张图片'%len(voc_train))
    voc_val = VOCSegmentation(root=root, year=year, image_set='val', crop_size=crop_size, use=use)
    print('已读取val, 共有%s张图片'%len(voc_val))
    train_iter = torch.utils.data.DataLoader(voc_train, batch_size=batch_size, shuffle=True, num_workers= num_workers)
    val_iter = torch.utils.data.DataLoader(voc_val, batch_size=batch_size, shuffle=True, num_workers= num_workers)
    return train_iter, val_iter

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