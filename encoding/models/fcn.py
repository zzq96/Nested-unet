# -*- coding: utf-8 -*-
# @Time    : 2018/9/19 16:56
# @Author  : HLin
# @Email   : linhua2017@ia.ac.cn
# @File    : encoder.py
# @Software: PyCharm

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
from .backbone import *
from ..nn.attention import Fuse_Attention
from ..utils.utils import get_upsampling_weight
import torchvision.models as models
# import torchvision

def get_fcn8s(dataset='vocaug', **kwargs):
    # infer number of classes
    from ..datasets import datasets, VOC2011Segmentation, VOC2012Segmentation, VOCAugSegmentation, ADE20KSegmentation
    model = FCN8s(datasets[dataset.lower()].NUM_CLASS, datasets[dataset.lower()].INPUT_CHANNELS, **kwargs)
    # if pretrained:
    #     from .model_store import get_model_file
    #     model.load_state_dict(torch.load(
    #         get_model_file('fcn_%s_%s'%(backbone, acronyms[dataset]), root=root)),
    #         strict= False)
    return model



class FCN8s(nn.Module):

    def __init__(self, num_classes, input_channels, fuse_attention = False, **kwargs):
        super(FCN8s, self).__init__()
        self.fuse_attention = fuse_attention
        #TODO:kwargs
        if self.fuse_attention:
            print('using fuse_attention!!')
            self.fuse_s16 = Fuse_Attention(deep_dim=21, shallow_dim=512)
            self.fuse_s8 = Fuse_Attention(deep_dim=21, shallow_dim=256)

        assert num_classes > 0
        self.conv1_1 = nn.Conv2d(input_channels, 64, 3, padding=100)
        self.relu1_1 = nn.ReLU(inplace=True)
        self.conv1_2 = nn.Conv2d(64, 64, 3, padding=1)
        self.relu1_2 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/2

        # conv2
        self.conv2_1 = nn.Conv2d(64, 128, 3, padding=1)
        self.relu2_1 = nn.ReLU(inplace=True)
        self.conv2_2 = nn.Conv2d(128, 128, 3, padding=1)
        self.relu2_2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/4

        # conv3
        self.conv3_1 = nn.Conv2d(128, 256, 3, padding=1)
        self.relu3_1 = nn.ReLU(inplace=True)
        self.conv3_2 = nn.Conv2d(256, 256, 3, padding=1)
        self.relu3_2 = nn.ReLU(inplace=True)
        self.conv3_3 = nn.Conv2d(256, 256, 3, padding=1)
        self.relu3_3 = nn.ReLU(inplace=True)
        self.pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/8

        # conv4
        self.conv4_1 = nn.Conv2d(256, 512, 3, padding=1)
        self.relu4_1 = nn.ReLU(inplace=True)
        self.conv4_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu4_2 = nn.ReLU(inplace=True)
        self.conv4_3 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu4_3 = nn.ReLU(inplace=True)
        self.pool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/16

        # conv5
        self.conv5_1 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu5_1 = nn.ReLU(inplace=True)
        self.conv5_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu5_2 = nn.ReLU(inplace=True)
        self.conv5_3 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu5_3 = nn.ReLU(inplace=True)
        self.pool5 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/32

        # fc6
        self.fc6 = nn.Conv2d(512,4096, 7)
        self.relu6 = nn.ReLU(inplace=True)
        self.drop6 = nn.Dropout2d()

        # fc7
        self.fc7 = nn.Conv2d(4096,4096, 1)
        self.relu7 = nn.ReLU(inplace=True)
        self.drop7 = nn.Dropout2d()

        self.score_fr1 = nn.Conv2d(4096, num_classes, 1)
        self.score_fr2 = nn.Conv2d(512, num_classes, 1)
        self.score_fr3 = nn.Conv2d(256, num_classes, 1)

        self.upscore_2x = nn.ConvTranspose2d(num_classes, num_classes, 4, stride=2,
                                          bias=False) 
        self.upscore_4x = nn.ConvTranspose2d(num_classes, num_classes, 4, stride=2,
                                          bias=False) 
        self.upscore_32x = nn.ConvTranspose2d(num_classes, num_classes, 16, stride=8,
                                          bias=False) 
                                        
        self._initialize_weights()
        
        vgg16 = models.vgg16(pretrained=True)
        self.copy_params_from_vgg16(vgg16)
        
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            
            if isinstance(m, nn.ConvTranspose2d):
                assert m.kernel_size[0] == m.kernel_size[1]
                # nn.init.kaiming_normal_(m.weight.data)
                initial_weight = get_upsampling_weight(
                    m.in_channels, m.out_channels, m.kernel_size[0])
                m.weight.data.copy_(initial_weight)

    def forward(self, X):
        h = X
        h = self.relu1_1(self.conv1_1(h))
        h = self.relu1_2(self.conv1_2(h))
        h = self.pool1(h)

        h = self.relu2_1(self.conv2_1(h))
        h = self.relu2_2(self.conv2_2(h))
        h = self.pool2(h)

        h = self.relu3_1(self.conv3_1(h))
        h = self.relu3_2(self.conv3_2(h))
        h = self.relu3_3(self.conv3_3(h))
        h = self.pool3(h)
        s8 = h

        h = self.relu4_1(self.conv4_1(h))
        h = self.relu4_2(self.conv4_2(h))
        h = self.relu4_3(self.conv4_3(h))
        h = self.pool4(h)
        s16 = h

        h = self.relu5_1(self.conv5_1(h))
        h = self.relu5_2(self.conv5_2(h))
        h = self.relu5_3(self.conv5_3(h))
        h = self.pool5(h)

        h = self.relu6(self.fc6(h))
        h = self.drop6(h)
        h = self.relu7(self.fc7(h))
        h = self.drop7(h)
        s32 = h

        s32_ = self.score_fr1(s32)

        s16_ = self.upscore_2x(s32_)        
        s16 = s16[:,:, 5 : 5 + s16_.size()[2], 5 : 5 + s16_.size()[3]]
        if self.fuse_attention :
            s16_ = self.fuse_s16(s16_, s16)
        else:
            s16 = self.score_fr2(s16)
            s16_ = s16 + s16_

        s8_ = self.upscore_4x(s16_)
        s8 = s8[:,:, 9 : 9 + s8_.size()[2], 9 : 9 + s8_.size()[3]]
        if self.fuse_attention :
            s8_ = self.fuse_s8(s8_, s8)
        else:
            s8 = self.score_fr3(s8)
            s8_ = s8 + s8_

        h = self.upscore_32x(s8_)
        h = h[:, :, 31:31 + X.size()[2], 31:31 + X.size()[3]].contiguous()

        return h

    def copy_params_from_vgg16(self,vgg16):
        features=[
            self.conv1_1, self.relu1_1,
            self.conv1_2, self.relu1_2,
            self.pool1,
            self.conv2_1, self.relu2_1,
            self.conv2_2, self.relu2_2,
            self.pool2,
            self.conv3_1, self.relu3_1,
            self.conv3_2, self.relu3_2,
            self.conv3_3, self.relu3_3,
            self.pool3,
            self.conv4_1, self.relu4_1,
            self.conv4_2, self.relu4_2,
            self.conv4_3, self.relu4_3,
            self.pool4,
            self.conv5_1, self.relu5_1,
            self.conv5_2, self.relu5_2,
            self.conv5_3, self.relu5_3,
            self.pool5
        ]
        for l1, l2 in zip(features, vgg16.features):
            if isinstance(l1, nn.Conv2d) and isinstance(l2, nn.Conv2d):
                assert l1.weight.data.shape == l2.weight.data.shape
                assert l1.bias.data.shape ==  l2.bias.data.shape
                l1.weight.data.copy_(l2.weight.data)
                l1.bias.data.copy_(l2.bias.data)
        for name, i in zip(['fc6', 'fc7'], [0, 3]):
            l1 = getattr(self, name)
            l2 = vgg16.classifier[i]
            l1.weight.data.copy_(l2.weight.data.view(l1.weight.data.shape))
            l1.bias.data.copy_(l2.bias.data.view(l1.bias.data.shape))

if __name__ =="__main__":
    model = DeepLab(output_stride=16, class_num=21, pretrained=True)
    print(model)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(1)
    device = torch.device('cuda')
    model = model.to(device)
    summary(model, (3, 513, 513), batch_size=8, device='cuda')