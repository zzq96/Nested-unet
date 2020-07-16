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
# import torchvision

import sys
sys.path.append(os.path.abspath('..'))

def _AsppConv(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bn_momentum=0.1):
    asppconv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, bias=False),
            nn.BatchNorm2d(out_channels, momentum=bn_momentum),
            nn.ReLU()
        )
    return asppconv

class AsppModule(nn.Module):
    def __init__(self, bn_momentum=0.1, output_stride=16):
        super(AsppModule, self).__init__()

        # output_stride choice
        if output_stride ==16:
            atrous_rates = [0, 6, 12, 18]
        elif output_stride == 8:
            atrous_rates = 2*[0, 12, 24, 36]
        else:
            raise Warning("output_stride must be 8 or 16!")
        # atrous_spatial_pyramid_pooling part
        self._atrous_convolution1 = _AsppConv(2048, 256, 1, 1, bn_momentum=bn_momentum)
        self._atrous_convolution2 = _AsppConv(2048, 256, 3, 1, padding=atrous_rates[1], dilation=atrous_rates[1]
                                              , bn_momentum=bn_momentum)
        self._atrous_convolution3 = _AsppConv(2048, 256, 3, 1, padding=atrous_rates[2], dilation=atrous_rates[2]
                                              , bn_momentum=bn_momentum)
        self._atrous_convolution4 = _AsppConv(2048, 256, 3, 1, padding=atrous_rates[3], dilation=atrous_rates[3]
                                              , bn_momentum=bn_momentum)

        #image_pooling part
        self._image_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Conv2d(2048, 256, kernel_size=1, bias=False),
            nn.BatchNorm2d(256, momentum=bn_momentum),
            nn.ReLU()
        )

        self.__init_weight()

    def forward(self, input):
        input1 = self._atrous_convolution1(input)
        input2 = self._atrous_convolution2(input)
        input3 = self._atrous_convolution3(input)
        input4 = self._atrous_convolution4(input)
        input5 = self._image_pool(input)
        input5 = F.interpolate(input=input5, size=input4.size()[2:4], mode='bilinear', align_corners=True)

        return torch.cat((input1, input2, input3, input4, input5), dim=1)

    def __init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

class Encoder(nn.Module):
    def __init__(self, bn_momentum=0.1, output_stride=16):
        super(Encoder, self).__init__()
        self.ASPP = AsppModule(bn_momentum=bn_momentum, output_stride=output_stride)
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(1280, 256, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(256, momentum=bn_momentum)
        self.dropout = nn.Dropout(0.5)

        self.__init_weight()

    def forward(self, input):
        input = self.ASPP(input)
        input = self.conv1(input)
        input = self.bn1(input)
        input = self.relu(input)
        input = self.dropout(input)
        return input


    def __init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

class Decoder(nn.Module):
    def __init__(self, class_num, bn_momentum=0.1, fuse_attention=False, **kwargs):
        super(Decoder, self).__init__()
        self.conv1 = nn.Conv2d(256, 48, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(48, momentum=bn_momentum)
        self.relu = nn.ReLU()
        # self.conv2 = SeparableConv2d(304, 256, kernel_size=3)
        # self.conv3 = SeparableConv2d(256, 256, kernel_size=3)
        if fuse_attention:
            print("trainning with fuse_attention!")
            self.fuse_attention = fuse_attention
            self.fuse  = Fuse_Attention(deep_dim=256, shallow_dim=48,reduce_rate=4)
            self.conv2 = nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False)
        else:
            self.conv2 = nn.Conv2d(304, 256, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(256, momentum=bn_momentum)
        self.dropout2 = nn.Dropout(0.5)
        self.conv3 = nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(256, momentum=bn_momentum)
        self.dropout3 = nn.Dropout(0.1)
        self.conv4 = nn.Conv2d(256, class_num, kernel_size=1)


        self._init_weight()

    def forward(self, x, low_level_feature):
        low_level_feature = self.conv1(low_level_feature)
        low_level_feature = self.bn1(low_level_feature)
        low_level_feature = self.relu(low_level_feature)
        x_4 = F.interpolate(x, size=low_level_feature.size()[2:4], mode='bilinear' ,align_corners=True)
        if self.fuse_attention:
            x_4_cat = self.fuse(x_4, low_level_feature)
        else:
            x_4_cat = torch.cat((x_4, low_level_feature), dim=1)
        x_4_cat = self.conv2(x_4_cat)
        x_4_cat = self.bn2(x_4_cat)
        x_4_cat = self.relu(x_4_cat)
        x_4_cat = self.dropout2(x_4_cat)
        x_4_cat = self.conv3(x_4_cat)
        x_4_cat = self.bn3(x_4_cat)
        x_4_cat = self.relu(x_4_cat)
        x_4_cat = self.dropout3(x_4_cat)
        x_4_cat = self.conv4(x_4_cat)

        return x_4_cat

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

class DeepLabV3Plus(nn.Module):
    def __init__(self, num_classes,input_channels=3, output_stride=16, pretrained=True, bn_momentum=0.1, freeze_bn=False, **kwargs):
        super(DeepLabV3Plus, self).__init__()
        self.Resnet101 = resnet101(bn_momentum, pretrained, output_stride=output_stride)
        self.encoder = Encoder(bn_momentum, output_stride)
        self.decoder = Decoder(num_classes, bn_momentum, **kwargs)
        #TODO: 每次epoch训练不是都会model.train（）吗
        if freeze_bn:
            self.freeze_bn()
            print("freeze bacth normalization successfully!")

    def forward(self, input):
        x, low_level_features = self.Resnet101(input)

        x = self.encoder(x)
        predict = self.decoder(x, low_level_features)
        output= F.interpolate(predict, size=input.size()[2:4], mode='bilinear', align_corners=True)
        return output

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

if __name__ =="__main__":
    model = DeepLab(output_stride=16, class_num=21, pretrained=True)
    print(model)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(1)
    device = torch.device('cuda')
    model = model.to(device)
    summary(model, (3, 513, 513), batch_size=8, device='cuda')