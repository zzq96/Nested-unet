import torch
import torch.nn as nn
import torchvision.models as models

vgg16 = models.resnet152(pretrained=True)

#打印出预训练模型的参数
print('vgg16:\n', vgg16) 
# print('vgg16:\n', list(vgg16.children())) 
# print('vgg16:\n', list(vgg16.modules())) 

for name, parm in vgg16.named_parameters():
    print(name)
# print('modified_features:\n', list(vgg16.parameters()))#打印修改后的模型参数
# modified_features = nn.Sequential(*list(vgg16.features.children())[:-1])
# print('modified_features:\n', list(vgg16.features.children()) )#打印修改后的模型参数
# modified_features = nn.Sequential(*list(vgg16.features.modules())[:-1])
# # to relu5_3
# print('modified_features:\n', list(vgg16.features.modules()))#打印修改后的模型参数