import torch
from torch import nn

class UnetBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super.__init__()
        self.conv1 = nn.Conv2d(in_channels, middle_channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(middle_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(middle_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, X):
        out = self.conv1(X)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(X)
        out = self.bn2(X)
        out = self.relu(X)

        return out
    

class Unet(nn.Module):
    def __init__(self, num_classes, in_channels=3):
        super.__init__()
        
        num_filter = [64, 128, 256, 512, 1024]

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)#github中Convtranspose2d和直接双线性插值的实现都有

        self.conv0_0 = UnetBlock(in_channels, num_filter[0], num_filter[0])
        self.conv1_0 = UnetBlock(num_filter[0], num_filter[1], num_filter[1])
        self.conv2_0 = UnetBlock(num_filter[1], num_filter[2], num_filter[2])
        self.conv3_0 = UnetBlock(num_filter[2], num_filter[3], num_filter[3])
        self.conv4_0 = UnetBlock(num_filter[3], num_filter[4], num_filter[4])

        self.conv3_1 = UnetBlock(num_filter[4] + num_filter[3], num_filter[3], num_filter[3])
        self.conv2_2 = UnetBlock(num_filter[3] + num_filter[2], num_filter[2], num_filter[2])
        self.conv1_3 = UnetBlock(num_filter[2] + num_filter[1], num_filter[1], num_filter[1])
        self.conv0_4 = UnetBlock(num_filter[1] + num_filter[0], num_filter[0], num_filter[0])

        self.final = nn.Conv2d(num_filter[0], num_classes, kernel_size=1, stride=1)

    def forward(self, X):

        x0_0 = self.conv0_0(X)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x3_0 = self.conv3_0(self.pool(x2_0))
        x4_0 = self.conv4_0(self.pool(x3_0))

        x3_1 = self.conv3_1(torch.cat(x3_0, self.up(x4_0), dim=1))
        x2_2 = self.conv2_2(torch.cat(x2_0, self.up(x3_1), dim=1))
        x1_3 = self.conv1_3(torch.cat(x1_0, self.up(x2_2), dim=1))
        x0_4 = self.conv0_4(torch.cat(x0_0, self.up(x1_3), dim=1))

        output = self.final(x0_4)
        return output

if __name__ == "__main__":
    pass
