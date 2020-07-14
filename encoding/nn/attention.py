from torch import nn
import torch
from torch.nn import functional as F

class Fuse_Attention(nn.Module):
    
    def __init__(self, deep_dim, shallow_dim):
        super(Fuse_Attention, self).__init__()
        assert shallow_dim >= 8

        self.query_conv = nn.Conv2d(in_channels=shallow_dim, out_channels=shallow_dim//4, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=shallow_dim, out_channels=shallow_dim//4, kernel_size=1)
        #self.value_conv = Conv2d(in_channels=deep_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)
    def forward(self, deep, shallow):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X (HxW) X (HxW)
        """
        assert deep.size()[2:]== shallow.size()[2:]

        m_batchsize, shallow_C, height, width = shallow.size()
        deep_C = deep.size()[1]
        proj_query = self.query_conv(shallow).view(m_batchsize, -1, width*height).permute(0, 2, 1)
        proj_key = self.key_conv(shallow).view(m_batchsize, -1, width*height)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        #proj_value = self.value_conv(x).view(m_batchsize, -1, width*height)

        #out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = torch.bmm(deep.view(m_batchsize, -1, width * height), attention.permute(0, 2, 1))
        out = out.view(m_batchsize, deep_C, height, width)

        out = self.gamma*out + deep
        return out