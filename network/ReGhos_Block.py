import torch
import torch.nn as nn
from copy import deepcopy

class ReGhos_Block(nn.Module):
    def __init__(self, block, kernel_size=3):
        super(ReGhos_Block, self).__init__()
        self.original_block = deepcopy(block)

        for param in self.original_block.parameters():
            param.requires_grad = False
        self.out_channels = self.original_block.out_channels
        self.Ghos_Mod = nn.Sequential(
        nn.Conv2d(self.out_channels, self.out_channels, kernel_size, 1, kernel_size//2, groups=self.out_channels, bias=False),
        )
        nn.init.dirac_(self.Ghos_Mod[0].weight, groups=self.out_channels)

    def forward(self, x):
        output = self.Ghos_Mod(self.original_block(x))
        return output

class Feat_Choice_Layer(nn.Module):
    def __init__(self, feature):
        super(Feat_Choice_Layer, self).__init__()
        self.feature = feature
        self.A = nn.Parameter(torch.Tensor(1, self.feature, 1, 1))
        self.A.data.uniform_(0, 0)

    def forward(self, Res, Ghost):
        return Res + self.A * Ghost

class Ghost_Block_Combine_base(nn.Module):
    def __init__(self, conv_layer, dw_size):
        super(Ghost_Block_Combine_base, self).__init__()
        
        self.conv_layer = deepcopy(conv_layer)
        for param in self.conv_layer.parameters():
            param.requires_grad = False

        self.out_channels = conv_layer.out_channels
        self.dw_size = dw_size

        self.ghost_layer = nn.Conv2d(self.out_channels, self.out_channels, self.dw_size, 1, self.dw_size//2, 
                                     groups=int(self.out_channels), bias=False)
    
    def forward(self, x):
        x1 = self.conv_layer(x)
        x2 = self.ghost_layer(x1)
        
        return x1, x2

class Ghost_Block_Combine(nn.Module):
    def __init__(self, conv_layer, dw_size):
        super(Ghost_Block_Combine, self).__init__()
        self.out_channels = conv_layer.out_channels
        self.conv = Ghost_Block_Combine_base(conv_layer, dw_size)

        self.combine = Feat_Choice_Layer(self.out_channels)
    
    def forward(self, x):
        x1, x2 = self.conv(x)
        out = self.combine(x1, x2)

        return out
    

class Residule_block(nn.Module):
    def __init__(self, conv_layer):
        super(Residule_block, self).__init__()
        self.conv_layer = deepcopy(conv_layer)
        for param in self.conv_layer.parameters():
            param.requires_grad = False

        self.out_channels = conv_layer.out_channels
        self.in_channels = conv_layer.in_channels

        self.re = nn.Conv2d(self.in_channels, self.out_channels, 1, stride=conv_layer.stride, bias=False)
    
    def forward(self, x):
        x1 = self.conv_layer(x)
        x2 = self.re(x)
        
        return x1 + x2