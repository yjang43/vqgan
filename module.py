import torch
import torch.nn as nn
import torch.nn.functional as F



class ResidualBlock(nn.Module):
    
    def __init__(self, dim=256):

        super().__init__()
        self.conv1 = nn.Conv2d(dim, dim, 3, 1, 1)
        self.conv2 = nn.Conv2d(dim, dim, 1, 1, 0)

    def forward(self, x):

        res = x
        h = F.relu(x)
        h = self.conv1(h)
        h = F.relu(h)
        h = self.conv2(h)
        out = h + res
        return out
    

class DownsampleBlock(nn.Module):

    def __init__(self, in_dim=3, out_dim=256):
        
        super().__init__()
        self.conv1 = nn.Conv2d(in_dim, out_dim, 3, 2, 1)
        self.conv2 = nn.Conv2d(out_dim, out_dim, 3, 2, 1)
    
    def forward(self, x):
        
        h = self.conv1(x)
        h = F.relu(h)
        out = self.conv2(h)
        return out


class UpsampleBlock(nn.Module):

    def __init__(self, in_dim=256, out_dim=3):
        
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.conv1 = nn.Conv2d(in_dim, in_dim, 3, 1, 1)
        self.conv2 = nn.Conv2d(in_dim, out_dim, 3, 1, 1)

    def forward(self, x):

        h = self.upsample(x)
        h = self.conv1(h)
        h = F.relu(h)
        h = self.upsample(h)
        out = self.conv2(h)
        return out
    

class Encoder(nn.Module):

    def __init__(self, in_dim=3, out_dim=256):

        super().__init__()
        self.downsample_block = DownsampleBlock(in_dim, out_dim)
        self.res_block1 = ResidualBlock(out_dim)
        self.res_block2 = ResidualBlock(out_dim)
    
    def forward(self, x):

        h = self.downsample_block(x)
        h = self.res_block1(h)
        out = self.res_block2(h)
        return out


class Decoder(nn.Module):

    def __init__(self, in_dim=256, out_dim=3):

        super().__init__()
        self.res_block1 = ResidualBlock(in_dim)
        self.res_block2 = ResidualBlock(in_dim)
        self.upsample_block = UpsampleBlock(in_dim, out_dim)
    
    def forward(self, x):

        h = self.res_block1(x)
        h = self.res_block2(h)
        out = self.upsample_block(h)
        return out

class Discriminator(nn.Module):

    def __init__(self, in_dim=3, hid_dim=256, out_dim=1):
        
        super().__init__()
        self.downsample1 = DownsampleBlock(3, 64)
        self.downsample2 = DownsampleBlock(64, 1)

    def forward(self, x):

        h = self.downsample1(x)
        h = F.relu(h)
        out = self.downsample2(h)
        return out
