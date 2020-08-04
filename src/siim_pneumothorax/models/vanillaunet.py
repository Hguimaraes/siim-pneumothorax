import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(ConvBlock, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.conv_block(x)

class DownConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DownConvBlock, self).__init__()
        self.conv_block = ConvBlock(in_ch, out_ch)
        self.max_pool = nn.MaxPool2d(2)

    def forward(self, x):
        x = self.max_pool(x)
        return self.conv_block(x)


class UpConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(UpConvBlock, self).__init__()
        self.conv_block = ConvBlock(in_ch, out_ch)
        self.tranpose_conv = nn.ConvTranspose2d(in_ch , in_ch // 2, kernel_size=2, stride=2)

    def forward(self, x1, x2):
        x1 = self.tranpose_conv(x1)
        x = torch.cat([x2, x1], dim=1)
        return self.conv_block(x)


class VanillaUNet(nn.Module):
    def __init__(self):
        super(VanillaUNet, self).__init__()
        self.input_block = ConvBlock(1, 64)
        self.down_block_1 = DownConvBlock(64, 128)
        self.down_block_2 = DownConvBlock(128, 256)
        self.down_block_3 = DownConvBlock(256, 512)
        self.down_block_4 = DownConvBlock(512, 1024)

        self.up_block_1 = UpConvBlock(1024, 512)
        self.up_block_2 = UpConvBlock(512, 256)
        self.up_block_3 = UpConvBlock(256, 128)
        self.up_block_4 = UpConvBlock(128, 64)
        self.out_block = nn.Conv2d(64, 1, kernel_size=1)

    
    def forward(self, x):
        x1 = self.input_block(x)
        x2 = self.down_block_1(x1)
        x3 = self.down_block_2(x2)
        x4 = self.down_block_3(x3)
        x5 = self.down_block_4(x4)

        x = self.up_block_1(x5, x4)
        x = self.up_block_2(x, x3)
        x = self.up_block_3(x, x2)
        x = self.up_block_4(x, x1)

        return self.out_block(x)