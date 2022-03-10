from turtle import forward
import torch, sys

from backbone import resnet
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class conv_bn_relu(nn.Module):
    def __init__(self, in_channels, out_channels, upsample=0):
        super(conv_bn_relu,self).__init__()
        self.conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation = 1, bias = False)
        self.bn = torch.nn.BatchNorm2d(out_channels)
        self.relu = torch.nn.ReLU()
        self.upsample = upsample

    def forward(self,x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        if self.upsample:
            x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=True)
        return x

class SegmentationBlock(nn.Module):
    def __init__(self, in_channels, out_channels, n_upsamples=0):
        super().__init__()

        blocks = [conv_bn_relu(in_channels, out_channels, upsample=bool(n_upsamples))]

        if n_upsamples > 1:
            for _ in range(1, n_upsamples):
                blocks.append(conv_bn_relu(out_channels, out_channels, upsample=True))

        self.block = nn.Sequential(*blocks)

    def forward(self, x):
        return self.block(x)

class FPNBlock(nn.Module):
    def __init__(self, pyramid_channels, skip_channels):
        super().__init__()
        self.skip_conv = nn.Conv2d(skip_channels, pyramid_channels, kernel_size=1)

    def forward(self, x, skip=None):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        skip = self.skip_conv(skip)
        x = x + skip
        return x

class SegmentationBlock(nn.Module):
    def __init__(self, in_channels, out_channels, n_upsamples=0):
        super().__init__()

        blocks = [conv_bn_relu(in_channels, out_channels, upsample=bool(n_upsamples))]

        if n_upsamples > 1:
            for _ in range(1, n_upsamples):
                blocks.append(conv_bn_relu(out_channels, out_channels, upsample=True))

        self.block = nn.Sequential(*blocks)

    def forward(self, x):
        return self.block(x)

class SegmentationHead(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, upsampling=4):
        super(SegmentationHead, self).__init__()

        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        self.upsampling = nn.UpsamplingBilinear2d(scale_factor=upsampling) if upsampling > 1 else nn.Identity()

    def forward(self, x):
        x = self.conv2d(x)
        x = self.upsampling(x)
        return x
        
class parsingNet(torch.nn.Module):
    def __init__(self, size=(288, 800), pretrained=True, backbone='50', cls_dim=(37, 10, 4), use_aux=False):
        super(parsingNet, self).__init__()

        self.size = size
        self.w = size[0]
        self.h = size[1]
        self.cls_dim = cls_dim # (num_gridding, num_cls_per_lane, num_of_lanes)
        # num_cls_per_lane is the number of row anchors
        self.total_dim = np.prod(cls_dim)

        # input : nchw,
        # output: (w+1) * sample_rows * 4 
        self.model = resnet(backbone, pretrained=pretrained)

        # Lateral layers
        self.p5 = conv_bn_relu(512, 128) if backbone in ['34','18'] else conv_bn_relu(2048, 128)
        self.p4 = FPNBlock(128, 256) if backbone in ['34','18'] else FPNBlock(128, 1024)
        self.p3 = FPNBlock(128, 128) if backbone in ['34','18'] else FPNBlock(128, 512)
        self.p2 = FPNBlock(128, 64) if backbone in ['34','18'] else FPNBlock(128, 256)

        # Smooth layers
        self.smooth5 = SegmentationBlock(128, 128, n_upsamples=3)
        self.smooth4 = SegmentationBlock(128, 128, n_upsamples=2)
        self.smooth3 = SegmentationBlock(128, 128, n_upsamples=1)
        self.smooth2 = SegmentationBlock(128, 128, n_upsamples=0)

        # Final layers output : n, num_of_lanes+1, h, w
        self.finallayer = SegmentationHead(128*4, cls_dim[-1]+1)

        initialize_weights(self.p5, self.p4, self.p3, self.p2,
                            self.smooth5, self.smooth4, self.smooth3, self.smooth2, self.finallayer)

    def forward(self, x):
        # n c h w - > n 2048 sh sw
        # Bottom-up
        c2, c3, c4, c5 = self.model(x)

        # Top-down
        p5 = self.p5(c5)
        p4 = self.p4(p5, c4)
        p3 = self.p3(p4, c3)
        p2 = self.p2(p3, c2)
        
        # Smooth
        p5 = self.smooth5(p5)
        p4 = self.smooth4(p4)
        p3 = self.smooth3(p3)
        p2 = self.smooth2(p2)

        seg = self.finallayer(torch.cat([p5, p4, p3, p2], dim=1))

        return seg

def initialize_weights(*models):
    for model in models:
        real_init_weights(model)

def real_init_weights(m):

    if isinstance(m, list):
        for mini_m in m:
            real_init_weights(mini_m)
    else:
        if isinstance(m, torch.nn.Conv2d):    
            torch.nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)
        elif isinstance(m, torch.nn.Linear):
            m.weight.data.normal_(0.0, std=0.01)
        elif isinstance(m, torch.nn.BatchNorm2d):
            torch.nn.init.constant_(m.weight, 1)
            torch.nn.init.constant_(m.bias, 0)
        elif isinstance(m,torch.nn.Module):
            for mini_m in m.children():
                real_init_weights(mini_m)
        else:
            print('unkonwn module', m)