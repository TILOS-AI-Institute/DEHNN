import os
import numpy as np
import pickle
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from typing import List
import sys
import pdb
from utils import *
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from tqdm import tqdm
    

class ResStem(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        dilation: int = 1,
        groups: int = 1,
        act_func: str = "GELU",
        norm_func: str = "BN",
        bias: bool = True,
    ) -> None:
        super().__init__()
        kernel_size = (kernel_size, kernel_size)
        stride = (stride, stride)
        dilation = (dilation, dilation)
        # same padding
        padding = [(dilation[i] * (kernel_size[i] - 1) + 1) // 2 for i in range(len(kernel_size))]

        # self.conv1 = BSConv2d(
        #     in_channels,
        #     out_channels // 2,
        #     kernel_size=kernel_size[0],
        #     stride=stride[0],
        #     dilation=dilation[0],
        #     bias=bias,
        # )
        self.conv1 = nn.Conv2d(
            in_channels,
            16,
            kernel_size=kernel_size[0],
            stride=stride[0],
            padding=padding[0],
            dilation=dilation[0],
        )

        # self.norm1 = nn.BatchNorm2d(out_channels // 2)
        self.norm1 = nn.BatchNorm2d(16)
        
        self.act1 = getattr(nn, act_func)()

        # self.conv2 = BSConv2d(
        #     out_channels // 2,
        #     out_channels,
        #     kernel_size=kernel_size[0],
        #     stride=stride[0],
        #     dilation=dilation[0],
        #     bias=bias,
        # )
        
        self.conv2 = nn.Conv2d(
            16,
            out_channels,
            kernel_size=kernel_size[0],
            stride=stride[0],
            padding=padding[0],
            dilation=dilation[0],
        )
        
        # self.bn2 = nn.BatchNorm2d(out_channels)

        self.norm2 = nn.BatchNorm2d(out_channels)
     
        # self.act2 = nn.ReLU(inplace=True)

        self.act2 = getattr(nn, act_func)()

    def forward(self, x):
        if isinstance(self.norm1, nn.LayerNorm):
            x = self.conv1(x)
            b, inc, h, w = x.shape
            x = x.permute(0, 2, 3, 1)
            x = self.norm1(x)
            x = x.permute(0, 3, 1, 2).contiguous()
            x = self.act1(x)
            
            x = self.conv2(x)
            b, inc, h, w = x.shape
            x = x.permute(0, 2, 3, 1)
            x = self.norm2(x)
            x = x.permute(0, 3, 1, 2).contiguous()
            x = self.act2(x)
            
            # x = self.act2(self.norm2(x)).reshape(b, h, w, -1).permute(0, 3, 1, 2).contiguous()
        else:
            x = self.act1(self.norm1(self.conv1(x)))
            x = self.act2(self.norm2(self.conv2(x)))
        return x


class BSConv2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        dilation: int = 1,
        bias: bool = True,
    ):
        super().__init__()
        kernel_size = (kernel_size, kernel_size)
        stride = (stride, stride)
        dilation = (dilation, dilation)
        # same padding
        padding = [(dilation[i] * (kernel_size[i] - 1) + 1) // 2 for i in range(len(kernel_size))]
        
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                dilation=1,
                groups=1,
                bias=bias,
            ),
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=out_channels,
                bias=bias,
            ),
        )

    def forward(self, x):
        return self.conv(x)