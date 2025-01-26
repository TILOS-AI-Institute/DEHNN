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
from models.stem import ResStem
    

def double_conv(in_channels, hidden, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, hidden, 3, padding=1),
        nn.BatchNorm2d(hidden),
        nn.ReLU(inplace=True),
        nn.Conv2d(hidden, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )

def feature_transform(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )
    
class UNet(nn.Module):
    def __init__(self, n_channels=33, n_classes=1, input_type=0):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.input_type = input_type
        # self.stem_out_dim = stem_out_dim
        self.dropout_rate = 0.0
        self.stem = ResStem(in_channels=32, out_channels=1) 
        self.feature_transform = feature_transform(1, 1) 

        dim = 16
        in_dim = 1

        if self.input_type == 0:
            in_dim = 2
        else:
            in_dim = 1

        self.dconv_down1 = double_conv(in_dim, dim, dim)
        self.dconv_down2 = double_conv(dim, dim * 2, dim * 2)
        self.dconv_down3 = double_conv(dim * 2, dim * 4, dim * 4)
        self.dconv_down4 = double_conv(dim * 4, dim * 8, dim * 8)

        # self.maxpool = nn.MaxPool2d(2)
        self.maxpool = nn.AvgPool2d(2, ceil_mode=True)  
        # self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.upsample1 = nn.ConvTranspose2d(dim * 8, dim * 8, kernel_size=2, stride=2)
        self.upsample2 = nn.ConvTranspose2d(dim * 4, dim * 4, kernel_size=2, stride=2)
        self.upsample3 = nn.ConvTranspose2d(dim * 2, dim * 2, kernel_size=2, stride=2)

        self.dconv_up3 = double_conv(dim * 12, dim * 8, dim * 4)
        self.dconv_up2 = double_conv(dim * 6, dim * 4, dim * 2)
        self.dconv_up1 = double_conv(dim * 3, dim * 2, dim)
        self.drop_out = nn.Dropout2d(self.dropout_rate)
        self.conv_last = nn.Conv2d(dim, self.n_classes, 1)
        
        ### a head with two conv layers
        self.head = nn.Sequential(  
            nn.Conv2d(dim, dim//2, 1, padding=0),
            nn.GELU(),
            nn.Conv2d(dim//2, self.n_classes, 1, padding=0),
            # nn.Sigmoid() # to make the output in the range of [0, 1]
            nn.ReLU() 
        )

        self.attn = nn.Sequential(
            nn.Conv2d(1, 1, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        """
        Forward pass of the U-Net model.
        """
        if self.input_type == 0:
            ### for the 32 channels from the feature maps use the stem, then concatenate with the rudy map
            x_feature = self.stem(x[:, :32, :, :])
            ### concatenate the rudy map
            x_rudy = self.feature_transform(x[:, 32:, :, :])
            x = torch.cat([x_feature, x_rudy], dim=1)
            ### apply attention to the feature map 
            # x =torch.cat([x_rudy * self.attn(x_feature), x_rudy], dim=1)
        elif self.input_type == 1:
            x = self.feature_transform(x)
        elif self.input_type == 2:
            x = self.stem(x)

        # if not self.rudy_only:
        #     ### for the 32 channels from the feature maps use the stem, then concatenate with the rudy map
        #     x_feature = self.stem(x[:, :32, :, :])
        #     ### concatenate the rudy map
        #     x_rudy = self.feature_transform(x[:, 32:, :, :])
        #     x = torch.cat([x_feature, x_rudy], dim=1)
        #     ### apply attention to the feature map 
        #     # x =torch.cat([x_rudy * self.attn(x_feature), x_rudy], dim=1)
        # else:
        #     x = self.feature_transform(x)

        conv1 = self.dconv_down1(x)
        x = self.maxpool(conv1)

        conv2 = self.dconv_down2(x)
        x = self.maxpool(conv2)

        conv3 = self.dconv_down3(x)
        x = self.maxpool(conv3)

        x = self.dconv_down4(x)

        conv3 = nn.functional.pad(conv3, (0, 0, 1, 0))
        x = self.upsample1(x)
        x = torch.cat([x, conv3], dim=1)

        # crop x
        x = x[:, :, :conv2.shape[2]//2, :conv2.shape[3]//2]
        x = self.dconv_up3(x)
        x = self.upsample2(x)
        # pad x
        x = nn.functional.pad(x, (0, 1, 0, 0))
        x = torch.cat([x, conv2], dim=1)

        x = self.dconv_up2(x)
        x = self.upsample3(x)
        x = torch.cat([x, conv1], dim=1)

        x = self.dconv_up1(x)

        # x = self.conv_last(x)  
        x = self.head(x)
        
        return x