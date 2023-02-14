import os
import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
import matplotlib.pyplot as plt
from time import time
import multiprocessing as mp
import numpy as np
from preprocess import CityScapesPreprocess

class Residual_Block(nn.Module):
    
    def __init__(self, in_channels, out_channels, stride):
        super(Residual_Block, self).__init__()
        
        self.relu = nn.ReLU()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1, stride=stride)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, stride=1)
        self.skip_conv = nn.Conv2d(in_channels, out_channels, 1, padding=0, stride=stride)
        
        self.in_1 = nn.InstanceNorm2d(in_channels, affine=True)
        self.in_2 = nn.InstanceNorm2d(out_channels, affine=True)

    def forward(self, x):
        
        res_block = self.relu(self.in_1(x))
        res_block = self.conv1(res_block)
        res_block = self.relu(self.in_2(res_block))
        res_block = self.conv2(res_block)
        s = self.skip_conv(x)
        skip = res_block + s
        return skip

class Decoder_Block(nn.Module):
    
    def __init__(self, in_channels, out_channels):
        super(Decoder_Block, self).__init__()

        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.res_block = Residual_Block(in_channels + out_channels, out_channels, stride=1)

    def forward(self, x, skip):
        
        dec_block = self.upsample(x)
        dec_block = torch.cat([dec_block, skip], axis=1)
        dec_block = self.res_block(dec_block)
        return dec_block

class double_conv(nn.Module):

    def __init__(self, in_channels,out_channels,kernel_size=3,stride=1,padding=1,maxpool=False):
        super(double_conv,self).__init__()

        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels,out_channels,kernel_size=kernel_size,stride=stride,padding=padding),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels,out_channels,kernel_size=kernel_size,stride=stride,padding=padding),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU()   
        )

    def forward(self, x):
        x = self.double_conv(x)
        return x

class max_down(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(max_down,self).__init__()

        self.max_down = nn.Sequential(
            nn.MaxPool2d(2),
            double_conv(in_channels,out_channels)
        )

    def forward(self, x):
        x = self.max_down(x)
        return x

class upsample(nn.Module):
    
    def __init__(self, in_channels, out_channels, in_conv=None, bilinear=False):
        super(upsample,self).__init__()

        if bilinear:
            self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)
        else:
            self.upsample = nn.ConvTranspose2d(in_channels//2, in_channels//2, kernel_size=2, stride=2)
        self.conv = double_conv(in_channels, out_channels)
        
        if in_conv != None:
            self.conv = double_conv(in_conv, out_channels)
            if bilinear:
                self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)
            else:
                self.upsample = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
    
    def forward(self, x1, x2):
        x1 = self.upsample(x1)
        diffX = x1.size()[2] - x2.size()[2]
        diffY = x1.size()[3] - x2.size()[3]
        x2 = F.pad(x2, (diffX // 2, int(diffX / 2),
                        diffY // 2, int(diffY / 2)))
        
        out = torch.cat([x2, x1], dim=1)
        out = self.conv(out)
        return out

def optimalWorkers():
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize(size=((1024,512))),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    dataset = CityScapesPreprocess("data", split='train', mode='fine', target_type='semantic', transform=transform, target_transform=transform)

    # print(type(dataset[0][0]), type(dataset[0][1]))

    for num_workers in range(2, mp.cpu_count(), 2):  
        train_loader = torch.utils.data.DataLoader(dataset,shuffle=True,num_workers=num_workers,batch_size=64,pin_memory=True)
        start = time()

        for epoch in range(1, 3):
            for i, data in enumerate(train_loader, 0):
                pass

        end = time()
        if (end-start) < passTime: 
            passTime = end - start
            cores = num_workers
        print("Finish with:{} second, num_workers={}".format(end - start, num_workers))
    return num_workers

if __name__=="__main__":
    optimalWorkers()