import os
import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
import matplotlib.pyplot as plt
from time import time
import multiprocessing as mp
import numpy as np
from utils.preprocess import CityScapesPreprocess



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
        torchvision.transforms.Resize(size=((512,1024))),
        torchvision.transforms.ToTensor(),
        # torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    dataset = CityScapesPreprocess("data", split='train', mode='fine', target_type='semantic', transform=transform, target_transform=transform)

    for num_workers in range(4, mp.cpu_count(), 2):  
        passTime = 100000000
        train_loader = torch.utils.data.DataLoader(dataset,shuffle=True,num_workers=num_workers,batch_size=64,pin_memory=True)
        start = time()

        for epoch in range(1, 3):
            for i, data in enumerate(train_loader, 0):
                pass
            print("Epoch #", epoch)

        end = time()
        if (end-start) < passTime: 
            passTime = end - start
            cores = num_workers
        print("Finish with:{} second, num_workers={}".format(end - start, num_workers))

    return cores

def train_model(model, data_loader, val_loader, epochs, steps_per_epoch, device, optim, iou, dice, precision, recall):
    outputs = []
    highest_dice = 0.0
    highest_iou = 0.0
    highest_prec = 0.0
    highest_rec = 0.0
    for epoch in range(epochs):
        print('-'*20)
        for i, (img, annotation) in enumerate(data_loader):
            img = img.to(device)
            annotation = annotation.to(device)
            
            output = model(img)
            iou_loss = iou(output, annotation)
            dice_loss = dice(output, annotation)
            precision_met = precision(output, annotation)
            recall_met = recall(output, annotation)
            
            optim.zero_grad()
            iou_loss.backward()
            optim.step()

            if highest_iou < 1-iou_loss.item():
                highest_iou = 1-iou_loss.item()

            if highest_dice < dice_loss:
                highest_dice = dice_loss

            if highest_prec < precision_met:
                highest_prec = precision_met

            if highest_rec < recall_met:
                highest_rec = recall_met    
            
            if (int(i+1))%(steps_per_epoch//5) == 0:
                print(f"epoch {epoch+1}/{epochs}, step {i+1}/{steps_per_epoch}, IoU score = {1-iou_loss.item():.4f}, Precision = {precision_met:.4f}, Recall = {recall_met:.4f}, F1/Dice score: {dice_loss:.4f}")
                
        model.eval()
        for img, annotation in val_loader:
            img = img.to(device)
            annotation = annotation.to(device)

            output = model(img)
            iou_loss = iou(output, annotation)
            dice_loss = dice(output, annotation)
            precision_met = precision(output, annotation)
            recall_met = recall(output, annotation)
        print(f"validation loss: IoU score = {1-iou_loss.item():.4f}, Precision = {precision_met:.4f}, Recall = {recall_met:.4f}, F1/Dice score: {dice_loss:.4f}")
        outputs.append((img, annotation, output))
    print("-"*20)
    print(f"highest values, IoU score = {highest_iou:.4f}, Precision = {highest_prec:.4f}, Recall = {highest_rec:.4f}, F1/Dice score: {highest_dice:.4f}")

    return model, outputs

#VARIABLES FOR ENCODING AND DECODING - MODIFY CLASSES HERE
void = [0, 1, 2, 3, 4, 5, 6, 9, 10, 14, 15, 16, 18, 29, 30, -1]
valid = [255, 7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33]
classes = ['unlabelled', 'road', 'sidewalk', 'building', 'wall', 'fence', 'pole', 'traffic_light', 'traffic_sign', 'vegetation', 'terrain', 'sky', 'person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle', 'bicycle']
colors = [[0, 0, 0],
          [128, 64, 128],
          [244, 35, 232],
          [70, 70, 70],
          [102, 102, 156],
          [190, 153, 153],
          [153, 153, 153],
          [250, 170, 30],
          [220, 220, 0],
          [107, 142, 35],
          [152, 251, 152],
          [0, 130, 180],
          [220, 20, 60],
          [255, 0, 0],
          [0, 0, 142],
          [0, 0, 70],
          [0, 60, 100],
          [0, 80, 100],
          [0, 0, 230],
          [119, 11, 32]]

def numClasses():
    return len(valid)

colorMap = dict(zip(valid, range(numClasses())))
reverseMap = dict(zip(range(numClasses()), colors))

def encodeMask(seg):
    for v in void:
        seg[seg == v] = 255 #change to background
    for v in valid:
        seg[seg == v] = colorMap[v]
    return seg

def decodeMask(seg):
    seg = seg.detach().cpu().numpy()
    r = seg.copy()
    g = seg.copy()
    b = seg.copy()
    
    for c in range(0, numClasses()):
        r[seg == c] = reverseMap[c][0]
        g[seg == c] = reverseMap[c][1]
        b[seg == c] = reverseMap[c][2]

    rgb = np.zeros((3, seg.shape[-2], seg.shape[-1])) #stitch everything back together
    rgb[0, :, :] = r / 255.0
    rgb[1, :, :] = g / 255.0
    rgb[2, :, :] = b / 255.0
    return rgb

def save_preds(output, path):
    temp = torch.zeros((output.shape[0],3,output.shape[1],output.shape[2]))
    for i in output:
        temp[i] = torch.from_numpy(decodeMask(i))
    grid = torchvision.utils.make_grid(temp, nrow=2)
    torchvision.utils.save_image(grid, path)
