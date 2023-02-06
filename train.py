import torch
import torchvision
from torch import nn
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F

# using torchvision to load in cityscapes data for faster preprocessing
dataset = torchvision.datasets.Cityscapes('data', split='train', mode='fine',
                     target_type='semantic')

img, seg = dataset[0]
print("image dimensions: ", img.size, "segmentation size: " seg.size)
fig,ax=plt.subplots(ncols=2,figsize=(12,8))
ax[0].imshow(dataset[0][0])
ax[1].imshow(dataset[0][1],cmap='gray')
plt.show()
