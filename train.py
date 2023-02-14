import torch
import torchvision
from torch import nn
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from utils.preprocess import CityScapesPreprocess


if __name__=="__main__":
    #PARAMS
    SEED = 42
    MIN, MAX = 0, 0
    WEIGHT_DECAY = 1e-6
    CROP = 0
    BATCH_SIZE = 64
    EPOCHS = 50
    LR = 1e-4
    CHECKPOINT_EPOCH = 0
    BEST_LOSS = 1e10
    # LOAD_MODEL = False
    # SAVE_MODEL = True
    # TRAINING =  False

    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize(size=((512,1024))),
        torchvision.transforms.ToTensor(),
        # torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    train = CityScapesPreprocess('data', split='train', mode='fine', target_type='semantic', transform=transform, target_transform=transform)
    # val = CityScapesPreprocess('data', split='val', mode='fine', target_type='semantic', transform=transform, target_transform=transform)

    # trainloader = torch.utils.data.DataLoader(train, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True) 
    # valloader = torch.utils.data.DataLoader(val, batch_size=BATCH_SIZE, shuffle=False)

    img, seg = train[0]
    print("image dimensions: ", img.size, "segmentation size: ", seg.size)
    fig,ax=plt.subplots(ncols=2,figsize=(12,8))
    print(torch.min(train[0][0]), torch.max(train[0][0]))

    ax[0].imshow(train[0][0].permute(1,2,0))
    ax[1].imshow(train[0][1].permute(1,2,0))
    plt.show()
