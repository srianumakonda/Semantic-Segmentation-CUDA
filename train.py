import torch
import torchvision
from torch import nn
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from model import CityScapesNetwork
from utils.preprocess import CityScapesPreprocess

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def encode_segmap(mask):
    #remove unwanted classes and recitify the labels of wanted classes
    for _validc in valid_classes:
        mask[mask == _validc] = class_map[_validc]
    return mask


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
    NUM_WORKERS = 12
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

    trainloader = torch.utils.data.DataLoader(train, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True) 
    # valloader = torch.utils.data.DataLoader(val, batch_size=BATCH_SIZE, shuffle=False)

    img, seg = train[0]

    print("image dimensions: ", img.shape, "segmentation size: ", seg.shape)
    print(torch.unique(seg))
    print(len(torch.unique(seg)))


    void_classes = [0, 1, 2, 3, 4, 5, 6, 9, 10, 14, 15, 16, 18, 29, 30, -1]
    valid_classes = [255, 7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33]
    class_names = ['unlabelled', 'road', 'sidewalk', 'building', 'wall', 'fence', 'pole', 'traffic_light', 'traffic_sign', 'vegetation', 'terrain', 'sky', 'person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle', 'bicycle']
    print(len(valid_classes), len(class_names))

    CityScapesNetwork = CityScapesNetwork()

    # fig,ax=plt.subplots(ncols=2,figsize=(12,8))


    # ax[0].imshow(img.permute(1,2,0))
    # ax[1].imshow(seg.permute(1,2,0))
    # plt.show()
