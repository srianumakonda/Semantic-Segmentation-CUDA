import torch
import torchvision
from torch import nn
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from utils import *
from model import CityScapesNetwork

torch.cuda.empty_cache()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if __name__=="__main__":
    #PARAMS
    SEED = 42
    WEIGHT_DECAY = 1e-6
    BATCH_SIZE = 16
    EPOCHS = 20
    LR = 1e-4
    CHECKPOINT_EPOCH = 0
    NUM_WORKERS = 12
    LOAD_MODEL = False
    SAVE_MODEL = True
    BEST_LOSS = 1e5 #defining some arbitary value
    # TRAINING =  True

    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize(size=((256,512))),
        torchvision.transforms.ToTensor(),
        # torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    target_transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize(size=((256,512))),
        torchvision.transforms.PILToTensor()
    ])

    train = CityScapesPreprocess('data', split='train', mode='fine', target_type='semantic', transform=transform, target_transform=target_transform)
    val = CityScapesPreprocess('data', split='val', mode='fine', target_type='semantic', transform=transform, target_transform=transform)

    trainloader = torch.utils.data.DataLoader(train, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True) 
    valloader = torch.utils.data.DataLoader(val, batch_size=BATCH_SIZE, shuffle=False)

    img, seg = train[100]

    print("image dimensions: ", img.shape, "segmentation size: ", seg.shape)
    print(torch.unique(seg))
    print(len(torch.unique(seg)))

    deseg = decodeMask(seg)
    print(deseg.shape)
    

    model = CityScapesNetwork(in_channels=3, out_channels=numClasses()).to(device)

    # fig,ax=plt.subplots(ncols=3,figsize=(12,8))
    # ax[0].imshow(img.permute(1,2,0))
    # ax[1].imshow(seg.permute(1,2,0))
    # ax[2].imshow(torch.Tensor(deseg).permute(1,2,0))
    # plt.show()

    criterion = DiceLoss()
    iou = IoULoss()
    # criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    if LOAD_MODEL:
        print("Loading models...")
        checkpoint = torch.load("saved_models/model.pth")
        model.load_state_dict(checkpoint['model_state'])
        optimizer.load_state_dict(checkpoint['optim_state'])
        model_loss = checkpoint['loss']
        CHECKPOINT_EPOCH = checkpoint['epoch']+1
        BEST_LOSS = model_loss
        print("Done!")
        print('-'*20) 

    for epoch in range(EPOCHS):
        print('-'*20)
        for i, (img, annotation) in enumerate(trainloader):
            img = img.to(device)
            annotation = annotation.to(device).long()
            
            output = encodeMask(model(img))
            output = torch.max(output,dim=1, keepdim=True)[0]

            loss = criterion(output, annotation)
            iou_loss = iou(output, annotation)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (int(i+1))%(len(train)//BATCH_SIZE//5) == 0:
                print(f"Epoch: {epoch+1}/{EPOCHS}, Step: {i+1}/{len(train)//BATCH_SIZE}, Dice loss: {loss.item():4f}, IOU: {1-iou_loss.item():4f}")

        running_val_loss = 0 
        running_iou_loss = 0
        path = f"predictions/epoch_{epoch+1}.png"
        model.eval() 

        with torch.no_grad(): 
            for idx, (img, annotation) in enumerate(valloader, 0):
                img = img.to(device)
                annotation = annotation.to(device).long()

                output = encodeMask(model(img))
                output = torch.max(output,dim=1, keepdim=True)[0]
                if idx==0:
                    save_preds(output, path)

                val_loss = criterion(output, annotation)
                val_iou = iou(output, annotation)
                running_val_loss += val_loss.item()
                running_iou_loss += val_iou.item()

        avg_val_loss = running_val_loss/len(valloader)
        avg_iou_loss = 1-running_iou_loss/len(valloader)

        print(f"Validation Loss: {avg_val_loss:4f}, IOU: {avg_iou_loss:4f}")
        

        if (avg_val_loss/len(valloader)) < BEST_LOSS: 
            if SAVE_MODEL:
                print("Saving model...")
                torch.save({
                    'epoch': epoch,
                    'model_state': model.state_dict(),
                    'optim_state': optimizer.state_dict(),
                    'loss': loss.item(),
                }, f"saved_models/model.pth")
                print("Done!")
                BEST_LOSS = avg_val_loss
                print('-'*20)

