import torch
import torchvision
import numpy as np
from PIL import Image
from collections import namedtuple
from torchvision.datasets import Cityscapes
from typing import Any, Callable, Dict, List, Optional, Union, Tuple
import utils
# import cv2

from inspect import getmembers, isfunction

class CityScapesPreprocess(Cityscapes):
    # super(CityScapesPreprocess, self).__init__()

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is a tuple of all target types if target_type is a list with more
            than one item. Otherwise, target is a json object if target_type="polygon", else the image segmentation.
        Note:
            I copied PyTorch's source code for preprocessing the CityScapes dataset and then personally modified it so that I can perform the __getitem__ operation, source code can be found here: https://pytorch.org/vision/main/_modules/torchvision/datasets/cityscapes.html#Cityscapes
            Also note that transforms.ToTensor() automatically does the normalization between 0->1 for me, which is why I don't need to do anything here; https://discuss.pytorch.org/t/does-pytorch-automatically-normalizes-image-to-0-1/40022
        """

        image = Image.open(self.images[index]).convert("RGB")

        targets: Any = []
        for i, t in enumerate(self.target_type):
            if t == "polygon":
                target = self._load_json(self.targets[index][i])
            else:
                target = Image.open(self.targets[index][i])
            targets.append(target)

        target = tuple(targets) if len(targets) > 1 else targets[0]

        if self.transforms is not None:
            image = self.transform(image) 

        if self.target_transform is not None:
            target = self.target_transform(target)

        # find better way to link all of this
        target = utils.encodeMask(target)
        target = torch.squeeze(target).long()

        # inspo from https://github.com/pytorch/vision/issues/9#issuecomment-304224800
        return image, target
