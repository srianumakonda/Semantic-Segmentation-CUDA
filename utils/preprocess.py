import torch
import torchvision
from PIL import image
import cv2

class CityScapesPreprocess():
    super(CityScapesPreprocess, self).__init__()

    def __init__(self, dir):
        
        """
        @param:
            root_dir (str): path of the input images from the CARLA simulator
            annotation_dir (str): path of the input annotation images
            factor(int): input of teh resize factor for input images        
        """