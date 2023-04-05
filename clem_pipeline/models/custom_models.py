from clem_pipeline.models.unet_module import UNet
from torch import nn 
import torch 
from enum import Enum


class ModelType(str, Enum):
    ONLYS1IDEA1 = "OnlyS1Idea1"

class OnlyS1Idea1(nn.Module):
    def __init__(self, bilinear=False):
        super(OnlyS1Idea1, self).__init__()
        self.unet = UNet(6, 1, bilinear)

    def forward(self, t2s1, t1s1, ts1, t2s2, t1s2, t2s2_mask, t1s2_mask):
        """Return a tensor of shape (batch_size, 256, 256) corresponding to the LAI of each image"""
        img_size = t2s1.shape[2]
        return self.unet(torch.cat([t2s1, t1s1, ts1], dim=1)).view(-1, img_size, img_size)
        


