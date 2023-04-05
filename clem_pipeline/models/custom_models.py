from clem_pipeline.models.unet_module import UNet
from torch import nn 
import torch 
from enum import Enum


class ModelType(str, Enum):
    S1_ONLY_IDEA1 = "s1_only_idea1"
    S1_MASK_IDEA2 = "s1_mask_idea2"
    S2_ONLY_IDEA3 = "s2_only_idea3"

def get_model(parameters):
    match parameters["model_type"]:
        case ModelType.S1_ONLY_IDEA1:
            return OnlyS1Idea1()
        case ModelType.S1_MASK_IDEA2:
            return S1MaskIdea2()
        case ModelType.S2_ONLY_IDEA3:
            return S2OnlyIdea3()
        case _:
            raise ValueError(f"Model type {parameters['model_type']} not implemented")

class OnlyS1Idea1(nn.Module):
    def __init__(self, bilinear=False):
        super(OnlyS1Idea1, self).__init__()
        self.unet = UNet(6, 1, bilinear)

    def forward(self, t2s1, t1s1, ts1, t2s2, t1s2, t2s2_mask, t1s2_mask, timestamp):
        """Return a tensor of shape (batch_size, 256, 256) corresponding to the LAI of each image"""
        img_size = t2s1.shape[2]
        return self.unet(torch.cat([t2s1, t1s1, ts1], dim=1)).view(-1, img_size, img_size)
        


class S1MaskIdea2(nn.Module):
    def __init__(self, mask_encoding_channels=4, bilinear=False):
        super(S1MaskIdea2, self).__init__()
        self.unet = UNet(6 + mask_encoding_channels*2, 1, bilinear)
        self.mask_conv = nn.Conv2d(10, mask_encoding_channels, kernel_size=1)

    def forward(self, t2s1, t1s1, ts1, t2s2, t1s2, t2s2_mask, t1s2_mask, timestamp):
        """Return a tensor of shape (batch_size, 256, 256) corresponding to the LAI of each image"""
        s1_concat = torch.cat([t2s1, t1s1, ts1], dim=1)
        t2s2_mask_encoding = self.mask_conv(t2s2_mask)
        t1s2_mask_encoding = self.mask_conv(t1s2_mask)
        s2_mask_encoding = torch.cat([t2s2_mask_encoding, t1s2_mask_encoding], dim=1)
        output = self.unet(torch.cat([s1_concat, s2_mask_encoding], dim=1))

        img_size = t2s1.shape[2]

        return output.view(-1, img_size, img_size)

class S2OnlyIdea3(nn.Module):
    def __init__(self, mask_encoding_channels=4, bilinear=False):
        super(S2OnlyIdea3, self).__init__()
        self.mask_conv = nn.Conv2d(10, mask_encoding_channels, kernel_size=1)
        self.unet = UNet(2*mask_encoding_channels+2, 1, bilinear)

    def forward(self, t2s1, t1s1, ts1, t2s2, t1s2, t2s2_mask, t1s2_mask, timestamp):
        """Return a tensor of shape (batch_size, 256, 256) corresponding to the LAI of each image"""
        t2s2_mask_encoding = self.mask_conv(t2s2_mask)
        t1s2_mask_encoding = self.mask_conv(t1s2_mask)

        all_past_s2 = torch.cat([t2s2[:, None, ...], t2s2_mask_encoding, t1s2[:, None, ...], t1s2_mask_encoding], dim=1)

        img_size = t2s1.shape[2]
        return self.unet(all_past_s2).view(-1, img_size, img_size)

if __name__=="__main__":
    from clem_pipeline.utils import parse_training_tensor

    # model = S1MaskIdea2()
    
    model = get_model({"model_type": ModelType.S2_ONLY_IDEA3})

    batch_size = 2
    data = torch.rand((batch_size, 3, 14, 256, 256))
    timestamp = torch.rand((batch_size, 2))

    t2s1, t1s1, ts1, t2s2, t1s2, ts2, t2s2_mask, t1s2_mask, ts2_binary_mask = parse_training_tensor(data)

    pred = model(t2s1, t1s1, ts1, t2s2, t1s2, t2s2_mask, t1s2_mask, timestamp)

    print(pred.shape)