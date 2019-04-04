import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import torchvision
#from .resnet import resnet50
from IPython import embed

class PCBModel(nn.Module):
    def __init__( self,last_conv_stride=1,last_conv_dilation=1,num_stripes=6,local_conv_out_channels=256,num_classes=751,loss={'softmax,metric'}):
        super(PCBModel, self).__init__()

        self.globe_conv5x = torchvision.models.resnet50(pretrained=False).layer4
        embed()
    def forward(self, x):
        x = self.globe_conv5x(x)
        return x

if __name__ =='__main__':
    model = PCBModel(num_classes=751)
