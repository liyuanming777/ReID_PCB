from __future__ import absolute_import
import torch
import torchvision

from torch import nn
from torch.nn import functional as F
from IPython import embed
from aligned import HorizontalMaxPool2d as h_pool
class ResNet50(nn.Module):
    def __init__(self,num_classes,loss={'softmax,metric'},aligned =False,**kwargs):
        super(ResNet50,self).__init__()
        resnet50 = torchvision.models.resnet50(pretrained=False)
        self.base = nn.Sequential(*list(resnet50.children())[:-2])
        #self.conv1 = nn.Conv2d(2048, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.classifier = nn.Linear(2048,num_classes)
        self.horizon_pool = h_pool.HorizontalMaxPool2d()

        self.aligned = aligned
        if self.aligned:
            self.bn = nn.BatchNorm2d(2048)
            self.relu = nn.ReLU()
            self.conv1 = nn.Conv2d(2048,128,kernel_size=1,stride=1,padding=0,bias=True)
        # for mo in self.base.layer4[0].modules():
        #     if isinstance(mo, nn.Conv2d):
        #         mo.stride = (1, 1)


    def forward(self, x):
        x = self.base(x) #32,2048,8,4
        if self.aligned:
            lf = self.bn(x)
            lf = self.relu(lf)
            lf = self.horizon_pool(lf)
            lf = self.conv1(lf)
        hrizon = x.size()[2]
        x = F.avg_pool2d(x,x.size()[2:]) #32,2048,1,1
        #x = self.conv1(x)
        f = x.view(x.size(0),-1) # 32,2048
        # f = 1.*f/(torch.norm(f,2,dim = -1,keepdim=True).expand_as(f)+1e-12) 归一化
        if not self.training:
            return f



        y = self.classifier(f)
        if self.aligned:
            return y,f,lf
        else:
            pass
    #embed()
        return y

class ResNet50_2(nn.Module):
    def __init__(self,num_classes,loss={'softmax,metric'},**kwargs):
        super(ResNet50_2,self).__init__()
        resnet50 = torchvision.models.resnet50(pretrained=True)
        self.base = nn.Sequential(*list(resnet50.children())[:-2])
        self.SA1 = SpatialAttn()


        self.conv_0 = nn.Conv2d(2048, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.conv_11 = nn.Conv2d(2048, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.conv_12 = nn.Conv2d(2048, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.conv_21 = nn.Conv2d(2048, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.conv_22 = nn.Conv2d(2048, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.conv_23 = nn.Conv2d(2048, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.conv_24 = nn.Conv2d(2048, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.classifier = nn.Linear(512,num_classes)
        self.classifier11 = nn.Linear(512, num_classes)
        #self.classifier12 = nn.Linear(512, num_classes)
        self.classifierZ1 = nn.Linear(512, num_classes)
        self.classifierZ2 = nn.Linear(512, num_classes)
        self.classifierZ3 = nn.Linear(512, num_classes)
        self.classifierZ4 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.base(x) #32,2048,8,4
        x_attn1 = self.SA1(x) #32,1,8,4
        x_c = x_attn1[:,0,:,:]#32,8,4
        x_c_w = torch.sum(x_c,dim=2) #32,8
        # x_c = torch.sum(x,dim=1)
        # x_c_w = torch.sum(x_c,dim=2) #32,8

        x_in = x*x_attn1
        # x_c_flatten  = x_c.view(x_c.size(0), -1)
        # x_c_flatten = F.softmax(x_c_flatten, dim=1)
        # x_c_restore = x_c_flatten.reshape([x_c.size(0),x_c.size(1),x_c.size(2)]) #32,8,4
        # for i in range(0,x.size(1)):
        #     x[:,i,:,:] = x[:,i,:,:]*x_c_restore

        # #N2_1 = torch.tensor[torch.sum(x_c_w[:,0:3],dim=1),torch.sum(x_c_w[:,4:7],dim =1)]
        # N2_1 = [torch.sum(x_c_w[:, 0:3], dim=1),torch.sum(x_c_w[:,4:7],dim =1)]
        # N2 = torch.tensor[N2_1]

        N2_1 = torch.sum(x_c_w[:,0:1],dim =1) #32
        N2_2 = torch.sum(x_c_w[:, 2:3], dim=1) #32
        N2_3 = torch.sum(x_c_w[:, 4:5], dim=1)  # 32
        N2_4 = torch.sum(x_c_w[:, 6:7], dim=1)  # 32
        N2 = torch.stack((N2_1,N2_2,N2_3,N2_4),dim=1) # 32 ,4
        N2_norm = F.softmax(N2,dim=1)
        N2 = torch.pow(N2_norm, 2).sum(dim=1, keepdim=True)
        N2 = torch.mean(N2)

        #
        # hrizon = x.size()[2]

        y = x_in
        y1 = y[:,:,0:3,:] #32 ,2048 ,4, 4
        y2 = y[:, :, 4:7, :]#32 ,2048 ,4, 4
        z = x_in
        z1 = z[:,:,0:1,:]#32 ,2048 ,2, 4
        z2 = z[:, :, 2:3, :]  # 32 ,2048 ,2, 4
        z3 = z[:, :, 4:5, :]  # 32 ,2048 ,2, 4
        z4 = z[:, :, 6:7, :]  # 32 ,2048 ,2, 4
        z1 = F.avg_pool2d(z1, z1.size()[2:]) # 32 ,2048 ,1, 1
        z2 = F.avg_pool2d(z2, z2.size()[2:])
        z3 = F.avg_pool2d(z3, z3.size()[2:])
        z4 = F.avg_pool2d(z4, z4.size()[2:])
        z_11 = self.conv_21(z1)
        z_12 = self.conv_22(z2)
        z_13 = self.conv_23(z3)
        z_14 = self.conv_24(z4)

        z_11_ = F.relu(z_11)
        z_12_ = F.relu(z_11)
        z_13_ = F.relu(z_11)
        z_14_ = F.relu(z_11)

        z_11_  = z_11_.view(z_11_.size(0), -1)
        z_12_ = z_12_.view(z_12_.size(0), -1)
        z_13_ = z_13_.view(z_13_.size(0), -1)
        z_14_ = z_14_.view(z_14_.size(0), -1)


        y1 = F.avg_pool2d(y1, y1.size()[2:])#  32,2048,1,1
        y2 = F.avg_pool2d(y2, y2.size()[2:]) # 32,2048,1,1
        y1 = self.conv_11(y1)
        y2 = self.conv_12(y2)
        f_11 = y1.view(y1.size(0), -1)  # 32,2048
        f_12 = y2.view(y2.size(0), -1)  # 32,2048
        x_in = F.avg_pool2d(x_in,x_in.size()[2:]) #32,2048,1,1
        x_in = self.conv_0(x_in)
        f_globe = F.relu(x_in)
        f = f_globe.view(f_globe.size(0),-1) # 32,512
        feature_total = torch.cat((f,f_11,f_12,z_11_,z_12_,z_13_,z_14_),1)
        feature_test = [z_11_,z_12_,z_13_,z_14_,N2_norm]
	#globe = f
        if not self.training:
            #return torch.cat((z_11,z_12,z_13,z_14),1)
            return  feature_test
        # f = 1.*f/(torch.norm(f,2,dim = -1,keepdim=True).expand_as(f)+1e-12) 归一化
        # feature_test  = [f,f_11,f_12,N2_norm]

        f = self.classifier(f)
        f_11= self.classifier11(f_11)
        f_12 = self.classifier11(f_12)
        f_21 = self.classifierZ1(z_11_)
        f_22 = self.classifierZ2(z_12_)
        f_23 = self.classifierZ3(z_13_)
        f_24 = self.classifierZ4(z_14_)
        feature_train = [f,f_11,f_12,f_21,f_22,f_23,f_24,N2_norm]
       # embed()
        return feature_train

class SpatialAttn(nn.Module):
    """Spatial Attention Layer"""
    def __init__(self):
        super(SpatialAttn, self).__init__()

    def forward(self, x):
        # global cross-channel averaging # e.g. 32,2048,24,8
        x = x.mean(1, keepdim=True)  # e.g. 32,1,24,8
        h = x.size(2)
        w = x.size(3)
        x = x.view(x.size(0),-1)     # e.g. 32,192
        z = x
        for b in range(x.size(0)):
            z[b] /= torch.sum(z[b])
        z = z.view(x.size(0),1,h,w)
        return z

if __name__ =='__main__':
    model = ResNet50(num_classes=751,aligned=True)
    imgs =torch.Tensor(32,3,256,128)
    y,f,lf = model(imgs)
    embed()
