from __future__ import absolute_import
import torch
import torchvision
from torch import nn
from torch.nn import init
from torch.nn import functional as F
import ResNet as at
from IPython import embed
# Defines the new fc layer and classification layer
# |--Linear--|--bn--|--relu--|--Linear--|
class ClassBlock(nn.Module):
    def __init__(self, input_dim, class_num, droprate, relu=False, bnorm=True, num_bottleneck=512, linear=True, return_f = False):
        super(ClassBlock, self).__init__()
        self.return_f = return_f
        add_block = []
        if linear:
            add_block += [nn.Linear(input_dim, num_bottleneck)]
        else:
            num_bottleneck = input_dim
        if bnorm:
            add_block += [nn.BatchNorm1d(num_bottleneck)]
        if relu:
            add_block += [nn.LeakyReLU(0.1)]
        if droprate>0:
            add_block += [nn.Dropout(p=droprate)]
        add_block = nn.Sequential(*add_block)
        add_block.apply(weights_init_kaiming)

        classifier = []
        classifier += [nn.Linear(num_bottleneck, class_num)]
        classifier = nn.Sequential(*classifier)
        classifier.apply(weights_init_classifier)

        self.add_block = add_block
        self.classifier = classifier
    def forward(self, x):
        x = self.add_block(x)
        if self.return_f:
            f = x
            x = self.classifier(x)
            return x,f
        else:
            x = self.classifier(x)
            return x
# Define the ResNet50-based Model
class ft_net(nn.Module):

    def __init__(self, num_classes, droprate=0.5, stride=2,loss={'softmax,metric'}):
        class_num = num_classes
        super(ft_net, self).__init__()
        model_ft = torchvision.models.resnet50(pretrained=False)
        # avg pooling to global pooling
        if stride == 1:
            self.model.layer4[0].downsample[0].stride = (1,1)
            self.model.layer4[0].conv2.stride = (1,1)
        model_ft.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.model = model_ft
        self.classifier = ClassBlock(2048, class_num, droprate)

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = self.model.avgpool(x)
        x = x.view(x.size(0), x.size(1))
        feature = x
        if not self.training:
            return feature
        x = self.classifier(x)
        return x

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in') # For old pytorch, you may use kaiming_normal.
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
        init.constant_(m.bias.data, 0.0)
    elif classname.find('BatchNorm1d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        init.normal_(m.weight.data, std=0.001)
        init.constant_(m.bias.data, 0.0)

#####PCB model
class PCB(nn.Module):
    def __init__(self, num_classes,loss={'softmax,metric'}):
        super(PCB, self).__init__()
        class_num = num_classes
        self.part = 4  # We cut the pool5 to 6 parts
        model_ft = torchvision.models.resnet50(pretrained=False)
        self.model = model_ft
        self.avgpool = nn.AdaptiveAvgPool2d((self.part, 1))
        self.dropout = nn.Dropout(p=0.5)
        # remove the final downsample
        self.model.layer4[0].downsample[0].stride = (1, 1)
        self.model.layer4[0].conv2.stride = (1, 1)
        # define 6 classifiers
        for i in range(self.part):
            name = 'classifier' + str(i)
            setattr(self, name, ClassBlock(2048, class_num, droprate=0.5, relu=False, bnorm=True, num_bottleneck=256))
        #全局特征分类器
        self.globePool = nn.AdaptiveAvgPool2d((1,1))
        self.classifier_globe = ClassBlock(2048, class_num, 0.5)
        # 第二层特征分类器
        self.classifier_21 = ClassBlock(2048, class_num, 0.5)
        self.classifier_22 = ClassBlock(2048, class_num, 0.5)

        self.SA1 = at.SpatialAttn()
    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)  # [32,2048,24,8]
        embed()
        #x_attn1 = self.SA1(x)
        #x = x*x_attn1
        x_globe = self.globePool(x)
        x = self.avgpool(x)

        feature_globe = x_globe.view(x_globe.size(0), -1)
        globe_f = self.classifier_globe(feature_globe)
        x = self.dropout(x)
        part = {}
        predict = {}
        # get six part feature batchsize*2048*6
        for i in range(self.part):
            part[i] = torch.squeeze(x[:, :, i])
            name = 'classifier' + str(i)
            c = getattr(self, name)
            predict[i] = c(part[i])

        if not self.training:
            return part,feature_globe
        # sum prediction
        # y = predict[0]
        # for i in range(self.part-1):
        #    y += predict[i+1]
        y = []
        for i in range(self.part):
            y.append(predict[i])
        return y,globe_f #返回局部特征和全局特征


if __name__ =='__main__':
    model = PCB(num_classes=751)
    model.train()
    imgs =torch.Tensor(32,3,384,128)
    f,f_globe = model(imgs)
    embed()
