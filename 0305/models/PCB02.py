import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import torchvision
#from .resnet import resnet50
from IPython import embed

class PCBModel(nn.Module):
    def __init__( self,last_conv_stride=1,last_conv_dilation=1,num_stripes=4,local_conv_out_channels=256,num_classes=751,loss={'softmax,metric'}):
        super(PCBModel, self).__init__()
        #通过去掉全连接层和分类层的resnet后得到的 base feature map
        self.model = torchvision.models.resnet50(pretrained=False)
        # remove the final downsample
        self.model.layer4[0].downsample[0].stride = (1, 1)
        self.model.layer4[0].conv2.stride = (1, 1)

        #空间注意力模块
        #self.SpAnt = SP_attention(k=4, c=2048, r=2048 / 8)
        self.classifier_2stage = nn.Linear(local_conv_out_channels, num_classes)
        self.SAT_L2=SAT_conv(2)
        self.SAT_L3=SAT_conv(4)
        self.classifier_2stageL2 = nn.Linear(local_conv_out_channels, num_classes)

        #全局单独的卷积层
        self.globe_conv5x = torchvision.models.resnet50(pretrained=False).layer4
        # 全局特征分类器
        self.globePool = nn.AdaptiveAvgPool2d((1, 1))
        self.globe_conv = nn.Sequential(
            # 进入的channel个数为 2048 输出的channel数为256（local_conv_out_channels=256）
            nn.Conv2d(2048, local_conv_out_channels, 1),
            nn.BatchNorm2d(local_conv_out_channels),
            nn.ReLU(inplace=True)
          )
        self.classifier_globe = nn.Linear(local_conv_out_channels, num_classes)
        init.normal_(self.classifier_globe.weight, std=0.001)
        init.constant_(self.classifier_globe.bias, 0)

        # 2层分类器
        self.L2avgpool = nn.AdaptiveAvgPool2d((2, 1))

        self.N2_conv1 = nn.Sequential(
            # 进入的channel个数为 2048 输出的channel数为256（local_conv_out_channels=256）
            nn.Conv2d(2048, local_conv_out_channels, 1),
            nn.BatchNorm2d(local_conv_out_channels),
            nn.ReLU(inplace=True)
        )
        self.N2_conv2 = nn.Sequential(
            # 进入的channel个数为 2048 输出的channel数为256（local_conv_out_channels=256）
            nn.Conv2d(2048, local_conv_out_channels, 1),
            nn.BatchNorm2d(local_conv_out_channels),
            nn.ReLU(inplace=True)
        )

        self.classifierZ1 = nn.Linear(local_conv_out_channels, num_classes)
        init.normal_(self.classifier_globe.weight, std=0.001)
        init.constant_(self.classifier_globe.bias, 0)

        self.classifierZ2 = nn.Linear(local_conv_out_channels, num_classes)
        init.normal_(self.classifier_globe.weight, std=0.001)
        init.constant_(self.classifier_globe.bias, 0)


        #将feature map 分成多个 stripes
        self.num_stripes = num_stripes
        self.dropout = nn.Dropout(p=0.5)

        self.local_conv_list = nn.ModuleList()
        for _ in range(num_stripes):
          self.local_conv_list.append(nn.Sequential(
            # 进入的channel个数为 2048 输出的channel数为256（local_conv_out_channels=256）
            nn.Conv2d(2048, local_conv_out_channels, 1),
            nn.BatchNorm2d(local_conv_out_channels),
            nn.ReLU(inplace=True)
          ))

        if num_classes > 0:
          self.fc_list = nn.ModuleList()
          # 构造num_stripes 个分类器 分别连接到经过local_conv_list之后的256通道的feature map后
          for _ in range(num_stripes):
            fc = nn.Linear(local_conv_out_channels, num_classes)
            init.normal_(fc.weight, std=0.001)
            init.constant_(fc.bias, 0)
            self.fc_list.append(fc)

    def forward(self, x):
        """
        Returns:
          local_feat_list: each member with shape [N, c]
          logits_list: each member with shape [N, num_classes]
        """
        # shape [N, C, H, W]

        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        # 全局的另外一个分支
        globe_x = x
        globe_x =self.globe_conv5x(globe_x)

        x = self.model.layer4(x)  # [32,2048,24,8]
        #feat = self.base(x)
        x = self.dropout(x)    #从这个地方拉一个attention模块出去，计算sa_list

        #sa_list = self.SpAnt(x)

        #全局特征
        globe_x =self.globePool(globe_x)
        globe_x = self.globe_conv(globe_x)
        globe_x = globe_x.view(globe_x.size(0),-1)
        out_globe_feature = 1.*globe_x/(torch.norm(globe_x,2,dim = -1,keepdim=True).expand_as(globe_x)+1e-12) #这个作为输出用于测试
        globe_x = self.dropout(globe_x)
        #分类向量
        globe_x_logits = self.classifier_globe(globe_x)


        #2层特征
        L2_feature = x
        SA_list2 = self.SAT_L2(x)
        L2_1 = F.avg_pool2d(
            L2_feature[:, :, 0* 12: (0 + 1) * 12-1, :],
            (12, L2_feature.size(-1)))
        L2_2 = F.avg_pool2d(
            L2_feature[:, :, 1* 12: (1 + 1) * 12-1, :],
            (12, L2_feature.size(-1)))
        # 降维
        L21 = self.N2_conv1(L2_1)
        L22 = self.N2_conv2(L2_2)
        #输出
        L21 = L21.view(L21.size(0), -1)
        L22 = L22.view(L22.size(0), -1 )
        L2_out1 = 1.*L21/(torch.norm(L21,2,dim = -1,keepdim=True).expand_as(L21)+1e-12)
        L2_out2 = 1. * L22 / (torch.norm(L22, 2, dim=-1, keepdim=True).expand_as(L22) + 1e-12)
        L2_feature_list = [] #这个仅仅用于测试
        L2_feature_list.append(L2_out1)
        L2_feature_list.append(L2_out2)
        #分类输出
        L2_logits = []
        L2_logits.append(self.classifierZ1(L21))
        L2_logits.append(self.classifierZ2(L22))
        #embed()

        #第三层特征
        #空间注意力
        # 从这个地方开始学习权重
        feat = x
        #sa_list = self.SpAnt(feat)
        SA_list3 = self.SAT_L3(feat)
        assert feat.size(2) % self.num_stripes == 0
        stripe_h = int(feat.size(2) / self.num_stripes)
        # 特征向量
        local_feat_list = []
        #可能可以涨点的特征向量
        out_loacl_list = []
        # 分类向量 由特征向量通过全连接得到
        logits_list = []
        for i in range(self.num_stripes):
          # shape [N, C, 1, 1]
          # 把每一个channel上的part区域做平均，得到图中的feature vector V
          local_feat = F.avg_pool2d(
            feat[:, :, i * stripe_h: (i + 1) * stripe_h, :],
            (stripe_h, feat.size(-1)))
          # shape [N, c, 1, 1]
          #得到图中的feature vector G
          local_feat = self.local_conv_list[i](local_feat)  #这个进行特征降维256
          # shape [N, c]
          # resize feature map 到[N, C]
          local_feat = local_feat.view(local_feat.size(0), -1)
          local_feat_list.append(local_feat)

          local_out = 1.*local_feat/(torch.norm(local_feat,2,dim = -1,keepdim=True).expand_as(local_feat)+1e-12)
          out_loacl_list.append(local_out)
          # f = 1.*f/(torch.norm(f,2,dim = -1,keepdim=True).expand_as(f)+1e-12) 归一化

          # 如果有全连接层 那么就让 local_feat 通过fc层
          if hasattr(self, 'fc_list'):
            logits_list.append(self.fc_list[i](local_feat))

        #第二阶段训练
        #--------------------------------------------
        Z2_total = (SA_list2[:, 0].reshape([-1, 1]).expand_as(local_feat_list[0])) *  L21 + \
                  (SA_list2[:, 1].reshape([-1, 1]).expand_as(local_feat_list[1])) *  L22


        Z3_total = (SA_list3[:, 0].reshape([-1, 1]).expand_as(local_feat_list[0])) * local_feat_list[0] + \
                   (SA_list3[:, 1].reshape([-1, 1]).expand_as(local_feat_list[1])) * local_feat_list[1] + \
                   (SA_list3[:, 2].reshape([-1, 1]).expand_as(local_feat_list[2])) * local_feat_list[2] + \
                   (SA_list3[:, 3].reshape([-1, 1]).expand_as(local_feat_list[3])) * local_feat_list[3]
        # -------------------------------------------

        z_total_Y3 = self.classifier_2stage(Z3_total)
        z_total_Y2 = self.classifier_2stageL2(Z2_total)
        # 如果不分类 仅进行metric learning 那么就return local_feat_list
        if not self.training:
            return out_loacl_list ,out_globe_feature,L2_feature_list ,Z2_total,Z3_total
        # 如果进行分类 那么就 return 分类完的特征
        return logits_list ,globe_x_logits,L2_logits,z_total_Y2,z_total_Y3


#自己定义的空间注意力模块
class SP_attention(nn.Module):
    def __init__(self, k,c,r,loss={'softmax,metric'}, **kwargs):
        super(SP_attention, self).__init__()
        self.r = c/8
        self.Pooling = nn.AdaptiveAvgPool3d((c, k,1))
        self.FC1 = nn.Linear(k*c,k*8)
        #self.ReLU = F.relu()
        self.FC2 = nn.Linear(8*k,k)
        #self.Sigmoid = F.sigmoid()

    def forward(self, x):    #32,2048,8,4
        x = self.Pooling(x)   #[batch_size,C,K,1]
        #embed()
        x = x.view(x.size(0),-1) #[batch_size,c*k]
        x = self.FC1(x)      #[b,k*8]
        x = F.relu(x)
        x = self.FC2(x)      #[b,k]
        x = F.sigmoid(x)
        return x
class SAT_conv(nn.Module):
    def __init__(self,output,loss={'softmax,metric'}, **kwargs):
        super(SAT_conv,self).__init__()
        self.conv = nn.Sequential(
            # 进入的channel个数为 2048 输出的channel数为256（local_conv_out_channels=256）
            nn.Conv2d(2048, 256, 1),
            nn.Conv2d(256, 1, 1),
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True)
        )
        self.pooling = nn.AdaptiveAvgPool2d((output, 1))
    def forward(self,x):
        x = self.conv(x) #32,1,24,8
        score = self.pooling(x)
        score = score.view(score.size(0),-1)
        score = F.softmax(score)
        return  score
class PCB111_model(nn.Module):
    def __init__(self,C=1):
        super(PCB111_model, self).__init__()
        self.conv1 = nn.Conv2d(1,1,kernel_size=(3,3), stride=2)
    def forward(self, x):

        x= self.conv1(x)
        return  x
if __name__ =='__main__':
    # model = PCBModel(num_classes=751)  32,2048,24,8
    # model.train()
    imgs =torch.Tensor(32,2048,24,8)
    #model = PCB111_model()
    model = SAT_conv(output=4)
    f = model(imgs)
    x = torch.ones(4,1)
    x = x.view(-1)
    x = F.softmax(x)
    embed()
