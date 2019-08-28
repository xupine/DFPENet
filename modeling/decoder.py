import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from modeling.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d


class Attention_block(nn.Module):
    def __init__(self,F_g,F_l,F_int):
        super(Attention_block,self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1,stride=1,padding=0,bias=False),
            nn.BatchNorm2d(F_int)
            )
        
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1,stride=1,padding=0,bias=False),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1,stride=1,padding=0,bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self,g,x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1+x1)
        psi = self.psi(psi)

        return x*psi

class Gates_block(nn.Module):
    def __init__(self,x_l):
        super(Gates_block,self).__init__()
        self.gli = nn.Sequential(
            nn.Conv2d(x_l, 1, kernel_size=1,stride=1,padding=0,bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
    def forward(self,x,g):
        gl = self.gli(x)
        p1 = (1+gl)*x
        p2 = (1+gl)*g
        p = torch.cat((p1,p2), dim=1)

        return p 


       
class Decoder(nn.Module):
    def __init__(self, num_classes, backbone, BatchNorm):
        super(Decoder, self).__init__()
        if backbone == 'resnet':
            low_level_inplanes1 = 64
            low_level_inplanes2 = 256
            low_level_inplanes3 = 512
        elif backbone == 'xception':
            low_level_inplanes1 = 64
            low_level_inplanes2 = 128
            low_level_inplanes3 = 256
        elif backbone == 'mobilenet':
            low_level_inplanes = 24
        else:
            raise NotImplementedError

        self.conv1 = nn.Conv2d(low_level_inplanes3, 192, 1, bias=False)
        self.bn1 = BatchNorm(192)
        self.Att3 = Attention_block(F_g=256,F_l=192,F_int=96)
        self.Gate3 = Gates_block(x_l=256)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(low_level_inplanes2, 96, 1, bias=False)
        self.bn2 = BatchNorm(96)
        self.Att2 = Attention_block(F_g=256,F_l=96,F_int=48)
        self.Gate2 = Gates_block(x_l=256)
        self.conv3 = nn.Conv2d(low_level_inplanes1, 24, 1, bias=False)
        self.bn3 = BatchNorm(24)
        self.Att1 = Attention_block(F_g=256,F_l=24,F_int=12)
        self.Gate1 = Gates_block(x_l=256)

        self.relu = nn.ReLU()
        self.third_conv=nn.Sequential(nn.Conv2d(448,256, kernel_size=3,stride=1,padding=1,bias=False),
                                       BatchNorm(256),
                                       nn.ReLU())
        self.second_conv=nn.Sequential(nn.Conv2d(608,256, kernel_size=3,stride=1,padding=1,bias=False),
                                       BatchNorm(256),
                                       nn.ReLU())       

        self.last_conv = nn.Sequential(nn.Conv2d(792, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       BatchNorm(256),
                                       nn.ReLU(),
                                       nn.Dropout(0.5),
                                       nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       BatchNorm(256),
                                       nn.ReLU(),
                                       nn.Dropout(0.1),
                                       nn.Conv2d(256, num_classes, kernel_size=1, stride=1))
        self._init_weight()


    def forward(self, x, low_level_feat1, low_level_feat2, low_level_feat3):
        low_level_feat3 = self.conv1(low_level_feat3)
        low_level_feat3 = self.bn1(low_level_feat3)
        low_level_feat3 = self.relu(low_level_feat3)
        
        
        x1 = F.interpolate(x, size=low_level_feat3.size()[2:], mode='bilinear', align_corners=True)
        low_level_feat3 = self.Att3(g=x1,x=low_level_feat3)
        x1 = self.Gate3(x=x1,g=low_level_feat3)
        x1 = self.third_conv(x1)

        low_level_feat2 = self.conv2(low_level_feat2)
        low_level_feat2 = self.bn2(low_level_feat2)
        low_level_feat2 = self.relu(low_level_feat2)
        x2 = F.interpolate(x1, size=low_level_feat2.size()[2:], mode='bilinear', align_corners=True)
        low_level_feat2 = self.Att2(g=x2,x=low_level_feat2)
        x2 = self.Gate2(x=x2,g=low_level_feat2)
        x21 = F.interpolate(x, size=low_level_feat2.size()[2:], mode='bilinear', align_corners=True)
        x2 = torch.cat((x2,x21), dim=1)
        x2 = self.second_conv(x2)

        low_level_feat1 = self.conv3(low_level_feat1)
        low_level_feat1 = self.bn3(low_level_feat1)
        low_level_feat1 = self.relu(low_level_feat1)
        x3 = F.interpolate(x2, size=low_level_feat1.size()[2:], mode='bilinear', align_corners=True)
        low_level_feat1 = self.Att1(g=x3,x=low_level_feat1)
        x3 = self.Gate1(x=x3,g=low_level_feat1)
        x31 = F.interpolate(x1, size=low_level_feat1.size()[2:], mode='bilinear', align_corners=True)
        x32 = F.interpolate(x, size=low_level_feat1.size()[2:], mode='bilinear', align_corners=True)
        x3 = torch.cat((x3,x31), dim=1)
        x3 = torch.cat((x3,x32), dim=1)
        x = self.last_conv(x3)

        return x

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, SynchronizedBatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

def build_decoder(num_classes, backbone, BatchNorm):
    return Decoder(num_classes, backbone, BatchNorm)