import torch
import math
import torch.nn as nn
from net.resnet_v2_sn import *
from net.l2norm import L2Norm
from net.devkit.ops import SwitchNorm2d

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class ACSPNet(nn.Module):
    def __init__(self):
        super(ACSPNet, self).__init__()

        self.resnetType = "resnetv2sn50"
        resnet = resnetv2sn50(pretrained = True)
        # resnet = resnetv2sn101(pretrained = True)
        self.conv1 = resnet.conv1
        self.sn1 = resnet.sn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        self.p3 = nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1)
        self.p4 = nn.ConvTranspose2d(1024, 256, kernel_size=4, stride=4, padding=0)
        self.p5 = nn.ConvTranspose2d(2048, 256, kernel_size=8, stride=8, padding=0)

        nn.init.xavier_normal_(self.p3.weight)
        nn.init.xavier_normal_(self.p4.weight)
        nn.init.xavier_normal_(self.p5.weight)
        nn.init.constant_(self.p3.bias, 0)
        nn.init.constant_(self.p4.bias, 0)
        nn.init.constant_(self.p5.bias, 0)

        self.p3_l2 = L2Norm(256, 10)
        self.p4_l2 = L2Norm(256, 10)
        self.p5_l2 = L2Norm(256, 10)

        # self.brand1 = Brand()
        # self.brand2 = Brand()
        # # self.brand3 = Brand()
        # self.v_brand = Brand()
        # self.se = SELayer(272, 16)
        # self.cls_refine_conv = nn.Conv2d(272, 1, kernel_size=1)

        self.feat = nn.Conv2d(768, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.feat_sn = SwitchNorm2d(256)
        self.feat_act = nn.ReLU(inplace=True)

        self.pos_conv = nn.Conv2d(256, 3, kernel_size=1)#(256, 1, kernel_size=1) --> (256, 3, kernel_size=1)
        self.reg_conv = nn.Conv2d(256, 3, kernel_size=1)#(256, 1, kernel_size=1) --> (256, 3, kernel_size=1)
        self.off_conv = nn.Conv2d(256, 6, kernel_size=1)



    def forward(self, x):#torch.Size([1, 3, 640, 1280])
        x = self.conv1(x)
        x = self.sn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)

        x = self.layer2(x)
        p3 = self.p3(x)
        p3 = self.p3_l2(p3)

        x = self.layer3(x)
        p4 = self.p4(x)
        p4 = self.p4_l2(p4)

        x = self.layer4(x)
        p5 = self.p5(x)
        p5 = self.p5_l2(p5)
        # print(p3.shape, p4.shape, p5.shape)
        cat = torch.cat([p3, p4, p5], dim=1)

        feat = self.feat(cat)
        feat = self.feat_sn(feat)
        feat = self.feat_act(feat)
        x_cls = self.pos_conv(feat)
        x_cls = torch.sigmoid(x_cls)  # torch.Size([1, 3, 160, 320])
        x_reg = self.reg_conv(feat)  # torch.Size([1, 2, 160, 320])
        x_off = self.off_conv(feat)  # torch.Size([1, 3, 160, 320])
        #brand_out_feat:torch.Size([1, 256, 160, 320])
        return x_cls, x_reg, x_off #

class Brand(nn.Module):
    def __init__(self):
        super(Brand, self).__init__()
        self.feat = nn.Conv2d(768, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.feat_sn = SwitchNorm2d(256)
        self.feat_act = nn.ReLU(inplace=True)

        # self.feat_add1 = nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1, bias=False)
        # self.feat_sn_add1 = SwitchNorm2d(256)
        # self.feat_act_add1 = nn.ReLU(inplace=True)
        #
        self.pos_conv = nn.Conv2d(256, 1, kernel_size=1)  # (256, 1, kernel_size=1) --> (256, 3, kernel_size=1)
        self.reg_conv = nn.Conv2d(256, 1, kernel_size=1)  # (256, 1, kernel_size=1) --> (256, 3, kernel_size=1)
        self.off_conv = nn.Conv2d(256, 2, kernel_size=1)

        nn.init.xavier_normal_(self.feat.weight)
        nn.init.xavier_normal_(self.pos_conv.weight)
        nn.init.xavier_normal_(self.reg_conv.weight)
        nn.init.xavier_normal_(self.off_conv.weight)

        nn.init.constant_(self.pos_conv.bias, -math.log(0.99 / 0.01))
        nn.init.constant_(self.reg_conv.bias, 0)
        nn.init.constant_(self.off_conv.bias, 0)

    def forward(self, x):  # 定义前向传播
        feat = self.feat(x)
        feat = self.feat_sn(feat)
        feat = self.feat_act(feat)
        # add layer1
        # feat = self.feat_add1(feat)  # torch.Size([1, 128, 160, 320])
        # feat = self.feat_sn_add1(feat)
        # feat = self.feat_act_add1(feat)

        x_cls = self.pos_conv(feat)
        x_cls = torch.sigmoid(x_cls)  # torch.Size([1, 3, 160, 320])
        x_reg = self.reg_conv(feat)  # torch.Size([1, 2, 160, 320])
        x_off = self.off_conv(feat)  # torch.Size([1, 3, 160, 320])

        return x_cls, x_reg, x_off