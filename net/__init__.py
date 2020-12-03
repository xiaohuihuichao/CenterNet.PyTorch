from .fpn import FPN, CBA
from .resnet import resnet18

import torch.nn as nn


class Net(nn.Module):
    def __init__(self, fpn_out_channel, num_classes):
        super().__init__()
        self.backbone = resnet18()
        self.fpn = FPN(self.backbone.out_channels_list, fpn_out_channel)
        
        self.hm_head = nn.Sequential(
            CBA(fpn_out_channel, 64, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(64, num_classes, kernel_size=1),
            nn.Sigmoid(),
        )
        
        self.xy_head = nn.Sequential(
            CBA(fpn_out_channel, 64, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(64, 2, kernel_size=1),
        )
        
        self.wh_head = nn.Sequential(
            CBA(fpn_out_channel, 64, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(64, 2, kernel_size=1),
        )
        
    def forward(self, x):
        x = self.backbone(x)
        x = self.fpn(x)
        
        hm_preds = self.hm_head(x)
        xy_preds = self.xy_head(x)
        wh_preds = self.wh_head(x)
        return hm_preds, xy_preds, wh_preds
    