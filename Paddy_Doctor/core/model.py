# -*- coding: utf-8 -*

# -------------------------------------------------------------------------------
# Author: LiuNing
# Contact: 2742229056@qq.com
# Software: PyCharm
# File: model.py
# Time: 6/26/19 6:57 PM
# Description: 
# -------------------------------------------------------------------------------

from core.loss import *
import torchvision.models as models
from core.utils import *
from core.seed import *
from block.senet import seresnet50, seresnext50_32x4d


########################################################################
# agriculture_model1
#
class agriculture_model1(nn.Module):
    def __init__(self, classes=10, stride=1):
        super(agriculture_model1, self).__init__()
        self.classes = classes

        model = models.resnet50(pretrained=True)
        # model = models(pretrained=True)
        if stride == 1:
            model.layer4[0].downsample[0].stride = (1, 1)
            model.layer4[0].conv2.stride = (1, 1)
        model.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.backbone = model
        self.fc7 = nn.Sequential(
            nn.Linear(2048, 512),
            nn.BatchNorm1d(512),
            nn.Dropout(0.5),
        )
        self.cls = nn.Linear(512, classes)

        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.loss = LabelSmoothingCrossEntropy(smoothing=0.1)

    def fix_params(self, is_training=True):
        for p in self.backbone.parameters():
            p.requires_grad = is_training

    def get_loss(self, logits, labels):
        loss = self.loss(logits[0], labels) + self.weight_loss(logits[1])
        return loss

    def features(self, x):
        # backbone
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)

        x = self.global_avg_pool(x)
        x = x.view(x.size(0), x.size(1))
        x = self.fc7(x)
        x = self.cls(x)
        return [x, ]

    def weight_loss(self, weight):
        mask = torch.ones_like(weight)
        for i in range(mask.shape[0]):
            for j in range(mask.shape[1]):
                mask[i, j, j] = -1
        nw = mask * weight
        tmp, _ = torch.max(nw, dim=1)
        tmp, _ = torch.max(tmp, dim=1)
        tmp2 = 0.000002 * torch.sum(torch.sum(nw, dim=1), dim=1)
        loss = torch.mean(tmp + tmp2)
        return loss

    def forward(self, x, label=None):
        # backbone
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)

        weight = x.view(x.size(0), x.size(1), -1)
        weight = l2_normalize(weight, dim=2)
        weight = torch.bmm(weight, torch.transpose(weight, 1, 2))

        x = self.global_avg_pool(x)
        x = x.view(x.size(0), x.size(1))
        x = self.fc7(x)
        x = self.cls(x)

        return [x, weight]
