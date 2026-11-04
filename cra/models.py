"""Backbones. All take [0,1] images, normalise internally, and expose
`features(x)` (penultimate representation used by every feature-space
detector), `forward(x)` and `head_weight()`."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


class Base(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("mean", torch.tensor(MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(STD).view(1, 3, 1, 1))

    def norm(self, x):
        return (x - self.mean) / self.std

    def forward(self, x):
        return self.head(self.features(x))

    def head_weight(self):
        return self.head.weight


class ResNet18(Base):
    def __init__(self, C, size):
        super().__init__()
        n = torchvision.models.resnet18(num_classes=C)
        n.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        n.maxpool = nn.Identity()
        self.body = nn.Sequential(n.conv1, n.bn1, n.relu, n.layer1, n.layer2, n.layer3, n.layer4,
                                  nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.head = nn.Linear(512, C)

    def features(self, x):
        return self.body(self.norm(x))


class VGG16(Base):
    def __init__(self, C, size):
        super().__init__()
        cfg = [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"]
        layers, c = [], 3
        for v in cfg:
            if v == "M":
                layers.append(nn.MaxPool2d(2))
            else:
                layers += [nn.Conv2d(c, v, 3, padding=1), nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                c = v
        self.body = nn.Sequential(*layers, nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.head = nn.Linear(512, C)

    def features(self, x):
        return self.body(self.norm(x))


class ViTS(Base):
    """ImageNet-pretrained ViT-S/16 fine-tuned at 224px (inputs upsampled on
    the fly). Features = pre-logit CLS representation."""

    def __init__(self, C, size, pretrained=True):
        super().__init__()
        import timm
        self.vit = timm.create_model("vit_small_patch16_224.augreg_in21k_ft_in1k",
                                     pretrained=pretrained, num_classes=0)
        self.head = nn.Linear(self.vit.num_features, C)
        self.mean.copy_(torch.tensor((0.5, 0.5, 0.5)).view(1, 3, 1, 1))
        self.std.copy_(torch.tensor((0.5, 0.5, 0.5)).view(1, 3, 1, 1))

    def features(self, x):
        x = F.interpolate(x, size=224, mode="bilinear", align_corners=False)
        return self.vit(self.norm(x))


ARCHS = {"resnet18": ResNet18, "vgg16": VGG16, "vit_s": ViTS}


def build_model(arch, C, size, **kw):
    return ARCHS[arch](C, size, **kw)
