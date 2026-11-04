"""Model definitions for CIFAR-10 adversarial robustness experiments."""

import torch
import torch.nn as nn
import torchvision

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2471, 0.2435, 0.2616)


class Normalize(nn.Module):
    def __init__(self, mean, std):
        super().__init__()
        self.register_buffer("mean", torch.tensor(mean).view(1, -1, 1, 1))
        self.register_buffer("std", torch.tensor(std).view(1, -1, 1, 1))

    def forward(self, x):
        return (x - self.mean) / self.std


class SmallCNN(nn.Module):
    """Compact CNN, fast enough for CPU smoke tests."""

    def __init__(self, num_classes=10):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 256), nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward_features(self, x):
        """Returns (logits, [f1, f2]) -- intermediate feature maps, used by
        attention/activation-based backdoor defenses (NAD, CBD, DBD, Trojan
        trigger optimization).
        """
        f1 = self.block1(x)
        f2 = self.block2(f1)
        logits = self.classifier(f2)
        return logits, [f1, f2]

    def forward(self, x):
        logits, _ = self.forward_features(x)
        return logits


class ResNet18CIFAR(nn.Module):
    """torchvision resnet18 adapted for 32x32 inputs (3x3 stem, no maxpool),
    exposing intermediate block outputs via forward_features.
    """

    def __init__(self, num_classes=10):
        super().__init__()
        net = torchvision.models.resnet18(num_classes=num_classes)
        net.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        net.maxpool = nn.Identity()
        self.net = net

    def forward_features(self, x):
        n = self.net
        x = n.relu(n.bn1(n.conv1(x)))
        x = n.maxpool(x)
        f1 = n.layer1(x)
        f2 = n.layer2(f1)
        f3 = n.layer3(f2)
        f4 = n.layer4(f3)
        pooled = torch.flatten(n.avgpool(f4), 1)
        logits = n.fc(pooled)
        return logits, [f1, f2, f3, f4]

    def forward(self, x):
        logits, _ = self.forward_features(x)
        return logits


class NormalizedModel(nn.Module):
    """Wraps a classifier so attacks/certification can operate directly on
    raw [0, 1] pixel inputs. Perturbation budgets (eps) are defined in pixel
    space in the literature (Madry et al. 2018, RobustBench); folding
    normalization into the model keeps that budget meaningful regardless of
    the network's internal preprocessing.
    """

    def __init__(self, net, mean=CIFAR10_MEAN, std=CIFAR10_STD):
        super().__init__()
        self.normalize = Normalize(mean, std)
        self.net = net

    def forward(self, x, return_features=False):
        x = self.normalize(x)
        if return_features:
            return self.net.forward_features(x)
        return self.net(x)


def build_model(arch="resnet18", num_classes=10):
    if arch == "resnet18":
        base = ResNet18CIFAR(num_classes)
    elif arch == "smallcnn":
        base = SmallCNN(num_classes)
    else:
        raise ValueError(f"unknown arch '{arch}'")
    return NormalizedModel(base)


def to_device(model, device):
    """Moves `model` to `device`, wrapping it in nn.DataParallel when more
    than one CUDA device is visible so training batches split across all of
    them. Do NOT use this for models that get passed to
    `src.backdoor_defenses.mcr_defense`: that function calls
    `torch.func.functional_call` directly against the model's own
    parameters to fit a Bezier curve in weight space, which requires the
    plain (unwrapped) module -- DataParallel's replicate/scatter forward
    isn't compatible with that.
    """
    model = model.to(device)
    if device.type == "cuda" and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    return model


def unwrap(model):
    """Inverse of to_device's DataParallel wrapping -- use before saving a
    checkpoint so it stays loadable regardless of how many GPUs trained it.
    """
    return model.module if isinstance(model, nn.DataParallel) else model
