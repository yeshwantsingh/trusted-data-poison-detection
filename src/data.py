"""CIFAR-10 data loading. Inputs are kept in raw [0, 1] pixel range (no
normalization here) since attacks/certification define their epsilon in
pixel space; see src/models.py for where normalization is applied instead.
"""

import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader, Subset


def get_cifar10_loaders(data_dir, batch_size=128, num_workers=4, subset=None):
    train_tf = T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
    ])
    test_tf = T.ToTensor()

    train_set = torchvision.datasets.CIFAR10(data_dir, train=True, download=True, transform=train_tf)
    test_set = torchvision.datasets.CIFAR10(data_dir, train=False, download=True, transform=test_tf)

    if subset is not None:
        train_set = Subset(train_set, range(min(subset, len(train_set))))
        test_set = Subset(test_set, range(min(max(subset // 5, 1), len(test_set))))

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True,
    )
    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    return train_loader, test_loader
