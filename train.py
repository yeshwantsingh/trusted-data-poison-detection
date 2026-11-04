"""Train a CIFAR-10 classifier with a chosen robustness method.

Examples:
  python train.py --method standard --arch resnet18 --epochs 60
  python train.py --method pgd      --arch resnet18 --epochs 60 --eps 0.0314
  python train.py --method trades   --arch resnet18 --epochs 60 --beta 6.0
  python train.py --method standard --arch resnet18 --epochs 60 --sigma 0.25   # for randomized smoothing

Smoke test on CPU (few minutes):
  python train.py --method pgd --arch smallcnn --epochs 1 --subset 512
"""

import argparse
import os

import torch
from torch.optim import SGD
from torch.optim.lr_scheduler import MultiStepLR

from src.data import get_cifar10_loaders
from src.defenses import (
    train_epoch_gaussian_augment,
    train_epoch_pgd_at,
    train_epoch_standard,
    train_epoch_trades,
)
from src.models import build_model, to_device, unwrap


def evaluate_clean(model, loader, device):
    model.eval()
    correct, n = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            correct += (model(x).argmax(1) == y).sum().item()
            n += x.size(0)
    return correct / n


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--method", choices=["standard", "pgd", "trades"], default="standard")
    p.add_argument("--arch", choices=["resnet18", "smallcnn"], default="resnet18")
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--eps", type=float, default=8 / 255, help="Linf attack budget for pgd/trades training")
    p.add_argument("--alpha", type=float, default=2 / 255, help="PGD step size")
    p.add_argument("--steps", type=int, default=7, help="PGD steps during training")
    p.add_argument("--beta", type=float, default=6.0, help="TRADES robustness weight")
    p.add_argument("--sigma", type=float, default=None,
                    help="If set, train with Gaussian-noise augmentation at this sigma "
                         "(base classifier for randomized smoothing certification)")
    p.add_argument("--data-dir", default="./data")
    p.add_argument("--out-dir", default="./checkpoints")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--subset", type=int, default=None, help="Use a small subset of the data for smoke tests")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)

    train_loader, test_loader = get_cifar10_loaders(
        args.data_dir, args.batch_size, num_workers=args.num_workers, subset=args.subset
    )
    model = to_device(build_model(args.arch), device)
    optimizer = SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    scheduler = MultiStepLR(
        optimizer, milestones=[max(1, int(args.epochs * 0.5)), max(2, int(args.epochs * 0.75))], gamma=0.1
    )

    print(f"device={device} method={args.method} arch={args.arch} epochs={args.epochs} sigma={args.sigma}", flush=True)

    for epoch in range(1, args.epochs + 1):
        if args.sigma is not None:
            loss, acc = train_epoch_gaussian_augment(model, train_loader, optimizer, device, args.sigma)
        elif args.method == "standard":
            loss, acc = train_epoch_standard(model, train_loader, optimizer, device)
        elif args.method == "pgd":
            loss, acc = train_epoch_pgd_at(model, train_loader, optimizer, device, args.eps, args.alpha, args.steps)
        else:
            loss, acc = train_epoch_trades(
                model, train_loader, optimizer, device, args.eps, args.alpha, args.steps, args.beta
            )
        scheduler.step()
        clean_acc = evaluate_clean(model, test_loader, device)
        print(
            f"epoch {epoch:3d}/{args.epochs} | train_loss {loss:.4f} | "
            f"train_acc {acc:.4f} | test_clean_acc {clean_acc:.4f}",
            flush=True,
        )

    tag = f"{args.arch}_{args.method}" + (f"_sigma{args.sigma}" if args.sigma is not None else "")
    ckpt_path = os.path.join(args.out_dir, f"{tag}.pt")
    torch.save(
        {"model": unwrap(model).state_dict(), "arch": args.arch, "method": args.method, "sigma": args.sigma},
        ckpt_path,
    )
    print(f"saved checkpoint to {ckpt_path}")


if __name__ == "__main__":
    main()
