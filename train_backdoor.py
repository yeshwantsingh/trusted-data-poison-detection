"""Poison a CIFAR-10 training set with a chosen backdoor attack and train a
classifier on it -- i.e., play the attacker who controls the training
pipeline, which is the threat model BadNets/Blend/SIG/Dynamic/WaNet/Trojan
all assume.

Examples:
  python train_backdoor.py --attack badnets --target-label 0 --arch resnet18 --epochs 60
  python train_backdoor.py --attack sig     --target-label 0 --arch resnet18 --epochs 60 --poison-rate 0.3
  python train_backdoor.py --attack trojan  --target-label 0 --arch resnet18 --epochs 60

Smoke test on CPU:
  python train_backdoor.py --attack wanet --arch smallcnn --epochs 1 --subset 512
"""

import argparse
import os

import torch
from torch.optim import SGD
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader

from src.backdoor_attacks import PoisonedDataset, build_trigger
from src.data import get_cifar10_loaders
from src.defenses import train_epoch_standard
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
    p.add_argument("--attack", choices=["badnets", "blend", "sig", "wanet", "dynamic", "trojan"], required=True)
    p.add_argument("--arch", choices=["resnet18", "smallcnn"], default="resnet18")
    p.add_argument("--target-label", type=int, default=0)
    p.add_argument("--poison-rate", type=float, default=0.1)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--pretrain-epochs", type=int, default=5,
                    help="clean epochs before optimizing the trigger (trojan only: it needs "
                         "a 'victim' reference model to run activation maximization against)")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--data-dir", default="./data")
    p.add_argument("--out-dir", default="./checkpoints")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--subset", type=int, default=None, help="use a small subset of the data for smoke tests")
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

    total_epochs = args.epochs
    if args.attack == "trojan":
        print(f"pretraining a clean reference model for {args.pretrain_epochs} epochs "
              f"before optimizing the Trojan trigger...", flush=True)
        for _ in range(args.pretrain_epochs):
            train_epoch_standard(model, train_loader, optimizer, device)
        trigger = build_trigger("trojan", reference_model=model, seed=args.seed)
        total_epochs = max(1, args.epochs - args.pretrain_epochs)
    else:
        trigger = build_trigger(args.attack, seed=args.seed)

    poisoned_dataset = PoisonedDataset(train_loader.dataset, trigger, args.target_label, args.poison_rate, seed=args.seed)
    poisoned_loader = DataLoader(
        poisoned_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=True
    )

    print(
        f"device={device} attack={args.attack} arch={args.arch} target={args.target_label} "
        f"poison_rate={args.poison_rate} epochs={total_epochs}",
        flush=True,
    )
    for epoch in range(1, total_epochs + 1):
        loss, acc = train_epoch_standard(model, poisoned_loader, optimizer, device)
        scheduler.step()
        clean_acc = evaluate_clean(model, test_loader, device)
        print(
            f"epoch {epoch:3d}/{total_epochs} | train_loss {loss:.4f} | "
            f"train_acc {acc:.4f} | test_clean_acc {clean_acc:.4f}",
            flush=True,
        )

    ckpt_path = os.path.join(args.out_dir, f"{args.arch}_backdoor_{args.attack}.pt")
    torch.save(
        {
            "model_object": unwrap(model).cpu(),
            "arch": args.arch,
            "attack": args.attack,
            "target_label": args.target_label,
            "poison_rate": args.poison_rate,
            "seed": args.seed,
            "trigger": trigger,
        },
        ckpt_path,
    )
    print(f"saved checkpoint to {ckpt_path}")


if __name__ == "__main__":
    main()
