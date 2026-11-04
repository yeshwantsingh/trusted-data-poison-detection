"""Evaluate clean accuracy and empirical robust accuracy (FGSM, PGD, C&W) of
a trained checkpoint.

Example:
  python evaluate.py --checkpoint ./checkpoints/resnet18_pgd.pt --max-batches 10
"""

import argparse

import torch

from src.attacks import cw_l2_attack, fgsm_attack, pgd_attack
from src.data import get_cifar10_loaders
from src.models import build_model


def robust_accuracy(model, loader, device, attack_fn, max_batches=None):
    model.eval()
    correct, n = 0, 0
    for i, (x, y) in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        x, y = x.to(device), y.to(device)
        x_adv = attack_fn(model, x, y)
        with torch.no_grad():
            correct += (model(x_adv).argmax(1) == y).sum().item()
        n += x.size(0)
    return correct / n


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data-dir", default="./data")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--eps", type=float, default=8 / 255)
    p.add_argument("--pgd-steps", type=int, default=20)
    p.add_argument("--pgd-alpha", type=float, default=2 / 255)
    p.add_argument("--cw-steps", type=int, default=200)
    p.add_argument("--cw-batches", type=int, default=5, help="C&W is slow; evaluate it on fewer batches by default")
    p.add_argument("--max-batches", type=int, default=None, help="Limit FGSM/PGD test batches too, for quick runs")
    p.add_argument("--subset", type=int, default=None)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=device)
    model = build_model(ckpt["arch"]).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    _, test_loader = get_cifar10_loaders(args.data_dir, args.batch_size, subset=args.subset)

    clean_acc = robust_accuracy(model, test_loader, device, lambda m, x, y: x, args.max_batches)
    fgsm_acc = robust_accuracy(
        model, test_loader, device, lambda m, x, y: fgsm_attack(m, x, y, args.eps), args.max_batches
    )
    pgd_acc = robust_accuracy(
        model, test_loader, device,
        lambda m, x, y: pgd_attack(m, x, y, args.eps, args.pgd_alpha, args.pgd_steps),
        args.max_batches,
    )
    cw_acc = robust_accuracy(
        model, test_loader, device,
        lambda m, x, y: cw_l2_attack(m, x, y, steps=args.cw_steps),
        args.max_batches if args.max_batches is not None else args.cw_batches,
    )

    print(f"checkpoint             : {args.checkpoint}")
    print(f"clean accuracy         : {clean_acc:.4f}")
    print(f"FGSM      (eps={args.eps:.4f}) : {fgsm_acc:.4f}")
    print(f"PGD-{args.pgd_steps:<3d} (eps={args.eps:.4f}) : {pgd_acc:.4f}")
    print(f"C&W L2    ({args.cw_steps} steps) : {cw_acc:.4f}")


if __name__ == "__main__":
    main()
