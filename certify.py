"""Certify a Gaussian-smoothed classifier (Cohen, Rosenfeld & Kolter, ICML 2019).

Expects a checkpoint trained with `train.py --sigma <sigma>` (Gaussian-noise
augmentation matching the certification sigma).

Example:
  python certify.py --checkpoint ./checkpoints/resnet18_standard_sigma0.25.pt \\
      --sigma 0.25 --num-images 200 --n 10000
"""

import argparse
import csv
import os
import time

import torch

from src.data import get_cifar10_loaders
from src.models import build_model
from src.smoothing import Smooth


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--sigma", type=float, required=True)
    p.add_argument("--data-dir", default="./data")
    p.add_argument("--num-images", type=int, default=100)
    p.add_argument("--n0", type=int, default=100, help="samples for class selection")
    p.add_argument("--n", type=int, default=1000, help="samples for probability estimation")
    p.add_argument("--alpha", type=float, default=0.001, help="1 - confidence")
    p.add_argument("--batch-size", type=int, default=400)
    p.add_argument("--skip", type=int, default=1, help="evaluate every `skip`-th test image")
    p.add_argument("--out-csv", default="./results/certify.csv")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=device)
    model = build_model(ckpt["arch"]).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    _, test_loader = get_cifar10_loaders(args.data_dir, batch_size=1)
    smooth = Smooth(model, num_classes=10, sigma=args.sigma, device=device)

    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    rows = []
    seen = 0
    for i, (x, y) in enumerate(test_loader):
        if i % args.skip != 0:
            continue
        if seen >= args.num_images:
            break
        seen += 1
        x, y = x.to(device), y.item()
        t0 = time.time()
        pred, radius = smooth.certify(x, args.n0, args.n, args.alpha, args.batch_size)
        elapsed = time.time() - t0
        correct = int(pred == y)
        rows.append(
            {"idx": i, "label": y, "predict": pred, "radius": radius, "correct": correct, "seconds": elapsed}
        )
        print(
            f"image {i:5d} | label {y} | predict {pred:2d} | radius {radius:.4f} | "
            f"correct {correct} | {elapsed:.1f}s",
            flush=True,
        )

    with open(args.out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    certified_at_0 = sum(r["correct"] for r in rows) / len(rows)
    print(f"wrote {args.out_csv}")
    print(f"certified accuracy at r=0 (~= clean accuracy of smoothed model): {certified_at_0:.4f}")


if __name__ == "__main__":
    main()
