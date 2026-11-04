"""Evaluate clean accuracy and Attack Success Rate (ASR) of a (possibly
defended) backdoor checkpoint.

ASR is the fraction of originally-non-target test images that get
classified as the target label once the trigger is stamped onto them --
a successful defense should keep clean accuracy high while driving ASR
down towards random-guess (1 / num_classes).

Examples:
  python evaluate_backdoor.py --checkpoint ./checkpoints/resnet18_backdoor_badnets.pt
  python evaluate_backdoor.py --checkpoint ./checkpoints/resnet18_backdoor_badnets_defended_abl.pt
"""

import argparse

import torch
from torch.utils.data import DataLoader

from src.backdoor_attacks import TriggeredEvalDataset
from src.data import get_cifar10_loaders


def accuracy(model, loader, device):
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
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data-dir", default="./data")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--subset", type=int, default=None)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # map_location="cpu": see defend_backdoor.py -- the pickled trigger's
    # tensors must stay CPU-resident since TriggeredEvalDataset poisons CPU
    # images, not device-resident ones.
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = ckpt["model_object"].to(device)
    model.eval()

    _, test_loader = get_cifar10_loaders(args.data_dir, args.batch_size, subset=args.subset)
    clean_acc = accuracy(model, test_loader, device)

    asr_dataset = TriggeredEvalDataset(test_loader.dataset, ckpt["trigger"], ckpt["target_label"])
    asr_loader = DataLoader(asr_dataset, batch_size=args.batch_size, shuffle=False)
    asr = accuracy(model, asr_loader, device)

    print(f"checkpoint      : {args.checkpoint}")
    print(f"attack          : {ckpt['attack']} (target label {ckpt['target_label']}, poison rate {ckpt['poison_rate']})")
    if "defense" in ckpt:
        print(f"defense applied : {ckpt['defense']}")
    print(f"clean accuracy  : {clean_acc:.4f}")
    print(f"attack success rate (ASR): {asr:.4f}  (lower is better after a defense; "
          f"~0.1 is random-guess for 10 classes)")


if __name__ == "__main__":
    main()
