"""Generate the experiment queues (see paper Sec. 'Experimental protocol').

G1 main     : {cifar10, gtsrb, tiny} x ResNet-18 x 9 attacks (+ no attack) x 3 seeds
G2 arch     : cifar10 x {VGG-16, ViT-S/16 (IN-pretrained, fine-tuned)} x 9 attacks (+none) x 3 seeds
G3 rate     : cifar10 x ResNet-18 x 6 standard attacks x rates {0.5,1,5,20}% x 3 seeds (10% is in G1)
Detection runs on every model (sweeps of reference size/contamination on G1 cifar10 + G2/G3 seed 0).
Remove-and-retrain on G1 (all seeds) for a fixed list of detectors/rules.
"""
import os
from .attacks import STANDARD, ADAPTIVE
from .run_backdoor import run_dir

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEEDS = [0, 1, 2]
RATE = {**{a: 0.1 for a in STANDARD}, **{a: 0.01 for a in ADAPTIVE}, "none": 0.0}
RETRAIN_FLAGS = [
    "oracle",
    "cafd|clean|cleancal", "cafd|train|topq", "maha|clean|cleancal", "maha|train|topq",
    "knn|clean|cleancal", "spectral|train|topq", "ac|train|native",
    "scan|clean|native", "asset|clean|native", "loss_early|none|topq",
]


def rel(p):
    return os.path.relpath(p, ROOT)


def main():
    runs = []  # (dataset, arch, attack, rate, seed, sweeps, retrain)
    for ds in ["cifar10", "gtsrb", "tiny"]:
        for atk in STANDARD + ADAPTIVE + ["none"]:
            for s in SEEDS:
                runs.append((ds, "resnet18", atk, RATE[atk], s, ds == "cifar10" or s == 0, atk != "none" and (ds == "cifar10" or s == 0)))
    for arch in ["vgg16", "vit_s"]:
        for atk in STANDARD + ADAPTIVE + ["none"]:
            for s in SEEDS:
                runs.append(("cifar10", arch, atk, RATE[atk], s, s == 0, False))
    for atk in STANDARD:
        for r in [0.005, 0.01, 0.05, 0.2]:
            for s in SEEDS:
                runs.append(("cifar10", "resnet18", atk, r, s, s == 0, False))
    os.makedirs(os.path.join(ROOT, "queue"), exist_ok=True)
    tr = open(os.path.join(ROOT, "queue", "train.txt"), "w")
    de = open(os.path.join(ROOT, "queue", "detect.txt"), "w")
    rt = open(os.path.join(ROOT, "queue", "retrain.txt"), "w")
    # cheap/high-value first: cifar10 main, then rate/arch, then gtsrb, tiny
    order = {"cifar10": 0, "gtsrb": 2, "tiny": 3}
    runs.sort(key=lambda r: (order[r[0]] + (1 if (r[1] != "resnet18" or r[3] not in (0.1, 0.01, 0.0)) else 0), r[4]))
    for ds, arch, atk, rate, s, sw, rtr in runs:
        d = run_dir(ds, arch, atk, rate, s)
        tr.write(f"{rel(d)}/meta.json\tpython -m cra.run_backdoor --dataset {ds} --arch {arch} --attack {atk} --rate {rate} --seed {s}\n")
        de.write(f"{rel(d)}/detect.json\t{rel(d)}/meta.json\tpython -m cra.detect {rel(d)}" + (" --sweeps" if sw else "") + "\n")
        if rtr:
            for f in RETRAIN_FLAGS:
                tag = f.replace("|", "__")
                rt.write(f"{rel(d)}/retrain/{tag}.json\t{rel(d)}/detect.json\tpython -m cra.retrain {rel(d)} --flag '{f}'\n")
    print(len(runs), "training runs")


if __name__ == "__main__":
    main()
