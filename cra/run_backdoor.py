"""Poison a dataset and train a victim model.

python -m cra.run_backdoor --dataset cifar10 --arch resnet18 --attack badnets --rate 0.1 --seed 0

Writes runs/<dataset>/<arch>/<attack>_r<rate>_s<seed>/ with
  model.pt   final weights           early.pt   epoch-2 weights (loss-early detector)
  poison.pt  modified training indices/images/labels, payload & cover indices,
             triggered test set (ASR), triggered pool images (contamination study)
  meta.json  clean accuracy, ASR, configuration
"""

import argparse
import json
import os
import time

import torch

from .attacks import build_attack, poison_training_set, triggered_test_set, _apply_u8
from .data import load_splits, default_target
from .train import EPOCHS, accuracy, train_model

RUNS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "runs")


def run_dir(dataset, arch, attack, rate, seed):
    return os.path.join(RUNS, dataset, arch, f"{attack}_r{rate:g}_s{seed}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="cifar10")
    p.add_argument("--arch", default="resnet18")
    p.add_argument("--attack", default="badnets")
    p.add_argument("--rate", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--epochs", type=int, default=None)
    a = p.parse_args()
    out = run_dir(a.dataset, a.arch, a.attack, a.rate, a.seed)
    if os.path.exists(os.path.join(out, "meta.json")):
        print("exists", out); return
    os.makedirs(out, exist_ok=True)
    logf = open(os.path.join(out, "train.log"), "a")

    def log(m):
        print(m, flush=True); logf.write(m + "\n"); logf.flush()

    dev = torch.device("cuda")
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    S = load_splits(a.dataset)
    C, size, hflip = S["num_classes"], S["size"], S["hflip"]
    target = default_target(a.dataset, S["train_y"])
    epochs = a.epochs or EPOCHS[(a.arch, a.dataset)]
    t0 = time.time()

    if a.attack == "none":
        xp, yp = S["train_x"], S["train_y"]
        payload = cover = torch.empty(0, dtype=torch.long)
        attack = None
    else:
        attack = build_attack(a.attack, size, C, target, a.seed)
        if a.attack == "trojan":
            log("training clean surrogate for trojan trigger optimisation")
            sur, _ = train_model(S["train_x"], S["train_y"], "resnet18", C, size, hflip, 5, a.seed + 7, dev, log=log)
            attack.fit(sur, S["train_x"][:5000].to(dev).float() / 255)
            del sur
        attack.to(dev)
        xp, yp, payload, cover = poison_training_set(attack, S["train_x"], S["train_y"], a.rate, a.seed, dev)
    log(f"{a.dataset}/{a.arch}/{a.attack} rate={a.rate} target={target} payload={payload.numel()} "
        f"cover={cover.numel()} N={yp.numel()} epochs={epochs}")

    if attack is not None:
        asr_x, asr_y = triggered_test_set(attack, S["test_x"], S["test_y"], dev)
    else:  # no attack: ASR measured with a BadNets patch, for reference only
        ref = build_attack("badnets", size, C, target, a.seed).to(dev)
        asr_x, asr_y = triggered_test_set(ref, S["test_x"], S["test_y"], dev)

    def ev(m):
        return f"CA={accuracy(m, S['test_x'], S['test_y'], dev):.4f} ASR={accuracy(m, asr_x, asr_y, dev):.4f}"

    model, early = train_model(xp, yp, a.arch, C, size, hflip, epochs, a.seed, dev,
                               snapshot_epoch=2, log=log, eval_fn=ev)
    ca = accuracy(model, S["test_x"], S["test_y"], dev)
    asr = accuracy(model, asr_x, asr_y, dev)

    changed = torch.where((xp != S["train_x"]).flatten(1).any(1) | (yp != S["train_y"]))[0]
    pool_trig = None
    if attack is not None:
        pool_trig = _apply_u8(attack.apply_test, S["pool_x"], dev)
    torch.save(model.state_dict(), os.path.join(out, "model.pt"))
    torch.save(early, os.path.join(out, "early.pt"))
    torch.save({"changed_idx": changed, "changed_x": xp[changed], "changed_y": yp[changed],
                "payload": payload, "cover": cover, "asr_x": asr_x, "asr_y": asr_y,
                "pool_trig": pool_trig, "target": target}, os.path.join(out, "poison.pt"))
    meta = dict(vars(a), epochs=epochs, target=target, clean_acc=ca, asr=asr,
                n_payload=int(payload.numel()), n_cover=int(cover.numel()), n_train=int(yp.numel()),
                minutes=(time.time() - t0) / 60)
    json.dump(meta, open(os.path.join(out, "meta.json"), "w"), indent=1)
    log(json.dumps(meta))


if __name__ == "__main__":
    main()
