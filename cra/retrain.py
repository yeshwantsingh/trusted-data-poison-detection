"""Remove-and-retrain: drop the samples a detector flagged and train a fresh
model from scratch on the rest (the standard downstream test for poison
isolation, e.g. [Tran+18, Chen+18, Qi+23]). The repair step is identical
for every detector, so differences are attributable to isolation alone.

python -m cra.retrain runs/cifar10/resnet18/badnets_r0.1_s0 --flag "cafd|clean|cleancal"
  special flags: "oracle" (remove exactly the payload), "none" (no removal)
"""

import argparse
import json
import os

import numpy as np
import torch

from .detect import load_run
from .train import accuracy, train_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--flag", required=True)
    a = ap.parse_args()
    tag = a.flag.replace("|", "__")
    od = os.path.join(a.run, "retrain"); os.makedirs(od, exist_ok=True)
    out = os.path.join(od, f"{tag}.json")
    if os.path.exists(out):
        return
    meta, S, P, xp, yp, gt = load_run(a.run)
    if a.flag == "oracle":
        removed = np.where(gt == 1)[0]
    elif a.flag == "none":
        removed = np.array([], dtype=int)
    elif "@" in a.flag:
        # e.g. "maha|clean|cleancal@0.2": recompute the clean-fitted score and apply the
        # conformal rule at a different alpha (same reference set as the main config)
        from .detect import FITTERS, MAIN_PER_CLASS, conformal_threshold, crossfit_ref_scores, make_reference
        from .models import build_model
        from .train import predict
        base, al = a.flag.split("@")
        sc = base.split("|")[0]
        dev0 = torch.device("cuda")
        m = build_model(meta["arch"], S["num_classes"], S["size"]).to(dev0)
        m.load_state_dict(torch.load(os.path.join(a.run, "model.pt"))); m.eval()
        Ft = predict(m, xp, dev0, feats=True).cpu().numpy().astype(np.float64)
        Fp = predict(m, S["pool_x"], dev0, feats=True).cpu().numpy().astype(np.float64)
        ypool = S["pool_y"].numpy()
        ri = make_reference(ypool, MAIN_PER_CLASS[meta["dataset"]], 0, S["num_classes"])
        fit = FITTERS[sc]
        s = fit(Fp[ri], ypool[ri], S["num_classes"])(Ft, yp.numpy())
        thr = conformal_threshold(crossfit_ref_scores(fit, Fp[ri], ypool[ri], S["num_classes"]), float(al))
        removed = np.where(s > thr)[0]
        del m
    else:
        removed = np.load(os.path.join(a.run, "flags.npz"))[tag]
    keep = np.ones(len(yp), dtype=bool); keep[removed] = False
    dev = torch.device("cuda")
    torch.backends.cudnn.benchmark = True
    model, _ = train_model(xp[keep], yp[keep], meta["arch"], S["num_classes"], S["size"], S["hflip"],
                           meta["epochs"], meta["seed"] + 100, dev, log=lambda m: None)
    res = dict(flag=a.flag, removed=int(len(removed)), removed_payload=int(gt[removed].sum()),
               payload=int(gt.sum()), clean_acc=accuracy(model, S["test_x"], S["test_y"], dev),
               asr=accuracy(model, P["asr_x"], P["asr_y"], dev))
    json.dump(res, open(out, "w"))
    print(res)


if __name__ == "__main__":
    main()
