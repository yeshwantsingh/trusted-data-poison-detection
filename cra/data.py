"""Dataset loading into in-memory uint8 tensors (N, 3, H, W).

Everything downstream (poisoning, GPU augmentation, feature extraction)
works on these tensors, so each dataset is decoded once and cached to
data/cache/<name>.pt.

Splits returned by `load_splits`:
  train   : the (to-be-poisoned) training set
  pool    : a class-balanced trusted-clean pool, DISJOINT from train, from
            which every clean reference set (any size) is drawn
  test    : the official test/val split, used only for clean accuracy and ASR
"""

import os

import numpy as np
import torch

ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
CACHE = os.path.join(ROOT, "cache")

INFO = {
    # name: (image size, num classes, pool images per class, horizontal flip ok)
    "cifar10": (32, 10, 100, True),
    "gtsrb": (32, 43, 25, False),   # traffic signs: flips change semantics
    "tiny": (64, 200, 25, True),
}


def _cifar10():
    import pickle
    d = os.path.join(ROOT, "cifar-10-batches-py")
    xs, ys = [], []
    for i in range(1, 6):
        with open(os.path.join(d, f"data_batch_{i}"), "rb") as f:
            b = pickle.load(f, encoding="bytes")
        xs.append(b[b"data"]); ys += b[b"labels"]
    with open(os.path.join(d, "test_batch"), "rb") as f:
        b = pickle.load(f, encoding="bytes")
    tx = np.concatenate(xs).reshape(-1, 3, 32, 32)
    vx = b[b"data"].reshape(-1, 3, 32, 32)
    return tx, np.array(ys), vx, np.array(b[b"labels"])


def _resize_stack(paths, size):
    from PIL import Image
    out = np.empty((len(paths), 3, size, size), dtype=np.uint8)
    for i, p in enumerate(paths):
        im = Image.open(p).convert("RGB")
        if im.size != (size, size):
            im = im.resize((size, size), Image.BILINEAR)
        out[i] = np.asarray(im).transpose(2, 0, 1)
    return out


def _gtsrb():
    """Official GTSRB train (26,640) / test (12,630) split, via the HF parquet
    mirror tanganke/gtsrb (the erda.dk host is unreachable from the cluster)."""
    import io
    import pyarrow.parquet as pq
    out = []
    for split in ("train", "test"):
        t = pq.read_table(os.path.join(ROOT, "gtsrb", f"{split}.parquet")).to_pydict()
        out += [_resize_stack([io.BytesIO(im["bytes"]) for im in t["image"]], 32), np.array(t["label"])]
    return tuple(out)


def _tiny():
    d = os.path.join(ROOT, "tiny", "tiny-imagenet-200")
    wnids = sorted(open(os.path.join(d, "wnids.txt")).read().split())
    idx = {w: i for i, w in enumerate(wnids)}
    tp, ty = [], []
    for w in wnids:
        imgdir = os.path.join(d, "train", w, "images")
        for fn in sorted(os.listdir(imgdir)):
            tp.append(os.path.join(imgdir, fn)); ty.append(idx[w])
    vp, vy = [], []
    for line in open(os.path.join(d, "val", "val_annotations.txt")):
        fn, w = line.split("\t")[:2]
        vp.append(os.path.join(d, "val", "images", fn)); vy.append(idx[w])
    return _resize_stack(tp, 64), np.array(ty), _resize_stack(vp, 64), np.array(vy)


def load_raw(name):
    os.makedirs(CACHE, exist_ok=True)
    path = os.path.join(CACHE, f"{name}.pt")
    if not os.path.exists(path):
        tx, ty, vx, vy = {"cifar10": _cifar10, "gtsrb": _gtsrb, "tiny": _tiny}[name]()
        torch.save({"tx": torch.from_numpy(tx), "ty": torch.from_numpy(ty).long(),
                    "vx": torch.from_numpy(vx), "vy": torch.from_numpy(vy).long()}, path + ".tmp")
        os.replace(path + ".tmp", path)
    return torch.load(path)


def load_splits(name, split_seed=0):
    """Fixed (seed-independent by default) train/pool split so that every run
    on a dataset shares the same trusted pool and the same training images."""
    size, C, per_class, _ = INFO[name]
    raw = load_raw(name)
    tx, ty = raw["tx"], raw["ty"]
    g = np.random.RandomState(1000 + split_seed)
    pool_idx = []
    for c in range(C):
        ids = np.where(ty.numpy() == c)[0]
        k = min(per_class, len(ids) // 5)  # never take more than 20% of a class
        pool_idx.append(g.choice(ids, k, replace=False))
    pool_idx = np.sort(np.concatenate(pool_idx))
    mask = np.ones(len(ty), dtype=bool); mask[pool_idx] = False
    train_idx = np.where(mask)[0]
    return {
        "train_x": tx[train_idx], "train_y": ty[train_idx],
        "pool_x": tx[pool_idx], "pool_y": ty[pool_idx],
        "test_x": raw["vx"], "test_y": raw["vy"],
        "size": size, "num_classes": C, "hflip": INFO[name][3],
    }


def default_target(name, train_y):
    """Target class = the most frequent training class (class 0 on balanced
    CIFAR-10/Tiny-ImageNet; a large class on GTSRB so clean-label attacks
    have enough target-class images to poison)."""
    return int(torch.bincount(train_y).argmax())
