"""Poison-sample detection with a controlled calibration ablation.

Every detector is a per-sample score s(x) (higher = more suspicious) built from
two separable ingredients:

  REFERENCE  where the score's statistics (centroids, covariance, PCA basis,
             singular vectors, neighbours, blend images) are fitted:
               train : the (poisoned) training set itself
               clean : a trusted clean reference set R (disjoint from training)
  THRESHOLD  how the flagged set is chosen from the scores:
               topq     : top q=5% of training samples (needs a guess of the rate)
               cleancal : s > 95th percentile of the scores of held-out clean
                          reference samples (cross-fitted when R is also the
                          reference, so clean scores are out-of-sample); targets
                          a 5% false-positive rate and uses no rate knowledge
Threshold-free metrics (AUROC, AUPRC, TPR at 5% FPR) are reported for every
score. Ground truth = payload samples (cover samples of adaptive attacks are
correctly labelled and count as clean).

python -m cra.detect runs/cifar10/resnet18/badnets_r0.1_s0 [--sweeps]
"""

import argparse
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans
from sklearn.covariance import LedoitWolf
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.mixture import GaussianMixture

from .data import load_splits
from .models import build_model
from .train import predict

Q = 0.05          # top-q rule
PCT = 95.0        # clean-calibrated rule (pre-registered; never tuned)
KFOLD = 10
MAIN_PER_CLASS = {"cifar10": 50, "gtsrb": 25, "tiny": 25}


# ---------------------------------------------------------------- metrics
def metrics(s, y, flag=None):
    s = np.nan_to_num(np.asarray(s, dtype=np.float64), nan=-1e9, posinf=1e9, neginf=-1e9)
    out = {}
    if 0 < y.sum() < len(y):
        out["auroc"] = float(roc_auc_score(y, s))
        out["auprc"] = float(average_precision_score(y, s))
        thr = np.percentile(s[y == 0], 95)
        out["tpr@fpr5"] = float((s[y == 1] > thr).mean())
    if flag is not None:
        tp = int((flag & (y == 1)).sum()); nf = int(flag.sum())
        out.update(precision=tp / max(nf, 1), recall=tp / max(int(y.sum()), 1),
                   flagged_frac=nf / len(y), fpr=int((flag & (y == 0)).sum()) / max(int((y == 0).sum()), 1))
    return out


def rule_topq(s):
    return s >= np.quantile(s, 1 - Q)


def conformal_threshold(s_ref, alpha=1 - PCT / 100):
    """Split-conformal outlier threshold [Bates+23]: the ceil((1-a)(n+1))-th
    smallest reference score. If clean training samples are exchangeable with
    the reference samples, P(clean sample flagged) <= alpha, for any poison
    rate (cross-fitted scores make this approximate, cf. jackknife+)."""
    s_ref = np.sort(np.asarray(s_ref))
    k = int(math.ceil((1 - alpha) * (len(s_ref) + 1)))
    return np.inf if k > len(s_ref) else s_ref[k - 1]


def rule_cleancal(s, s_ref):
    return s > conformal_threshold(s_ref)


# ---------------------------------------------------------------- reference-fitted feature scores
class PCAFit:
    def __init__(self, X, k):
        self.mu = X.mean(0)
        _, _, Vt = np.linalg.svd(X - self.mu, full_matrices=False)
        self.V = Vt[:min(k, Vt.shape[0])].T

    def __call__(self, X):
        return (X - self.mu) @ self.V


def fit_cafd(Fr, yr, C, k=10):
    """Clean-anchored (or train-anchored) feature deviation: distance to the
    class centroid in a k-dim PCA basis, normalised by mean class spread."""
    P = PCAFit(Fr, k)
    Z = P(Fr)
    mus, sig = np.zeros((C, Z.shape[1])), np.ones(C)
    gmu = Z.mean(0)
    for c in range(C):
        Zc = Z[yr == c]
        if len(Zc) == 0:
            mus[c] = gmu; continue
        mus[c] = Zc.mean(0)
        sig[c] = max(np.linalg.norm(Zc - mus[c], axis=1).mean(), 1e-6) if len(Zc) > 1 else 1.0
    return lambda X, y: np.linalg.norm(P(X) - mus[y], axis=1) / sig[y]


def fit_maha(Fr, yr, C, k=64):
    """Class-conditional Mahalanobis with tied Ledoit-Wolf covariance [Lee+18]."""
    k = min(k, max(2, len(Fr) // 4))
    P = PCAFit(Fr, k)
    Z = P(Fr)
    mus = np.zeros((C, Z.shape[1])); gmu = Z.mean(0)
    for c in range(C):
        mus[c] = Z[yr == c].mean(0) if (yr == c).any() else gmu
    R = Z - mus[yr]
    prec = LedoitWolf().fit(R).precision_

    def score(X, y):
        D = P(X) - mus[y]
        return np.einsum("ij,jk,ik->i", D, prec, D)
    return score


def fit_spectral(Fr, yr, C):
    """Spectral signatures [Tran+18]: squared projection on the top singular
    vector of the class-centred representations."""
    mus, vs = {}, {}
    for c in range(C):
        Fc = Fr[yr == c]
        if len(Fc) < 2:
            continue
        mus[c] = Fc.mean(0)
        _, _, Vt = np.linalg.svd(Fc - mus[c], full_matrices=False)
        vs[c] = Vt[0]

    def score(X, y):
        s = np.zeros(len(X))
        for c in np.unique(y):
            if c in mus:
                m = y == c
                s[m] = ((X[m] - mus[c]) @ vs[c]) ** 2
        return s
    return score


def fit_knn(Fr, yr, C, k=5, exclude_self=False):
    """Cosine distance to the k-th nearest same-class reference sample."""
    Fn = torch.from_numpy(Fr).float().cuda()
    Fn = F.normalize(Fn, dim=1)
    yr_t = torch.from_numpy(yr).cuda()

    def score(X, y):
        Xn = F.normalize(torch.from_numpy(X).float().cuda(), dim=1)
        yt = torch.from_numpy(y).cuda()
        out = torch.zeros(len(X), device="cuda")
        for c in torch.unique(yt):
            m = yt == c
            R = Fn[yr_t == c]
            if R.size(0) == 0:
                continue
            kk = min(k + int(exclude_self), R.size(0))
            for i0 in range(0, int(m.sum()), 8192):
                idx = torch.where(m)[0][i0:i0 + 8192]
                d = 1 - Xn[idx] @ R.T
                out[idx] = d.topk(kk, dim=1, largest=False).values[:, -1]
        return out.cpu().numpy()
    return score


FITTERS = {"cafd": fit_cafd, "maha": fit_maha, "spectral": fit_spectral, "knn": fit_knn}


def crossfit_ref_scores(fitter, Fr, yr, C, seed=0):
    """Out-of-sample scores for the reference set itself (K-fold, stratified;
    exact leave-one-out for k-NN, whose score depends on the reference size)."""
    if fitter is fit_knn:
        return fit_knn(Fr, yr, C, exclude_self=True)(Fr, yr)
    rng = np.random.RandomState(seed)
    fold = np.zeros(len(yr), dtype=int)
    for c in np.unique(yr):
        ids = np.where(yr == c)[0]
        fold[rng.permutation(ids)] = np.arange(len(ids)) % KFOLD
    s = np.zeros(len(yr))
    for k in range(KFOLD):
        te = fold == k
        if te.any() and (~te).any():
            s[te] = fitter(Fr[~te], yr[~te], C)(Fr[te], yr[te])
    return s


# ---------------------------------------------------------------- activation clustering (train-only by design)
def activation_clustering(Ft, yt, C, seed=0):
    """[Chen+18]: per class, ICA/PCA-10 + 2-means; soft score = how much
    closer a sample is to the smaller cluster than to the larger one; native
    rule flags the smaller cluster when its relative size < 0.35."""
    s = np.zeros(len(yt)); native = np.zeros(len(yt), dtype=bool)
    for c in range(C):
        m = np.where(yt == c)[0]
        if len(m) < 10:
            continue
        Z = PCAFit(Ft[m], 10)(Ft[m])
        km = KMeans(2, n_init=3, random_state=seed).fit(Z)
        small = int(np.bincount(km.labels_, minlength=2).argmin())
        d = km.transform(Z)
        s[m] = d[:, 1 - small] - d[:, small]
        if (km.labels_ == small).mean() < 0.35:
            native[m] = km.labels_ == small
    return s, native


# ---------------------------------------------------------------- SCAn (needs clean data by design)
def scan(Ft, yt, Fr, yr, C, iters=10):
    """Simplified SCAn [Tang+21]: within-class covariance S_e estimated on the
    clean reference; each class of the training set is untangled into two
    subgroups by alternating Fisher-direction assignment; the per-class
    log-likelihood-ratio J is robustly standardised across classes (median /
    MAD) and classes with J* > e^2 are flagged (minority subgroup)."""
    P = PCAFit(Fr, min(32, max(2, len(Fr) // 4)))
    Zr, Zt = P(Fr), P(Ft)
    mus = np.stack([Zr[yr == c].mean(0) if (yr == c).any() else Zr.mean(0) for c in range(C)])
    prec = LedoitWolf().fit(Zr - mus[yr]).precision_
    J = np.zeros(C); minority = np.zeros(len(yt), dtype=bool); margin = np.zeros(len(yt))
    for c in range(C):
        m = np.where(yt == c)[0]
        if len(m) < 4:
            continue
        Z = Zt[m]
        mu = Z.mean(0)
        g = (((Z - mu) @ np.linalg.svd(Z - mu, full_matrices=False)[2][0]) > 0)
        for _ in range(iters):
            if g.all() or (~g).all():
                break
            m1, m0 = Z[g].mean(0), Z[~g].mean(0)
            w = prec @ (m1 - m0)
            gn = (Z - (m1 + m0) / 2) @ w > 0
            if (gn == g).all():
                break
            g = gn
        if g.all() or (~g).all():
            continue
        m1, m0 = Z[g].mean(0), Z[~g].mean(0)
        q = lambda D: np.einsum("ij,jk,ik->i", D, prec, D)
        J[c] = 0.5 * (q(Z - mu).sum() - q(Z[g] - m1).sum() - q(Z[~g] - m0).sum()) / len(m)
        small = g if g.mean() < 0.5 else ~g
        minority[m] = small
        w = prec @ (m1 - m0)
        proj = (Z - (m1 + m0) / 2) @ w
        margin[m] = proj if g.mean() < 0.5 else -proj
    med = np.median(J); mad = np.median(np.abs(J - med)) * 1.4826 + 1e-12
    Js = (J - med) / mad
    s = Js[yt] + 1e-3 * np.tanh(margin / (np.abs(margin).std() + 1e-12))
    s[~minority] = Js[yt][~minority] - 1e3
    native = minority & (Js[yt] > math.e ** 2)
    return s, native, Js


# ---------------------------------------------------------------- model-based scores
@torch.no_grad()
def per_sample_loss(model, X, Y, dev):
    logits = predict(model, X, dev)
    return F.cross_entropy(logits, Y.to(dev), reduction="none").cpu().numpy()


@torch.no_grad()
def strip_entropy(model, X, blends_u8, dev, n=16, bs=192, seed=0):
    """STRIP [Gao+19]: entropy of predictions on x superimposed (alpha=0.5)
    with n reference images; low entropy = trigger dominates."""
    model.eval()
    g = torch.Generator(device="cpu").manual_seed(seed)
    B = blends_u8.to(dev).float() / 255
    out = []
    for i in range(0, X.size(0), bs):
        xb = X[i:i + bs].to(dev).float() / 255
        idx = torch.randint(0, B.size(0), (xb.size(0), n), generator=g).to(dev)
        mix = 0.5 * xb[:, None] + 0.5 * B[idx]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            p = F.softmax(model(mix.flatten(0, 1)).float(), 1).view(xb.size(0), n, -1).mean(1)
        out.append(-(p * (p + 1e-12).log()).sum(1).cpu())
    return torch.cat(out).numpy()


def dct_matrix(n):
    k = torch.arange(n).float()
    M = torch.cos(math.pi * (2 * k[None] + 1) * k[:, None] / (2 * n)) * math.sqrt(2 / n)
    M[0] /= math.sqrt(2)
    return M


@torch.no_grad()
def freq_score(X, dev, bs=4096):
    """Model-free high-frequency energy ratio of the grayscale 2-D DCT
    (simplified frequency heuristic in the spirit of [Zeng+21])."""
    H = X.size(-1)
    D = dct_matrix(H).to(dev)
    u = torch.arange(H, device=dev)
    hi = ((u[:, None] + u[None]) >= H).float()
    out = []
    for i in range(0, X.size(0), bs):
        g = (X[i:i + bs].to(dev).float() / 255).mean(1)
        c = D @ g @ D.T
        e = c ** 2
        out.append(((e * hi).sum((1, 2)) / (e.sum((1, 2)) + 1e-12)).cpu())
    return torch.cat(out).numpy()


def asset_offset(model, Xt, Yt, Xr, Yr, dev, steps=300, lr=1e-3, bs=256, seed=0):
    """ASSET-style offset [Pan+23] (reimplementation): alternately minimise the
    loss on the clean reference set and maximise the (clipped) loss on the
    training set; clean training samples follow the reference set and keep
    low loss, poisoned samples are pushed to high loss. Score = final loss;
    native rule = high-loss component of a 2-GMM on log-loss."""
    import copy
    m = copy.deepcopy(model).train()
    opt = torch.optim.SGD(m.parameters(), lr=lr, momentum=0.9)
    g = torch.Generator().manual_seed(seed)
    Xr_d, Yr_d = Xr.to(dev), Yr.to(dev)
    for _ in range(steps):
        it = torch.randint(0, Xt.size(0), (bs,), generator=g)
        ir = torch.randint(0, Xr.size(0), (min(bs, Xr.size(0)),), generator=g).to(dev)
        xt = Xt[it].to(dev).float() / 255; yt = Yt[it].to(dev)
        xr = Xr_d[ir].float() / 255; yr = Yr_d[ir]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            l_r = F.cross_entropy(m(xr), yr)
            l_t = F.cross_entropy(m(xt), yt, reduction="none").clamp(max=5.0).mean()
            loss = l_r - l_t
        opt.zero_grad(); loss.backward(); opt.step()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            l2 = F.cross_entropy(m(xr), yr)
        opt.zero_grad(); l2.backward(); opt.step()
    s = per_sample_loss(m, Xt, Yt, dev)
    ls = np.log(s + 1e-6).reshape(-1, 1)
    gm = GaussianMixture(2, random_state=seed).fit(ls)
    hi = int(np.argmax(gm.means_.ravel()))
    native = gm.predict(ls) == hi
    del m
    return s, native


# ---------------------------------------------------------------- reference-set construction
def make_reference(pool_y, per_class, seed, C):
    rng = np.random.RandomState(seed)
    idx = []
    for c in range(C):
        ids = np.where(pool_y == c)[0]
        idx.append(rng.choice(ids, min(per_class, len(ids)), replace=False))
    return np.sort(np.concatenate(idx))


def contaminate(ref_idx, Fpool, Fpool_trig, ypool, target, frac, seed, clean_label):
    """Replace a `frac` fraction of the reference set by triggered versions of
    reference images (labelled with the target class for dirty-label attacks)
    -- the defender's 'trusted' set is silently poisoned."""
    Fr = Fpool[ref_idx].copy(); yr = ypool[ref_idx].copy()
    if frac <= 0 or Fpool_trig is None:
        return Fr, yr
    rng = np.random.RandomState(seed + 5)
    cand = np.arange(len(ref_idx))
    if clean_label:
        cand = cand[yr == target]
    n = min(int(round(frac * len(ref_idx))), len(cand))
    pick = rng.choice(cand, n, replace=False)
    Fr[pick] = Fpool_trig[ref_idx[pick]]
    yr[pick] = target
    return Fr, yr


# ---------------------------------------------------------------- main
def load_run(rd):
    meta = json.load(open(os.path.join(rd, "meta.json")))
    S = load_splits(meta["dataset"])
    P = torch.load(os.path.join(rd, "poison.pt"))
    xp, yp = S["train_x"].clone(), S["train_y"].clone()
    xp[P["changed_idx"]] = P["changed_x"]; yp[P["changed_idx"]] = P["changed_y"]
    gt = np.zeros(len(yp), dtype=np.int64); gt[P["payload"].numpy()] = 1
    return meta, S, P, xp, yp, gt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--sweeps", action="store_true", help="also run reference-size and contamination sweeps")
    ap.add_argument("--ref-seeds", type=int, default=3)
    a = ap.parse_args()
    out_path = os.path.join(a.run, "detect.json")
    t0 = time.time()
    dev = torch.device("cuda")
    meta, S, P, xp, yp, gt = load_run(a.run)
    C, size = S["num_classes"], S["size"]
    target = meta["target"]
    clean_label = meta["attack"] == "sig"
    model = build_model(meta["arch"], C, size, **({"pretrained": False} if meta["arch"] == "vit_s" else {})).to(dev)
    model.load_state_dict(torch.load(os.path.join(a.run, "model.pt")))
    model.eval()
    early = build_model(meta["arch"], C, size, **({"pretrained": False} if meta["arch"] == "vit_s" else {})).to(dev)
    early.load_state_dict(torch.load(os.path.join(a.run, "early.pt"))); early.eval()

    Ft = predict(model, xp, dev, feats=True).cpu().numpy().astype(np.float64)
    Fpool = predict(model, S["pool_x"], dev, feats=True).cpu().numpy().astype(np.float64)
    Fpool_trig = None
    if P.get("pool_trig") is not None:
        Fpool_trig = predict(model, P["pool_trig"], dev, feats=True).cpu().numpy().astype(np.float64)
    yt, ypool = yp.numpy(), S["pool_y"].numpy()
    results, flags = [], {}

    def record(score, ref, cfg, s_train, s_ref=None, native=None, extra=None):
        row = dict(score=score, ref=ref, **cfg, **metrics(s_train, gt))
        row["topq"] = metrics(s_train, gt, rule_topq(s_train))
        if s_ref is not None:
            f = rule_cleancal(s_train, s_ref)
            row["cleancal"] = metrics(s_train, gt, f)
            row["alpha_sweep"] = {str(al): metrics(s_train, gt, s_train > conformal_threshold(s_ref, al))
                                  for al in (0.01, 0.02, 0.05, 0.1, 0.2)}
            if cfg.get("main"):
                flags[f"{score}|{ref}|cleancal"] = np.where(f)[0]
        if native is not None:
            row["native"] = metrics(s_train, gt, native)
            if cfg.get("main"):
                flags[f"{score}|{ref}|native"] = np.where(native)[0]
        if cfg.get("main"):
            flags[f"{score}|{ref}|topq"] = np.where(rule_topq(s_train))[0]
        if extra:
            row.update(extra)
        results.append(row)

    main_k = MAIN_PER_CLASS[meta["dataset"]]
    configs = [dict(per_class=main_k, contam=0.0, ref_seed=r, main=(r == 0)) for r in range(a.ref_seeds)]
    if a.sweeps:
        configs += [dict(per_class=k, contam=0.0, ref_seed=0, main=False)
                    for k in (5, 10, 25, 50, 100) if k != main_k and k <= int(np.bincount(ypool).min())]
        configs += [dict(per_class=main_k, contam=c, ref_seed=0, main=False) for c in (0.01, 0.05, 0.1, 0.2)]

    # ---- reference-free scores (only the threshold can use clean data)
    ref0 = make_reference(ypool, main_k, 0, C)
    l_final = per_sample_loss(model, xp, yp, dev)
    l_final_ref = per_sample_loss(model, S["pool_x"][ref0], S["pool_y"][ref0], dev)
    record("loss", "none", configs[0], -l_final, -l_final_ref)
    l_early = per_sample_loss(early, xp, yp, dev)
    l_early_ref = per_sample_loss(early, S["pool_x"][ref0], S["pool_y"][ref0], dev)
    record("loss_early", "none", configs[0], -l_early, -l_early_ref)
    fq = freq_score(xp, dev); fq_ref = freq_score(S["pool_x"][ref0], dev)
    record("freq", "none", configs[0], fq, fq_ref)
    ac, ac_native = activation_clustering(Ft, yt, C)
    Zref_ac = None
    record("ac", "train", configs[0], ac, None, ac_native)

    # ---- STRIP: blend images from training set (train-ref) vs clean reference (clean-ref)
    n_bl = 8 if meta["arch"] == "vit_s" else 16
    rng = np.random.RandomState(0)
    tr_bl = xp[torch.from_numpy(rng.choice(len(yp), len(ref0), replace=False))]
    for ref, bl in (("train", tr_bl), ("clean", S["pool_x"][ref0])):
        e = strip_entropy(model, xp, bl, dev, n=n_bl)
        if ref == "clean":   # clean entropy distribution: blend each ref image with the *other* ref images
            e_ref = strip_entropy(model, S["pool_x"][ref0], bl, dev, n=n_bl, seed=1)
        else:
            e_ref = strip_entropy(model, S["pool_x"][ref0], bl, dev, n=n_bl, seed=1)
        record("strip", ref, configs[0], -e, -e_ref)

    # ---- reference-fitted feature scores: train-fit vs clean-fit
    for name, fitter in FITTERS.items():
        if name == "knn":
            s_tr_fit = fit_knn(Ft, yt, C, exclude_self=True)(Ft, yt)
        else:
            s_tr_fit = fitter(Ft, yt, C)(Ft, yt)
        tfit = fitter(Ft, yt, C) if name != "knn" else fit_knn(Ft, yt, C, exclude_self=False)
        record(name, "train", configs[0], s_tr_fit, tfit(Fpool[ref0], ypool[ref0]))
        for cfg in configs:
            ridx = make_reference(ypool, cfg["per_class"], cfg["ref_seed"], C)
            Fr, yr = contaminate(ridx, Fpool, Fpool_trig, ypool, target, cfg["contam"], cfg["ref_seed"], clean_label)
            s = fitter(Fr, yr, C)(Ft, yt)
            s_ref = crossfit_ref_scores(fitter, Fr, yr, C, seed=cfg["ref_seed"])
            record(name, "clean", cfg, s, s_ref)

    # ---- SCAn and ASSET (clean data required by design)
    for cfg in configs:
        ridx = make_reference(ypool, cfg["per_class"], cfg["ref_seed"], C)
        Fr, yr = contaminate(ridx, Fpool, Fpool_trig, ypool, target, cfg["contam"], cfg["ref_seed"], clean_label)
        s, native, Js = scan(Ft, yt, Fr, yr, C)
        record("scan", "clean", cfg, s, None, native, {"J_target": float(Js[target]), "J_max_other": float(np.delete(Js, target).max())})
    cfg = configs[0]
    s, native = asset_offset(model, xp, yp, S["pool_x"][ref0], S["pool_y"][ref0], dev)
    record("asset", "clean", cfg, s, None, native)

    json.dump({"meta": meta, "results": results, "seconds": time.time() - t0}, open(out_path + ".tmp", "w"))
    np.savez_compressed(os.path.join(a.run, "flags.npz"), **{k.replace("|", "__"): v for k, v in flags.items()})
    os.replace(out_path + ".tmp", out_path)
    print(f"done {a.run} {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
