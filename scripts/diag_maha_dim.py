"""Diagnostic: Mahalanobis(clean) AUROC vs PCA dimension, plus residual (off-subspace) energy."""
import sys, numpy as np, torch
sys.path.insert(0, "/home/shubhi.sharma/yesh/adversarial-robustness")
from cra.detect import load_run, make_reference, PCAFit, MAIN_PER_CLASS
from cra.models import build_model
from cra.train import predict
from sklearn.covariance import LedoitWolf
from sklearn.metrics import roc_auc_score
import os
for rd in sys.argv[1:]:
    meta, S, P, xp, yp, gt = load_run(rd)
    C = S["num_classes"]; dev = torch.device("cuda")
    m = build_model(meta["arch"], C, S["size"]).to(dev); m.load_state_dict(torch.load(os.path.join(rd, "model.pt"))); m.eval()
    Ft = predict(m, xp, dev, feats=True).cpu().numpy().astype(np.float64)
    Fp = predict(m, S["pool_x"], dev, feats=True).cpu().numpy().astype(np.float64)
    yt, yp_ = yp.numpy(), S["pool_y"].numpy()
    ri = make_reference(yp_, MAIN_PER_CLASS[meta["dataset"]], 0, C); Fr, yr = Fp[ri], yp_[ri]
    out = []
    for k in [16, 64, 256, Fr.shape[1]]:
        P_ = PCAFit(Fr, k); Z = P_(Fr); mus = np.stack([Z[yr == c].mean(0) for c in range(C)])
        prec = LedoitWolf().fit(Z - mus[yr]).precision_
        D = P_(Ft) - mus[yt]; s = np.einsum("ij,jk,ik->i", D, prec, D)
        out.append(f"k={k}:{roc_auc_score(gt, s):.3f}")
    P_ = PCAFit(Fr, 64); res = np.linalg.norm((Ft - P_.mu) - P_(Ft) @ P_.V.T, axis=1)
    out.append(f"residual64:{roc_auc_score(gt, res):.3f}")
    print(meta["dataset"], meta["attack"], " ".join(out), flush=True)
