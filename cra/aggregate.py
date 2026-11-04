"""Aggregate runs/*/detect.json and retrain/*.json into CSV tables + LaTeX
snippets in results/cra/. Every number in the revised paper is produced here.

python -m cra.aggregate
"""

import glob
import json
import os
from collections import defaultdict

import numpy as np
from scipy.stats import wilcoxon

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results", "cra")
ATTACKS = ["badnets", "blend", "sig", "wanet", "icp", "trojan", "adap_blend", "adap_patch", "tact"]
ANAME = {"badnets": "BadNets", "blend": "Blend", "sig": "SIG", "wanet": "WaNet", "icp": "ICP",
         "trojan": "TrojanNN", "adap_blend": "Adap-Blend", "adap_patch": "Adap-Patch", "tact": "TaCT",
         "none": "No attack"}
SNAME = {"loss": "Loss (final)", "loss_early": "Loss (early)", "freq": "Frequency", "ac": "Act. clustering",
         "strip": "STRIP", "spectral": "Spectral sig.", "cafd": "Feature deviation", "maha": "Mahalanobis",
         "knn": "$k$-NN", "scan": "SCAn", "asset": "ASSET"}


def load():
    rows, metas, retr = [], {}, []
    for f in glob.glob(os.path.join(ROOT, "runs", "*", "*", "*", "detect.json")):
        d = json.load(open(f))
        m = d["meta"]
        key = (m["dataset"], m["arch"], m["attack"], m["rate"], m["seed"])
        metas[key] = m
        for r in d["results"]:
            rows.append({**r, "dataset": m["dataset"], "arch": m["arch"], "attack": m["attack"],
                         "rate": m["rate"], "seed": m["seed"]})
    for f in glob.glob(os.path.join(ROOT, "runs", "*", "*", "*", "retrain", "*.json")):
        m = json.load(open(os.path.join(os.path.dirname(os.path.dirname(f)), "meta.json")))
        retr.append({**json.load(open(f)), "dataset": m["dataset"], "arch": m["arch"], "attack": m["attack"],
                     "rate": m["rate"], "seed": m["seed"], "orig_asr": m["asr"], "orig_ca": m["clean_acc"]})
    return rows, metas, retr


def sel(rows, **kw):
    out = []
    for r in rows:
        ok = True
        for k, v in kw.items():
            rv = r.get(k)
            if callable(v):
                ok = v(rv)
            elif isinstance(v, (list, tuple, set)):
                ok = rv in v
            else:
                ok = rv == v
            if not ok:
                break
        if ok:
            out.append(r)
    return out


def get(r, path):
    for p in path.split("."):
        if r is None or p not in r:
            return np.nan
        r = r[p]
    return r


def ms(vals, pct=False, d=3):
    v = np.array([x for x in vals if x is not None and not np.isnan(x)], dtype=float)
    if len(v) == 0:
        return "--"
    f = 100 if pct else 1
    dd = 1 if pct else d
    if len(v) == 1:
        return f"{v[0] * f:.{dd}f}"
    return f"{v.mean() * f:.{dd}f}$\\pm${v.std(ddof=1) * f:.{dd}f}"


MAIN_RATE = {**{a: 0.1 for a in ATTACKS[:6]}, **{a: 0.01 for a in ATTACKS[6:]}, "none": 0.0}


def main_cfg(r):
    """The main reference configuration (or reference-free/train rows) at the attack's main rate."""
    return r["rate"] == MAIN_RATE[r["attack"]] and r.get("contam", 0.0) == 0.0 and r.get("ref_seed", 0) == 0 and (
        r["ref"] != "clean" or r.get("main", False))


def write(name, text):
    os.makedirs(OUT, exist_ok=True)
    open(os.path.join(OUT, name), "w").write(text)


def table_ablation(rows, ds="cifar10", arch="resnet18"):
    """2x2 factorial (reference x threshold) per score, pooled over the 9
    attacks x seeds. Paired Wilcoxon on AUROC (clean vs train reference)."""
    base = [r for r in sel(rows, dataset=ds, arch=arch, attack=ATTACKS) if main_cfg(r)
            and r["rate"] in (0.1, 0.01)]
    lines, csv = [], ["score,ref,auroc,tpr5,topq_f1,cleancal_f1,cleancal_fpr,n"]
    stats = {}
    for sc in ["loss", "loss_early", "freq", "strip", "spectral", "knn", "maha", "cafd"]:
        for ref in (["none"] if sc in ("loss", "loss_early", "freq") else ["train", "clean"]):
            R = sel(base, score=sc, ref=ref)
            if not R:
                continue
            f1 = lambda r, k: (2 * get(r, k + ".precision") * get(r, k + ".recall") /
                               max(get(r, k + ".precision") + get(r, k + ".recall"), 1e-12))
            row = dict(au=[get(r, "auroc") for r in R], tp=[get(r, "tpr@fpr5") for r in R],
                       tq=[f1(r, "topq") for r in R], cc=[f1(r, "cleancal") for r in R],
                       fp=[get(r, "cleancal.fpr") for r in R])
            stats[(sc, ref)] = {(r["attack"], r["seed"]): get(r, "auroc") for r in R}
            lines.append(f"{SNAME[sc]} & {ref} & {ms(row['au'])} & {ms(row['tp'], True)} & "
                         f"{ms(row['tq'], True)} & {ms(row['cc'], True)} & {ms(row['fp'], True)} \\\\")
            csv.append(f"{sc},{ref},{np.nanmean(row['au']):.4f},{np.nanmean(row['tp']):.4f},"
                       f"{np.nanmean(row['tq']):.4f},{np.nanmean(row['cc']):.4f},{np.nanmean(row['fp']):.4f},{len(R)}")
    tests = ["score,median_delta_auroc,wilcoxon_p,n"]
    for sc in ["strip", "spectral", "knn", "maha", "cafd"]:
        a, b = stats.get((sc, "clean"), {}), stats.get((sc, "train"), {})
        k = sorted(set(a) & set(b))
        if len(k) >= 6:
            d = np.array([a[x] - b[x] for x in k])
            p = wilcoxon(d).pvalue if np.any(d != 0) else 1.0
            tests.append(f"{sc},{np.median(d):.4f},{p:.2e},{len(k)}")
    write(f"ablation_{ds}_{arch}.tex", "\n".join(lines))
    write(f"ablation_{ds}_{arch}.csv", "\n".join(csv))
    write(f"ablation_tests_{ds}_{arch}.csv", "\n".join(tests))


DET = [("loss_early", "none", "topq"), ("ac", "train", "native"), ("spectral", "train", "topq"),
       ("strip", "clean", "cleancal"), ("scan", "clean", "native"), ("asset", "clean", "native"),
       ("knn", "clean", "cleancal"), ("maha", "clean", "cleancal"), ("cafd", "clean", "cleancal")]


def table_per_attack(rows, ds, arch="resnet18", metric="auroc"):
    base = [r for r in sel(rows, dataset=ds, arch=arch) if main_cfg(r) and r["rate"] in (0.1, 0.01, 0.0)]
    lines, csv = [], ["attack," + ",".join(f"{s}|{r}" for s, r, _ in DET)]
    for atk in ATTACKS:
        cells, vals = [], []
        for sc, ref, rule in DET:
            R = sel(base, attack=atk, score=sc, ref=ref)
            v = [get(r, metric if metric != "rule_f1" else "") for r in R] if metric != "rule_f1" else [
                2 * get(r, rule + ".precision") * get(r, rule + ".recall") /
                max(get(r, rule + ".precision") + get(r, rule + ".recall"), 1e-12) for r in R]
            cells.append(v); vals.append(np.nanmean(v) if len(v) else np.nan)
        if all(len(c) == 0 for c in cells):
            continue
        best = np.nanmax(vals) if not np.all(np.isnan(vals)) else None
        txt = []
        for c, v in zip(cells, vals):
            s = ms(c, pct=metric != "auroc")
            txt.append(f"\\textbf{{{s}}}" if best is not None and v == best else s)
        lines.append(f"{ANAME[atk]} & " + " & ".join(txt) + " \\\\")
        csv.append(atk + "," + ",".join(f"{v:.4f}" for v in vals))
    write(f"per_attack_{metric}_{ds}_{arch}.tex", "\n".join(lines))
    write(f"per_attack_{metric}_{ds}_{arch}.csv", "\n".join(csv))


def table_operating(rows, ds="cifar10", arch="resnet18"):
    """Precision / recall / FPR of the decision rule of each detector."""
    base = [r for r in sel(rows, dataset=ds, arch=arch, attack=ATTACKS + ["none"]) if main_cfg(r)
            and r["rate"] in (0.1, 0.01, 0.0)]
    lines = []
    for atk in ATTACKS + ["none"]:
        cells = []
        for sc, ref, rule in DET:
            R = sel(base, attack=atk, score=sc, ref=ref)
            if atk == "none":
                cells.append(ms([get(r, rule + ".flagged_frac") for r in R], True))
            else:
                cells.append(ms([get(r, rule + ".precision") for r in R], True).split("$")[0] + "/" +
                             ms([get(r, rule + ".recall") for r in R], True).split("$")[0])
        lines.append(f"{ANAME[atk]} & " + " & ".join(cells) + " \\\\")
    write(f"operating_{ds}_{arch}.tex", "\n".join(lines))


def table_arch(rows):
    lines = []
    for arch in ["resnet18", "vgg16", "vit_s"]:
        base = [r for r in sel(rows, dataset="cifar10", arch=arch, attack=ATTACKS) if main_cfg(r)
                and r["rate"] in (0.1, 0.01)]
        cells = []
        for sc, ref, rule in DET:
            cells.append(ms([get(r, "auroc") for r in sel(base, score=sc, ref=ref)]))
        lines.append({"resnet18": "ResNet-18", "vgg16": "VGG-16", "vit_s": "ViT-S/16 (pretr.)"}[arch] + " & " + " & ".join(cells) + " \\\\")
    for ds in ["cifar10", "gtsrb", "tiny"]:
        base = [r for r in sel(rows, dataset=ds, arch="resnet18", attack=ATTACKS) if main_cfg(r)
                and r["rate"] in (0.1, 0.01)]
        cells = [ms([get(r, "auroc") for r in sel(base, score=sc, ref=ref)]) for sc, ref, _ in DET]
        lines.append({"cifar10": "\\midrule CIFAR-10", "gtsrb": "GTSRB", "tiny": "Tiny-ImageNet"}[ds] + " & " + " & ".join(cells) + " \\\\")
    write("generality.tex", "\n".join(lines))


def rate_cfg(r):
    return r.get("contam", 0.0) == 0.0 and r.get("ref_seed", 0) == 0 and (r["ref"] != "clean" or r.get("main", False))


def table_rate(rows):
    """AUROC and operating points vs poison rate (CIFAR-10, ResNet-18, 6 standard attacks)."""
    std = ATTACKS[:6]
    csv = ["score,ref,rule,rate,auroc,precision,recall,fpr,n"]
    for sc, ref, rule in DET + [("cafd", "clean", "topq"), ("cafd", "train", "topq")]:
        for rate in [0.005, 0.01, 0.05, 0.1, 0.2]:
            R = [r for r in sel(rows, dataset="cifar10", arch="resnet18", attack=std, rate=rate, score=sc, ref=ref)
                 if rate_cfg(r)]
            if not R:
                continue
            csv.append(f"{sc},{ref},{rule},{rate},{np.nanmean([get(r, 'auroc') for r in R]):.4f},"
                       f"{np.nanmean([get(r, rule + '.precision') for r in R]):.4f},"
                       f"{np.nanmean([get(r, rule + '.recall') for r in R]):.4f},"
                       f"{np.nanmean([get(r, rule + '.fpr') for r in R]):.4f},{len(R)}")
    write("rate.csv", "\n".join(csv))


def table_refsize(rows):
    csv = ["score,per_class,contam,auroc,recall,fpr,n"]
    for sc in ["cafd", "maha", "knn", "scan"]:
        rule = "native" if sc == "scan" else "cleancal"
        R = [r for r in sel(rows, dataset="cifar10", arch="resnet18", attack=ATTACKS, score=sc, ref="clean",
                            ref_seed=0) if r["rate"] in (0.1, 0.01)]
        for key in sorted({(r["per_class"], r["contam"]) for r in R}):
            Rk = [r for r in R if (r["per_class"], r["contam"]) == key]
            csv.append(f"{sc},{key[0]},{key[1]},{np.nanmean([get(r, 'auroc') for r in Rk]):.4f},"
                       f"{np.nanmean([get(r, rule + '.recall') for r in Rk]):.4f},"
                       f"{np.nanmean([get(r, rule + '.fpr') for r in Rk]):.4f},{len(Rk)}")
    write("refsize_contam.csv", "\n".join(csv))


def table_retrain(retr, metas):
    flags = ["none", "oracle", "loss_early|none|topq", "ac|train|native", "spectral|train|topq",
             "strip|clean|cleancal", "scan|clean|native", "asset|clean|native", "knn|clean|cleancal",
             "maha|clean|cleancal", "cafd|train|topq", "cafd|clean|cleancal", "maha|train|topq",
             "maha|clean|cleancal@0.1", "maha|clean|cleancal@0.2"]
    csv = ["dataset,attack,flag,clean_acc,asr,removed_frac,removed_payload_frac,n"]
    lines = defaultdict(list)
    for ds in ["cifar10", "gtsrb", "tiny"]:
        for atk in ATTACKS:
            for fl in flags:
                if fl == "none":
                    M = [m for k, m in metas.items() if k[0] == ds and k[1] == "resnet18" and k[2] == atk
                         and k[3] in (0.1, 0.01) and os.path.exists(os.path.join(
                             ROOT, "runs", ds, "resnet18", f"{atk}_r{k[3]:g}_s{k[4]}", "retrain"))]
                    R = [dict(clean_acc=m["clean_acc"], asr=m["asr"], removed=0, removed_payload=0,
                              payload=m["n_payload"], n_train=m["n_train"]) for m in M]
                else:
                    R = sel(retr, dataset=ds, arch="resnet18", attack=atk, flag=fl)
                if not R:
                    continue
                N = metas[next(k for k in metas if k[0] == ds and k[2] == atk)]["n_train"]
                csv.append(f"{ds},{atk},{fl},{np.mean([r['clean_acc'] for r in R]):.4f},"
                           f"{np.mean([r['asr'] for r in R]):.4f},{np.mean([r['removed'] / N for r in R]):.4f},"
                           f"{np.mean([r['removed_payload'] / max(r['payload'], 1) for r in R]):.4f},{len(R)}")
                lines[(ds, fl)].append((atk, ms([r["clean_acc"] for r in R], True), ms([r["asr"] for r in R], True)))
    write("retrain.csv", "\n".join(csv))


def table_models(metas):
    csv = ["dataset,arch,attack,rate,seed,clean_acc,asr,n_payload,n_cover"]
    for k in sorted(metas):
        m = metas[k]
        csv.append(f"{k[0]},{k[1]},{k[2]},{k[3]},{k[4]},{m['clean_acc']:.4f},{m['asr']:.4f},{m['n_payload']},{m['n_cover']}")
    write("models.csv", "\n".join(csv))


def main():
    rows, metas, retr = load()
    print(len(rows), "detection rows,", len(metas), "runs,", len(retr), "retrains")
    table_models(metas)
    for ds in ["cifar10", "gtsrb", "tiny"]:
        table_ablation(rows, ds)
        table_per_attack(rows, ds)
        table_per_attack(rows, ds, metric="rule_f1")
        table_operating(rows, ds)
    for arch in ["vgg16", "vit_s"]:
        table_ablation(rows, "cifar10", arch)
        table_per_attack(rows, "cifar10", arch)
    table_arch(rows)
    table_rate(rows)
    table_refsize(rows)
    table_retrain(retr, metas)
    print("wrote", OUT)




# ------------------------------------------------------------------ LaTeX for sweeps, retrain, figures
def tex_rate():
    import csv
    R = list(csv.DictReader(open(os.path.join(OUT, "rate.csv"))))
    pick = [("maha", "clean", "cleancal"), ("scan", "clean", "native"), ("loss_early", "none", "topq"),
            ("spectral", "train", "topq"), ("ac", "train", "native")]
    lines = []
    for rate in ["0.005", "0.01", "0.05", "0.1", "0.2"]:
        cells = []
        for sc, ref, rule in pick:
            r = [x for x in R if (x["score"], x["ref"], x["rule"], x["rate"]) == (sc, ref, rule, rate)]
            if not r:
                cells.append("--"); continue
            r = r[0]
            c = f"{100 * float(r['precision']):.0f}/{100 * float(r['recall']):.0f}"
            if sc == "maha":
                c = f"{float(r['auroc']):.3f} & " + c + f" & {100 * float(r['fpr']):.1f}"
            cells.append(c)
        lines.append(f"{100 * float(rate):g}\\% & " + " & ".join(cells) + " \\\\")
    write("rate.tex", "\n".join(lines))


def tex_refsize():
    import csv
    R = list(csv.DictReader(open(os.path.join(OUT, "refsize_contam.csv"))))
    lines = []
    keys = [("5", "0.0"), ("10", "0.0"), ("25", "0.0"), ("50", "0.0"), ("100", "0.0"),
            ("50", "0.01"), ("50", "0.05"), ("50", "0.1"), ("50", "0.2")]
    for k, c in keys:
        cells = []
        for sc in ["maha", "knn", "cafd", "scan"]:
            r = [x for x in R if (x["score"], x["per_class"], x["contam"]) == (sc, k, c)]
            cells.append(f"{float(r[0]['auroc']):.3f} / {100 * float(r[0]['recall']):.0f}" if r else "--")
        lab = f"$m={k}$" if c == "0.0" else f"$m=50$, {100 * float(c):g}\\% poisoned"
        lines.append(lab + " & " + " & ".join(cells) + " \\\\")
    write("refsize.tex", "\n".join(lines))


RFLAGS = [("none", "None"), ("oracle", "Oracle"), ("loss_early|none|topq", "Loss-E"), ("ac|train|native", "AC"),
          ("spectral|train|topq", "SS"), ("scan|clean|native", "SCAn"), ("asset|clean|native", "ASSET"),
          ("cafd|train|topq", "FD$^{\\D}$"), ("cafd|clean|cleancal", "FD$^{\\R}$"), ("maha|train|topq", "Maha$^{\\D}$"),
          ("knn|clean|cleancal", "$k$NN$^{\\R}$"), ("maha|clean|cleancal", "Maha$^{\\R}$")]


def tex_retrain():
    import csv
    p = os.path.join(OUT, "retrain.csv")
    if not os.path.exists(p):
        return
    R = list(csv.DictReader(open(p)))
    for ds in ["cifar10", "gtsrb", "tiny"]:
        lines = []
        for atk in ATTACKS + ["mean"]:
            cells = []
            for fl, _ in RFLAGS:
                if atk == "mean":
                    v = [float(x["asr"]) for x in R if x["dataset"] == ds and x["flag"] == fl]
                    ca = [float(x["clean_acc"]) for x in R if x["dataset"] == ds and x["flag"] == fl]
                    cells.append(f"{100 * np.mean(v):.1f} ({100 * np.mean(ca):.1f})" if v else "--")
                else:
                    r = [x for x in R if (x["dataset"], x["attack"], x["flag"]) == (ds, atk, fl)]
                    cells.append(f"{100 * float(r[0]['asr']):.1f}" if r else "--")
            lines.append(("\\midrule Mean ASR (CA) " if atk == "mean" else ANAME[atk]) + " & " + " & ".join(cells) + " \\\\")
        write(f"retrain_{ds}.tex", "\n".join(lines))


def tex_retrain_alpha():
    """Remove-and-retrain with clean Mahalanobis at alpha = 5/10/20% (CIFAR-10)."""
    import csv
    p = os.path.join(OUT, "retrain.csv")
    R = [x for x in csv.DictReader(open(p)) if x["dataset"] == "cifar10"]
    fl = [("oracle", "Oracle"), ("maha|clean|cleancal", "5"), ("maha|clean|cleancal@0.1", "10"), ("maha|clean|cleancal@0.2", "20")]
    lines = []
    for atk in ATTACKS + ["mean"]:
        cells = []
        for f, _ in fl:
            rr = [x for x in R if x["flag"] == f and (atk == "mean" or x["attack"] == atk)]
            if not rr:
                cells.append("--"); continue
            if atk == "mean":
                cells.append(f"{100 * np.mean([float(x['asr']) for x in rr]):.1f} ({100 * np.mean([float(x['clean_acc']) for x in rr]):.1f}; "
                             f"{100 * np.mean([float(x['removed_frac']) for x in rr]):.1f}\\%)")
            else:
                cells.append(f"{100 * float(rr[0]['asr']):.1f}")
        lines.append(("\\midrule Mean ASR (CA; removed)" if atk == "mean" else ANAME[atk]) + " & " + " & ".join(cells) + " \\\\")
    write("retrain_alpha.tex", "\n".join(lines))


def figures():
    import csv
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    R = list(csv.DictReader(open(os.path.join(OUT, "rate.csv"))))
    fig, ax = plt.subplots(1, 3, figsize=(10, 2.8))
    styles = {("maha", "clean", "cleancal"): ("Maha$^R$ + conformal", "C0", "-"),
              ("knn", "clean", "cleancal"): ("$k$NN$^R$ + conformal", "C1", "-"),
              ("scan", "clean", "native"): ("SCAn (native)", "C2", "--"),
              ("loss_early", "none", "topq"): ("Loss-early + top-5%", "C3", ":"),
              ("spectral", "train", "topq"): ("Spectral$^D$ + top-5%", "C4", ":")}
    for key, (lab, col, ls) in styles.items():
        rr = sorted([x for x in R if (x["score"], x["ref"], x["rule"]) == key], key=lambda x: float(x["rate"]))
        if not rr:
            continue
        xs = [100 * float(x["rate"]) for x in rr]
        for i, m in enumerate(["precision", "recall", "fpr"]):
            ax[i].plot(xs, [float(x[m]) for x in rr], ls, color=col, marker="o", ms=3, label=lab)
    for i, t in enumerate(["Precision", "Recall", "False-positive rate"]):
        ax[i].set_xscale("log"); ax[i].set_xlabel("payload rate (%)"); ax[i].set_title(t, fontsize=9)
        ax[i].set_xticks([0.5, 1, 5, 10, 20]); ax[i].set_xticklabels(["0.5", "1", "5", "10", "20"])
    ax[2].axhline(0.05, color="gray", lw=0.8, ls="--")
    ax[2].text(0.55, 0.055, r"$\alpha$", color="gray", fontsize=8)
    ax[0].legend(fontsize=6.5, loc="lower right")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_rate.pdf")); plt.close(fig)

    C = list(csv.DictReader(open(os.path.join(OUT, "refsize_contam.csv"))))
    fig, ax = plt.subplots(1, 2, figsize=(7, 2.6))
    for sc, lab, col in [("maha", "Maha$^R$", "C0"), ("knn", "$k$NN$^R$", "C1"), ("cafd", "FD$^R$", "C5"), ("scan", "SCAn", "C2")]:
        s = sorted([x for x in C if x["score"] == sc and x["contam"] == "0.0"], key=lambda x: int(x["per_class"]))
        ax[0].plot([int(x["per_class"]) for x in s], [float(x["auroc"]) for x in s], marker="o", ms=3, color=col, label=lab)
        c = sorted([x for x in C if x["score"] == sc and x["per_class"] == "50"], key=lambda x: float(x["contam"]))
        ax[1].plot([100 * float(x["contam"]) for x in c], [float(x["auroc"]) for x in c], marker="o", ms=3, color=col, label=lab)
    ax[0].set_xscale("log"); ax[0].set_xlabel("trusted images per class $m$"); ax[0].set_ylabel("AUROC")
    ax[0].set_xticks([5, 10, 25, 50, 100]); ax[0].set_xticklabels(["5", "10", "25", "50", "100"])
    ax[1].set_xlabel("poisoned share of trusted set (%)"); ax[0].legend(fontsize=7)
    for a in ax:
        a.axhline(0.5, color="gray", lw=0.6, ls="--")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_refsize.pdf")); plt.close(fig)


def tex_alpha():
    """Realised FPR and recall of the conformal rule vs target alpha (CIFAR-10/R18 main runs)."""
    rows, _, _ = load()
    base = [r for r in sel(rows, dataset="cifar10", arch="resnet18", attack=ATTACKS) if main_cfg(r) and "alpha_sweep" in r]
    lines = []
    for sc, ref in [("maha", "clean"), ("knn", "clean"), ("cafd", "clean"), ("loss_early", "none"), ("strip", "clean")]:
        R = sel(base, score=sc, ref=ref)
        cells = []
        for al in ["0.01", "0.05", "0.1", "0.2"]:
            f = np.nanmean([r["alpha_sweep"][al]["fpr"] for r in R]) if R else np.nan
            rc = np.nanmean([r["alpha_sweep"][al]["recall"] for r in R]) if R else np.nan
            cells.append(f"{100 * f:.1f} / {100 * rc:.0f}")
        sup = "$^{\\R}$" if ref == "clean" else ""
        lines.append(SNAME[sc] + sup + " & " + " & ".join(cells) + " \\\\")
    write("alpha.tex", "\n".join(lines))




def tex_numbers_and_attacks():
    import csv
    A = {(r["score"], r["ref"]): r for r in csv.DictReader(open(os.path.join(OUT, "ablation_cifar10_resnet18.csv")))}
    names = {("maha", "train"): "MahaTrain", ("maha", "clean"): "MahaClean", ("knn", "train"): "KnnTrain",
             ("knn", "clean"): "KnnClean", ("cafd", "train"): "CafdTrain", ("cafd", "clean"): "CafdClean",
             ("spectral", "train"): "SsTrain", ("spectral", "clean"): "SsClean"}
    out = [f"\\newcommand{{\\{v}AUROC}}{{{float(A[k]['auroc']):.3f}}}" for k, v in names.items() if k in A]
    write("numbers.tex", "\n".join(out))
    M = list(csv.DictReader(open(os.path.join(OUT, "models.csv"))))
    params = {"badnets": "3$\\times$3 white patch", "blend": "noise pattern, $\\alpha{=}0.2$",
              "sig": "sinusoid $\\Delta{=}20/255$, clean-label", "wanet": "warp $k{=}4$, $s{=}0.5$, noise mode",
              "icp": "input-conditioned 8$\\times$8 patch", "trojan": "activation-maximised patch",
              "adap_blend": "partial blend, cover samples", "adap_patch": "4 patches, cover samples",
              "tact": "source-specific patch, cover"}
    lines = []
    for atk in ATTACKS + ["none"]:
        cells = []
        for ds in ["cifar10", "gtsrb", "tiny"]:
            rr = [r for r in M if r["dataset"] == ds and r["arch"] == "resnet18" and r["attack"] == atk
                  and float(r["rate"]) == MAIN_RATE[atk]]
            cells.append(f"{100 * np.mean([float(r['clean_acc']) for r in rr]):.1f} / {100 * np.mean([float(r['asr']) for r in rr]):.1f}"
                         if rr else "--")
        lines.append(f"{ANAME[atk]} & {params.get(atk, '--')} & " + " & ".join(cells) + " \\\\")
    write("attacks.tex", "\n".join(lines))


if __name__ == "__main__":
    main(); tex_retrain_alpha(); tex_rate(); tex_numbers_and_attacks(); tex_refsize(); tex_retrain(); figures(); tex_alpha()
