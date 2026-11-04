# Adversarial Robustness of Neural Networks

Reference implementations of the core attacks, defenses, and certification
methods surveyed in [`literature-survey.md`](literature-survey.md), evaluated
on CIFAR-10. Each script maps to a specific paper — see the docstring at the
top of each file.

## Layout

```
src/
  models.py             SmallCNN + CIFAR-style ResNet-18 (forward_features for attention-based defenses)
  data.py               CIFAR-10 loaders
  attacks.py            evasion attacks: FGSM, PGD (Linf/L2), C&W L2
  defenses.py           evasion defenses: standard / PGD-AT (Madry) / TRADES / Gaussian augmentation
  smoothing.py          randomized smoothing certification (Cohen et al.)
  backdoor_attacks.py   poisoning attacks: BadNets, Blend, SIG, Dynamic, WaNet, Trojan
  backdoor_defenses.py  poisoning defenses: ABL, AIBD, CBD, DBD, MCR, NAD
train.py                train an evasion-robust model with a chosen method
evaluate.py             clean + empirical robust accuracy (FGSM/PGD/C&W)
certify.py              certified accuracy via randomized smoothing
plot_certified_accuracy.py   certified-accuracy-vs-radius curve
train_backdoor.py       poison CIFAR-10 with a chosen attack and train on it
defend_backdoor.py      apply a backdoor defense to a train_backdoor.py checkpoint
evaluate_backdoor.py    clean accuracy + Attack Success Rate (ASR)
scripts/                PBS job scripts (this cluster uses PBS, not Slurm)
```

The two families are separate threat models sharing infrastructure: evasion
attacks (`attacks.py`) perturb an input against a fixed, honestly-trained
model at *test* time; backdoor attacks (`backdoor_attacks.py`) poison part of
the *training* set so an otherwise normal model learns a hidden trigger. See
`literature-survey.md` §2–§10 (evasion) and §12 (backdoor) for the papers
behind each.

## Environment

Everything runs in the existing `torch-gpu` conda env (PyTorch 2.13,
torchvision 0.28; matplotlib/scipy/tqdm added for this project):

```bash
source /home/soft/anaconda3/etc/profile.d/conda.sh
conda activate torch-gpu
```

There is no GPU on the login node — `nvidia-smi` fails there by design. Use
the PBS `gpu` queue for real training runs (see below); the login node is
fine for CPU smoke tests with `--subset`.

## Quick smoke test (CPU, minutes not hours)

```bash
python train.py --method pgd --arch smallcnn --epochs 1 --subset 256 --num-workers 0
python evaluate.py --checkpoint ./checkpoints/smallcnn_pgd.pt --subset 256 --max-batches 2 --cw-steps 20

# backdoor side:
python train_backdoor.py --attack badnets --arch smallcnn --epochs 1 --subset 512 --num-workers 0
python evaluate_backdoor.py --checkpoint ./checkpoints/smallcnn_backdoor_badnets.pt --subset 300
python defend_backdoor.py --checkpoint ./checkpoints/smallcnn_backdoor_badnets.pt --method abl --subset 512 --num-workers 0
python evaluate_backdoor.py --checkpoint ./checkpoints/smallcnn_backdoor_badnets_defended_abl.pt --subset 300
```

## Full training runs (GPU, via PBS)

```bash
# Standard (non-robust) baseline
qsub -v METHOD=standard,ARCH=resnet18,EPOCHS=60 scripts/submit_train.sh

# PGD adversarial training (Madry et al. 2018)
qsub -v METHOD=pgd,ARCH=resnet18,EPOCHS=60 scripts/submit_train.sh

# TRADES (Zhang et al. 2019)
qsub -v METHOD=trades,ARCH=resnet18,EPOCHS=60 scripts/submit_train.sh

# Gaussian-noise base classifier for randomized smoothing (Cohen et al. 2019)
qsub -v METHOD=standard,ARCH=resnet18,EPOCHS=60,SIGMA=0.25 scripts/submit_train.sh
```

Checkpoints land in `./checkpoints/<arch>_<method>[_sigma<sigma>].pt`.

## Evaluating empirical robustness

```bash
qsub -v CHECKPOINT=./checkpoints/resnet18_pgd.pt scripts/submit_evaluate.sh
# or directly:
python evaluate.py --checkpoint ./checkpoints/resnet18_pgd.pt \
    --eps 0.0314 --pgd-steps 20 --cw-steps 200
```

Reports clean accuracy plus accuracy under FGSM, PGD-20, and C&W-L2. C&W is
run on a small batch subset by default (`--cw-batches`, default 5) since its
inner optimization is far slower than the gradient-sign attacks — pass
`--max-batches` to bound all four evaluations consistently.

**On trusting these numbers**: per Athalye et al.'s obfuscated-gradients
finding (see survey §4), any *new* defense's robustness numbers should be
re-checked with attacks adapted to that specific defense, not just this
generic attack suite. PGD-20/C&W here are a reasonable sanity check for the
defenses implemented in this repo (they don't rely on gradient masking), but
are not a substitute for AutoAttack/adaptive evaluation on novel defenses.

## Certified robustness (randomized smoothing)

Requires a checkpoint trained with `--sigma` (Gaussian-noise augmented, so
the base classifier matches the noise level used at certification time):

```bash
qsub -v CHECKPOINT=./checkpoints/resnet18_standard_sigma0.25.pt,SIGMA=0.25 scripts/submit_certify.sh
```

This runs `certify.py` (writes a per-image CSV of certified L2 radii) then
`plot_certified_accuracy.py` (renders the standard certified-accuracy-vs-radius
curve). Certification cost scales with `--n` (Monte Carlo samples per image,
default 1000 quick / 10000+ for publication-quality numbers) — expect
certify.py to take from minutes to hours depending on `--num-images` and `--n`.

## Backdoor (data-poisoning) attacks and defenses

```bash
# Poison + train (the attacker's step). Attacks: badnets, blend, sig, wanet, dynamic, trojan.
qsub -v ATTACK=wanet,ARCH=resnet18,EPOCHS=60,TARGET=0,POISON_RATE=0.1 scripts/submit_train_backdoor.sh

# Check clean accuracy + Attack Success Rate (ASR) before defending:
python evaluate_backdoor.py --checkpoint ./checkpoints/resnet18_backdoor_wanet.pt

# Apply a defense and compare before/after. Defenses: abl, aibd, cbd, dbd, mcr, nad, hybrid, neural_cleanse.
qsub -v CHECKPOINT=./checkpoints/resnet18_backdoor_wanet.pt,METHOD=nad scripts/submit_defend_backdoor.sh
```

ASR is the fraction of originally-non-target-class test images that get
classified as the target label once the trigger is stamped on them; a
successful defense keeps clean accuracy high while pushing ASR down toward
random-guess (~10% for CIFAR-10). ABL/AIBD/CBD/DBD retrain (or continue
training) on the reconstructed poisoned set; MCR/NAD only need the
checkpointed model plus a small clean holdout carved from the test set
(`--clean-holdout`, default 500 images) — see the module docstrings in
`src/backdoor_defenses.py` for which assumption each one makes and
`defend_backdoor.py`'s docstring for how that maps onto the CLI.

**Every attack/defense here is a scoped simplification of its paper** — e.g.
Dynamic uses a frozen (not jointly/adversarially trained) generator, DBD's
self-supervised stage is a lightweight SimCLR rather than full MoCo, MCR's
curve fitting skips the multi-point loss-surface visualization the paper
uses to *choose* a repair point (this repo always takes the midpoint). Each
class/function's docstring says exactly what was simplified and why — read
those before citing numbers from this repo as representative of a paper's
reported results.

## Experimental findings: mapping every attack to a defense that solves it

`train_backdoor.py` was run for all 6 attacks (resnet18, 60 epochs, target
label 0, 10% poison rate) and `defend_backdoor.py` for all 6 defenses
against each, producing `results/backdoor_experiment_summary.csv` (42
rows: 6 undefended baselines + 36 attack x defense combinations). Clean
accuracy / Attack Success Rate (ASR; lower is better after a defense,
~0.10 is random-guess for CIFAR-10's 10 classes):

| Attack | none | ABL | AIBD | CBD | DBD | MCR | NAD |
|---|---|---|---|---|---|---|---|
| BadNets | 0.93/1.00 | 0.93/1.00 | 0.90/1.00 | 0.93/1.00 | 0.22/0.61 | 0.92/0.99 | **0.92/0.32** |
| Blend | 0.94/1.00 | 0.92/1.00 | 0.89/1.00 | 0.93/1.00 | 0.20/0.84 | 0.92/0.42 | **0.92/0.01** |
| SIG | 0.94/0.95 | 0.93/0.96 | 0.92/0.96 | 0.94/0.95 | 0.37/0.03 | 0.92/1.00 | 0.92/0.97 |
| WaNet | 0.92/0.08 | 0.91/0.08 | 0.85/0.07 | 0.91/0.08 | 0.24/0.63 | 0.91/0.01 | 0.91/0.01 |
| Dynamic | 0.94/1.00 | 0.93/1.00 | 0.89/1.00 | 0.93/1.00 | 0.31/0.97 | 0.92/1.00 | 0.91/1.00 |
| Trojan | 0.94/1.00 | 0.93/1.00 | 0.90/1.00 | 0.93/1.00 | 0.33/1.00 | 0.92/1.00 | 0.92/1.00 |

(format: clean accuracy / ASR)

Takeaways at this repo's default settings (not necessarily each paper's own
hyperparameters):

- BadNets/Blend/Dynamic/Trojan implant near-perfect backdoors (ASR ~1.00);
  **WaNet barely implants one at all** (ASR ~0.08) at this poison
  rate/epoch budget, so it isn't a fair comparison point for the defenses.
- **NAD is the only defense with a real, consistent effect**, and only
  against BadNets/Blend.
- **ABL, AIBD, CBD do essentially nothing** against any attack here — ASR
  stays at the undefended baseline. Their isolation mechanisms need more
  epochs/tuned thresholds than these defaults to separate poisoned from
  clean samples at a 10% poison rate.
- **DBD is broken at this scale** — clean accuracy collapses to 0.20-0.37
  for every attack; its 3-stage pipeline needs far more than 20
  self-supervised-pretraining epochs to recover a usable classifier from
  scratch.

### `hybrid_defense`: combining what worked

Given NAD was the only defense that moved the needle (and only for two of
six attacks), `src/backdoor_defenses.py::hybrid_defense` combines AIBD-style
isolation, CBD-style confidence reweighting, and NAD-style attention
distillation into one defense (see its docstring for the full mechanism).
Four iterations were needed to find a stable design -- each failure mode is
informative enough that they're kept in the docstring and summarized here:

| Version | Mechanism | Result |
|---|---|---|
| v1 | Separate full-epoch pass over the *entire* isolated pool, 100% wrong-label | ASR -> ~0, but clean accuracy collapsed to 0.15-0.61 |
| v2 | Same, but bounded to one same-sized batch per clean batch | Accuracy partially recovered (0.41-0.75), ASR mixed |
| v3 | Isolated samples drawn at their *natural* rate within normal batches | Accuracy fully recovered (0.90-0.93), but the correction was diluted enough by the 95%+ correct majority per batch that ASR barely moved |
| v4 | `WeightedRandomSampler` oversampling isolated indices to ~30%/batch | Middle ground: partial ASR reduction on some attacks, ~5-8pt accuracy cost |

A follow-up grid over `gamma` (correction loss weight) and
`oversample_fraction` (`results/hybrid_grid_summary.csv`) showed **oversample
rate is the more effective lever than loss weight**, and pushing it to 50%
gives the best hybrid numbers found:

| Attack | NAD alone | hybrid (gamma=1, oversample=0.5) |
|---|---|---|
| BadNets | **0.32** | 0.99 |
| Blend | 0.01 | **0.005** |
| SIG | 0.97 | 0.93 |
| WaNet | **0.01** | 0.05 |
| Dynamic | 1.00 | **0.06** |
| Trojan | 1.00 | 0.66 |

(ASR only; hybrid's clean accuracy is consistently ~5-15pt lower than NAD's
~0.91-0.92 across all six)

**Conclusion so far: hybrid is not a uniform improvement over NAD.** It is a
clear, substantial win specifically for Dynamic (1.00 -> 0.06 ASR, something
no individual defense achieved) and a modest one for Trojan, but it is worse
than plain NAD on BadNets, Blend, and WaNet, and neither approach touches
SIG. BadNets and SIG stay stuck above 0.9 ASR across *every* hybrid
configuration tried (gamma in {1, 3}, oversample_fraction in {0.3, 0.5}) --
that flatness across a 2D hyperparameter sweep is itself evidence the
bottleneck is isolation *quality* for those two attacks (the AIBD-style
loss-ranking step likely isn't finding the real poisoned samples), not a
tunable loss weight.

### Alternative isolation signals: cluster, ensemble, frequency

`hybrid_defense`'s `isolation_method` is swappable. Two more signals were
tried, each targeting a different property of how a trigger manifests:

- **`cluster`** — activation clustering (Chen et al., 2018): within each
  class's samples, k-means (k=2) on a PCA projection of penultimate-layer
  features, flagging the smaller/well-separated cluster.
- **`frequency`** — `_isolate_by_frequency`, adapting Zeng et al. (2021)'s
  observation that many backdoor triggers leave a narrow, unnaturally clean
  spike in an image's spatial-frequency spectrum. **This is the only
  isolation method in the repo that doesn't touch the model at all** — it
  scores raw pixel statistics (FFT peakiness of each image's column-intensity
  profile) instead of anything the model learned.

Comparing `loss` vs. `cluster` vs. `frequency` (all with gamma=1,
oversample_fraction=0.5) found each is blind to different attacks:

| Attack | `loss` (best) | `cluster` | `frequency` |
|---|---|---|---|
| BadNets | 0.99 | 1.00 | 0.71 |
| Blend | **0.005** | 0.99 | 0.54 |
| SIG | 0.93 | 0.96 | **0.07** |
| WaNet | 0.05 | 0.33 | 0.27 |
| Dynamic | **0.06** | 1.00 | 0.52 |
| Trojan | 0.66 | **0.01** | 0.63 |

(ASR only)

**`frequency` is the one that finally solves SIG** (0.93 -> 0.07 ASR, by far
the best result any method found for it across the whole investigation) --
confirming the diagnosis: SIG is the repo's only *clean-label* attack (its
10% poison rate applies only within the target class, ~1% of the full
dataset, and labels are never falsified), so it has no "wrong label" or
"anomalous prediction" signature for loss-ranking, clustering, or Neural
Cleanse (below) to exploit. Its poisoned images don't behave strangely to
the model at all -- they just *look* subtly different, which is exactly what
a model-free pixel-statistics signal is positioned to catch and everything
else in this file is blind to. `cluster` does dramatically better on Trojan
specifically (0.66 -> 0.01) because TrojanNN's trigger is literally built by
activation maximization, so it produces an unusually sharp feature-space
cluster. An `ensemble` mode (union of all flagged samples across methods)
was also tried and found to *dilute* rather than combine strengths -- e.g.
Blend regressed from loss-alone's 0.005 to 0.40 -- because a plain union has
no way to weight which signal is actually right for a given attack, so the
"wrong" method's noise interferes with the "right" method's precision.

### Neural Cleanse: a fundamentally different, non-isolation defense

Every defense above tries to identify *which training samples* are
poisoned. `src/backdoor_defenses.py::neural_cleanse_defense` (Wang et al.,
2019) doesn't -- it reverse-engineers, per class, the smallest mask+pattern
that flips any clean image to that class, flags classes whose trigger is an
outlier via a Median Absolute Deviation test, then unlearns by fine-tuning
trigger-stamped clean images toward their true label. Run standalone
against all six attacks:

| Attack | ASR / clean acc | Flagged class |
|---|---|---|
| BadNets | **0.03 / 0.89** | [0] -- correct |
| Blend | 0.01 / 0.88 | [0, 3] -- correct + 1 false positive |
| SIG | 0.93 / 0.86 | [0, 9] -- correct target, still couldn't fix it |
| WaNet | 0.01 / 0.90 | [0] -- correct |
| Dynamic | 1.00 / 0.94 | **[] -- missed entirely** |
| Trojan | 0.89 / 0.89 | **[5] -- wrong class** |

**Neural Cleanse is the best result found for BadNets across the entire
investigation** (0.03 ASR). It operates on the model's decision geometry
rather than pooled per-sample features, so BadNets' small 4x4 corner patch
-- invisible to loss-ranking and activation clustering after global average
pooling dilutes it -- is exactly what its optimization is built to find
directly in pixel space. Its failures are just as mechanistically clean:
Dynamic's trigger varies per input, violating Neural Cleanse's core
assumption of *one universal* trigger per class, so there's nothing
consistent for it to reverse-engineer; and it misidentified Trojan's target
class, plausibly because Trojan's optimization-built trigger doesn't distort
the decision boundary by an unusually small amount even though it does
distort the *feature space* unusually (which is exactly what `cluster`
catches instead).

Folding Neural Cleanse into `hybrid_defense` as a 4th loss term
(`use_neural_cleanse=True`, weighted by `delta`) was also tried, combined
with `loss` isolation:

| Attack | Standalone NC | loss-hybrid alone | loss-hybrid + NC combined |
|---|---|---|---|
| BadNets | **0.03** | 0.99 | 0.58 |
| Blend | 0.01 | 0.005 | **0.003** |
| Dynamic | 1.00 | **0.06** | 0.16 |
| Trojan | 0.89 | 0.66 | 0.37 |

Combining **hurt** BadNets specifically (0.03 standalone -> 0.58 combined)
and diluted Dynamic's already-good result (0.06 -> 0.16): folding in the
loss-isolation and NAD-distillation terms means the optimizer balances four
competing losses per step, and since loss-based isolation is actively
*wrong* for BadNets (that's the whole reason Neural Cleanse was needed), its
noisy correction interferes with Neural Cleanse's otherwise-clean signal in
the same gradient step. It did help Trojan (0.66 -> 0.37, though still far
behind `cluster`'s 0.01) and gave Blend a marginal edge. **The lesson:
combining signals isn't automatically better -- when one signal is already
excellent for a given attack and another is actively wrong for it in the
same optimization, combining can make things worse than the better signal
alone.**

### The complete picture: every attack, solved by something different

Across ABL, AIBD, CBD, DBD, MCR, NAD, four `hybrid_defense` isolation modes,
standalone and combined Neural Cleanse, and a 2D hyperparameter grid, every
one of the six attacks now has a defense that meaningfully breaks it -- but
no single method solves more than two, and the "right" method traces
directly back to that attack's trigger mechanism:

| Attack | Winning defense | ASR | Why that mechanism matches |
|---|---|---|---|
| BadNets | Neural Cleanse (standalone) | 0.03 | Small universal patch -- found via decision-geometry search, invisible to pooled features |
| Blend | `hybrid` (loss isolation + Neural Cleanse) | 0.003 | Full-image pattern -- both label-anomaly and decision-geometry signals agree |
| SIG | `hybrid` (frequency isolation) | 0.07 | Clean-label periodic pattern -- only a model-free pixel-statistics signal sees it |
| WaNet | NAD (simplest defense in the file) | 0.01 | Geometric warp -- attention distillation alone suffices, no isolation needed |
| Dynamic | `hybrid` (loss isolation, no Neural Cleanse) | 0.06 | Per-input trigger -- loss-anomaly still fires per-sample even though the pattern varies |
| Trojan | `hybrid` (activation-clustering isolation) | 0.01 | Activation-maximized trigger -- feature-space clustering is exactly what it's built to trigger |

The overarching finding of this whole investigation: **there is no universal
backdoor defense at this problem scale.** Every isolation/detection
mechanism here is precision-matched to a specific trigger mechanism, and the
combinations that were tried to unify them (`ensemble` isolation, folding
Neural Cleanse into `hybrid`) diluted rather than combined their individual
strengths. A real defender who doesn't already know which trigger family
they're facing can't simply pick the "best" method -- they would need to run
several and take whichever one's diagnostic signal (isolation confidence,
Neural Cleanse's anomaly index, etc.) looks most convincing, which is itself
an unsolved meta-problem this repo doesn't attempt.

## What's implemented vs. what's in the survey

| Survey section | Implemented here |
|---|---|
| FGSM, PGD, C&W attacks | `src/attacks.py` |
| Obfuscated-gradients caution | Documented above; not a specific defense to break |
| PGD adversarial training, TRADES | `src/defenses.py` |
| Randomized smoothing certification | `src/smoothing.py`, `certify.py` |
| AutoAttack / RobustBench-style standardized eval | Not implemented (would pull in the `autoattack` package — out of scope here; `evaluate.py`'s PGD-20/C&W is a lighter-weight stand-in) |
| IBP / Lipschitz certification | Not implemented |
| Diffusion-based purification, ViT robustness, LLM/VLM robustness | Not implemented — these are newer, less settled directions noted in the survey as future work, not reference algorithms with a single canonical implementation |
| BadNets, Blend, SIG, Dynamic, WaNet, Trojan backdoor attacks | `src/backdoor_attacks.py` |
| ABL, AIBD, CBD, DBD, MCR, NAD backdoor defenses | `src/backdoor_defenses.py` |
| Neural Cleanse (Wang et al. 2019) | `src/backdoor_defenses.py::neural_cleanse_defense` — best standalone result found for BadNets |
| Activation clustering (Chen et al. 2018) / frequency-domain analysis (Zeng et al. 2021), as isolation signals | `src/backdoor_defenses.py::_isolate_by_activation_clustering` / `_isolate_by_frequency` |
| Combined/hybrid backdoor defense (not from a single paper) | `src/backdoor_defenses.py::hybrid_defense` — see "Experimental findings" above for the full per-attack comparison |

## Notes on hyperparameters

- `eps=8/255` (Linf) is the standard CIFAR-10 budget used by Madry et al.,
  TRADES, and RobustBench.
- C&W's confidence weight `c` is escalated per-sample (geometric search over
  a handful of rounds) rather than fixed, since a too-small fixed `c` can
  silently fail to find any adversarial example and understate attack
  strength — see the docstring in `src/attacks.py::cw_l2_attack`.
- `sigma` for randomized smoothing must match between training
  (`train.py --sigma`) and certification (`certify.py --sigma`); Cohen et
  al. report sigma in {0.12, 0.25, 0.5, 1.0} for CIFAR-10.
