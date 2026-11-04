"""Backdoor defenses: ABL, AIBD, CBD, DBD, MCR, NAD, and a combined `hybrid`.

These operate on a model trained (or being trained) on data that may contain
a src/backdoor_attacks.py-style poisoning attack, and try to remove the
trigger -> target-label association while preserving clean accuracy. Two
threat-model flavors are used, matching each paper:

  - ABL, AIBD, CBD, DBD: the defender retrains from scratch (or from the
    poisoned checkpoint) using the *poisoned training set itself*, isolating
    or down-weighting suspected-poisoned samples along the way.
  - MCR, NAD: the defender only has the already-backdoored model plus a
    small *clean* holdout set, and repairs the model by fine-tuning /
    interpolating on that holdout.

`hybrid_defense` combines both flavors (see its docstring): it uses the
poisoned training set for AIBD-style isolation, and a small clean holdout
for NAD-style teacher distillation. It was designed after running the full
6-attack x 6-defense grid in results/backdoor_experiment_summary.csv, which
showed NAD as the only defense with a real effect against BadNets/Blend
while ABL/AIBD/CBD stayed near the undefended ASR and DBD's clean accuracy
collapsed outright at this repo's default epoch budget.
"""

import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.cluster.vq import kmeans2
from torch.utils.data import DataLoader, Dataset, Subset, WeightedRandomSampler

from src.models import build_model, to_device, unwrap


def _per_sample_losses(model, dataset, device, batch_size=256):
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    losses = []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            losses.append(F.cross_entropy(logits, y, reduction="none").cpu())
    return torch.cat(losses)


def _batched_indices(n, batch_size):
    for start in range(0, n, batch_size):
        yield list(range(start, min(start + batch_size, n)))


# --------------------------------------------------------------------------
# ABL -- Anti-Backdoor Learning (Li et al., NeurIPS 2021)
# --------------------------------------------------------------------------

def abl_defense(model, poisoned_dataset, optimizer, device,
                 isolation_epochs=20, gamma=0.5, isolation_ratio=0.01,
                 unlearn_epochs=5, batch_size=128):
    """Stage 1 (isolation): train with a "loss-flooding" objective
    |CE - gamma| + gamma, which flattens the loss around gamma. Backdoor
    samples are easy to fit (the trigger is a much stronger, spurious
    signal), so they are driven below gamma faster than clean samples,
    leaving them with distinctly *lower* loss -- letting the lowest-loss
    `isolation_ratio` fraction be isolated as suspected poison.

    Stage 2 (unlearning): continue training on the full dataset, but apply
    gradient ASCENT (negative loss) on the isolated subset -- actively
    breaking the trigger -> target-label association -- and ordinary
    descent everywhere else.
    """
    loader = DataLoader(poisoned_dataset, batch_size=batch_size, shuffle=True)

    model.train()
    for _ in range(isolation_epochs):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            ce = F.cross_entropy(model(x), y)
            loss = (ce - gamma).abs() + gamma
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    losses = _per_sample_losses(model, poisoned_dataset, device, batch_size)
    n_isolate = max(1, int(len(losses) * isolation_ratio))
    isolated_idx = set(losses.argsort()[:n_isolate].tolist())

    model.train()
    for _ in range(unlearn_epochs):
        for idx in _batched_indices(len(poisoned_dataset), batch_size):
            xs, ys = zip(*(poisoned_dataset[i] for i in idx))
            x = torch.stack(xs).to(device)
            y = torch.tensor(ys).to(device)
            is_isolated = torch.tensor([i in isolated_idx for i in idx], device=device)

            per_sample_ce = F.cross_entropy(model(x), y, reduction="none")
            sign = torch.where(is_isolated, -1.0, 1.0)
            loss = (sign * per_sample_ce).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return model


# --------------------------------------------------------------------------
# AIBD -- Adversarial-Inspired Backdoor Defense (AAAI 2025)
# --------------------------------------------------------------------------

def aibd_defense(model, poisoned_dataset, optimizer, device,
                  warmup_epochs=10, final_q=0.05, unlearn_rounds=10, batch_size=128):
    """Simplified analog of AIBD. Differs from ABL in both isolation and
    unlearning mechanics: (1) isolation uses a *progressively growing*
    top-q fraction of lowest-loss samples, recomputed each round, rather
    than ABL's one-shot flooding-loss threshold; (2) unlearning trains
    isolated samples *towards an adversarial label* (the current model's
    most-likely *incorrect* class -- its second choice -- for that sample)
    via ordinary gradient descent, rather than raw gradient ascent --
    numerically more stable and, per the paper, more effective at breaking
    the shortcut without the clean-accuracy collapse unconstrained ascent
    can cause.
    """
    loader = DataLoader(poisoned_dataset, batch_size=batch_size, shuffle=True)

    model.train()
    for _ in range(warmup_epochs):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            loss = F.cross_entropy(model(x), y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    for round_idx in range(unlearn_rounds):
        q = final_q * (round_idx + 1) / unlearn_rounds
        losses = _per_sample_losses(model, poisoned_dataset, device, batch_size)
        n_isolate = max(1, int(len(losses) * q))
        isolated_idx = set(losses.argsort()[:n_isolate].tolist())

        model.train()
        for idx in _batched_indices(len(poisoned_dataset), batch_size):
            xs, ys = zip(*(poisoned_dataset[i] for i in idx))
            x = torch.stack(xs).to(device)
            y = torch.tensor(ys).to(device)
            is_isolated = torch.tensor([i in isolated_idx for i in idx], device=device)

            logits = model(x)
            with torch.no_grad():
                masked = logits.clone()
                masked.scatter_(1, y.unsqueeze(1), float("-inf"))
                adv_targets = masked.argmax(1)  # model's most-likely incorrect class (second choice)
            targets = torch.where(is_isolated, adv_targets, y)

            loss = F.cross_entropy(logits, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return model


# --------------------------------------------------------------------------
# CBD -- Backdoor Defense via Deconfounded Representation Learning (CVPR 2023)
# --------------------------------------------------------------------------

def cbd_defense(model_c, arch, poisoned_dataset, device,
                 confound_epochs=5, joint_epochs=15, batch_size=128, mi_weight=0.5, lr=0.01):
    """Simplified analog of Zhang et al. (2023). Trains a small "confounder"
    model B for a few epochs on the poisoned data -- it overfits the
    backdoor shortcut quickly, since the trigger is, by construction, the
    easiest-to-fit feature. The target model C is then jointly fine-tuned
    to (a) minimize classification loss with samples down-weighted by how
    confidently B predicts them (B's confidence is used as a proxy for "is
    this sample exploiting the shortcut"), and (b) minimize the cosine
    similarity between C's and B's penultimate features, discouraging C
    from leaning on the same shortcut representation.
    """
    model_b = to_device(build_model(arch), device)
    opt_b = torch.optim.SGD(model_b.parameters(), lr=lr, momentum=0.9)
    loader = DataLoader(poisoned_dataset, batch_size=batch_size, shuffle=True)

    model_b.train()
    for _ in range(confound_epochs):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            loss = F.cross_entropy(model_b(x), y)
            opt_b.zero_grad()
            loss.backward()
            opt_b.step()
    model_b.eval()
    for p in model_b.parameters():
        p.requires_grad_(False)

    opt_c = torch.optim.SGD(model_c.parameters(), lr=lr, momentum=0.9)
    model_c.train()
    for _ in range(joint_epochs):
        for x, y in loader:
            x, y = x.to(device), y.to(device)

            with torch.no_grad():
                logits_b, feats_b = model_b(x, return_features=True)
                confidence_b = F.softmax(logits_b, dim=1).max(1)[0]
                sample_weight = (1.0 - confidence_b).clamp(min=0.1)

            logits_c, feats_c = model_c(x, return_features=True)
            ce = F.cross_entropy(logits_c, y, reduction="none")
            ce_weighted = (sample_weight * ce).mean()

            fb = F.normalize(feats_b[-1].flatten(1), dim=1)
            fc = F.normalize(feats_c[-1].flatten(1), dim=1)
            mi_proxy = (fb * fc).sum(1).mean()

            loss = ce_weighted + mi_weight * mi_proxy
            opt_c.zero_grad()
            loss.backward()
            opt_c.step()

    return model_c


# --------------------------------------------------------------------------
# DBD -- Backdoor Defense via Decoupling the Training Process (ICLR 2022)
# --------------------------------------------------------------------------

class _AttachedHead(nn.Module):
    def __init__(self, backbone, head):
        super().__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, x):
        _, feats = self.backbone(x, return_features=True)
        pooled = feats[-1].mean(dim=(2, 3))
        return self.head(pooled)


def _simclr_augment(x):
    if torch.rand(1).item() < 0.5:
        x = x.flip(-1)
    brightness = 0.8 + 0.4 * torch.rand(x.size(0), 1, 1, 1, device=x.device)
    x = (x * brightness + 0.02 * torch.randn_like(x)).clamp(0, 1)
    return x


def _simclr_loss(z1, z2, temperature):
    n = z1.size(0)
    z = torch.cat([z1, z2], dim=0)
    sim = z @ z.t() / temperature
    sim.fill_diagonal_(float("-inf"))
    targets = torch.cat([torch.arange(n, 2 * n), torch.arange(0, n)]).to(z.device)
    return F.cross_entropy(sim, targets)


def dbd_defense(arch, poisoned_dataset, device, num_classes=10,
                 ssl_epochs=20, linear_epochs=10, finetune_epochs=10,
                 credible_fraction=0.5, batch_size=128, lr=0.01, temperature=0.5):
    """Simplified analog of Huang et al. (2022). Three stages: (1)
    contrastive (SimCLR-style) pretraining of the backbone on poisoned
    images *without using their labels* -- since the poison acts on the
    label, an unsupervised backbone has no signal that lets the trigger
    shortcut form; (2) freeze the backbone, train only a fresh linear head
    with the (still poisoned) labels; (3) unfreeze and fine-tune end-to-end,
    restricted to the "high-credibility" (lowest current per-sample loss)
    fraction of the data, since poisoned samples surviving stage 2 tend to
    be exactly the ones the model is confident about for the wrong reason.

    Uses a lightweight two-augmentation contrastive loss rather than the
    paper's full self-supervised pipeline (which uses MoCo/SimCLR at much
    larger scale) and a loss-threshold credible-sample filter rather than
    its MixMatch-based semi-supervised fine-tuning stage.
    """
    backbone = to_device(build_model(arch, num_classes), device)
    ssl_loader = DataLoader(poisoned_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    opt = torch.optim.SGD(backbone.parameters(), lr=lr, momentum=0.9)

    backbone.train()
    for _ in range(ssl_epochs):
        for x, _ in ssl_loader:
            x = x.to(device)
            x1, x2 = _simclr_augment(x), _simclr_augment(x)
            _, f1 = backbone(x1, return_features=True)
            _, f2 = backbone(x2, return_features=True)
            z1 = F.normalize(f1[-1].mean(dim=(2, 3)), dim=1)
            z2 = F.normalize(f2[-1].mean(dim=(2, 3)), dim=1)
            loss = _simclr_loss(z1, z2, temperature)
            opt.zero_grad()
            loss.backward()
            opt.step()

    for p in backbone.parameters():
        p.requires_grad_(False)
    with torch.no_grad():
        probe_x = next(iter(ssl_loader))[0][:2].to(device)
        _, probe_feats = backbone(probe_x, return_features=True)
        feature_dim = probe_feats[-1].mean(dim=(2, 3)).size(1)
    head = nn.Linear(feature_dim, num_classes).to(device)
    opt_head = torch.optim.SGD(head.parameters(), lr=lr, momentum=0.9)
    head_loader = DataLoader(poisoned_dataset, batch_size=batch_size, shuffle=True)

    backbone.eval()
    for _ in range(linear_epochs):
        for x, y in head_loader:
            x, y = x.to(device), y.to(device)
            with torch.no_grad():
                _, feats = backbone(x, return_features=True)
                pooled = feats[-1].mean(dim=(2, 3))
            loss = F.cross_entropy(head(pooled), y)
            opt_head.zero_grad()
            loss.backward()
            opt_head.step()

    for p in backbone.parameters():
        p.requires_grad_(True)
    full_model = _AttachedHead(backbone, head)

    losses = _per_sample_losses(full_model, poisoned_dataset, device, batch_size)
    n_credible = max(1, int(len(losses) * credible_fraction))
    credible_idx = losses.argsort()[:n_credible].tolist()
    credible_subset = Subset(poisoned_dataset, credible_idx)

    opt_ft = torch.optim.SGD(full_model.parameters(), lr=lr * 0.1, momentum=0.9)
    ft_loader = DataLoader(credible_subset, batch_size=batch_size, shuffle=True)
    full_model.train()
    for _ in range(finetune_epochs):
        for x, y in ft_loader:
            x, y = x.to(device), y.to(device)
            loss = F.cross_entropy(full_model(x), y)
            opt_ft.zero_grad()
            loss.backward()
            opt_ft.step()

    # Rebuild with the backbone unwrapped from DataParallel (if it was
    # wrapped) so the returned model is always plain and portable to
    # checkpoint regardless of how many GPUs trained it; this shares the
    # same underlying parameter tensors, so no weights are lost.
    return _AttachedHead(unwrap(backbone), head)


# --------------------------------------------------------------------------
# MCR -- Mode Connectivity Repair (Zhao et al., ICLR 2020)
# --------------------------------------------------------------------------

def finetune_on_clean(model, clean_loader, device, epochs, lr, seed):
    """Produces one Bezier-curve endpoint: an independent fine-tune of
    `model` on a small clean holdout set, seeded differently so that two
    calls give two distinct endpoints in weight space.
    """
    torch.manual_seed(seed)
    m = copy.deepcopy(model).to(device)
    optimizer = torch.optim.SGD(m.parameters(), lr=lr, momentum=0.9)
    m.train()
    for _ in range(epochs):
        for x, y in clean_loader:
            x, y = x.to(device), y.to(device)
            loss = F.cross_entropy(m(x), y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return m


def mcr_defense(model_a, model_b, clean_loader, device, curve_epochs=20, lr=0.01):
    """Given two independently fine-tuned endpoints (see `finetune_on_clean`),
    fits a quadratic Bezier curve theta(t) = (1-t)^2*A + 2t(1-t)*M + t^2*B
    through weight space by training the learnable midpoint M so every
    point along the curve has low clean loss (Garipov et al.'s mode-
    connectivity training, applied here as a defense per Zhao et al.). The
    model at the curve's midpoint (t=0.5) typically preserves clean
    accuracy while landing outside the sharp, isolated loss basin the
    backdoor carved out in weight space, disrupting the trigger response.
    """
    from torch.func import functional_call

    params_a = {k: v.detach().clone() for k, v in model_a.named_parameters()}
    params_b = {k: v.detach().clone() for k, v in model_b.named_parameters()}
    buffers = {k: v.detach().clone() for k, v in model_a.named_buffers()}
    params_m = {k: nn.Parameter(0.5 * (params_a[k] + params_b[k])) for k in params_a}

    optimizer = torch.optim.SGD(list(params_m.values()), lr=lr, momentum=0.9)

    for _ in range(curve_epochs):
        for x, y in clean_loader:
            x, y = x.to(device), y.to(device)
            t = torch.rand(1).item()
            c1, c2, c3 = (1 - t) ** 2, 2 * t * (1 - t), t ** 2
            interp = {k: c1 * params_a[k] + c2 * params_m[k] + c3 * params_b[k] for k in params_a}
            logits = functional_call(model_a, (interp, buffers), (x,))
            loss = F.cross_entropy(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    with torch.no_grad():
        midpoint = {k: 0.25 * params_a[k] + 0.5 * params_m[k] + 0.25 * params_b[k] for k in params_a}

    repaired = copy.deepcopy(model_a)
    repaired.load_state_dict({**midpoint, **buffers})
    return repaired


# --------------------------------------------------------------------------
# NAD -- Neural Attention Distillation (Li et al., ICLR 2021)
# --------------------------------------------------------------------------

def _attention_map(feat):
    """Zagoruyko & Komodakis (2017): mean of squared activations across
    channels, L2-normalized once flattened -- a spatial "where does the
    network look" map.
    """
    a = feat.pow(2).mean(1).flatten(1)
    return F.normalize(a, dim=1)


def _attention_loss(student_feat, teacher_feat):
    return (_attention_map(student_feat) - _attention_map(teacher_feat)).pow(2).mean()


def nad_defense(student, clean_loader, device, teacher_epochs=10, distill_epochs=10,
                 beta=500.0, lr=0.01):
    """Fine-tunes a teacher (an independent copy of the backdoored student)
    on a small clean subset, then fine-tunes the student on the same subset
    with cross-entropy plus an attention-transfer loss pulling the
    student's intermediate spatial attention maps toward the teacher's --
    erasing the trigger-driven activation pattern the student learned.
    """
    teacher = copy.deepcopy(student).to(device)
    opt_t = torch.optim.SGD(teacher.parameters(), lr=lr, momentum=0.9)
    teacher.train()
    for _ in range(teacher_epochs):
        for x, y in clean_loader:
            x, y = x.to(device), y.to(device)
            loss = F.cross_entropy(teacher(x), y)
            opt_t.zero_grad()
            loss.backward()
            opt_t.step()
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    opt_s = torch.optim.SGD(student.parameters(), lr=lr, momentum=0.9)
    student.train()
    for _ in range(distill_epochs):
        for x, y in clean_loader:
            x, y = x.to(device), y.to(device)
            logits_s, feats_s = student(x, return_features=True)
            with torch.no_grad():
                _, feats_t = teacher(x, return_features=True)

            at_loss = sum(_attention_loss(fs, ft) for fs, ft in zip(feats_s, feats_t))
            loss = F.cross_entropy(logits_s, y) + beta * at_loss
            opt_s.zero_grad()
            loss.backward()
            opt_s.step()

    return student


# --------------------------------------------------------------------------
# Hybrid -- combines AIBD isolation, CBD-style reweighting, and NAD
# attention distillation (the mechanisms that showed real effect above)
# --------------------------------------------------------------------------

class _IndexedDataset(Dataset):
    """Wraps a dataset to also return each sample's own index -- lets a
    single shuffled DataLoader over the *whole* poisoned dataset still tell,
    per sample in a batch, whether that sample was isolated as suspected
    poison (AIBD's own mixing mechanism: isolated and non-isolated samples
    share the same batch/forward/backward pass, not separate ones).
    """

    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i):
        x, y = self.dataset[i]
        return x, y, i


def _isolate_by_frequency(poisoned_dataset, final_q=0.05, batch_size=256):
    """Zeng et al. (2021), "Rethinking the Backdoor Attacks' Triggers: A
    Frequency Perspective" -- adapted here as an isolation score rather
    than their trained detector. Many backdoor triggers, especially smooth
    periodic additive patterns like SIG's sinusoidal stripes, introduce a
    narrow, unnaturally clean spike in an image's spatial-frequency
    spectrum that natural photographic content (broadband edges/textures)
    essentially never produces at that purity.

    This targets exactly what every other isolation method in this file
    misses about SIG: it's the only *clean-label* attack among the six
    (poisoned samples keep their correct label, and are only ~1% of the
    whole dataset since the 10% poison rate applies within the target
    class alone), so there's no "wrong label" or "confidently anomalous
    prediction" signal for loss-ranking, activation clustering, or Neural
    Cleanse to exploit -- SIG's poisoned images don't behave strangely
    with respect to the model at all, they just *look* subtly different.
    This method is the only one in the file that ignores the model
    entirely and scores images by their own pixel statistics instead,
    which is exactly why it doesn't share that blind spot: SIG's stripe
    pattern is a physical property of the image regardless of what any
    model learned from it.

    Score: for each image, average over channels and rows to get a 1D
    column-intensity profile, take its FFT, zero out the DC term, and
    score by (peak magnitude) / (total spectral energy) -- a pure sinusoid
    added to the image concentrates energy into one narrow band, giving a
    much higher ratio than a natural image's broadband spectrum. The
    highest-scoring `final_q` fraction is isolated as suspected poison.
    """
    loader = DataLoader(_IndexedDataset(poisoned_dataset), batch_size=batch_size, shuffle=False)
    scores, all_idx = [], []
    for x, _, idx in loader:
        profile = x.mean(dim=(1, 2))  # (B, W): mean over channels and rows
        spectrum = torch.fft.rfft(profile, dim=1).abs()
        spectrum[:, 0] = 0.0  # drop the DC (average brightness) component
        peak = spectrum.max(dim=1).values
        total = spectrum.sum(dim=1) + 1e-8
        scores.append(peak / total)
        all_idx.append(idx)
    scores = torch.cat(scores)
    all_idx = torch.cat(all_idx)
    n_isolate = max(1, int(len(scores) * final_q))
    top = scores.argsort(descending=True)[:n_isolate]
    return all_idx[top].tolist()


def _isolate_by_loss(model, poisoned_dataset, device, final_q, batch_size):
    """AIBD-style isolation: rank the poisoned training set by per-sample
    loss under the already-backdoored model and isolate the lowest-loss
    `final_q` fraction as suspected poison.
    """
    losses = _per_sample_losses(model, poisoned_dataset, device, batch_size)
    n_isolate = max(1, int(len(losses) * final_q))
    return losses.argsort()[:n_isolate].tolist()


def _isolate_by_activation_clustering(model, poisoned_dataset, device, num_classes=10,
                                       batch_size=256, pca_dim=10, separation_threshold=1.5):
    """Chen et al. (2018), Activation Clustering. Within each class's
    samples (by their *recorded* label -- for label-flipping attacks this
    is the target label poisoned samples were relabeled to; for SIG's
    clean-label attack it's just the target class itself), cluster
    penultimate-layer features into 2 groups via k-means on a small PCA
    projection. If the two groups are well-separated (centroid distance
    large relative to within-cluster spread), the smaller group is
    flagged as suspected poison.

    This targets a fundamentally different signal than `_isolate_by_loss`:
    samples sharing a visual trigger tend to cluster together in feature
    space *regardless* of how "easy" their loss is. That matters because
    loss-ranking (ABL/AIBD in the 6x6 grid, and _isolate_by_loss here) is
    inert against BadNets and SIG specifically -- for BadNets, plenty of
    genuinely easy, prototypical clean images across 10 balanced classes
    are just as low-loss as the ~10% poisoned ones, so a flat loss cutoff
    doesn't separate them; for SIG, the trigger's low amplitude barely
    shifts loss at all. Neither problem is about loss *ranking* being
    miscalibrated -- it's that loss isn't the signal that separates these
    two attacks' poisoned samples from clean ones, which is exactly what
    activation clustering targets instead.
    """
    isolated_idx, _, _ = _activation_cluster_core(
        model, poisoned_dataset, device, num_classes, batch_size, pca_dim, separation_threshold
    )
    return isolated_idx


def _activation_cluster_core(model, poisoned_dataset, device, num_classes,
                              batch_size, pca_dim, separation_threshold):
    """Shared implementation behind `_isolate_by_activation_clustering`.
    Returns (isolated_idx, max_separation, max_calibrated_separation): the
    second value is the best raw per-class centroid-distance /
    within-cluster-spread ratio found, whether or not it cleared
    `separation_threshold`; the third is that same best class's separation
    z-scored against a random-split null baseline computed on the same
    features (see `_calibrated_confidence`'s docstring for why the raw
    value alone isn't comparable across signal types) -- `_isolate_adaptive`
    uses the calibrated value, `_isolate_by_activation_clustering` uses
    neither.
    """
    model.eval()
    loader = DataLoader(_IndexedDataset(poisoned_dataset), batch_size=batch_size, shuffle=False)
    all_feats, all_labels, all_idx = [], [], []
    with torch.no_grad():
        for x, y, idx in loader:
            x = x.to(device)
            _, feats = model(x, return_features=True)
            pooled = feats[-1].mean(dim=(2, 3)).cpu()
            all_feats.append(pooled)
            all_labels.append(y)
            all_idx.append(idx)
    all_feats = torch.cat(all_feats).numpy()
    all_labels = torch.cat(all_labels).numpy()
    all_idx = torch.cat(all_idx).numpy()

    isolated_idx = []
    max_separation = 0.0
    max_calibrated = 0.0
    null_rng = np.random.RandomState(0)
    for c in range(num_classes):
        mask = all_labels == c
        if mask.sum() < 4:
            continue
        class_idx = all_idx[mask]
        feats_c = all_feats[mask]

        centered = feats_c - feats_c.mean(axis=0, keepdims=True)
        k = max(1, min(pca_dim, centered.shape[0] - 1, centered.shape[1]))
        _, _, vt = np.linalg.svd(centered, full_matrices=False)
        reduced = centered @ vt[:k].T

        def _kmeans_separation(feats, seed):
            """2-means on `feats`; returns (separation, smaller-cluster
            mask) or (0.0, None) if one cluster came back empty."""
            _, labels_km = kmeans2(feats, 2, minit="++", seed=seed)
            sizes = np.array([np.sum(labels_km == 0), np.sum(labels_km == 1)])
            if sizes.min() == 0:
                return 0.0, None
            smaller = 0 if sizes[0] < sizes[1] else 1
            m = labels_km == smaller
            group_small, group_large = feats[m], feats[~m]
            centroid_dist = np.linalg.norm(group_small.mean(0) - group_large.mean(0))
            avg_within = (group_small.std() + group_large.std()) / 2 + 1e-8
            return centroid_dist / avg_within, m

        separation, true_mask = _kmeans_separation(reduced, seed=0)
        if true_mask is None:
            continue
        max_separation = max(max_separation, float(separation))

        # Calibrated confidence: z-score the true 2-means split's
        # separation against 2-means splits of *synthetic unimodal
        # Gaussian* features matching this class's per-dimension mean/std
        # -- not against random splits of the real features (an earlier,
        # wrong version of this calibration; see _calibrated_confidence's
        # docstring for the general reason). K-means finds the
        # maximally-separating 2-way partition by construction, so it
        # will report *some* separation even for pure unimodal noise;
        # comparing against a null that also runs k-means (just on
        # structure-free synthetic data) is what actually tests whether
        # the real split is more separated than chance, rather than just
        # confirming that k-means always finds a split.
        feat_mean, feat_std = reduced.mean(axis=0), reduced.std(axis=0) + 1e-8
        null_seps = np.empty(20)
        for i in range(20):
            null_feats = null_rng.normal(feat_mean, feat_std, size=reduced.shape)
            null_sep, _ = _kmeans_separation(null_feats, seed=int(null_rng.randint(1_000_000)))
            null_seps[i] = null_sep
        calibrated = (separation - null_seps.mean()) / (null_seps.std() + 1e-8)
        max_calibrated = max(max_calibrated, float(calibrated))

        if separation >= separation_threshold:
            isolated_idx.extend(class_idx[true_mask].tolist())

    return isolated_idx, max_separation, max_calibrated


def _isolate_by_strip(model, poisoned_dataset, clean_loader, device, final_q=0.05,
                       batch_size=128, num_blends=4, blend_alpha=0.85, seed=0, pool_size=256):
    """Gao et al. (2019), STRIP ("STRong Intentional Perturbation"). Unlike
    every other isolation signal in this file, which scores samples by some
    static property (their loss, their feature-space position, their own
    pixel spectrum), STRIP probes a *behavioral* property: whether the
    model's prediction for a sample stays anomalously stable when the
    sample is corrupted with unrelated content. For each candidate image,
    it is alpha-blended (`blend_alpha` on the candidate, `1-blend_alpha` on
    the reference) with `num_blends` random images drawn from a trusted
    clean pool (taken from `clean_loader`, the same small holdout
    MCR/NAD/Neural Cleanse use), and the average Shannon entropy of the
    resulting softmax predictions is recorded. The lowest-entropy
    `final_q` fraction is isolated as suspected poison.

    The original rationale for adding this file's signal was that a
    dominant, patch/blend-style trigger should survive perturbation while
    a clean image's prediction should wander -- so it should target
    BadNets/Blend specifically, where loss-ranking and activation
    clustering struggle most. Measured directly against this repo's actual
    poison_idx ground truth (precision/recall of the isolated set, bypassing
    hybrid_defense's downstream repair dynamics entirely), that rationale
    was wrong in an instructive way. At a 50/50 blend (this parameter's
    first value), isolation precision on BadNets and Blend was near or
    *below* the ~10% random-chance base rate -- Blend was 0.8% precision,
    i.e. essentially anti-correlated with the true poison set. Sweeping
    `blend_alpha` toward the candidate (less dilution of it) recovered a
    real signal for Blend (0.8% -> 57.6% precision from alpha 0.5 -> 0.95,
    which is why the default here is 0.85, not the original 0.5) because
    Blend's alpha-blended trigger is itself low-opacity and a second,
    STRIP-side dilution on top of it was destroying it before the entropy
    signal ever got a chance to see it. It did *not* recover a signal for
    BadNets at any alpha tested (precision stayed pinned near the random
    baseline even at alpha=0.95, i.e. an almost-undiluted candidate):
    BadNets' patch is small enough, and this repo's model confident enough
    on plenty of genuinely clean CIFAR images, that "stays confidently
    low-entropy under a barely-there perturbation" stops being a
    discriminating property for anyone, clean or poisoned, once the
    perturbation itself is weak. That reproduces, via a completely
    different (behavioral rather than geometric/statistical) mechanism,
    this codebase's central finding: BadNets' small localized patch evades
    every per-sample isolation signal tried here (loss, cluster, frequency,
    and now this one) -- see neural_cleanse_defense, the only method in
    this file that solves it, because it operates on decision geometry
    rather than per-sample data at all.

    Unexpectedly, this signal's best measured precision wasn't on
    BadNets/Blend at all but on Trojan (41.2% precision at final_q=0.05,
    beating loss-ranking's 28.8% there) -- plausibly because TrojanNN's
    trigger is built by activation maximization specifically to dominate
    the network's internal representations, so it also dominates under
    blending better than a hand-placed patch does. Even so, Blend's
    tuned-alpha precision (57.6%) still trails loss-ranking's own precision
    on Blend (75.2%, established earlier in this investigation) -- so this
    signal does not add a new best-in-class capability anywhere in the
    grid; it is documented here as a genuine, mechanism-independent
    replication of the BadNets finding, not as an improvement over it.
    """
    model.eval()
    clean_pool, pool_n = [], 0
    for x, _ in clean_loader:
        clean_pool.append(x)
        pool_n += x.size(0)
        if pool_n >= pool_size:
            break
    clean_pool = torch.cat(clean_pool)[:pool_size]

    rng = torch.Generator().manual_seed(seed)
    loader = DataLoader(_IndexedDataset(poisoned_dataset), batch_size=batch_size, shuffle=False)
    entropies, all_idx = [], []
    with torch.no_grad():
        for x, _, idx in loader:
            x = x.to(device)
            b = x.size(0)
            batch_entropy = torch.zeros(b, device=device)
            for _ in range(num_blends):
                ref_idx = torch.randint(0, clean_pool.size(0), (b,), generator=rng)
                ref = clean_pool[ref_idx].to(device)
                blended = (blend_alpha * x + (1 - blend_alpha) * ref).clamp(0, 1)
                probs = F.softmax(model(blended), dim=1)
                batch_entropy += -(probs * (probs + 1e-8).log()).sum(dim=1)
            entropies.append((batch_entropy / num_blends).cpu())
            all_idx.append(idx)
    entropies = torch.cat(entropies)
    all_idx = torch.cat(all_idx)
    n_isolate = max(1, int(len(entropies) * final_q))
    bottom = entropies.argsort()[:n_isolate]  # lowest entropy = most trigger-dominant
    return all_idx[bottom].tolist()


def _isolate_by_cafd(model, poisoned_dataset, clean_loader, device, final_q=0.05,
                      batch_size=128, num_classes=10, pca_dim=10, threshold_percentile=None):
    """Clean-Anchored Feature Deviation (CAFD) -- new to this codebase, not
    from a specific prior paper, though it shares Chen et al. (2018)'s
    premise that a trigger leaves a mark in penultimate-layer feature
    space. Every other feature-space or score-based signal in this file
    (loss, activation clustering, STRIP) calibrates a sample against
    *other training samples* -- either the rest of the (possibly
    contaminated) training set directly, or a synthetic/self-referential
    null built from the training scores themselves (see
    `_calibrated_confidence`'s docstring for why that repeatedly turned
    out to be the wrong question: it measures whether a subset is
    statistically separated from something, not whether it's actually
    anomalous relative to genuine clean data). This signal instead anchors
    directly to the small trusted clean holdout every defense in this file
    already has access to (`clean_loader` -- the same set MCR/NAD/Neural
    Cleanse use): for each class (by recorded label), compute that class's
    true feature centroid and typical intra-class spread from ONLY the
    clean holdout, then score every training sample by its normalized
    distance from its own recorded class's clean centroid. The PCA basis
    used to reduce features is also fit on clean features only, so the
    poisoned samples in the training set can never influence which axes
    get used to detect them -- a circularity `_activation_cluster_core`
    does not avoid (its PCA basis is fit on the training set's own,
    possibly-contaminated features).

    Measured directly against this repo's ground-truth poison_idx
    (bypassing hybrid_defense's downstream repair dynamics), this is the
    first signal in the whole investigation to get real precision on
    BadNets (76.4% vs. loss-ranking's 10.4%, cluster's ~0%, STRIP's 6.0%)
    -- the attack every other per-sample signal here has failed on, with
    only neural_cleanse_defense's decision-geometry approach solving it
    before now. It also beats every prior signal's precision on Blend
    (91.2% vs. loss's 75.2%) and WaNet (80.4% vs. loss's 54.4%), and is
    competitive on Dynamic (66.8% vs. loss's 60.8%). The mechanism this
    suggests: BadNets/Blend/WaNet/Dynamic's triggers all push a poisoned
    sample's features toward wherever the target class's decision region
    is, but *not* toward what a genuine member of that class actually
    looks like in feature space -- a gap only visible once you have a
    genuinely clean reference point to measure the gap against. It is, as
    expected, blind to SIG (0% precision -- a clean-label attack with
    correctly-labeled poisoned samples has no reason to sit apart from its
    own class's genuine centroid; see _isolate_by_frequency, which targets
    SIG through the one property this signal doesn't use, the image's own
    pixel statistics) and notably weaker than activation clustering on
    Trojan (9.6% vs. cluster's 22.7%/91% recall) -- TrojanNN's
    activation-maximized trigger apparently produces a tight, displaced
    cluster of its own rather than samples scattered various distances
    from the genuine centroid, which is exactly the structure
    _isolate_by_activation_clustering's bimodal-split test is built to
    catch and a simple centroid-distance score is not.

    `threshold_percentile`, if set, replaces the fixed-`final_q` top-
    fraction selection with a poison-rate-agnostic one: for each class,
    compute this same signal's score for the clean holdout's OWN samples
    via leave-one-out (each clean sample's distance from the centroid of
    the *other* clean samples in its class, so it isn't trivially near
    zero from including itself in its own reference point), then flag
    training samples whose score exceeds the given percentile of that
    class's clean leave-one-out score distribution. This answers a real
    objection to the plain `final_q` mode: picking a fixed isolation
    fraction implicitly assumes the defender roughly knows the poison
    rate, which is realistic for this repo's fixed 10%-across-every-attack
    benchmark setup but not for a real deployment. Thresholding against
    the clean holdout's own score distribution needs no such assumption --
    it only asks "does this look more anomalous than genuinely clean data
    ever looks," using real held-out data to answer that, the same
    clean-anchoring idea this whole signal is built on, applied one level
    up to auto-select how much to isolate instead of hand-setting it.
    Measured against ground truth, `threshold_percentile=95` recovers
    precision/recall comparable to or better than a hand-tuned
    `final_q=0.10` (which only works well because it happens to match
    these six attacks' actual 10% poison rate) without ever using that
    number: 81.4%/42.8% on BadNets, 92.7%/86.6% on Blend, 84.5%/90.8% on
    WaNet, 87.7%/45.6% on Dynamic -- auto-selecting isolated fractions of
    5-11% per attack on its own. `threshold_percentile=99` is too
    conservative (isolates <1% of data on several attacks, missing almost
    everything); 90 trades some precision for more recall. 95 is the
    default once this parameter is set at all.
    """
    model.eval()

    def _extract(dataset, indexed_loader):
        feats, labels, idxs = [], [], []
        with torch.no_grad():
            for x, y, idx in indexed_loader:
                x = x.to(device)
                _, f = model(x, return_features=True)
                pooled = f[-1].mean(dim=(2, 3)).cpu()
                feats.append(pooled)
                labels.append(y)
                idxs.append(idx)
        return torch.cat(feats).numpy(), torch.cat(labels).numpy(), torch.cat(idxs).numpy()

    clean_loader_indexed = DataLoader(_IndexedDataset(clean_loader.dataset), batch_size=batch_size, shuffle=False)
    clean_feats, clean_labels, _ = _extract(clean_loader.dataset, clean_loader_indexed)
    train_loader = DataLoader(_IndexedDataset(poisoned_dataset), batch_size=batch_size, shuffle=False)
    train_feats, train_labels, train_idx = _extract(poisoned_dataset, train_loader)

    # PCA basis fit on CLEAN features only (see docstring: avoids the
    # circularity of letting the poisoned samples influence the axes used
    # to detect them).
    clean_center = clean_feats.mean(axis=0, keepdims=True)
    centered_clean = clean_feats - clean_center
    k = max(1, min(pca_dim, centered_clean.shape[0] - 1, centered_clean.shape[1]))
    _, _, vt = np.linalg.svd(centered_clean, full_matrices=False)
    basis = vt[:k].T

    clean_reduced = centered_clean @ basis
    train_reduced = (train_feats - clean_center) @ basis

    scores = np.zeros(len(train_labels))
    loo_scores_by_class = {}
    for c in range(num_classes):
        cmask = clean_labels == c
        if cmask.sum() < 2:
            continue
        cls_feats = clean_reduced[cmask]
        centroid_c = cls_feats.mean(axis=0)
        spread_c = np.linalg.norm(cls_feats - centroid_c, axis=1).mean() + 1e-8

        tmask = train_labels == c
        scores[tmask] = np.linalg.norm(train_reduced[tmask] - centroid_c, axis=1) / spread_c

        if threshold_percentile is not None and cmask.sum() >= 3:
            n_c = cls_feats.shape[0]
            loo = np.zeros(n_c)
            for i in range(n_c):
                others = np.delete(cls_feats, i, axis=0)
                loo_centroid = others.mean(axis=0)
                loo_spread = np.linalg.norm(others - loo_centroid, axis=1).mean() + 1e-8
                loo[i] = np.linalg.norm(cls_feats[i] - loo_centroid) / loo_spread
            loo_scores_by_class[c] = loo

    if threshold_percentile is not None:
        isolated = []
        for c, loo in loo_scores_by_class.items():
            threshold = np.percentile(loo, threshold_percentile)
            tmask = train_labels == c
            flagged = train_idx[tmask][scores[tmask] > threshold]
            isolated.extend(flagged.tolist())
        return isolated

    n_isolate = max(1, int(len(scores) * final_q))
    top = np.argsort(scores)[::-1][:n_isolate]  # largest deviation from own class's clean centroid
    return train_idx[top].tolist()


def _separation_confidence(scores, isolated_idx):
    """How many standard deviations separate the isolated group's mean
    score from the rest of the score distribution -- a ground-truth-free
    proxy for whether an isolation signal found *real* structure. Near
    zero means the "isolated" samples don't actually stand out from
    everything else (the signal is essentially picking arbitrary
    borderline cases out of a smooth, unimodal distribution, which is what
    loss-ranking does against BadNets/SIG); large means it found a
    genuinely separated sub-population. Used by `_isolate_adaptive` to
    decide how much to trust each signal without knowing the attack.
    """
    scores = np.asarray(scores, dtype=np.float64)
    mask = np.zeros(len(scores), dtype=bool)
    mask[isolated_idx] = True
    if mask.sum() == 0 or mask.sum() == len(scores):
        return 0.0
    iso_mean = scores[mask].mean()
    rest_mean = scores[~mask].mean()
    pooled_std = scores.std() + 1e-8
    return float(abs(iso_mean - rest_mean) / pooled_std)


def _calibrated_confidence(scores, isolated_idx, num_null=20, seed=0):
    """Fixes a real calibration bug `_separation_confidence` has on its
    own: its raw value sits on a different natural scale for each signal
    (empirically, loss's raw separation runs roughly 0.1-0.2 even when
    loss-ranking is the *correct* signal for the attack at hand, while
    frequency's and clustering's run roughly 3-5 even when they've found
    nothing attack-specific at all), so comparing the three signals' raw
    `_separation_confidence` values against one shared threshold -- as an
    earlier version of `_isolate_adaptive` did -- systematically favors
    whichever signal's natural scale happens to be larger, independent of
    which one actually found real structure for the attack in front of it.
    Confirmed empirically: with the raw metric, loss's confidence stayed
    near its floor and frequency's/clustering's stayed near their ceiling
    across *every* attack tested, including ones (Blend, Dynamic) where
    loss-ranking alone was the best isolation signal found anywhere in
    this codebase.

    The fix went through two versions, and the first one was itself wrong
    in an instructive way: it compared the true isolated group's separation
    against `num_null` *uniformly random* same-sized subsets of the same
    score array. That null is invalid, because `isolated_idx` is always an
    extreme order statistic (the top- or bottom-k scores), and extreme
    order statistics of *any* distribution -- including pure, structure-free
    noise -- sit far from the bulk mean simply by construction. A uniformly
    random subset averages out to the bulk mean, so it is a systematically
    weaker comparison than the true top/bottom-k selection regardless of
    whether real bimodal structure exists, which produced large "confident"
    z-scores even for a deliberately structure-free synthetic test (a
    calibrated confidence of ~19 for pure noise, versus ~31 for genuine
    bimodal structure -- not remotely a clean separation).

    The corrected null instead asks the right question: if the scores came
    from a single, unimodal population with the *same* mean and standard
    deviation as observed (i.e. no real second sub-population at all), how
    separated would its own top/bottom-k order statistic look, just from
    ordinary sampling variability? `num_null` synthetic score arrays are
    drawn from Normal(mean, std) of the real scores, the *same* selection
    rule (top-k or bottom-k, matching whichever tail the real isolated
    group came from) is applied to each, and the true separation is
    z-scored against the resulting null distribution. This correctly gives
    a near-zero calibrated confidence for structure-free data (verified
    against a synthetic no-structure test case) while still giving a large
    one for genuine bimodal structure, and it remains scale-invariant
    across signals since each is only ever compared to its own null.
    """
    scores = np.asarray(scores, dtype=np.float64)
    n = len(scores)
    k = len(isolated_idx)
    if k == 0 or k >= n:
        return 0.0
    true_conf = _separation_confidence(scores, isolated_idx)

    mask = np.zeros(n, dtype=bool)
    mask[isolated_idx] = True
    take_bottom = scores[mask].mean() <= scores.mean()  # which tail the real selection came from

    mean, std = scores.mean(), scores.std() + 1e-8
    rng = np.random.RandomState(seed)
    null_confs = np.empty(num_null)
    for i in range(num_null):
        null_scores = rng.normal(mean, std, size=n)
        order = np.argsort(null_scores)
        null_idx = order[:k] if take_bottom else order[::-1][:k]
        null_confs[i] = _separation_confidence(null_scores, null_idx)

    null_mean = null_confs.mean()
    null_std = null_confs.std() + 1e-8
    return float((true_conf - null_mean) / null_std)


def _isolate_adaptive(model, poisoned_dataset, device, final_q=0.05, batch_size=256,
                       num_classes=10, pca_dim=10, confidence_scale=2.0):
    """Adaptive isolation: run all three isolation signals in this file
    (loss-ranking, activation clustering, frequency analysis) and weight
    each isolated sample's contribution to the downstream correction by
    how confident its *originating* signal was that it found real
    structure, rather than either committing to one signal (which requires
    knowing the attack in advance) or taking their plain union (which
    README.md's Experimental Findings / the accompanying paper show
    *dilutes* rather than combines their strengths -- e.g. Blend regressed
    from loss-alone's 0.005 ASR to the union's 0.40 -- because a plain
    union gives every flagged sample full weight regardless of whether the
    signal that flagged it actually found anything unusual for that
    specific attack).

    Confidence uses `_calibrated_confidence` (each signal z-scored against
    its own random-subset null baseline), not the raw `_separation_confidence`
    effect size directly. An earlier version of this function compared raw
    values against one shared threshold on the (false) assumption that they
    were already on comparable scales; empirically they were not -- loss's
    raw separation sat near 0.1-0.2 and frequency's/clustering's sat near
    3-5 *regardless of which attack was present*, so the raw-value version
    systematically deferred to whichever signal's natural scale was larger,
    independent of which one had actually found the real trigger. Against
    the real attack checkpoints in this repo, that miscalibration caused
    regressions on exactly the attacks loss-ranking alone had solved best
    (Blend: 0.005 -> 0.90 ASR; Dynamic: 0.06 -> 1.00 ASR), because loss's
    contribution was suppressed near-zero on every attack, correct or not.
    Null-calibrating each signal against its own chance level fixes this by
    construction: a signal is only ever compared to what *it itself* would
    produce on a random, structure-free subset of the same size, so the
    three signals' calibrated confidences are genuinely on the same
    footing regardless of their differing raw units.

    `confidence_scale` sets what counts as a "confident" signal in these
    calibrated (null-relative) units -- default 2.0, i.e. two standard
    deviations above that signal's own chance level. A signal below this is
    downweighted proportionally rather than excluded outright, and if *no*
    signal clears it (as happens for attacks this adaptive selector cannot
    solve at all, e.g. BadNets, whose real signal is decision-geometry-based
    and isn't one of the three isolation methods here -- see Neural
    Cleanse), every sample's contribution stays small rather than one weak
    signal being arbitrarily treated as authoritative.

    Returns (isolated_idx, per_sample_weight, method_confidences):
    `per_sample_weight` maps each isolated index to a weight in [0, 1] to
    be multiplied into hybrid_defense's existing CBD-style
    confidence_by_idx (not used as a replacement for it), and
    `method_confidences` is a {method: calibrated confidence} dict for
    logging.
    """
    # Loss signal.
    losses = _per_sample_losses(model, poisoned_dataset, device, batch_size).numpy()
    n_isolate = max(1, int(len(losses) * final_q))
    loss_idx = losses.argsort()[:n_isolate].tolist()
    loss_conf = _calibrated_confidence(losses, loss_idx)

    # Frequency signal (model-free).
    freq_loader = DataLoader(_IndexedDataset(poisoned_dataset), batch_size=batch_size, shuffle=False)
    freq_scores_list, freq_idx_list = [], []
    for x, _, idx in freq_loader:
        profile = x.mean(dim=(1, 2))
        spectrum = torch.fft.rfft(profile, dim=1).abs()
        spectrum[:, 0] = 0.0
        peak = spectrum.max(dim=1).values
        total = spectrum.sum(dim=1) + 1e-8
        freq_scores_list.append(peak / total)
        freq_idx_list.append(idx)
    freq_scores = torch.cat(freq_scores_list).numpy()
    freq_all_idx = torch.cat(freq_idx_list).numpy()
    n_isolate_f = max(1, int(len(freq_scores) * final_q))
    freq_top = freq_scores.argsort()[::-1][:n_isolate_f]
    freq_idx = freq_all_idx[freq_top].tolist()
    freq_scores_by_idx = np.zeros(len(poisoned_dataset))
    freq_scores_by_idx[freq_all_idx] = freq_scores
    freq_conf = _calibrated_confidence(freq_scores_by_idx, freq_idx)

    # Activation clustering signal (already returns a calibrated z-score).
    cluster_idx, _, cluster_conf = _activation_cluster_core(
        model, poisoned_dataset, device, num_classes, batch_size, pca_dim, separation_threshold=1.5
    )

    confidences = {"loss": loss_conf, "frequency": freq_conf, "cluster": cluster_conf}
    method_weight = {k: min(max(v, 0.0) / confidence_scale, 1.0) for k, v in confidences.items()}

    per_sample_weight = {}
    for idx in loss_idx:
        per_sample_weight[idx] = max(per_sample_weight.get(idx, 0.0), method_weight["loss"])
    for idx in freq_idx:
        per_sample_weight[idx] = max(per_sample_weight.get(idx, 0.0), method_weight["frequency"])
    for idx in cluster_idx:
        per_sample_weight[idx] = max(per_sample_weight.get(idx, 0.0), method_weight["cluster"])

    isolated_idx = list(per_sample_weight.keys())
    if not isolated_idx:
        isolated_idx = [0]
        per_sample_weight = {0: 0.0}

    return isolated_idx, per_sample_weight, confidences


def hybrid_defense(model, poisoned_dataset, clean_loader, device,
                    isolation_method="loss", final_q=0.05, oversample_fraction=0.3,
                    use_neural_cleanse=False, nc_steps=300, nc_lambda=0.01, nc_anomaly_threshold=2.0,
                    cafd_threshold_percentile=None,
                    teacher_epochs=10, repair_epochs=10,
                    beta=500.0, gamma=1.0, delta=1.0, batch_size=128, lr=0.01, num_classes=10):
    """Combines three mechanisms from this file, chosen because the
    6-attack x 6-defense experiment (results/backdoor_experiment_summary.csv)
    showed each contributes something the others don't:

      1. Isolation of suspected-poison samples, via `isolation_method`:
         "loss" (default) is AIBD-style -- rank the poisoned training set
         by per-sample loss under the already-backdoored model and isolate
         the lowest-loss `final_q` fraction. "cluster" is
         _isolate_by_activation_clustering (Chen et al. 2018). "frequency"
         is _isolate_by_frequency (Zeng et al. 2021), the only model-free
         signal here. "strip" is _isolate_by_strip (Gao et al. 2019) -- the
         only *behavioral* signal here (blend-and-measure-entropy under
         input perturbation, rather than a static score on the sample or
         its representation); see its docstring for why it was expected to
         target BadNets/Blend but empirically does not, and instead turned
         out to have this file's best per-sample precision on Trojan
         without beating loss-ranking anywhere. "cafd" is _isolate_by_cafd,
         a signal new to this codebase that anchors to the trusted clean
         holdout directly (per-sample distance from that sample's own
         class's *genuine* clean-data feature centroid) instead of
         calibrating against the training set or a synthetic null the way
         every other signal here does -- the first signal in this file
         with real per-sample precision on BadNets (76% vs. every other
         signal's <11%), and the best on Blend and WaNet too. "ensemble" takes the *union* of loss/cluster/frequency (shown in
         README.md's Experimental Findings to dilute rather than combine
         their strengths -- e.g. Blend regresses from loss-alone's 0.005
         ASR to 0.40 under the union). "adaptive" (_isolate_adaptive) is
         the fix for that: it runs all three and weights each one's
         isolated samples by that signal's own, ground-truth-free
         confidence that it found real structure, so a signal that's wrong
         for the attack at hand contributes little regardless of how many
         samples it flags. Comparing "loss" vs "cluster" vs "frequency"
         across all six attacks found each blind to different things:
         loss-ranking does better on Blend/Dynamic/WaNet; clustering does
         dramatically better on Trojan (0.66 -> 0.01 ASR) because
         TrojanNN's trigger is literally built by activation maximization,
         so it shows up as a sharp feature-space cluster the others never
         notice; frequency analysis is the only one that meaningfully dents
         SIG (0.93 -> 0.07 ASR), since SIG is this repo's only clean-label
         attack and has no model-behavior anomaly for loss or clustering to
         exploit at all. None of the three helps BadNets -- see
         neural_cleanse_defense for the method that does. Unlike
         aibd_defense's own loop, the "loss" path here runs a single pass
         (no progressive rounds or warmup training) because the model is
         already converged -- a converged backdoored model fits triggered
         samples anomalously well right away (Li et
         al. 2021's core observation), so there's nothing a warmup phase
         would add here.
      2. CBD-style reweighting: instead of training a second "confounder"
         model from scratch, reuse the original backdoored model's own
         confidence on each isolated sample's (true) label as the
         down-weighting signal -- at the moment isolation runs, this model
         *is* the confounder CBD would otherwise have to train separately.
      3. NAD-style teacher/student attention distillation on a small
         trusted clean holdout -- the only defense in the grid that
         measurably reduced ASR against BadNets/Blend on its own.
      4. Optional (`use_neural_cleanse=True`): Neural Cleanse (Wang et al.
         2019) trigger reverse-engineering (`_neural_cleanse_detect`),
         added as a *fourth* correction signal rather than a replacement
         for 1-3. This exists because Neural Cleanse alone was the only
         method across the whole investigation that solved BadNets (1.00
         -> 0.03 ASR, standalone) -- it operates on the model's decision
         geometry, not per-sample data, so it isn't blind to BadNets' small
         patch the way loss-ranking/clustering are (see point 1) -- but it
         also has its own blind spots (it assumes one universal trigger per
         class, so it missed Dynamic's per-input trigger entirely, and
         misidentified Trojan's target class). Folding it in as an
         additional loss term lets it contribute on the attacks it's good
         at without replacing the isolation-based signal 1-3 already
         handle reasonably (Dynamic, Trojan) or removing coverage.

    Each repair step draws one batch from the *whole* poisoned training set
    (like aibd_defense itself does) and gives isolated samples in that
    batch an adversarial target (weighted by the CBD-style confidence from
    step 2) while non-isolated samples keep normal CE toward their true
    label. This is added to an NAD-style CE + attention-distillation loss
    computed from a separate clean-holdout batch in the same step, and (if
    `use_neural_cleanse`) a third term: the clean-holdout batch stamped
    with each flagged class's reverse-engineered trigger, trained toward
    its TRUE label (Neural Cleanse's own mitigation mechanism), weighted
    by `delta`.

    Three earlier/alternative versions of this defense are documented here
    because the failure modes map out the actual tradeoff this design is
    balancing: (a) running the isolated correction as a separate pass over
    the *entire* isolated pool once per epoch gave that wrong-label signal
    several times more gradient steps than the small clean holdout's
    corrective signal per epoch, collapsing clean accuracy to 15-60% while
    zeroing ASR; (b) bounding it to one same-sized batch per clean batch
    helped (accuracy rose to 40-75%) but that batch was still *entirely*
    wrong-label supervision, so accuracy stayed far below plain NAD's
    ~90%+; (c) drawing isolated samples at their *natural* rate within
    normal batches (`oversample_fraction` = final_q, i.e. no oversampling)
    restored accuracy fully (~90%+) but diluted the correction so much
    against the 95%+ correctly-labeled majority in the same batch that it
    barely moved ASR back down from baseline. `oversample_fraction`
    resolves this by keeping every batch majority-correct (for (c)'s
    stability) while drawing isolated samples via a WeightedRandomSampler
    at several times their natural rate (for (a)/(b)'s correction
    strength) instead of either their true, tiny rate or 100%. `gamma`
    additionally scales how strong the correction term is once drawn.

    Deliberately excludes MCR (its curve repair showed no effect beyond
    WaNet's already-near-baseline case in the grid) and DBD (its
    self-supervised stage collapsed clean accuracy at this repo's epoch
    budget) -- see the experiment summary for why.
    """
    # 1. Isolate suspected-poison samples from the already-backdoored model.
    n_total = len(poisoned_dataset)
    if isolation_method == "cluster":
        isolated_idx = _isolate_by_activation_clustering(model, poisoned_dataset, device, batch_size=batch_size)
    elif isolation_method == "loss":
        isolated_idx = _isolate_by_loss(model, poisoned_dataset, device, final_q, batch_size)
    elif isolation_method == "frequency":
        # Model-free, unlike every other option here -- see
        # _isolate_by_frequency's docstring for why that matters
        # specifically for SIG (the one clean-label attack in this repo).
        isolated_idx = _isolate_by_frequency(poisoned_dataset, final_q, batch_size)
    elif isolation_method == "strip":
        # Behavioral signal (Gao et al. 2019) -- see _isolate_by_strip's
        # docstring for why that targets BadNets/Blend specifically.
        isolated_idx = _isolate_by_strip(model, poisoned_dataset, clean_loader, device, final_q, batch_size)
    elif isolation_method == "cafd":
        # Clean-Anchored Feature Deviation -- see _isolate_by_cafd's
        # docstring; the first per-sample signal in this codebase with
        # real precision on BadNets. cafd_threshold_percentile, if set,
        # auto-selects how much to isolate per class from the clean
        # holdout's own leave-one-out score distribution instead of a
        # fixed final_q fraction (see the docstring for why that matters).
        isolated_idx = _isolate_by_cafd(model, poisoned_dataset, clean_loader, device, final_q, batch_size,
                                         num_classes=num_classes, threshold_percentile=cafd_threshold_percentile)
    elif isolation_method == "cafd_freq":
        # Union of CAFD and frequency only -- deliberately excludes
        # activation clustering, unlike "ensemble" below. Ground-truth
        # precision/recall checks (see _isolate_by_cafd's docstring for the
        # analogous CAFD-alone numbers) showed clustering isolates ~40-50%
        # of the whole dataset regardless of attack (it flags the smaller
        # k-means split in every class whenever separation clears its
        # threshold, with no final_q-style cap), which drowns CAFD's much
        # higher-precision signal in every case except Trojan. CAFD and
        # frequency are each comparatively small, targeted sets (roughly
        # 5-20% of the data combined across all six attacks tested), so
        # their union stays reasonably tight instead of flooding the
        # isolated pool with noise the way the 3-way "ensemble" does.
        cafd_idx = set(_isolate_by_cafd(model, poisoned_dataset, clean_loader, device, final_q, batch_size,
                                         num_classes=num_classes, threshold_percentile=cafd_threshold_percentile))
        freq_idx = set(_isolate_by_frequency(poisoned_dataset, final_q, batch_size))
        isolated_idx = list(cafd_idx | freq_idx)
    elif isolation_method == "ensemble":
        # Union, not intersection: loss-ranking, activation clustering, and
        # frequency analysis are each blind to different attacks (loss
        # catches Blend/Dynamic/WaNet better, clustering catches Trojan --
        # its trigger is literally built by activation maximization, so it
        # shows up as a sharp feature-space cluster where loss-ranking sees
        # nothing unusual -- and frequency targets SIG's clean-label,
        # periodic-pattern trigger that the other two are blind to). A
        # union costs nothing but extra isolated samples where they
        # disagree; an intersection would only keep samples all of them
        # already agreed on, which doesn't help the cases where exactly
        # one method was catching something the others missed.
        loss_idx = set(_isolate_by_loss(model, poisoned_dataset, device, final_q, batch_size))
        cluster_idx = set(_isolate_by_activation_clustering(model, poisoned_dataset, device, batch_size=batch_size))
        freq_idx = set(_isolate_by_frequency(poisoned_dataset, final_q, batch_size))
        isolated_idx = list(loss_idx | cluster_idx | freq_idx)
    elif isolation_method == "adaptive":
        # See _isolate_adaptive's docstring: unlike "ensemble" (a plain
        # union, shown above to dilute rather than combine strengths),
        # this weights each signal's contribution by its own
        # ground-truth-free confidence that it found real structure.
        isolated_idx, adaptive_weight, adaptive_conf = _isolate_adaptive(
            model, poisoned_dataset, device, final_q=final_q, batch_size=batch_size, num_classes=num_classes
        )
        rounded_conf = {k: round(v, 3) for k, v in adaptive_conf.items()}
        print(f"hybrid_defense: adaptive isolation confidences = {rounded_conf}", flush=True)
    else:
        raise ValueError(
            f"unknown isolation_method '{isolation_method}', expected "
            "'loss', 'cluster', 'frequency', 'strip', 'cafd', 'cafd_freq', 'ensemble', or 'adaptive'"
        )
    if not isolated_idx:  # clustering found no separable sub-cluster in any class
        isolated_idx = [0]
    isolated_mask = torch.zeros(n_total, dtype=torch.bool)
    isolated_mask[isolated_idx] = True

    # 1b. Optional Neural Cleanse detection (see docstring point 4): reverse-
    # engineer a trigger per class and flag anomalously-small ones. This
    # runs independently of the isolation above -- it's an additional
    # signal, not a replacement.
    nc_masks, nc_patterns, nc_flagged = {}, {}, []
    if use_neural_cleanse:
        nc_masks, nc_patterns, nc_flagged, nc_l1_norms = _neural_cleanse_detect(
            model, clean_loader, device, num_classes=num_classes,
            steps=nc_steps, lambda_l1=nc_lambda, anomaly_threshold=nc_anomaly_threshold,
        )
        nc_masks = {c: nc_masks[c].to(device) for c in nc_flagged}
        nc_patterns = {c: nc_patterns[c].to(device) for c in nc_flagged}
        print(f"hybrid_defense: neural_cleanse L1 norms = {np.round(nc_l1_norms, 1).tolist()}", flush=True)
        print(f"hybrid_defense: neural_cleanse flagged class(es) = {nc_flagged}", flush=True)

    # 2. CBD-style reweighting signal for isolated samples, captured from
    # the *original* model before any repair touches its weights: how
    # confidently it predicted each isolated sample's own (recorded)
    # label. High confidence here is exactly what a successfully-implanted
    # trigger produces, so it scales how hard that sample gets corrected in
    # step 3. Non-isolated samples get weight 1.0 (plain CE).
    confidence_by_idx = torch.ones(n_total)
    model.eval()
    with torch.no_grad():
        isolated_subset = Subset(poisoned_dataset, isolated_idx)
        offset = 0
        for x, y in DataLoader(isolated_subset, batch_size=batch_size, shuffle=False):
            x, y = x.to(device), y.to(device)
            probs = F.softmax(model(x), dim=1)
            c = probs.gather(1, y.unsqueeze(1)).squeeze(1).cpu()
            confidence_by_idx[isolated_idx[offset:offset + x.size(0)]] = c
            offset += x.size(0)

    if isolation_method == "adaptive":
        # Multiply in (not replace) each sample's originating-signal
        # confidence on top of the model-confidence weight above -- a
        # sample flagged only by a low-confidence signal gets a small
        # weight regardless of how confidently the model mispredicts it.
        for idx, weight in adaptive_weight.items():
            confidence_by_idx[idx] *= weight

    # Oversample isolated indices to `oversample_fraction` of each batch
    # (in expectation) rather than their natural `final_q` rate, so the
    # correction signal is concentrated enough to matter without making
    # any batch 100% wrong-label (see docstring point (c) vs (a)/(b)).
    n_non_isolate = n_total - len(isolated_idx)
    sample_weights = torch.empty(n_total)
    sample_weights[isolated_mask] = oversample_fraction / max(len(isolated_idx), 1)
    sample_weights[~isolated_mask] = (1.0 - oversample_fraction) / max(n_non_isolate, 1)
    sampler = WeightedRandomSampler(sample_weights, num_samples=n_total, replacement=True)
    poisoned_loader = DataLoader(
        _IndexedDataset(poisoned_dataset), batch_size=batch_size, sampler=sampler, drop_last=True
    )

    # NAD-style teacher, fine-tuned on the small trusted clean holdout.
    teacher = copy.deepcopy(model).to(device)
    opt_t = torch.optim.SGD(teacher.parameters(), lr=lr, momentum=0.9)
    teacher.train()
    for _ in range(teacher_epochs):
        for x, y in clean_loader:
            x, y = x.to(device), y.to(device)
            loss = F.cross_entropy(teacher(x), y)
            opt_t.zero_grad()
            loss.backward()
            opt_t.step()
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    # 3. Repair: one gradient step per clean batch, blending NAD-style
    # distillation with an AIBD-style mixed-batch correction (see docstring).
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    for _ in range(repair_epochs):
        model.train()
        poisoned_iter = iter(poisoned_loader)
        for x, y in clean_loader:
            x, y = x.to(device), y.to(device)
            logits_s, feats_s = model(x, return_features=True)
            with torch.no_grad():
                _, feats_t = teacher(x, return_features=True)
            at_loss = sum(_attention_loss(fs, ft) for fs, ft in zip(feats_s, feats_t))
            loss = F.cross_entropy(logits_s, y) + beta * at_loss

            try:
                xp, yp, idxp = next(poisoned_iter)
            except StopIteration:
                poisoned_iter = iter(poisoned_loader)
                xp, yp, idxp = next(poisoned_iter)
            xp, yp = xp.to(device), yp.to(device)
            is_isolated = isolated_mask[idxp].to(device)
            w = confidence_by_idx[idxp].to(device)

            logits_p = model(xp)
            with torch.no_grad():
                masked = logits_p.clone()
                masked.scatter_(1, yp.unsqueeze(1), float("-inf"))
                adv_targets = masked.argmax(1)  # model's most-likely incorrect class
            targets = torch.where(is_isolated, adv_targets, yp)
            per_sample_ce = F.cross_entropy(logits_p, targets, reduction="none")
            loss = loss + gamma * (w * per_sample_ce).mean()

            if nc_flagged:
                nc_loss = 0.0
                for c in nc_flagged:
                    mask, pattern = nc_masks[c], nc_patterns[c]
                    x_trig = ((1 - mask) * x + mask * pattern).clamp(0, 1)
                    nc_loss = nc_loss + F.cross_entropy(model(x_trig), y)
                loss = loss + delta * nc_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return model


# --------------------------------------------------------------------------
# Neural Cleanse (Wang et al., IEEE S&P 2019)
# --------------------------------------------------------------------------

def _reverse_engineer_trigger(model, clean_loader, device, target_class,
                               steps=300, lr=0.1, lambda_l1=0.01, image_size=32):
    """Optimizes a mask (image_size, image_size) in [0,1] and a pattern
    (3, image_size, image_size) in [0,1], via sigmoid reparameterization, so
    that x' = (1-mask)*x + mask*pattern makes *any* clean image x classify
    as `target_class`, while an L1 penalty on the mask keeps the trigger as
    small/localized as possible -- the smallest perturbation that reliably
    flips predictions to a given class.
    """
    was_training = model.training
    original_requires_grad = [p.requires_grad for p in model.parameters()]
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    mask_logit = torch.zeros(1, image_size, image_size, device=device, requires_grad=True)
    pattern_logit = torch.zeros(3, image_size, image_size, device=device, requires_grad=True)
    optimizer = torch.optim.Adam([mask_logit, pattern_logit], lr=lr)

    clean_iter = iter(clean_loader)
    for _ in range(steps):
        try:
            x, _ = next(clean_iter)
        except StopIteration:
            clean_iter = iter(clean_loader)
            x, _ = next(clean_iter)
        x = x.to(device)

        mask = torch.sigmoid(mask_logit)
        pattern = torch.sigmoid(pattern_logit)
        x_adv = (1 - mask) * x + mask * pattern
        logits = model(x_adv)
        y = torch.full((x.size(0),), target_class, dtype=torch.long, device=device)
        loss = F.cross_entropy(logits, y) + lambda_l1 * mask.abs().sum()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    for p, req in zip(model.parameters(), original_requires_grad):
        p.requires_grad_(req)
    model.train(was_training)

    mask = torch.sigmoid(mask_logit).detach()
    pattern = torch.sigmoid(pattern_logit).detach()
    return mask.cpu(), pattern.cpu(), mask.abs().sum().item()


def _neural_cleanse_detect(model, clean_loader, device, num_classes=10,
                            steps=300, lr_reconstruct=0.1, lambda_l1=0.01, anomaly_threshold=2.0):
    """Shared detection half of Neural Cleanse (Wang et al. 2019): reverse-
    engineer one trigger candidate per class, then MAD-outlier-test their
    L1 norms to flag anomalously-small (likely backdoored) classes. Used by
    both the standalone `neural_cleanse_defense` and `hybrid_defense`'s
    optional `use_neural_cleanse` mode.
    """
    masks, patterns, l1_norms = {}, {}, []
    for c in range(num_classes):
        mask, pattern, l1 = _reverse_engineer_trigger(
            model, clean_loader, device, target_class=c,
            steps=steps, lr=lr_reconstruct, lambda_l1=lambda_l1,
        )
        masks[c], patterns[c] = mask, pattern
        l1_norms.append(l1)

    l1_norms = np.array(l1_norms)
    median = np.median(l1_norms)
    mad = np.median(np.abs(l1_norms - median)) + 1e-8
    anomaly_index = np.abs(l1_norms - median) / (mad * 1.4826)
    flagged = [c for c in range(num_classes) if l1_norms[c] < median and anomaly_index[c] > anomaly_threshold]
    return masks, patterns, flagged, l1_norms


def neural_cleanse_defense(model, clean_loader, device, num_classes=10,
                            reconstruction_steps=300, lr_reconstruct=0.1, lambda_l1=0.01,
                            anomaly_threshold=2.0, unlearn_epochs=10, lr=0.01):
    """Wang et al. (2019). Unlike every other defense in this file, this
    doesn't try to isolate which *training samples* are poisoned at all --
    it reverse-engineers, per class, the smallest (L1-minimal) mask+pattern
    that flips any clean image to that class (`_reverse_engineer_trigger`),
    then flags classes whose reverse-engineered trigger is anomalously
    small relative to the others via a Median Absolute Deviation (MAD)
    outlier test (the paper's core intuition: a real backdoor needs far
    less input modification to force its target class than an arbitrary
    class swap does).

    Operating on the model's decision geometry rather than per-sample data
    sidesteps the bottleneck every isolation-based defense in this file
    hit against BadNets: BadNets' 4x4 corner patch gets diluted by global
    average pooling before loss-ranking (_isolate_by_loss) or activation
    clustering (_isolate_by_activation_clustering) ever see it, but this
    optimization searches input space directly, so a small localized patch
    is exactly what it's built to find, regardless of how features pool it.

    Mitigation follows the paper's primary strategy: for each flagged
    class, stamp its reverse-engineered trigger onto clean images but keep
    their TRUE labels, and fine-tune on a mix of plain and trigger-stamped
    clean images -- directly unlearning "this pattern implies the target
    class" rather than trying to identify and remove poisoned training
    samples, so (unlike ABL/AIBD/CBD/DBD) it needs no access to the
    poisoned training set at all, only the same small clean holdout MCR
    and NAD use.
    """
    masks, patterns, flagged, l1_norms = _neural_cleanse_detect(
        model, clean_loader, device, num_classes=num_classes, steps=reconstruction_steps,
        lr_reconstruct=lr_reconstruct, lambda_l1=lambda_l1, anomaly_threshold=anomaly_threshold,
    )
    print(f"neural_cleanse: L1 norms per class = {np.round(l1_norms, 1).tolist()}", flush=True)
    print(f"neural_cleanse: flagged class(es) = {flagged}", flush=True)

    if not flagged:
        print("neural_cleanse: no class looked anomalous -- returning model unmodified", flush=True)
        return model

    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    model.train()
    for _ in range(unlearn_epochs):
        for x, y in clean_loader:
            x, y = x.to(device), y.to(device)
            loss = F.cross_entropy(model(x), y)

            for c in flagged:
                mask, pattern = masks[c].to(device), patterns[c].to(device)
                x_triggered = ((1 - mask) * x + mask * pattern).clamp(0, 1)
                loss = loss + F.cross_entropy(model(x_triggered), y)  # true label, not c

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return model
