"""Backdoor (data-poisoning) attacks: BadNets, Blend, SIG, Dynamic, WaNet,
Trojan.

Unlike src/attacks.py (test-time evasion attacks against a fixed model),
these attacks operate at *training* time: a trigger is stamped onto a
fraction of the training set (usually with the label flipped to a chosen
target class), so that any model trained on the poisoned data learns a
spurious trigger -> target-class association it did not have before. At
test time, the same trigger reliably flips the prediction of an otherwise
normally-behaving model.

Each trigger exposes `.poison(x)` for a single [0,1] CHW image tensor and a
`.clean_label` flag (True only for SIG, whose poisoned samples keep their
ground-truth label -- see class docstring).
"""

import math
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset


def _generator(seed):
    return torch.Generator().manual_seed(seed)


class BadNetsTrigger:
    """Gu et al. (2017): a small, fixed, input-independent patch (here a
    random black/white checkerboard) stamped in a corner of the image.
    """

    clean_label = False

    def __init__(self, image_size=32, patch_size=4, seed=0):
        g = _generator(seed)
        self.mask = torch.zeros(1, image_size, image_size)
        self.mask[:, -patch_size:, -patch_size:] = 1.0
        pattern = (torch.rand(3, image_size, image_size, generator=g) > 0.5).float()
        self.pattern = pattern

    def poison(self, x):
        return x * (1 - self.mask) + self.pattern * self.mask


class BlendTrigger:
    """Chen et al. (2017): alpha-blend a fixed, full-image random pattern
    into the input rather than replacing a localized patch, trading
    visibility for coverage of the whole image.
    """

    clean_label = False

    def __init__(self, image_size=32, alpha=0.15, seed=1):
        g = _generator(seed)
        self.alpha = alpha
        self.pattern = torch.rand(3, image_size, image_size, generator=g)

    def poison(self, x):
        return ((1 - self.alpha) * x + self.alpha * self.pattern).clamp(0, 1)


class SIGTrigger:
    """Barni, Kallas & Tondi (2019): a clean-label attack. A smooth
    sinusoidal stripe pattern is added to samples that *already* belong to
    the target class, and their label is left untouched -- the model learns
    "this sinusoidal pattern co-occurs with class c" without any label
    ever being falsified, which makes the attack much harder to catch by
    inspecting label/content mismatches.
    """

    clean_label = True

    def __init__(self, image_size=32, delta=0.15, frequency=6, seed=None):
        # seed is unused (the sinusoidal pattern is fully deterministic) --
        # accepted only so callers can pass `seed=` uniformly across all
        # non-Trojan trigger constructors (see build_trigger).
        cols = torch.arange(image_size).float()
        stripe = delta * torch.sin(2 * math.pi * frequency * cols / image_size)
        self.stripe = stripe.view(1, 1, image_size).expand(3, image_size, image_size)

    def poison(self, x):
        return (x + self.stripe).clamp(0, 1)


class WaNetTrigger:
    """Nguyen & Tran (2021): warps the image with a fixed, smooth random
    displacement field (bicubic-upsampled from a coarse k x k control grid)
    instead of adding a visible pattern -- the trigger is a *geometric*
    transform, which survives JPEG-style perceptual inspection far better
    than an additive patch or blend.
    """

    clean_label = False

    def __init__(self, image_size=32, k=4, warp_strength=0.01, seed=2):
        g = _generator(seed)
        field = torch.rand(1, 2, k, k, generator=g) * 2 - 1
        field = F.interpolate(field, size=(image_size, image_size), mode="bicubic", align_corners=True)
        field = field / field.abs().amax()
        self.flow = field[0].permute(1, 2, 0) * warp_strength  # (H, W, 2), small smooth displacement

        ys, xs = torch.meshgrid(
            torch.linspace(-1, 1, image_size), torch.linspace(-1, 1, image_size), indexing="ij"
        )
        self.base_grid = torch.stack([xs, ys], dim=-1)  # (H, W, 2), grid_sample's (x, y) order

    def poison(self, x):
        grid = (self.base_grid + self.flow).clamp(-1, 1).unsqueeze(0)
        warped = F.grid_sample(
            x.unsqueeze(0), grid, align_corners=True, mode="bilinear", padding_mode="reflection"
        )
        return warped.squeeze(0).clamp(0, 1)


class DynamicTrigger:
    """Simplified analog of Nguyen & Tran's Input-Aware Dynamic Backdoor
    Attack (NeurIPS 2020). The defining property of that attack is that the
    trigger *varies per input* rather than being fixed (as in BadNets), so
    defenses that assume one static pattern fail. Here a small, randomly
    initialized and frozen conv network maps each input to its own trigger
    content, stamped into a fixed corner region.

    The original paper jointly trains the generator adversarially, together
    with a cross-trigger diversity loss, against the classifier being
    attacked -- that bilevel optimization is out of scope here; this keeps
    the input-conditioned trigger *content* without the joint training loop.
    """

    clean_label = False

    def __init__(self, image_size=32, patch_size=8, seed=3):
        torch.manual_seed(seed)
        self.generator = nn.Sequential(
            nn.Conv2d(3, 8, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(8, 3, 3, padding=1), nn.Sigmoid(),
        ).eval()
        for p in self.generator.parameters():
            p.requires_grad_(False)
        self.image_size = image_size
        self.mask = torch.zeros(1, image_size, image_size)
        self.mask[:, -patch_size:, -patch_size:] = 1.0

    def poison(self, x):
        with torch.no_grad():
            pattern = self.generator(x.unsqueeze(0)).squeeze(0)
        return x * (1 - self.mask) + pattern * self.mask


class TrojanTrigger:
    """Liu et al. (2018), TrojanNN: rather than an arbitrary fixed pattern,
    the trigger's *pixel content* is optimized via activation maximization
    to strongly excite an internal neuron/channel of a reference model, then
    stamped like a BadNets patch. The original paper also reverse-engineers
    substitute training data for attackers without training-set access;
    here the attacker is assumed to poison the real training set directly
    (consistent with the other attacks in this module), so only the
    trigger-optimization step is adopted.
    """

    clean_label = False

    def __init__(self, reference_model, image_size=32, patch_size=6, feature_layer=1,
                 steps=200, lr=0.1, seed=4):
        device = next(reference_model.parameters()).device
        g = torch.Generator().manual_seed(seed)

        mask = torch.zeros(1, image_size, image_size)
        mask[:, :patch_size, :patch_size] = 1.0
        self.mask = mask.to(device)

        patch = (torch.rand(3, image_size, image_size, generator=g) * self.mask.cpu()).to(device)
        patch.requires_grad_(True)

        optimizer = torch.optim.Adam([patch], lr=lr)
        background = torch.rand(8, 3, image_size, image_size, device=device)

        # Freeze the reference model and force eval mode (so BatchNorm
        # running stats aren't corrupted by the synthetic `background`
        # batch) only for the duration of trigger optimization; both are
        # restored afterwards since the caller may keep training this same
        # model (e.g. train_backdoor.py's Trojan branch continues fine-
        # tuning the very model used as the activation-maximization target).
        was_training = reference_model.training
        original_requires_grad = [p.requires_grad for p in reference_model.parameters()]
        reference_model.eval()
        for p in reference_model.parameters():
            p.requires_grad_(False)

        for _ in range(steps):
            batch = background * (1 - self.mask) + patch * self.mask
            _, feats = reference_model(batch, return_features=True)
            layer_idx = min(feature_layer, len(feats) - 1)
            activation = feats[layer_idx].mean()
            loss = -activation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                patch.clamp_(0, 1)

        for p, req in zip(reference_model.parameters(), original_requires_grad):
            p.requires_grad_(req)
        reference_model.train(was_training)

        self.pattern = (patch.detach() * self.mask).cpu()
        self.mask = self.mask.cpu()

    def poison(self, x):
        return x * (1 - self.mask) + self.pattern * self.mask


TRIGGER_CLASSES = {
    "badnets": BadNetsTrigger,
    "blend": BlendTrigger,
    "sig": SIGTrigger,
    "wanet": WaNetTrigger,
    "dynamic": DynamicTrigger,
}


def build_trigger(name, image_size=32, reference_model=None, **kwargs):
    if name == "trojan":
        if reference_model is None:
            raise ValueError("trojan trigger requires a reference_model to optimize against")
        return TrojanTrigger(reference_model, image_size=image_size, **kwargs)
    if name not in TRIGGER_CLASSES:
        raise ValueError(f"unknown attack '{name}', expected one of "
                          f"{sorted(TRIGGER_CLASSES) + ['trojan']}")
    return TRIGGER_CLASSES[name](image_size=image_size, **kwargs)


def _targets(dataset):
    if hasattr(dataset, "targets"):
        return list(dataset.targets)
    return [dataset[i][1] for i in range(len(dataset))]


class PoisonedDataset(Dataset):
    """Wraps a clean dataset, poisoning a `poison_rate` fraction of samples:
    for clean-label attacks (SIG), the fraction is drawn only from samples
    already labeled `target_label` and labels are left untouched; otherwise
    it is drawn from the whole dataset and poisoned samples are relabeled
    to `target_label`.
    """

    def __init__(self, base_dataset, trigger, target_label, poison_rate=0.1, seed=0):
        self.base = base_dataset
        self.trigger = trigger
        self.target_label = target_label

        rng = random.Random(seed)
        if getattr(trigger, "clean_label", False):
            candidates = [i for i, y in enumerate(_targets(base_dataset)) if y == target_label]
        else:
            candidates = list(range(len(base_dataset)))
        n_poison = int(len(candidates) * poison_rate)
        self.poison_idx = set(rng.sample(candidates, min(n_poison, len(candidates))))

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        x, y = self.base[i]
        if i in self.poison_idx:
            x = self.trigger.poison(x)
            if not getattr(self.trigger, "clean_label", False):
                y = self.target_label
        return x, y


class TriggeredEvalDataset(Dataset):
    """Stamps `trigger` onto every sample of `dataset` (excluding samples
    already labeled `target_label`, by default) and relabels them to
    `target_label` -- used to measure Attack Success Rate (ASR): the
    fraction of such triggered, originally-non-target images the model
    (mis)classifies as the target class.
    """

    def __init__(self, dataset, trigger, target_label, exclude_target=True):
        self.dataset = dataset
        self.trigger = trigger
        self.target_label = target_label
        targets = _targets(dataset)
        if exclude_target:
            self.indices = [i for i, y in enumerate(targets) if y != target_label]
        else:
            self.indices = list(range(len(dataset)))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, j):
        x, _ = self.dataset[self.indices[j]]
        return self.trigger.poison(x), self.target_label
