"""Vectorised backdoor poisoning attacks on float [0,1] image batches.

Every attack is *data-only*: the attacker modifies (and, for dirty-label
attacks, relabels) a subset of the training set and has no control over
training. Poisoning is applied once, before the defender's augmentation.

Parameters follow the original papers / BackdoorBench defaults unless noted:
  badnets   3x3 (32px) / 6x6 (64px) white patch, bottom-right             [Gu+17]
  blend     fixed uniform-noise pattern, alpha=0.2                          [Chen+17]
  sig       sinusoid delta=20/255, f=6, clean-label                         [Barni+19]
  wanet     k=4, s=0.5, noise mode with cross_ratio=2                       [Nguyen&Tran21]
  icp       input-conditioned patch: frozen random conv generator writes an
            input-dependent pattern into an 8x8 corner region. This is a
            *simplified* stand-in for Input-Aware Dynamic [Nguyen&Tran20]
            (no jointly-trained generator) and is named accordingly.
  trojan    TrojanNN-style patch optimised by activation maximisation of a
            penultimate unit of a surrogate model trained on clean data     [Liu+18]
Adaptive attacks designed to suppress latent separability:
  adap_blend  blend pattern split in 4x4 pieces, each payload/cover sample
              gets a random 50% of pieces at alpha=0.15; test uses the full
              pattern at alpha=0.2; cover samples keep their label          [Qi+23]
  adap_patch  4 distinct patch triggers at fixed locations, payload/cover
              samples get one (or two) at opacity 0.5; test stamps all four
              at full opacity; cover samples keep their label               [Qi+23]
  tact        source-specific: only class `src` is relabelled to target;
              the same trigger on `cover` classes keeps the true label     [Tang+21]
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _g(seed):
    return torch.Generator().manual_seed(int(seed))


class Attack:
    clean_label = False
    cover_ratio = 0.0          # cover samples per payload sample
    name = "base"

    def __init__(self, size, num_classes, target, seed):
        self.size, self.C, self.target, self.seed = size, num_classes, target, seed

    # train-time trigger for payload samples
    def apply_train(self, x, gen):
        return self.apply_test(x)

    # trigger used on cover samples (label kept)
    def apply_cover(self, x, gen):
        return self.apply_train(x, gen)

    def apply_test(self, x):
        raise NotImplementedError

    def payload_candidates(self, y):
        if self.clean_label:
            return torch.where(y == self.target)[0]
        return torch.where(y != self.target)[0]

    def cover_candidates(self, y):
        return torch.where(y != self.target)[0]

    def asr_mask(self, y):
        """Which test images count towards attack success rate."""
        return y != self.target

    def to(self, device):
        for k, v in list(vars(self).items()):
            if torch.is_tensor(v):
                setattr(self, k, v.to(device))
            elif isinstance(v, nn.Module):
                v.to(device)
        return self


class BadNets(Attack):
    name = "badnets"

    def __init__(self, size, C, target, seed):
        super().__init__(size, C, target, seed)
        p = max(3, size // 10)
        self.mask = torch.zeros(1, 1, size, size)
        self.mask[..., -p - 1:-1, -p - 1:-1] = 1.0

    def apply_test(self, x):
        return x * (1 - self.mask) + self.mask


class Blend(Attack):
    name = "blend"
    alpha = 0.2

    def __init__(self, size, C, target, seed):
        super().__init__(size, C, target, seed)
        self.pattern = torch.rand(1, 3, size, size, generator=_g(seed + 11))

    def apply_test(self, x):
        return (1 - self.alpha) * x + self.alpha * self.pattern


class SIG(Attack):
    name = "sig"
    clean_label = True

    def __init__(self, size, C, target, seed, delta=20 / 255, freq=6):
        super().__init__(size, C, target, seed)
        cols = torch.arange(size).float()
        s = delta * torch.sin(2 * math.pi * freq * cols / size)
        self.stripe = s.view(1, 1, 1, size).expand(1, 3, size, size).contiguous()

    def apply_test(self, x):
        return (x + self.stripe).clamp(0, 1)


class WaNet(Attack):
    name = "wanet"
    cover_ratio = 2.0

    def __init__(self, size, C, target, seed, k=4, s=0.5):
        super().__init__(size, C, target, seed)
        ins = torch.rand(1, 2, k, k, generator=_g(seed + 22)) * 2 - 1
        ins = ins / ins.abs().mean()
        noise = F.interpolate(ins, size=size, mode="bicubic", align_corners=True).permute(0, 2, 3, 1)
        a = torch.linspace(-1, 1, size)
        xx, yy = torch.meshgrid(a, a, indexing="ij")
        ident = torch.stack((yy, xx), 2)[None]
        self.grid = (ident + s * noise / size).clamp(-1, 1)

    def _warp(self, x, grid):
        return F.grid_sample(x, grid.expand(x.size(0), -1, -1, -1), align_corners=True).clamp(0, 1)

    def apply_test(self, x):
        return self._warp(x, self.grid)

    def apply_cover(self, x, gen):
        n = x.size(0)
        r = (torch.rand(n, self.size, self.size, 2, generator=gen) * 2 - 1).to(x.device)
        grid = (self.grid + r / self.size).clamp(-1, 1)
        return F.grid_sample(x, grid, align_corners=True).clamp(0, 1)


class ICP(Attack):
    """Input-conditioned patch (simplified Input-Aware Dynamic)."""
    name = "icp"

    def __init__(self, size, C, target, seed):
        super().__init__(size, C, target, seed)
        torch.manual_seed(seed + 33)
        self.gen = nn.Sequential(nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(),
                                 nn.Conv2d(16, 3, 3, padding=1), nn.Sigmoid()).eval()
        for p in self.gen.parameters():
            p.requires_grad_(False)
        p = size // 4
        self.mask = torch.zeros(1, 1, size, size)
        self.mask[..., -p:, -p:] = 1.0

    @torch.no_grad()
    def apply_test(self, x):
        out = []
        for i in range(0, x.size(0), 1024):
            xb = x[i:i + 1024]
            pat = (self.gen(xb) > 0.5).float()   # binarised, strongly input-dependent
            out.append(xb * (1 - self.mask) + pat * self.mask)
        return torch.cat(out)


class Trojan(Attack):
    name = "trojan"

    def __init__(self, size, C, target, seed):
        super().__init__(size, C, target, seed)
        p = size // 5
        self.mask = torch.zeros(1, 1, size, size)
        self.mask[..., :p, :p] = 1.0
        self.pattern = torch.rand(1, 3, size, size, generator=_g(seed + 44))

    def fit(self, surrogate, images, steps=300, lr=0.1):
        """Activation-maximise the most target-relevant penultimate unit of a
        clean surrogate (Liu et al. select the unit most connected to the
        trigger layer; we pick the unit with largest weight to the target)."""
        dev = images.device
        surrogate.eval()
        unit = int(surrogate.head_weight()[self.target].argmax())
        pat = self.pattern.to(dev).clone().requires_grad_(True)
        m = self.mask.to(dev)
        opt = torch.optim.Adam([pat], lr=lr)
        for _ in range(steps):
            idx = torch.randint(0, images.size(0), (128,), device=dev)
            xb = images[idx] * (1 - m) + pat.clamp(0, 1) * m
            f = surrogate.features(xb)
            loss = -f[:, unit].mean()
            opt.zero_grad(); loss.backward(); opt.step()
        self.pattern = pat.detach().clamp(0, 1).cpu()
        self.unit = unit
        return self

    def apply_test(self, x):
        return x * (1 - self.mask) + self.pattern * self.mask


class AdapBlend(Attack):
    name = "adap_blend"
    cover_ratio = 1.0

    def __init__(self, size, C, target, seed, pieces=4, train_alpha=0.15, test_alpha=0.2, rate=0.5):
        super().__init__(size, C, target, seed)
        self.pattern = torch.rand(1, 3, size, size, generator=_g(seed + 55))
        self.pieces, self.ta, self.va, self.rate = pieces, train_alpha, test_alpha, rate

    def _piece_mask(self, n, gen, device):
        k = self.pieces
        keep = (torch.rand(n, 1, k, k, generator=gen) < self.rate).float().to(device)
        return F.interpolate(keep, size=self.size, mode="nearest")

    def apply_train(self, x, gen):
        m = self._piece_mask(x.size(0), gen, x.device) * self.ta
        return (1 - m) * x + m * self.pattern

    def apply_test(self, x):
        return (1 - self.va) * x + self.va * self.pattern


class AdapPatch(Attack):
    name = "adap_patch"
    cover_ratio = 1.0

    def __init__(self, size, C, target, seed, n_patches=4, train_opacity=0.5):
        super().__init__(size, C, target, seed)
        g = _g(seed + 66)
        p = max(4, size // 8)
        self.masks = torch.zeros(n_patches, 1, size, size)
        corners = [(1, 1), (1, size - p - 1), (size - p - 1, 1), (size - p - 1, size - p - 1)]
        for i, (r, c) in enumerate(corners[:n_patches]):
            self.masks[i, :, r:r + p, c:c + p] = 1.0
        self.patterns = (torch.rand(n_patches, 3, size, size, generator=g) > 0.5).float()
        self.op = train_opacity

    def apply_train(self, x, gen):
        n, k = x.size(0), self.masks.size(0)
        out = x.clone()
        for _ in range(2):  # up to two random patches per sample
            which = torch.randint(0, k, (n,), generator=gen).to(x.device)
            use = (torch.rand(n, generator=gen) < 0.75).float().to(x.device).view(n, 1, 1, 1)
            m = self.masks[which] * self.op * use
            out = (1 - m) * out + m * self.patterns[which]
        return out

    def apply_test(self, x):
        out = x
        for i in range(self.masks.size(0)):
            m = self.masks[i:i + 1]
            out = (1 - m) * out + m * self.patterns[i:i + 1]
        return out


class TaCT(Attack):
    name = "tact"
    cover_ratio = 1.0

    def __init__(self, size, C, target, seed):
        super().__init__(size, C, target, seed)
        self.src = (target + 1) % C
        self.cover = [(target + 2) % C, (target + 3) % C]
        p = max(4, size // 8)
        self.mask = torch.zeros(1, 1, size, size)
        self.mask[..., -p - 1:-1, -p - 1:-1] = 1.0
        self.pattern = (torch.rand(1, 3, size, size, generator=_g(seed + 77)) > 0.5).float()

    def apply_test(self, x):
        return x * (1 - self.mask) + self.pattern * self.mask

    def payload_candidates(self, y):
        return torch.where(y == self.src)[0]

    def cover_candidates(self, y):
        return torch.where((y == self.cover[0]) | (y == self.cover[1]))[0]

    def asr_mask(self, y):
        return y == self.src


ATTACKS = {c.name: c for c in [BadNets, Blend, SIG, WaNet, ICP, Trojan, AdapBlend, AdapPatch, TaCT]}
STANDARD = ["badnets", "blend", "sig", "wanet", "icp", "trojan"]
ADAPTIVE = ["adap_blend", "adap_patch", "tact"]


def build_attack(name, size, C, target, seed):
    return ATTACKS[name](size, C, target, seed)


def _apply_u8(fn, x_u8, device, bs=2048):
    out = []
    for i in range(0, x_u8.size(0), bs):
        xb = x_u8[i:i + bs].to(device).float() / 255
        out.append((fn(xb).clamp(0, 1) * 255).round().byte().cpu())
    return torch.cat(out) if out else x_u8.clone()


def poison_training_set(attack, x, y, rate, seed, device):
    """Returns (x_poisoned uint8, y_poisoned, payload_idx, cover_idx).

    `rate` is the payload fraction of the whole training set. Clean-label
    payload is capped at 50% of the target class; TaCT payload is capped at
    50% of the source class."""
    gen = torch.Generator().manual_seed(int(seed) + 999)
    N = y.numel()
    n = int(round(rate * N))
    cand = attack.payload_candidates(y)
    if attack.clean_label or attack.name == "tact":
        n = min(n, cand.numel() // 2)
    n = max(n, 1)
    payload = cand[torch.randperm(cand.numel(), generator=gen)[:n]]
    cover = torch.empty(0, dtype=torch.long)
    if attack.cover_ratio > 0:
        cc = attack.cover_candidates(y)
        cc = cc[~torch.isin(cc, payload)]
        nc = min(int(round(attack.cover_ratio * n)), cc.numel())
        cover = cc[torch.randperm(cc.numel(), generator=gen)[:nc]]
    xp, yp = x.clone(), y.clone()
    xp[payload] = _apply_u8(lambda z: attack.apply_train(z, gen), x[payload], device)
    if not attack.clean_label:
        yp[payload] = attack.target
    if cover.numel():
        xp[cover] = _apply_u8(lambda z: attack.apply_cover(z, gen), x[cover], device)
    return xp, yp, payload.sort().values, cover.sort().values


def triggered_test_set(attack, x, y, device):
    m = attack.asr_mask(y)
    xs = _apply_u8(attack.apply_test, x[m], device)
    return xs, torch.full((xs.size(0),), attack.target, dtype=torch.long)
