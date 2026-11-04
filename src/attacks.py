"""Adversarial attacks: FGSM, PGD (Linf/L2), and C&W L2.

References:
  - Goodfellow, Shlens & Szegedy, "Explaining and Harnessing Adversarial
    Examples" (2015) -- FGSM.
  - Madry et al., "Towards Deep Learning Models Resistant to Adversarial
    Attacks" (2018) -- PGD.
  - Carlini & Wagner, "Towards Evaluating the Robustness of Neural
    Networks" (2017) -- C&W.

All attacks assume inputs in [0, 1] pixel range (see src/models.py).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def fgsm_attack(model, x, y, eps, loss_fn=None):
    loss_fn = loss_fn or nn.CrossEntropyLoss()
    x = x.clone().detach().requires_grad_(True)
    loss = loss_fn(model(x), y)
    grad = torch.autograd.grad(loss, x)[0]
    x_adv = x.detach() + eps * grad.sign()
    return x_adv.clamp(0, 1).detach()


def _normalize_l2(x, eps=1e-12):
    flat = x.view(x.size(0), -1)
    norm = flat.norm(dim=1).clamp(min=eps)
    return (flat / norm.view(-1, 1)).view_as(x)


def pgd_attack(model, x, y, eps, alpha, steps, norm="linf", random_start=True, loss_fn=None):
    """Madry et al. (2018) projected gradient descent attack."""
    loss_fn = loss_fn or nn.CrossEntropyLoss()
    x_orig = x.clone().detach()

    if random_start:
        if norm == "linf":
            delta = torch.empty_like(x_orig).uniform_(-eps, eps)
        else:
            delta = _normalize_l2(torch.randn_like(x_orig))
            delta = delta * eps * torch.rand(x_orig.size(0), 1, 1, 1, device=x_orig.device)
        x_adv = (x_orig + delta).clamp(0, 1)
    else:
        x_adv = x_orig.clone()

    for _ in range(steps):
        x_adv = x_adv.clone().detach().requires_grad_(True)
        loss = loss_fn(model(x_adv), y)
        grad = torch.autograd.grad(loss, x_adv)[0]

        if norm == "linf":
            step = alpha * grad.sign()
            x_adv = x_adv.detach() + step
            delta = (x_adv - x_orig).clamp(-eps, eps)
        else:
            step = alpha * _normalize_l2(grad)
            x_adv = x_adv.detach() + step
            delta = x_adv - x_orig
            delta_norms = delta.view(x_orig.size(0), -1).norm(dim=1)
            factor = (eps / delta_norms.clamp(min=1e-12)).clamp(max=1.0)
            delta = delta * factor.view(-1, 1, 1, 1)

        x_adv = (x_orig + delta).clamp(0, 1)

    return x_adv.detach()


def cw_l2_attack(model, x, y, num_classes=10, c=1.0, kappa=0.0, steps=100, lr=0.01,
                  c_search_rounds=6, c_search_factor=10.0):
    """Carlini & Wagner (2017) untargeted L2 attack, tanh-space parameterization.

    The original paper picks c via an outer binary search so that each
    perturbation is close to minimal-L2. A single fixed c is unreliable:
    too small and the L2 penalty dominates so the attack never flips the
    label (silently under-reporting attack strength); too large and it
    always flips but with needlessly large perturbations. Instead of a
    full binary search, `c` is escalated geometrically per-sample only for
    samples that haven't yet been misclassified -- cheaper than a binary
    search (no wasted high-c rounds once a sample succeeds) while still
    guaranteeing the attack finds *some* adversarial example whenever one
    is reachable within `steps * c_search_rounds` iterations.
    """
    x = x.clone().detach()
    B = x.size(0)
    y_onehot = F.one_hot(y, num_classes=num_classes).float()

    best_adv = x.clone()
    best_l2 = torch.full((B,), float("inf"), device=x.device)
    found = torch.zeros(B, dtype=torch.bool, device=x.device)
    cur_c = torch.full((B,), c, device=x.device)

    for _ in range(c_search_rounds):
        w = torch.atanh((x * 2 - 1).clamp(-1 + 1e-6, 1 - 1e-6)).detach().requires_grad_(True)
        optimizer = torch.optim.Adam([w], lr=lr)

        for _ in range(steps):
            adv = 0.5 * (torch.tanh(w) + 1)
            l2 = (adv - x).view(B, -1).pow(2).sum(1)
            logits = model(adv)

            real = (logits * y_onehot).sum(1)
            other = (logits - y_onehot * 1e4).max(1)[0]
            f_loss = torch.clamp(real - other, min=-kappa)

            loss = (l2 + cur_c * f_loss).sum()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                pred = logits.argmax(1)
                misclassified = pred != y
                improved = misclassified & (l2 < best_l2)
                best_l2[improved] = l2[improved]
                best_adv[improved] = adv[improved]
                found |= misclassified

        if bool(found.all()):
            break
        cur_c = torch.where(found, cur_c, cur_c * c_search_factor)

    return best_adv.detach()
