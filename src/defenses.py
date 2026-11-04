"""Training-time defenses: standard training, PGD adversarial training
(Madry et al. 2018), TRADES (Zhang et al. 2019), and Gaussian-noise
augmentation for randomized smoothing (Cohen et al. 2019).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.attacks import pgd_attack


def train_epoch_standard(model, loader, optimizer, device):
    model.train()
    total_loss, correct, n = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        n += x.size(0)
    return total_loss / n, correct / n


def train_epoch_pgd_at(model, loader, optimizer, device, eps, alpha, steps):
    """Madry et al. (2018): min-max adversarial training. Each batch is
    replaced by its PGD adversarial counterpart before the weight update.
    """
    model.train()
    total_loss, correct, n = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        model.eval()
        x_adv = pgd_attack(model, x, y, eps=eps, alpha=alpha, steps=steps)
        model.train()

        optimizer.zero_grad()
        logits = model(x_adv)
        loss = F.cross_entropy(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        n += x.size(0)
    return total_loss / n, correct / n


def train_epoch_trades(model, loader, optimizer, device, eps, alpha, steps, beta):
    """Zhang et al. (2019) TRADES: natural CE loss + beta * KL(f(x_adv) || f(x)),
    where x_adv maximizes the KL divergence within the eps-ball (rather than
    the training loss, as in PGD-AT).
    """
    model.train()
    total_loss, correct, n = 0.0, 0, 0
    kl = nn.KLDivLoss(reduction="batchmean")

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        model.eval()
        x_adv = (x.detach() + 0.001 * torch.randn_like(x)).clamp(0, 1)
        for _ in range(steps):
            x_adv.requires_grad_(True)
            with torch.enable_grad():
                loss_kl = kl(F.log_softmax(model(x_adv), dim=1), F.softmax(model(x), dim=1))
            grad = torch.autograd.grad(loss_kl, x_adv)[0]
            x_adv = x_adv.detach() + alpha * grad.sign()
            x_adv = torch.min(torch.max(x_adv, x - eps), x + eps).clamp(0, 1)
        model.train()

        optimizer.zero_grad()
        logits_natural = model(x)
        logits_adv = model(x_adv)
        loss_natural = F.cross_entropy(logits_natural, y)
        loss_robust = kl(F.log_softmax(logits_adv, dim=1), F.softmax(logits_natural, dim=1))
        loss = loss_natural + beta * loss_robust
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)
        correct += (logits_natural.argmax(1) == y).sum().item()
        n += x.size(0)
    return total_loss / n, correct / n


def train_epoch_gaussian_augment(model, loader, optimizer, device, sigma):
    """Cohen et al. (2019): train the base classifier on inputs perturbed by
    N(0, sigma^2 I) so it matches the noise distribution used at
    randomized-smoothing certification time.
    """
    model.train()
    total_loss, correct, n = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        x_noisy = (x + torch.randn_like(x) * sigma).clamp(0, 1)
        optimizer.zero_grad()
        logits = model(x_noisy)
        loss = F.cross_entropy(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        n += x.size(0)
    return total_loss / n, correct / n
