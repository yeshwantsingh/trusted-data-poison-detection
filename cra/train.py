"""Fast GPU-resident trainer: the whole uint8 dataset lives on the GPU,
augmentation (random crop + optional flip) is vectorised, bf16 autocast."""

import math
import time

import torch
import torch.nn.functional as F

from .models import build_model

HP = {
    # arch: (optimizer, lr, wd, batch)
    "resnet18": ("sgd", 0.1, 5e-4, 256),
    "vgg16": ("sgd", 0.05, 5e-4, 256),
    "vit_s": ("adamw", 1e-4, 0.05, 128),
}
EPOCHS = {("resnet18", "cifar10"): 40, ("vgg16", "cifar10"): 40, ("vit_s", "cifar10"): 8,
          ("resnet18", "gtsrb"): 30, ("vgg16", "gtsrb"): 30, ("vit_s", "gtsrb"): 6,
          ("resnet18", "tiny"): 30, ("vgg16", "tiny"): 30, ("vit_s", "tiny"): 6}


def augment(xb, hflip, pad):
    B, C, H, W = xb.shape
    xp = F.pad(xb, (pad, pad, pad, pad), mode="reflect")
    oy = torch.randint(0, 2 * pad + 1, (B,), device=xb.device)
    ox = torch.randint(0, 2 * pad + 1, (B,), device=xb.device)
    ar = torch.arange(H, device=xb.device)
    ir = (oy[:, None] + ar)[:, None, :, None].expand(B, C, H, W + 2 * pad)
    x1 = torch.gather(xp, 2, ir)
    ic = (ox[:, None] + ar)[:, None, None, :].expand(B, C, H, W)
    x2 = torch.gather(x1, 3, ic)
    if hflip:
        f = torch.rand(B, device=xb.device) < 0.5
        x2 = torch.where(f[:, None, None, None], x2.flip(3), x2)
    return x2


@torch.no_grad()
def predict(model, x_u8, device, bs=1024, feats=False):
    model.eval()
    outs = []
    for i in range(0, x_u8.size(0), bs):
        xb = x_u8[i:i + bs].to(device, non_blocking=True).float() / 255
        with torch.autocast("cuda", dtype=torch.bfloat16):
            f = model.features(xb)
            o = f if feats else model.head(f)
        outs.append(o.float())
    return torch.cat(outs)


def accuracy(model, x, y, device):
    if x.size(0) == 0:
        return float("nan")
    return (predict(model, x, device).argmax(1).cpu() == y).float().mean().item()


def train_model(x_u8, y, arch, C, size, hflip, epochs, seed, device, snapshot_epoch=None,
                log=print, eval_fn=None):
    torch.manual_seed(seed)
    model = build_model(arch, C, size).to(device).to(memory_format=torch.channels_last)
    opt_name, lr, wd, bs = HP[arch]
    if opt_name == "sgd":
        opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, nesterov=True, weight_decay=wd)
    else:
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    X = x_u8.to(device)
    Y = y.to(device)
    N = X.size(0)
    steps_per_epoch = N // bs
    total = epochs * steps_per_epoch
    warm = steps_per_epoch
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm))))
    pad = size // 8
    snapshot = None
    g = torch.Generator(device=device).manual_seed(seed)
    for ep in range(1, epochs + 1):
        model.train()
        t0 = time.time()
        perm = torch.randperm(N, device=device, generator=g)
        tot_loss = 0.0
        for s in range(steps_per_epoch):
            idx = perm[s * bs:(s + 1) * bs]
            xb = augment(X[idx].float() / 255, hflip, pad).contiguous(memory_format=torch.channels_last)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = F.cross_entropy(model(xb), Y[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            tot_loss += loss.item() if s % 50 == 0 else 0.0
        msg = f"epoch {ep}/{epochs} loss~{tot_loss / max(1, math.ceil(steps_per_epoch / 50)):.3f} {time.time() - t0:.1f}s"
        if eval_fn is not None and (ep == epochs or ep % 10 == 0):
            msg += " " + eval_fn(model)
        log(msg)
        if snapshot_epoch is not None and ep == snapshot_epoch:
            snapshot = {k: v.detach().clone().cpu() for k, v in model.state_dict().items()}
    del X, Y
    return model, snapshot
