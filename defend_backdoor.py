"""Apply a backdoor defense to a checkpoint produced by train_backdoor.py.

ABL and AIBD continue training the checkpointed model on the *same poisoned
training set* it was trained on, isolating/unlearning suspected-poisoned
samples along the way. CBD and DBD are full alternative training
*procedures* (per their papers) rather than post-hoc repairs -- they
retrain from scratch on the reconstructed poisoned set; --checkpoint is
used only to recover the attack configuration (trigger, target label,
poison rate) needed to rebuild that poisoned set identically, not the
checkpoint's weights. MCR and NAD repair the checkpointed model's weights
using only a small clean holdout carved from the test set. `hybrid` uses
both: AIBD-style isolation on the poisoned set plus NAD-style distillation
on the clean holdout -- see src/backdoor_defenses.py::hybrid_defense for
why this combination was chosen (it follows directly from which of the six
individual defenses actually moved ASR in results/backdoor_experiment_summary.csv).
`neural_cleanse` needs only the clean holdout too (like MCR/NAD), but
doesn't isolate training samples at all -- it reverse-engineers a per-class
trigger candidate directly from the model's decision geometry; see
src/backdoor_defenses.py::neural_cleanse_defense for why that sidesteps the
bottleneck every isolation-based defense here hit against BadNets.

Examples:
  python defend_backdoor.py --checkpoint ./checkpoints/resnet18_backdoor_badnets.pt --method abl
  python defend_backdoor.py --checkpoint ./checkpoints/resnet18_backdoor_wanet.pt   --method nad
  python defend_backdoor.py --checkpoint ./checkpoints/resnet18_backdoor_blend.pt   --method hybrid
  python defend_backdoor.py --checkpoint ./checkpoints/resnet18_backdoor_badnets.pt --method neural_cleanse
"""

import argparse
import os

import torch
from torch.optim import SGD
from torch.utils.data import DataLoader, Subset

from src.backdoor_attacks import PoisonedDataset
from src.backdoor_defenses import (
    abl_defense,
    aibd_defense,
    cbd_defense,
    dbd_defense,
    finetune_on_clean,
    hybrid_defense,
    mcr_defense,
    nad_defense,
    neural_cleanse_defense,
)
from src.data import get_cifar10_loaders
from src.models import to_device, unwrap


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True, help="a checkpoint produced by train_backdoor.py")
    p.add_argument("--method",
                    choices=["abl", "aibd", "cbd", "dbd", "mcr", "nad", "hybrid", "neural_cleanse"],
                    required=True)
    p.add_argument("--data-dir", default="./data")
    p.add_argument("--out-dir", default="./checkpoints")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--clean-holdout", type=int, default=500, help="images carved from the test set (MCR/NAD/hybrid)")
    p.add_argument("--finetune-epochs", type=int, default=5, help="MCR endpoint fine-tuning epochs")
    p.add_argument("--final-q", type=float, default=0.05, help="isolation fraction (AIBD/hybrid)")
    p.add_argument("--gamma", type=float, default=1.0, help="isolated-correction loss weight (hybrid)")
    p.add_argument("--oversample-fraction", type=float, default=0.3,
                    help="target fraction of isolated samples per batch (hybrid)")
    p.add_argument("--isolation-method",
                    choices=["loss", "cluster", "frequency", "strip", "cafd", "cafd_freq", "ensemble", "adaptive"],
                    default="loss",
                    help="how hybrid isolates suspected-poison samples: AIBD-style loss "
                         "ranking, Chen et al. activation clustering, Zeng et al. "
                         "frequency-domain analysis (model-free; targets SIG specifically), "
                         "Gao et al. STRIP blend-and-measure-entropy (behavioral; best "
                         "per-sample precision on Trojan, not BadNets/Blend as originally "
                         "expected -- see _isolate_by_strip), Clean-Anchored Feature Deviation "
                         "(cafd; per-sample distance from the clean holdout's own class "
                         "centroid -- the first signal with real precision on BadNets, see "
                         "_isolate_by_cafd), a plain union of loss/cluster/frequency "
                         "(ensemble), or a confidence-weighted combination of loss/cluster/"
                         "frequency (adaptive; see _isolate_adaptive)")
    p.add_argument("--nc-steps", type=int, default=300, help="trigger reconstruction steps per class (neural_cleanse)")
    p.add_argument("--nc-lambda", type=float, default=0.01, help="mask L1 penalty weight (neural_cleanse)")
    p.add_argument("--nc-anomaly-threshold", type=float, default=2.0,
                    help="MAD anomaly-index cutoff for flagging a class (neural_cleanse)")
    p.add_argument("--nc-unlearn-epochs", type=int, default=10,
                    help="fine-tuning epochs on trigger-stamped clean data (neural_cleanse)")
    p.add_argument("--use-neural-cleanse", action="store_true",
                    help="fold Neural Cleanse in as a 4th signal inside hybrid (see hybrid_defense docstring)")
    p.add_argument("--cafd-threshold-percentile", type=float, default=None,
                    help="if set (e.g. 95), cafd isolation auto-selects how much to isolate per class "
                         "from the clean holdout's own leave-one-out score distribution instead of a "
                         "fixed --final-q fraction -- doesn't require knowing the true poison rate "
                         "(see _isolate_by_cafd)")
    p.add_argument("--delta", type=float, default=1.0, help="neural-cleanse correction loss weight (hybrid)")
    p.add_argument("--subset", type=int, default=None,
                    help="cap on the poisoned training set (ABL/AIBD/CBD/DBD), for smoke tests")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)

    # map_location="cpu": the pickled trigger's tensors (mask/pattern) must
    # stay on CPU regardless of `device`, since PoisonedDataset.__getitem__
    # poisons CPU images (dataset items are always CPU before the training
    # loop's own .to(device) call) -- mapping the whole checkpoint to a CUDA
    # device would make the trigger's tensors CUDA-resident and crash with a
    # device mismatch the moment a poisoned sample is fetched.
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model_plain = ckpt["model_object"].to(device)
    # MCR fits its Bezier curve via torch.func.functional_call directly on
    # model parameters, which needs the plain (unwrapped) module -- every
    # other defense here (including neural_cleanse) just calls the model as
    # a normal nn.Module, which is DataParallel-compatible.
    model = model_plain if args.method == "mcr" else to_device(model_plain, device)

    train_loader, test_loader = get_cifar10_loaders(
        args.data_dir, args.batch_size, num_workers=args.num_workers, subset=args.subset
    )
    poisoned_dataset = PoisonedDataset(
        train_loader.dataset, ckpt["trigger"], ckpt["target_label"], ckpt["poison_rate"], seed=ckpt["seed"]
    )
    clean_holdout = Subset(test_loader.dataset, range(min(args.clean_holdout, len(test_loader.dataset))))
    clean_loader = DataLoader(clean_holdout, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    print(f"device={device} method={args.method} checkpoint={args.checkpoint} attack={ckpt['attack']}", flush=True)

    if args.method == "abl":
        optimizer = SGD(model.parameters(), lr=args.lr, momentum=0.9)
        repaired = abl_defense(model, poisoned_dataset, optimizer, device, batch_size=args.batch_size)
    elif args.method == "aibd":
        optimizer = SGD(model.parameters(), lr=args.lr, momentum=0.9)
        repaired = aibd_defense(model, poisoned_dataset, optimizer, device,
                                 final_q=args.final_q, batch_size=args.batch_size)
    elif args.method == "cbd":
        repaired = cbd_defense(model, ckpt["arch"], poisoned_dataset, device, lr=args.lr, batch_size=args.batch_size)
    elif args.method == "dbd":
        repaired = dbd_defense(ckpt["arch"], poisoned_dataset, device, lr=args.lr, batch_size=args.batch_size)
    elif args.method == "mcr":
        model_a = finetune_on_clean(model, clean_loader, device, epochs=args.finetune_epochs, lr=args.lr, seed=1)
        model_b = finetune_on_clean(model, clean_loader, device, epochs=args.finetune_epochs, lr=args.lr, seed=2)
        repaired = mcr_defense(model_a, model_b, clean_loader, device, lr=args.lr)
    elif args.method == "hybrid":
        repaired = hybrid_defense(model, poisoned_dataset, clean_loader, device,
                                   isolation_method=args.isolation_method,
                                   final_q=args.final_q, oversample_fraction=args.oversample_fraction,
                                   use_neural_cleanse=args.use_neural_cleanse,
                                   nc_steps=args.nc_steps, nc_lambda=args.nc_lambda,
                                   nc_anomaly_threshold=args.nc_anomaly_threshold,
                                   cafd_threshold_percentile=args.cafd_threshold_percentile,
                                   gamma=args.gamma, delta=args.delta, lr=args.lr, batch_size=args.batch_size)
    elif args.method == "neural_cleanse":
        repaired = neural_cleanse_defense(model, clean_loader, device,
                                           reconstruction_steps=args.nc_steps, lambda_l1=args.nc_lambda,
                                           anomaly_threshold=args.nc_anomaly_threshold,
                                           unlearn_epochs=args.nc_unlearn_epochs, lr=args.lr)
    else:  # nad
        repaired = nad_defense(model, clean_loader, device, lr=args.lr)

    tag = os.path.splitext(os.path.basename(args.checkpoint))[0]
    out_path = os.path.join(args.out_dir, f"{tag}_defended_{args.method}.pt")
    torch.save(
        {
            "model_object": unwrap(repaired).cpu(),
            "arch": ckpt["arch"],
            "attack": ckpt["attack"],
            "target_label": ckpt["target_label"],
            "poison_rate": ckpt["poison_rate"],
            "seed": ckpt["seed"],
            "trigger": ckpt["trigger"],
            "defense": args.method,
        },
        out_path,
    )
    print(f"saved defended checkpoint to {out_path}")


if __name__ == "__main__":
    main()
