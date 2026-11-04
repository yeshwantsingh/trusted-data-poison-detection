# Literature Survey: Adversarial Robustness of Neural Networks

*Compiled: 2026-09-11*

## 1. Scope

This survey covers the core literature on adversarial examples and robustness for
neural networks: how they are generated (attacks), how models are hardened against
them (empirical and certified defenses), how robustness is measured (benchmarks),
and where the field is heading (diffusion-based purification, transformers,
foundation models). §2–§10 cover *evasion* attacks (test-time perturbations against
a fixed, honestly-trained model); §12 covers the related but distinct *backdoor /
data-poisoning* threat model, where the attacker instead controls part of the
training pipeline.

## 2. Foundational Work

- **Szegedy et al., "Intriguing Properties of Neural Networks" (ICLR 2014)** —
  first showed that imperceptible, adversarially-crafted perturbations cause
  consistent misclassification, coining the term *adversarial examples* and
  revealing that this behavior generalizes across models and datasets. ([arXiv](https://arxiv.org/abs/1312.6199))
- **Goodfellow, Shlens & Szegedy, "Explaining and Harnessing Adversarial
  Examples" (ICLR 2015)** — introduced the **Fast Gradient Sign Method (FGSM)**
  and argued that adversarial vulnerability stems from the locally linear
  behavior of high-dimensional models rather than nonlinearity or overfitting.
  ([arXiv:1412.6572](https://arxiv.org/abs/1412.6572))
- **Kurakin, Goodfellow & Bengio, "Adversarial Examples in the Physical
  World" (2016)** — extended FGSM to an iterative attack (BIM/I-FGSM) and showed
  adversarial examples survive camera capture, establishing physical-world
  threats.
- **Madry et al., "Towards Deep Learning Models Resistant to Adversarial
  Attacks" (ICLR 2018)** — framed adversarial training as a min-max
  (saddle-point) optimization problem and introduced **Projected Gradient
  Descent (PGD)** as a strong first-order attack; PGD adversarial training
  remains a standard baseline defense.
- **Carlini & Wagner, "Towards Evaluating the Robustness of Neural
  Networks" (IEEE S&P 2017)** — the **C&W attack**, an optimization-based
  attack that defeated several then state-of-the-art defenses and became a
  standard tool for robustness evaluation.

## 3. Threat Models

- **Perturbation norms**: most work bounds perturbations by an L∞, L2, or L1
  ball of radius ε around the input; L0 (sparse) and semantic/physical
  perturbations are also studied.
- **White-box vs. black-box**: white-box attackers have full access to model
  gradients (FGSM, PGD, C&W); black-box attackers rely on transferability or
  query-based estimation of gradients (e.g., NES, SPSA, boundary attacks).
- **Transferability**: adversarial examples crafted on a surrogate model often
  fool unrelated target models, enabling practical black-box attacks and
  underpinning ensemble/query-free attack methods.

## 4. Attacks

- **Gradient-based, single/multi-step**: FGSM, BIM/I-FGSM, PGD, MI-FGSM
  (momentum), and C&W remain the core white-box toolkit.
- **Ensemble/standardized evaluation — AutoAttack** (Croce & Hein, 2020):
  a parameter-free ensemble of four diverse attacks (two variants of APGD,
  FAB, and Square Attack) built specifically to catch the gradient-masking
  failures that made earlier ad-hoc evaluations overstate robustness; it is
  the de facto evaluation standard today (used by RobustBench, below).
- **Obfuscated Gradients — Athalye, Carlini & Wagner, "Obfuscated Gradients
  Give a False Sense of Security" (ICML 2018, best paper)** — showed that 7 of
  9 ICLR 2018 defenses relied on some form of gradient obfuscation
  (shattered, stochastic, or vanishing/exploding gradients) rather than true
  robustness, and broke 6 of them with adapted attacks (BPDA, EOT). This paper
  fundamentally changed how robustness claims must be evaluated — defenses
  must be tested against *adaptive* attacks, not just off-the-shelf ones.
  ([GitHub](https://github.com/anishathalye/obfuscated-gradients))

## 5. Empirical Defenses

- **Adversarial training (Madry et al., 2018)** — training on PGD-generated
  adversarial examples; still the strongest empirical defense family, but
  computationally expensive (an inner PGD loop per training step) and prone
  to **robust overfitting**, where robust test accuracy degrades after a
  point even as robust train accuracy keeps rising.
- **TRADES (Zhang et al., ICML 2019)** — decomposes robust error into natural
  error plus a boundary term, explicitly trading off clean accuracy for
  robustness via a tunable regularization weight; strong empirical and
  certified-robustness results when combined with smoothing.
  ([GitHub](https://github.com/yaodongyu/TRADES))
- **MART, AWP, and related variants** — refine adversarial training with
  misclassification-aware losses or weight-perturbation regularization to
  improve robust generalization.
- **"Revisiting Adversarial Training at Scale" (CVPR 2024)** — modern work
  scaling adversarial training to large architectures/datasets, reflecting
  continued interest in making adversarial training practical at ImageNet
  scale and beyond.
- **Caution — defenses without adaptive evaluation are unreliable**: per
  Athalye et al. (§4), any new empirical defense should be stress-tested with
  adaptive/obfuscation-aware attacks before its robustness numbers are trusted.

## 6. Certified / Provable Robustness

- **Randomized Smoothing (Cohen, Rosenfeld & Kolter, ICML 2019)** — converts
  any base classifier into a certifiably robust one under L2 perturbations by
  classifying the majority vote over Gaussian-noised copies of the input,
  with a certified radius derived from the Neyman–Pearson lemma; became the
  dominant certification technique due to its model-agnostic, scalable
  nature. ([PMLR](https://proceedings.mlr.press/v97/cohen19c/cohen19c.pdf))
- Follow-on work improves the certified radius/training efficiency: tighter
  certification via regularized training, "backward smoothing" for efficient
  robust training, and feature-level stochastic smoothing.
- **Interval Bound Propagation (IBP) / Lipschitz-bound methods** — alternative
  certification routes that propagate input perturbation bounds through the
  network (IBP) or bound the network's global/local Lipschitz constant,
  trading tightness for computational cost.
- **Formal verification perspective**: Meng et al., "Adversarial Robustness
  of Deep Neural Networks: A Survey from a Formal Verification Perspective"
  (IEEE TDSC, 2024) frames robustness certification as a verification
  problem and surveys SMT/MILP/abstract-interpretation-based verifiers.
  ([arXiv:2206.12227](https://arxiv.org/abs/2206.12227))

## 7. Theory: Why Does the Robustness–Accuracy Tradeoff Exist?

- **Tsipras et al., "Robustness May Be at Odds with Accuracy" (ICLR 2019)** —
  argues, with a theoretical model and empirical support, that robust and
  standard generalization may be fundamentally in tension: robust models tend
  to rely on more human-aligned but less predictive features.
  ([arXiv:1805.12152](https://arxiv.org/abs/1805.12152))
- **Ilyas et al., "Adversarial Examples Are Not Bugs, They Are Features"
  (NeurIPS 2019)** — reframes adversarial vulnerability as a consequence of
  models exploiting highly predictive but human-imperceptible ("non-robust")
  features present in standard datasets, rather than a flaw unique to a given
  architecture or training procedure. ([PDF](https://gradientscience.org/adv.pdf))

## 8. Benchmarks & Standardized Evaluation

- **RobustBench (Croce et al., NeurIPS Datasets & Benchmarks 2021)** — a
  standardized leaderboard (robustbench.github.io) and library evaluating
  models against **AutoAttack** under fixed L∞/L2 threat models on
  CIFAR-10/100 and ImageNet, explicitly created because ad hoc self-reported
  robustness numbers were frequently overestimated (via gradient masking or
  weak evaluation). It also studies robustness's relationship to distribution
  shift, calibration, OOD detection, fairness, and transferability.
  ([arXiv:2010.09670](https://arxiv.org/abs/2010.09670))
- Related benchmarking work (e.g., "A Comprehensive Study on Robustness of
  Image Classification Models," 2023) extends comparisons across architecture
  families and corruption/attack types.

## 9. Emerging Directions (2023–2026)

- **Diffusion-based adversarial purification**: use a diffusion model to
  partially noise-and-denoise an input before classification, removing
  adversarial perturbations while preserving semantic content (Nie et al.,
  "Diffusion Models for Adversarial Purification," ICML 2022, and many 2024–25
  follow-ups: frequency-domain purification, heterogeneous-noise purification,
  purification for vision-language models, and robustness analyses such as
  "DiffHammer" (NeurIPS 2024) probing whether purification robustness holds
  under adaptive EOT-based attacks).
- **Vision Transformers vs. CNN robustness**: active comparison of whether
  ViT-style architectures are inherently more/less robust than convolutional
  networks, and how attention-based inductive biases affect adversarial
  vulnerability.
- **Robustness of LLMs / multimodal foundation models**: the field is
  extending classical adversarial-robustness concepts (perturbation attacks,
  certification, transferability) to text and vision-language models —
  jailbreaks, typographic/diffusion-generated adversarial images against
  VLMs, and surveys such as "Survey of Adversarial Robustness in Multimodal
  Large Language Models" (2025) and "Robustness in Large Language Models: A
  Survey of Mitigation Strategies and Evaluation Metrics" (2025). This is a
  fast-moving, less settled area compared to the classical vision literature.
- **Data-centric robustness**: "It Is All About Data: A Survey on the Effects
  of Data on Adversarial Robustness" (2023) surveys how dataset size, quality,
  and augmentation affect robust generalization, complementing
  architecture/training-centric approaches.

## 10. Open Problems

1. **Scalable certified robustness** — certification methods (randomized
   smoothing, IBP) still lag far behind empirical adversarial training in
   the accuracy/robustness they can guarantee at ImageNet scale.
2. **Robust overfitting** — adversarial training's robust generalization gap
   is not fully understood or solved.
3. **Evaluation trustworthiness** — obfuscated gradients (§4) show that
   robustness claims are easy to overstate; adaptive evaluation (AutoAttack,
   manual adaptive attacks) should be treated as mandatory, not optional.
4. **Beyond Lp balls** — real-world threats (physical perturbations, semantic
   changes, natural distribution shift) are only partially captured by
   Lp-norm-bounded threat models.
5. **Robustness in foundation models** — certification and even reliable
   empirical defenses for LLMs/VLMs are much less mature than for image
   classifiers.

## 11. Key References

| Year | Paper | Contribution |
|---|---|---|
| 2014 | Szegedy et al., *Intriguing Properties of Neural Networks* | Discovery of adversarial examples |
| 2015 | Goodfellow et al., *Explaining and Harnessing Adversarial Examples* | FGSM |
| 2017 | Carlini & Wagner, *Towards Evaluating the Robustness of Neural Networks* | C&W attack |
| 2018 | Madry et al., *Towards Deep Learning Models Resistant to Adversarial Attacks* | PGD, adversarial training as min-max |
| 2018 | Athalye et al., *Obfuscated Gradients Give a False Sense of Security* | Adaptive-attack evaluation standard |
| 2019 | Cohen et al., *Certified Adversarial Robustness via Randomized Smoothing* | Certified L2 robustness |
| 2019 | Zhang et al., *TRADES* | Robustness/accuracy tradeoff loss |
| 2019 | Tsipras et al., *Robustness May Be at Odds with Accuracy* | Theory of the tradeoff |
| 2019 | Ilyas et al., *Adversarial Examples Are Not Bugs, They Are Features* | Non-robust features hypothesis |
| 2020 | Croce & Hein, *AutoAttack* | Standardized strong attack ensemble |
| 2021 | Croce et al., *RobustBench* | Standardized robustness leaderboard |
| 2022 | Nie et al., *Diffusion Models for Adversarial Purification* | Diffusion-based purification |
| 2024 | Meng et al., *Formal Verification Perspective Survey* | Certification via verification |
| 2024 | *Revisiting Adversarial Training at Scale* (CVPR) | Scaling adversarial training |
| 2025 | *Survey of Adversarial Robustness in Multimodal LLMs* | Extending robustness to VLMs |
| 2017 | Gu et al., *BadNets* | First data-poisoning backdoor attack |
| 2017 | Chen et al., *Targeted Backdoor Attacks Using Data Poisoning* | Blended (alpha-blend) trigger |
| 2018 | Liu et al., *Trojaning Attack on Neural Networks* | Activation-maximization trigger optimization |
| 2018 | Chen et al., *Detecting Backdoor Attacks on DNNs by Activation Clustering* | Feature-space isolation via k-means |
| 2019 | Barni, Kallas & Tondi, *A New Backdoor Attack* | Clean-label (SIG) backdoor |
| 2019 | Wang et al., *Neural Cleanse* | Per-class trigger reverse-engineering, model-geometry (not sample) based |
| 2020 | Nguyen & Tran, *Input-Aware Dynamic Backdoor Attack* | Per-input, generator-driven trigger |
| 2020 | Zhao et al., *Bridging Mode Connectivity and Adversarial Robustness* | MCR backdoor repair |
| 2021 | Nguyen & Tran, *WaNet* | Imperceptible warping-based trigger |
| 2021 | Li et al., *Neural Attention Distillation* | NAD backdoor erasure |
| 2021 | Li et al., *Anti-Backdoor Learning* | ABL isolate-then-unlearn defense |
| 2021 | Zeng et al., *Rethinking the Backdoor Attacks' Triggers: A Frequency Perspective* | Model-free, frequency-domain trigger detection |
| 2022 | Huang et al., *Backdoor Defense via Decoupling the Training Process* | DBD self-supervised decoupling |
| 2023 | Zhang et al., *Backdoor Defense via Deconfounded Representation Learning* | CBD causal defense |
| 2025 | *Adversarial-Inspired Backdoor Defense* (AAAI) | AIBD progressive isolation + adversarial-label unlearning |

## 12. Backdoor (Data-Poisoning / Trojan) Attacks and Defenses

Unlike §2–§10's evasion attacks, a backdoor (or "Trojan") attacker controls part of
the *training* pipeline: they poison a fraction of the training set so that any
model trained on it learns a spurious trigger → target-label association it
otherwise would not have. At test time, a normally-behaving model misclassifies
any input containing the trigger as the attacker's chosen target class. This repo's
`src/backdoor_attacks.py` and `src/backdoor_defenses.py` implement reference
versions of six attacks and six defenses from this literature; see `README.md` for
how to run them and exactly which paper-specific simplifications each one makes.

**Attacks** (increasing stealth, decreasing "does the trigger look wrong" risk):

- **BadNets (Gu et al., 2017)** — the original backdoor attack: stamp a small,
  fixed, input-independent patch onto poisoned samples and flip their label. Easy
  to implement, easy to *see* if you inspect the data.
- **Blend (Chen et al., 2017)** — alpha-blend a fixed full-image pattern into the
  input instead of localizing to a patch, trading a visible patch for a faint
  image-wide watermark.
- **SIG (Barni, Kallas & Tondi, 2019)** — a *clean-label* attack: a smooth
  sinusoidal-stripe pattern is added only to samples that already belong to the
  target class, and labels are never falsified, defeating any defense that looks
  for label/content mismatches.
- **Dynamic / Input-Aware (Nguyen & Tran, 2020)** — the trigger *itself* varies
  per input (generated by a small network conditioned on the image), breaking the
  "one static pattern" assumption most detection defenses rely on.
- **WaNet (Nguyen & Tran, 2021)** — a *geometric* trigger: a smooth, imperceptible
  image warp rather than an additive pattern, which survives visual/perceptual
  inspection far better than a patch or blend.
- **Trojan / TrojanNN (Liu et al., 2018)** — the trigger's pixel content (not just
  its presence) is optimized via activation maximization to strongly excite a
  chosen internal neuron of a reference model, rather than using arbitrary noise.

**Defenses** (roughly: retrain-time isolation/unlearning vs. post-hoc repair from a
small clean set):

- **ABL (Li et al., NeurIPS 2021)** — trains with a loss-flooding objective so
  backdoor samples (easy to fit) end up with distinctly lower loss than clean ones,
  isolates the lowest-loss fraction, then unlearns via gradient *ascent* on exactly
  that subset.
- **AIBD (AAAI 2025)** — similar isolate-then-break-correlation structure to ABL,
  but with progressive top-q isolation (recomputed each round, growing over time)
  and adversarial-label descent instead of raw ascent.
- **CBD (Zhang et al., CVPR 2023)** — trains a deliberately shortcut-prone
  "confounder" model, then trains the target model to both down-weight samples the
  confounder is confident about and minimize feature similarity with it.
- **DBD (Huang et al., ICLR 2022)** — decouples training into (1) label-free
  self-supervised backbone pretraining, (2) linear-head-only training on poisoned
  labels, (3) fine-tuning restricted to the lowest-loss ("high-credibility")
  subset — since an unsupervised backbone has no signal the trigger-label
  correlation can exploit.
- **MCR (Zhao et al., ICLR 2020)** — fits a low-loss quadratic Bezier curve in
  weight space between two independently clean-fine-tuned copies of the
  backdoored model; the curve's midpoint tends to fall outside the sharp, isolated
  loss basin the backdoor carved out.
- **NAD (Li et al., ICLR 2021)** — fine-tunes a teacher on a small clean set, then
  distills the teacher's intermediate spatial *attention* maps into the backdoored
  student, erasing the trigger-driven activation pattern.
- **Neural Cleanse (Wang et al., IEEE S&P 2019)** — the odd one out: doesn't try to
  identify poisoned *training samples* at all. Reverse-engineers, per class, the
  smallest mask+pattern that flips any clean image to that class, then flags
  classes whose trigger is an outlier via a Median Absolute Deviation test across
  classes, and unlearns by fine-tuning trigger-stamped clean images toward their
  true label. Operates on the model's decision geometry rather than per-sample
  data or pooled features.
- **Frequency-domain detection (Zeng et al., ICCV 2021, "Rethinking the Backdoor
  Attacks' Triggers: A Frequency Perspective")** — observes that many backdoor
  triggers leave a narrow, unnaturally clean spike in an image's spatial-frequency
  spectrum that natural photographic content doesn't produce. Unlike every defense
  above, it's a pure image-statistics signal — it doesn't touch the model at all.

Running all six attacks against all six defenses (`results/backdoor_experiment_summary.csv`,
detailed in `README.md`) found NAD to be the only defense with a real effect
at this repo's default settings, and only against BadNets/Blend; ABL, AIBD,
and CBD stayed at the undefended baseline ASR, and DBD's clean accuracy
collapsed outright. This repo's `hybrid_defense` (not from any single
paper) combines AIBD's isolation, CBD's confidence reweighting, and NAD's
attention distillation into one defense to test whether the mechanisms
that showed *partial* signal individually could be combined into something
better; its isolation step is itself swappable between AIBD-style loss
ranking, activation clustering (Chen et al., 2018), and the frequency-domain
signal above. Combined with a standalone run of Neural Cleanse, the full
investigation (documented in `README.md`, ten-plus configurations across
every attack) converged on a striking conclusion: **there is no universal
backdoor defense at this problem scale.** Every one of the six attacks was
eventually broken by *some* method, but no single method solved more than
two, and which one worked traces directly back to the trigger's own
mechanism — Neural Cleanse (standalone) solved BadNets because it searches
decision geometry directly rather than pooled features that dilute a small
patch; frequency-domain isolation solved SIG because it's the repo's only
*clean-label* attack, with no "wrong label" or "anomalous prediction"
signature for any model-behavior-based method to exploit; activation
clustering solved Trojan because its trigger is literally built by
activation maximization, producing an unusually sharp feature cluster;
loss-ranking solved Dynamic because per-sample loss anomalies still fire
even though the trigger itself varies per input; and NAD alone sufficed for
WaNet. Attempts to *unify* these signals — an isolation-method ensemble
(union of flagged samples), and folding Neural Cleanse into `hybrid_defense`
as a fourth loss term — both diluted rather than combined their component
strengths (e.g. folding Neural Cleanse in made BadNets *worse*, 0.03 -> 0.58
ASR, because loss-ranking's noise for that attack interferes with Neural
Cleanse's otherwise-clean correction in the same gradient step). A real
defender facing an unknown trigger can't simply pick "the best" method from
this set — they would need signal from several at once and a way to judge
which one's diagnosis to trust, which is itself an unsolved problem this
repo does not attempt.

## Sources Consulted

- [Adversarial Robustness of Deep Neural Networks: A Survey from a Formal Verification Perspective](https://arxiv.org/abs/2206.12227)
- [A Survey of Neural Network Robustness Assessment in Image Recognition](https://arxiv.org/pdf/2404.08285)
- [It Is All About Data: A Survey on the Effects of Data on Adversarial Robustness](https://arxiv.org/pdf/2303.09767)
- [Certified Adversarial Robustness via Randomized Smoothing](https://proceedings.mlr.press/v97/cohen19c/cohen19c.pdf)
- [TRADES GitHub](https://github.com/yaodongyu/TRADES)
- [RobustBench (arXiv:2010.09670)](https://arxiv.org/abs/2010.09670)
- [Diffusion Models for Adversarial Purification](https://arxiv.org/pdf/2205.07460)
- [DiffHammer: Rethinking the Robustness of Diffusion-Based Purification (NeurIPS 2024)](https://proceedings.neurips.cc/paper_files/paper/2024/file/a2fac827e992d55dcfdd4263e98528f4-Paper-Conference.pdf)
- [Explaining and Harnessing Adversarial Examples (arXiv:1412.6572)](https://arxiv.org/abs/1412.6572)
- [Obfuscated Gradients Give a False Sense of Security (GitHub)](https://github.com/anishathalye/obfuscated-gradients)
- [Robustness May Be at Odds with Accuracy (arXiv:1805.12152)](https://arxiv.org/pdf/1805.12152)
- [Adversarial Examples Are Not Bugs, They Are Features](https://gradientscience.org/adv.pdf)
- [Survey of Adversarial Robustness in Multimodal Large Language Models](https://arxiv.org/abs/2503.13962)
- [Robustness in Large Language Models: A Survey of Mitigation Strategies and Evaluation Metrics](https://arxiv.org/pdf/2505.18658)
- [Anti-Backdoor Learning: Training Clean Models on Poisoned Data](https://arxiv.org/abs/2110.11571)
- [Adversarial-Inspired Backdoor Defense via Bridging Backdoor and Adversarial Attacks (AAAI 2025)](https://ojs.aaai.org/index.php/AAAI/article/view/33030)
- [Backdoor Defense via Deconfounded Representation Learning (CBD, CVPR 2023)](https://arxiv.org/abs/2303.06818)
- [Backdoor Defense via Decoupling the Training Process (DBD, ICLR 2022)](https://arxiv.org/pdf/2202.03423)
- [Bridging Mode Connectivity in Loss Landscapes and Adversarial Robustness (MCR, ICLR 2020)](https://arxiv.org/abs/2005.00060)
- [Neural Attention Distillation: Erasing Backdoor Triggers from Deep Neural Networks (NAD, ICLR 2021)](https://arxiv.org/abs/2101.05930)
- [WaNet -- Imperceptible Warping-based Backdoor Attack (ICLR 2021)](https://github.com/VinAIResearch/Warping-based_Backdoor_Attack-release)
- [Input-Aware Dynamic Backdoor Attack (NeurIPS 2020)](https://proceedings.neurips.cc/paper/2020/hash/234e691320c0ad5b45ee3c96d0d7b8f8-Abstract.html)
- [Trojaning Attack on Neural Networks (NDSS 2018)](https://www.ndss-symposium.org/wp-content/uploads/2018/02/ndss2018_03A-5_Liu_paper.pdf)
- [Targeted Backdoor Attacks on Deep Learning Systems Using Data Poisoning (Blend, 2017)](https://arxiv.org/abs/1712.05526)
- [Rethinking the Backdoor Attacks' Triggers: A Frequency Perspective (Zeng et al., ICCV 2021)](https://arxiv.org/abs/2104.03413)
- [Neural Cleanse: Identifying and Mitigating Backdoor Attacks in Neural Networks (Wang et al., IEEE S&P 2019)](https://sites.cs.ucsb.edu/~bolunwang/assets/docs/backdoor-sp19.pdf)
