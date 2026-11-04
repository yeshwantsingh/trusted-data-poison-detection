"""Certified robustness via randomized smoothing (Cohen, Rosenfeld & Kolter,
"Certified Adversarial Robustness via Randomized Smoothing", ICML 2019).

Given a base classifier f, the smoothed classifier g(x) = argmax_c
P(f(x + N(0, sigma^2 I)) = c). `certify` returns a class and an L2 radius
such that g is provably constant (and correct, if the returned class
matches the true label) within that radius, with confidence 1 - alpha.
"""

import numpy as np
import torch
from scipy.stats import beta as beta_dist
from scipy.stats import binomtest, norm


class Smooth:
    ABSTAIN = -1

    def __init__(self, model, num_classes, sigma, device):
        self.model = model
        self.num_classes = num_classes
        self.sigma = sigma
        self.device = device

    def _sample_noise(self, x, num, batch_size):
        counts = np.zeros(self.num_classes, dtype=int)
        remaining = num
        with torch.no_grad():
            while remaining > 0:
                this_batch = min(batch_size, remaining)
                remaining -= this_batch
                batch = x.repeat(this_batch, 1, 1, 1)
                noise = torch.randn_like(batch) * self.sigma
                preds = self.model((batch + noise).clamp(0, 1)).argmax(1)
                counts += np.bincount(preds.cpu().numpy(), minlength=self.num_classes)
        return counts

    def certify(self, x, n0, n, alpha, batch_size):
        """Two-stage certification: use n0 samples to guess the top class,
        n samples to lower-bound its probability, then convert that bound
        to a certified L2 radius via the Gaussian CDF (Cohen et al., Thm 1).
        """
        counts_selection = self._sample_noise(x, n0, batch_size)
        c_hat = int(counts_selection.argmax())
        counts_estimation = self._sample_noise(x, n, batch_size)
        n_a = int(counts_estimation[c_hat])
        p_a_lower = self._lower_confidence_bound(n_a, n, alpha)
        if p_a_lower < 0.5:
            return Smooth.ABSTAIN, 0.0
        radius = self.sigma * norm.ppf(p_a_lower)
        return c_hat, radius

    def predict(self, x, n, alpha, batch_size):
        """Monte Carlo prediction with a hypothesis test to decide whether
        to abstain (used when only a label, not a certified radius, is needed).
        """
        counts = self._sample_noise(x, n, batch_size)
        top2 = counts.argsort()[::-1][:2]
        n_a, n_b = int(counts[top2[0]]), int(counts[top2[1]])
        if binomtest(n_a, n_a + n_b, 0.5).pvalue > alpha:
            return Smooth.ABSTAIN
        return int(top2[0])

    @staticmethod
    def _lower_confidence_bound(n_a, n, alpha):
        """One-sided Clopper-Pearson lower bound at confidence 1 - alpha."""
        if n_a == 0:
            return 0.0
        return float(beta_dist.ppf(alpha, n_a, n - n_a + 1))
