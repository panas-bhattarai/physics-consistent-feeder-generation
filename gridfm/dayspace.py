"""Day-subspace representation (M4, "physics by representation").

M3's finding: at MV scale the AC power-flow manifold is locally near-affine, so the
linear span of solved feeder-days is approximately physics-consistent (the day-PCA
Gaussian baseline sat AT the representation floor). v1 therefore generates inside
that span: a feeder-day (already in time-PCA node coefficients, (N, dx)) is projected
onto a KD-dim day-level basis fitted on the feeder's train days; the generative model
(flow or Gaussian) lives in the whitened score space.

Consequences, all inherited by construction:
  * physics ~ at the representation floor (samples never leave the feasible span)
  * feeder-wide cross-node correlation (the day basis carries it)
  * conditioning on any subset of observed node-profiles is LINEAR in the scores ->
    exact linear-Gaussian posterior (the pseudo-measurement experiment, notebook 04), with
    the flow adding non-Gaussian correction on top.

Cost, stated honestly: the day basis is FEEDER-SPECIFIC (fitted on that feeder's
days). v1 gives up zero-shot topology transfer in exchange for physics; how much
target-feeder data the basis needs is exactly M5's transfer question.
"""
import numpy as np

KD = 64


class DaySpace:
    """Whitened day-PCA for one feeder. Operates on flattened time-PCA coeffs."""

    def __init__(self, kd=KD):
        self.kd = kd

    def fit(self, coeffs):
        """coeffs (D, N, dx) train days."""
        d = coeffs.shape[0]
        self.n, self.dx = coeffs.shape[1], coeffs.shape[2]
        X = coeffs.reshape(d, -1).astype(np.float64)
        self.mean = X.mean(axis=0)
        Xc = X - self.mean
        _, sv, vt = np.linalg.svd(Xc, full_matrices=False)
        self.comps = vt[: self.kd]                       # (KD, M)
        self.scale = sv[: self.kd] / np.sqrt(max(d - 1, 1))
        self.explained = float(((sv[: self.kd] ** 2).sum()) / (sv ** 2).sum())
        return self

    def encode(self, coeffs):
        """(D, N, dx) -> whitened scores (D, KD), unit variance on train data."""
        X = coeffs.reshape(coeffs.shape[0], -1).astype(np.float64)
        return ((X - self.mean) @ self.comps.T / self.scale).astype(np.float32)

    def decode(self, s):
        """(D, KD) -> (D, N, dx)"""
        X = (s.astype(np.float64) * self.scale) @ self.comps + self.mean
        return X.reshape(len(s), self.n, self.dx).astype(np.float32)

    # ------------------------------------------------------------------
    # linear observation model: observing entries `idx` of the flattened
    # day vector x = mean + (s*scale) @ comps  ->  obs = A s + b
    # ------------------------------------------------------------------
    def obs_operator(self, idx):
        A = (self.comps[:, idx] * self.scale[:, None]).T   # (n_obs, KD)
        b = self.mean[idx]
        return A, b

    def conditional_gaussian(self, idx, obs, sigma):
        """Exact posterior of s ~ N(0, I) given obs = A s + b + eps, eps~N(0,sigma^2).

        Returns (mu (KD,), chol (KD, KD)) for sampling mu + chol @ z."""
        A, b = self.obs_operator(idx)
        prec = np.eye(self.kd) + (A.T @ A) / sigma ** 2
        cov = np.linalg.inv(prec)
        mu = cov @ A.T @ (obs - b) / sigma ** 2
        return mu, np.linalg.cholesky(cov)


def node_obs_indices(n_nodes, dx, obs_nodes):
    """Flattened indices of ALL coeff dims of the observed nodes."""
    return np.concatenate([np.arange(n * dx, (n + 1) * dx) for n in sorted(obs_nodes)])
