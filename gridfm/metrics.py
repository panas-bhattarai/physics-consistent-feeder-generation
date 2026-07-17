"""Fidelity metrics for synthetic feeder-days (M2).

Follows the validation methodology of Cramer et al., "Validation methods for
energy time series scenarios from deep generative models" (IEEE Access 2022):
compare synthetic vs. real on (i) marginal distributions, (ii) autocorrelation,
(iii) power spectral density, (iv) cross-series correlation — extended here with
the network axis their price/wind signals didn't have (per-bus channels,
feeder-head aggregate, reverse-flow share). Physics metrics arrive in M3; this
module grades *statistics only*.

All comparisons are real VALIDATION days (never trained on) vs. generated days,
in physical units.

Baselines:
  * independent  — each node's day drawn independently from the real train pool
                   (correct marginals by construction, no cross-node correlation)
  * gaussian     — multivariate normal fitted to the flattened train coefficients
                   (correct second moments at best, no tails/multimodality)
"""
import numpy as np

DYN_COLS = ["Pd", "Qd", "Pg", "Qg", "Vm", "Va"]


# ---------------------------------------------------------------------------
# baselines (operate in physical day-tensor space (D, N, C, 96))
# ---------------------------------------------------------------------------
def sample_independent(days_pool, n_samples, rng):
    """Node-independent bootstrap: node i's day is a random real day OF NODE i."""
    d, n = days_pool.shape[:2]
    out = np.empty((n_samples, *days_pool.shape[1:]), dtype=days_pool.dtype)
    for i in range(n):
        out[:, i] = days_pool[rng.randint(0, d, n_samples), i]
    return out


def fit_gaussian_coeffs(coeffs):
    """coeffs (D, N, dx) -> mean + Cholesky of covariance over flattened dims."""
    d = coeffs.shape[0]
    X = coeffs.reshape(d, -1).astype(np.float64)
    mu = X.mean(axis=0)
    Xc = X - mu
    cov = (Xc.T @ Xc) / (d - 1) + 1e-4 * np.eye(X.shape[1])
    return mu, np.linalg.cholesky(cov)


def sample_gaussian_coeffs(mu, chol, n_samples, shape_nd, rng):
    z = rng.standard_normal((n_samples, mu.shape[0]))
    return (mu + z @ chol.T).reshape(n_samples, *shape_nd).astype(np.float32)


# ---------------------------------------------------------------------------
# metric primitives
# ---------------------------------------------------------------------------
def w1(a, b, n_q=200):
    """1-Wasserstein distance via quantile functions."""
    q = np.linspace(0.005, 0.995, n_q)
    return float(np.abs(np.quantile(a, q) - np.quantile(b, q)).mean())


def autocorr(x, max_lag):
    x = (x - x.mean()) / (x.std() + 1e-12)
    n = len(x)
    return np.array([1.0] + [float(np.dot(x[:-l], x[l:]) / (n - l))
                             for l in range(1, max_lag + 1)])


def head_series(days, ref_row):
    """(D,N,C,96) -> concatenated feeder-head P series (D*96,)."""
    return days[:, ref_row, 2, :].reshape(-1)


def psd(x, nseg=96 * 4):
    """Mean periodogram over segments (simple Welch, hann window)."""
    nseg = min(nseg, len(x))
    n_win = len(x) // nseg
    w = np.hanning(nseg)
    segs = x[: n_win * nseg].reshape(n_win, nseg) * w
    p = (np.abs(np.fft.rfft(segs, axis=1)) ** 2).mean(axis=0)
    return p / p.sum()


def cross_node_corr(days, channel=0, keep=None, min_std=1e-3):
    """Correlation matrix of per-node time series (channel).

    `keep` selects the node subset; if None it is derived from this data
    (std > min_std). Pass the REAL data's keep when scoring generated data so
    both matrices cover the same nodes."""
    d, n = days.shape[:2]
    X = days[:, :, channel, :].transpose(1, 0, 2).reshape(n, -1)
    if keep is None:
        keep = X.std(axis=1) > min_std
    Xk = X[keep]
    sd = Xk.std(axis=1, keepdims=True)
    Xk = (Xk - Xk.mean(axis=1, keepdims=True)) / np.maximum(sd, 1e-12)
    return (Xk @ Xk.T) / Xk.shape[1], keep


# ---------------------------------------------------------------------------
# the scorecard
# ---------------------------------------------------------------------------
def scorecard(real_days, gen_days, ref_row, label=""):
    """Both inputs physical (D,N,C,96). Returns dict of scalar metrics
    (lower = better for every entry)."""
    out = {"label": label}

    for c, name in enumerate(DYN_COLS):                 # (i) marginals
        out[f"W1_{name}"] = w1(real_days[:, :, c, :].ravel(),
                               gen_days[:, :, c, :].ravel())

    rh, gh = head_series(real_days, ref_row), head_series(gen_days, ref_row)
    lag = 96
    out["ACF_head_rmse"] = float(np.sqrt(np.mean(                 # (ii)
        (autocorr(rh, lag) - autocorr(gh, lag)) ** 2)))
    pr, pg = psd(rh), psd(gh)
    out["PSD_head_l1"] = float(np.abs(pr - pg).sum())             # (iii)
    out["W1_head_ramps"] = w1(np.diff(rh), np.diff(gh))

    cr, keep = cross_node_corr(real_days)                         # (iv)
    cg, _ = cross_node_corr(gen_days, keep=keep)                  # same node set
    iu = np.triu_indices_from(cr, k=1)
    out["XCorr_Pd_rmse"] = float(np.sqrt(np.mean((cr[iu] - cg[iu]) ** 2)))

    out["revflow_real_pct"] = 100 * float((rh < 0).mean())
    out["revflow_gen_pct"] = 100 * float((gh < 0).mean())
    out["revflow_abs_err_pct"] = abs(out["revflow_real_pct"] - out["revflow_gen_pct"])
    return out
