"""Feeder-day dataset for the generative model (M2).

Turns an M1 feeder-year npz into model-ready samples:

  one sample = one feeder-day = (N nodes, C=6 channels, 96 steps)
  -> per-node NAMEPLATE per-unit normalization (transfer-safe: every scale is
     derivable from the static conditioning features, never from the target
     feeder's own operating data)
  -> shared time-PCA basis per channel (96 -> K coefficients), fitted once on
     pooled node-days of the TRAIN feeders' TRAIN days only
  -> flattened per-node coefficient vector (N, C*K) the flow models jointly.

Normalization (ASSUMPTIONS, logged in notebook 02):
  Pd, Qd   / max(peak_Pd, peak_Qd at that bus, floor 0.05 MW/MVAr)
  Pg, Qg   / max(inst_DER at that bus, floor 0.05)  -- REF bus instead uses the
             feeder's total peak load (grid import is bounded by what the feeder
             can consume; derivable from static)
  Vm       (Vm - 1.0) / 0.05      (five percent band around nominal)
  Va       Va / 0.05 rad          (~2.9 degrees)
A small dequantization noise (sigma=0.01 in normalized units) is added during
training so constant channels (e.g. Pd on load-free buses) don't produce a
singular density. Sampling adds no noise.
"""
import json
from pathlib import Path

import numpy as np

DYN_C = 6                    # Pd Qd Pg Qg Vm Va
STEPS = 96
SCALE_FLOOR = 0.05
VM_SCALE = 0.05
VA_SCALE = 0.05
DEQUANT_SIGMA = 0.01
VAL_FRACTION = 0.15
SEED = 42


def load_year(path):
    z = np.load(path)
    return dict(dyn=z["dyn"], static=z["static"], edge_index=z["edge_index"],
                edge_attr=z["edge_attr"], bus_ids=z["bus_ids"],
                resid=z["resid_mva"], meta=json.loads(str(z["meta"])))


def node_scales(static):
    """(N, C) per-node scale factors, derived from static nameplate ONLY."""
    n = static.shape[0]
    peak_pd, peak_qd, inst_der, _, _, ref = static.T
    total_peak = float(peak_pd.sum())
    s = np.zeros((n, DYN_C))
    s[:, 0] = np.maximum(peak_pd, SCALE_FLOOR)
    s[:, 1] = np.maximum(peak_qd, SCALE_FLOOR)
    gen_scale = np.maximum(inst_der, SCALE_FLOOR)
    gen_scale[ref > 0] = max(total_peak, SCALE_FLOOR)
    s[:, 2] = gen_scale
    s[:, 3] = gen_scale
    s[:, 4] = VM_SCALE
    s[:, 5] = VA_SCALE
    return s


def year_to_days(dyn):
    """(T, N, C) -> (D, N, C, 96) day tensor."""
    T, n, c = dyn.shape
    d = T // STEPS
    return dyn[:d * STEPS].reshape(d, STEPS, n, c).transpose(0, 2, 3, 1)


def normalize_days(days, static):
    """physical (D,N,C,96) -> normalized per-unit float32."""
    s = node_scales(static)[None, :, :, None]
    out = days.copy()
    out[:, :, 4, :] -= 1.0
    return (out / s).astype(np.float32)


def denormalize_days(days_pu, static):
    s = node_scales(static)[None, :, :, None]
    out = days_pu.astype(np.float64) * s
    out[:, :, 4, :] += 1.0
    return out


def split_days(n_days, seed=SEED):
    rng = np.random.RandomState(seed)
    idx = rng.permutation(n_days)
    n_val = int(round(VAL_FRACTION * n_days))
    return np.sort(idx[n_val:]), np.sort(idx[:n_val])     # train, val


class TimePCA:
    """Shared 96->K basis per channel, fitted on pooled node-days (train only)."""

    def __init__(self, k=16):
        self.k = k
        self.mean = np.zeros((DYN_C, STEPS), dtype=np.float32)
        self.basis = np.zeros((DYN_C, k, STEPS), dtype=np.float32)
        self.explained = np.zeros((DYN_C, k), dtype=np.float32)

    def fit(self, days_pu_list):
        for c in range(DYN_C):
            X = np.concatenate([d[:, :, c, :].reshape(-1, STEPS) for d in days_pu_list])
            self.mean[c] = X.mean(axis=0)
            Xc = X - self.mean[c]
            # economy SVD on (samples, 96); components = rows of Vt
            _, sv, vt = np.linalg.svd(Xc, full_matrices=False)
            self.basis[c] = vt[: self.k]
            ev = (sv ** 2) / (sv ** 2).sum()
            self.explained[c] = ev[: self.k]
        return self

    def encode(self, days_pu):
        """(D,N,C,96) -> (D,N,C*K)"""
        d, n = days_pu.shape[:2]
        out = np.empty((d, n, DYN_C, self.k), dtype=np.float32)
        for c in range(DYN_C):
            out[:, :, c, :] = (days_pu[:, :, c, :] - self.mean[c]) @ self.basis[c].T
        return out.reshape(d, n, DYN_C * self.k)

    def decode(self, coeffs):
        """(D,N,C*K) -> (D,N,C,96)"""
        d, n = coeffs.shape[:2]
        co = coeffs.reshape(d, n, DYN_C, self.k)
        out = np.empty((d, n, DYN_C, STEPS), dtype=np.float32)
        for c in range(DYN_C):
            out[:, :, c, :] = co[:, :, c, :] @ self.basis[c] + self.mean[c]
        return out

    def save(self, path):
        np.savez(path, k=self.k, mean=self.mean, basis=self.basis,
                 explained=self.explained)

    @classmethod
    def load(cls, path):
        z = np.load(path)
        p = cls(int(z["k"]))
        p.mean, p.basis, p.explained = z["mean"], z["basis"], z["explained"]
        return p


def depth_parity_mask(edge_index, n, ref_row):
    """BFS depth parity from the REF bus -> the coupling mask (True = even set).
    On a radial (bipartite) feeder this is the natural 2-coloring."""
    from collections import deque
    adj = [[] for _ in range(n)]
    for a, b in edge_index.T:
        adj[int(a)].append(int(b))
    depth = np.full(n, -1)
    depth[ref_row] = 0
    q = deque([ref_row])
    while q:
        u = q.popleft()
        for v in adj[u]:
            if depth[v] < 0:
                depth[v] = depth[u] + 1
                q.append(v)
    depth[depth < 0] = 0
    return (depth % 2 == 0)


def build_feeder(path, pca=None, k=16):
    """Load one M1 npz and return everything the flow needs for that feeder."""
    y = load_year(path)
    days = year_to_days(y["dyn"])                     # (366, N, 6, 96) physical
    days_pu = normalize_days(days, y["static"])
    tr, va = split_days(days.shape[0])
    ref_row = int(np.argmax(y["static"][:, 5]))
    mask = depth_parity_mask(y["edge_index"], y["static"].shape[0], ref_row)
    return dict(
        name=y["meta"]["feeder"], days=days, days_pu=days_pu, train_idx=tr,
        val_idx=va, static=y["static"], edge_index=y["edge_index"],
        edge_attr=y["edge_attr"], mask=mask, ref_row=ref_row, meta=y["meta"], pca=pca,
    )
