"""Differentiable AC power-balance residual for generated feeder-days (M3).

The physics loss / metric: for a generated state (Pd, Qd, Pg, Qg, Vm, Va) at every
15-min step, the nodal apparent-power mismatch against the feeder's full Ybus

    dP_i = Vm_i * sum_j Vm_j (G_ij cos th_ij + B_ij sin th_ij) * base - (Pg_i - Pd_i)
    dQ_i = Vm_i * sum_j Vm_j (G_ij sin th_ij - B_ij cos th_ij) * base - (Qg_i - Qd_i)
    |dS| = sqrt(dP^2 + dQ^2)                                        [MVA]

— the same quantity the M1 gate checks on real solver states (there: <=8e-8 MVA in
float64; float32 storage floor ~5e-3 MVA; the K=16 PCA representation adds its own,
larger floor, measured in notebook 03). Ybus is rebuilt through the exact M1 path
(pandapower internal Ybus incl. line charging and transformer taps — series-only edge
attributes would be wrong).

All ops are plain torch tensor algebra -> differentiable end to end, so the residual
serves both as the M3 training penalty (through the flow's inverse pass) and as the
evaluation metric.
"""
import numpy as np
import torch

from .dataset import DYN_C, STEPS, node_scales


def feeder_ybus(feeder_key):
    """(G, B, base_mva) dense numpy for one feeder, via the M1 extraction path."""
    import pandapower as pp

    from .simbench_data import extract_features, load_feeder

    net = load_feeder(feeder_key)
    pp.runpp(net, numba=True)
    node_df, _, ybus_df, base_mva = extract_features(net, 0)
    nb = len(node_df)
    pos = {int(b): i for i, b in enumerate(node_df.bus.values)}
    G = np.zeros((nb, nb))
    B = np.zeros((nb, nb))
    for i, j, g, b in zip(ybus_df.i, ybus_df.j, ybus_df.G, ybus_df.B):
        G[pos[int(i)], pos[int(j)]] = g
        B[pos[int(i)], pos[int(j)]] = b
    return G, B, float(base_mva)


class PhysicsHead(torch.nn.Module):
    """Per-feeder container: Ybus + PCA basis + nameplate scales, all as buffers,
    turning flow coefficients into physical days and physical days into residuals."""

    def __init__(self, feeder, pca):
        super().__init__()
        G, B, base = feeder_ybus(feeder["name"])
        self.base_mva = base
        self.register_buffer("G", torch.as_tensor(G, dtype=torch.float32))
        self.register_buffer("Bm", torch.as_tensor(B, dtype=torch.float32))
        self.register_buffer("basis", torch.as_tensor(pca.basis))     # (C, K, 96)
        self.register_buffer("mean", torch.as_tensor(pca.mean))       # (C, 96)
        s = node_scales(feeder["static"]).astype(np.float32)
        self.register_buffer("scales", torch.as_tensor(s))            # (N, C)
        self.k = pca.k

    def decode(self, coeffs):
        """(B, N, C*K) coefficients -> physical days (B, N, C, 96)."""
        b, n, _ = coeffs.shape
        co = coeffs.reshape(b, n, DYN_C, self.k)
        days_pu = torch.einsum("bnck,cks->bncs", co, self.basis) + self.mean
        days = days_pu * self.scales[None, :, :, None]
        days = days.clone()
        days[:, :, 4, :] = days[:, :, 4, :] + 1.0                     # Vm offset
        return days

    def residual_mva(self, days):
        """(B, N, C, 96) physical -> |dS| (B, 96, N) in MVA."""
        pd_, qd = days[:, :, 0, :], days[:, :, 1, :]
        pg, qg = days[:, :, 2, :], days[:, :, 3, :]
        vm, va = days[:, :, 4, :], days[:, :, 5, :]
        vr = (vm * torch.cos(va)).permute(0, 2, 1)                    # (B, 96, N)
        vi = (vm * torch.sin(va)).permute(0, 2, 1)
        ir = vr @ self.G.T - vi @ self.Bm.T
        ii = vr @ self.Bm.T + vi @ self.G.T
        p_calc = (vr * ir + vi * ii) * self.base_mva
        q_calc = (vi * ir - vr * ii) * self.base_mva
        dp = p_calc - (pg - pd_).permute(0, 2, 1)
        dq = q_calc - (qg - qd).permute(0, 2, 1)
        return torch.sqrt(dp ** 2 + dq ** 2 + 1e-12)

    def loss(self, coeffs):
        """Mean squared residual (MVA^2) of decoded coefficients — the M3 penalty."""
        r = self.residual_mva(self.decode(coeffs))
        return (r ** 2).mean()


def residual_stats(days_np, head):
    """Numpy days (D,N,C,96) -> dict of residual statistics in MVA."""
    with torch.no_grad():
        r = head.residual_mva(torch.as_tensor(
            days_np, dtype=torch.float32, device=head.G.device))
    r = r.cpu().numpy()
    return {"resid_mean_mva": float(r.mean()),
            "resid_p95_mva": float(np.quantile(r, 0.95)),
            "resid_max_mva": float(r.max())}
