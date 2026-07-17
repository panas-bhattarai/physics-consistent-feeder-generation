"""v1 generator: a small conditional RealNVP over whitened day-scores (M4).

The scores are already ~N(0, I) on train data (whitened day-PCA), so the Gaussian
baseline is literally the flow's base distribution — everything the flow learns is
the NON-Gaussian structure: tails (storm days), skew, multimodality (weather
regimes). One flow is shared across feeders; feeder identity enters as a context
vector pooled from the static/RWPE node features (a new feeder with a fitted
DaySpace reuses the same weights — the transfer cost is the basis, not the flow).

Also provides Langevin posterior refinement for conditional generation: start at
the exact linear-Gaussian posterior samples, then refine under
log p_flow(s) - ||A s + b - obs||^2 / (2 sigma^2).
"""
import numpy as np
import torch
import torch.nn as nn

S_MAX = 3.0


class MLPCoupling(nn.Module):
    def __init__(self, dim, d_ctx, hidden=128):
        super().__init__()
        self.register_buffer("mask", torch.zeros(dim))
        self.net = nn.Sequential(
            nn.Linear(dim + d_ctx, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, 2 * dim))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def set_mask(self, m):
        self.mask.copy_(torch.as_tensor(m, dtype=torch.float32))

    def _st(self, x_vis, ctx):
        s, t = self.net(torch.cat([x_vis, ctx], dim=-1)).chunk(2, dim=-1)
        return S_MAX * torch.tanh(s), t

    def forward(self, x, ctx):
        vis = x * self.mask
        s, t = self._st(vis, ctx)
        tm = 1 - self.mask
        y = vis + tm * (x * torch.exp(s) + t)
        return y, (s * tm).sum(dim=-1)

    def inverse(self, y, ctx):
        vis = y * self.mask
        s, t = self._st(vis, ctx)
        tm = 1 - self.mask
        return vis + tm * ((y - t) * torch.exp(-s))


class DayFlow(nn.Module):
    def __init__(self, dim, d_ctx, n_layers=8, hidden=128, seed=0):
        super().__init__()
        self.dim = dim
        self.layers = nn.ModuleList(
            [MLPCoupling(dim, d_ctx, hidden) for _ in range(n_layers)])
        rng = np.random.RandomState(seed)
        for i, l in enumerate(self.layers):
            m = np.zeros(dim)
            m[rng.permutation(dim)[: dim // 2]] = 1     # random fixed half-masks
            l.set_mask(m)

    def log_prob(self, x, ctx):
        total = torch.zeros(x.shape[0], device=x.device)
        z = x
        for l in self.layers:
            z, ld = l(z, ctx)
            total = total + ld
        logp = -0.5 * (z ** 2 + np.log(2 * np.pi)).sum(dim=-1)
        return logp + total

    def inverse_z(self, z, ctx):
        for l in reversed(self.layers):
            z = l.inverse(z, ctx)
        return z

    @torch.no_grad()
    def sample(self, n, ctx1):
        z = torch.randn(n, self.dim, device=ctx1.device)
        return self.inverse_z(z, ctx1.expand(n, -1))


def feeder_context(feeder):
    """Pooled static+RWPE context vector (44,) for one feeder."""
    from .encoding import rwpe

    ei = torch.as_tensor(feeder["edge_index"], dtype=torch.long)
    ea = torch.as_tensor(feeder["edge_attr"], dtype=torch.float64)
    n = feeder["static"].shape[0]
    pe = rwpe(ei, ea, n).float()
    st = torch.as_tensor(feeder["static"], dtype=torch.float32)
    st = torch.cat([torch.log1p(st[:, :3]), st[:, 3:]], dim=1)
    feats = torch.cat([st, pe], dim=1)                   # (N, 22)
    return torch.cat([feats.mean(0), feats.std(0)])      # (44,)


def langevin_refine(flow, ctx, s_init, A, b, obs, sigma, steps=150, eps=2e-4):
    """SGLD refinement of posterior samples. All torch, on flow's device."""
    dev = next(flow.parameters()).device
    s = torch.as_tensor(s_init, dtype=torch.float32, device=dev).clone()
    A_t = torch.as_tensor(A, dtype=torch.float32, device=dev)
    b_t = torch.as_tensor(b, dtype=torch.float32, device=dev)
    o_t = torch.as_tensor(obs, dtype=torch.float32, device=dev)
    c = ctx.to(dev).expand(s.shape[0], -1)
    for _ in range(steps):
        s.requires_grad_(True)
        lp = flow.log_prob(s, c).sum()
        ll = -(((s @ A_t.T + b_t) - o_t) ** 2).sum() / (2 * sigma ** 2)
        g = torch.autograd.grad(lp + ll, s)[0]
        with torch.no_grad():
            s = s + 0.5 * eps * g + np.sqrt(eps) * torch.randn_like(s)
    return s.detach().cpu().numpy()
