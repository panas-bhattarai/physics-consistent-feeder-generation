"""Graph-conditioned normalizing flow over feeder-days (M2, generator v0).

One sample x = (N, D) — N buses, D = C*K time-PCA coefficients per bus. The flow
maps x <-> z (same shape), z ~ N(0, I), through a stack of affine coupling layers
whose node sets alternate along the feeder's BFS depth parity (the natural 2-coloring of a radial network — radial feeders are bipartite):

    y_A = x_A * exp(s) + t,   y_B = x_B
    (s, t) = conditioner( x_B, static, RWPE, global context )

The conditioner is a small message-passing GNN over the feeder graph (series
admittance as edge features) plus a mean-pooled global context vector — local
electrical structure AND feeder-wide drivers (one weather field moves every DER
unit, M1 fig01). Exact log-likelihood by construction (Jacobian = sum of s).

Design notes:
  * s is tanh-clamped (|s| <= S_MAX) — standard RealNVP stabilization.
  * ActNorm (per-dim affine, data-initialized) before the couplings.
  * All feeder-specific information enters through the conditioning features;
    the weights are shared across feeders (the foundation-model property).
"""
import torch
import torch.nn as nn

S_MAX = 2.0


class ActNorm(nn.Module):
    """Per-dimension affine with data-dependent init (Glow)."""

    def __init__(self, dim):
        super().__init__()
        self.log_s = nn.Parameter(torch.zeros(dim))
        self.b = nn.Parameter(torch.zeros(dim))
        self.register_buffer("initialized", torch.tensor(0, dtype=torch.uint8))

    def forward(self, x):                       # x (B, N, D)
        if self.initialized.item() == 0 and self.training:
            with torch.no_grad():
                flat = x.reshape(-1, x.shape[-1])
                self.b.copy_(-flat.mean(0))
                self.log_s.copy_(-torch.log(flat.std(0).clamp(min=1e-6)))
                self.initialized.fill_(1)
        z = (x + self.b) * torch.exp(self.log_s)
        logdet = self.log_s.sum() * x.shape[1]  # per sample: sum over N nodes
        return z, logdet

    def inverse(self, z):
        return z * torch.exp(-self.log_s) - self.b


class GNNConditioner(nn.Module):
    """(masked x, cond) -> per-node (s, t). Two rounds of admittance-weighted
    message passing + a global mean context."""

    def __init__(self, d_x, d_cond, hidden=128, rounds=2):
        super().__init__()
        self.rounds = rounds
        self.embed = nn.Sequential(nn.Linear(d_x + d_cond + 1, hidden), nn.SiLU())
        self.msg = nn.ModuleList([
            nn.Sequential(nn.Linear(2 * hidden + 2, hidden), nn.SiLU())
            for _ in range(rounds)])
        self.upd = nn.ModuleList([
            nn.Sequential(nn.Linear(2 * hidden, hidden), nn.SiLU())
            for _ in range(rounds)])
        self.out = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.SiLU(),
                                 nn.Linear(hidden, 2 * d_x))
        nn.init.zeros_(self.out[-1].weight)
        nn.init.zeros_(self.out[-1].bias)       # start at identity transform

    def forward(self, x_vis, cond, mask_f, edge_index, edge_attr):
        # x_vis (B,N,Dx) already zeroed on the transformed set; mask_f (N,1)
        B, N, _ = x_vis.shape
        m = mask_f.expand(B, N, 1)
        h = self.embed(torch.cat([x_vis, cond.expand(B, -1, -1), m], dim=-1))
        src, dst = edge_index[0], edge_index[1]
        ea = edge_attr.unsqueeze(0).expand(B, -1, -1)
        for r in range(self.rounds):
            msg = self.msg[r](torch.cat([h[:, src], h[:, dst], ea], dim=-1))
            agg = torch.zeros_like(h)
            agg.index_add_(1, dst, msg)
            h = self.upd[r](torch.cat([h, agg], dim=-1))
        g = h.mean(dim=1, keepdim=True).expand(-1, N, -1)   # global context
        s, t = self.out(torch.cat([h, g], dim=-1)).chunk(2, dim=-1)
        return S_MAX * torch.tanh(s), t


class Coupling(nn.Module):
    def __init__(self, d_x, d_cond, hidden=128):
        super().__init__()
        self.net = GNNConditioner(d_x, d_cond, hidden)

    def forward(self, x, cond, tmask, edge_index, edge_attr):
        # tmask (N,1) float: 1 = node is TRANSFORMED this layer
        vis = x * (1 - tmask)                           # conditioner sees the rest
        s, t = self.net(vis, cond, tmask, edge_index, edge_attr)
        y = vis + tmask * (x * torch.exp(s) + t)
        logdet = (s * tmask).sum(dim=(1, 2))
        return y, logdet

    def inverse(self, y, cond, tmask, edge_index, edge_attr):
        vis = y * (1 - tmask)
        s, t = self.net(vis, cond, tmask, edge_index, edge_attr)
        x = vis + tmask * ((y - t) * torch.exp(-s))
        return x


class GraphFlow(nn.Module):
    def __init__(self, d_x, d_cond, n_layers=8, hidden=128):
        super().__init__()
        self.actnorm = ActNorm(d_x)
        self.layers = nn.ModuleList(
            [Coupling(d_x, d_cond, hidden) for _ in range(n_layers)])
        self.d_x = d_x

    def _tmasks(self, mask):
        """Alternate the transformed set: even-depth nodes, then odd-depth."""
        even = mask.float().unsqueeze(-1)
        return [even if i % 2 == 0 else 1 - even for i in range(len(self.layers))]

    def log_prob(self, x, g):
        """x (B,N,D); g = dict(cond, mask, edge_index, edge_attr). Returns (B,)."""
        z, logdet = self.actnorm(x)
        total = logdet if logdet.dim() else logdet.expand(x.shape[0])
        for layer, tm in zip(self.layers, self._tmasks(g["mask"])):
            z, ld = layer(z, g["cond"], tm, g["edge_index"], g["edge_attr"])
            total = total + ld
        logp_z = -0.5 * (z ** 2 + torch.log(torch.tensor(2 * torch.pi))).sum(dim=(1, 2))
        return logp_z + total

    def inverse_z(self, z, g):
        """z -> x through the inverse flow, WITH gradients (used by the M3
        physics penalty, which backpropagates through sampling)."""
        for layer, tm in zip(reversed(self.layers), reversed(self._tmasks(g["mask"]))):
            z = layer.inverse(z, g["cond"], tm, g["edge_index"], g["edge_attr"])
        return self.actnorm.inverse(z)

    @torch.no_grad()
    def sample(self, n_samples, g):
        N = g["mask"].shape[0]
        z = torch.randn(n_samples, N, self.d_x, device=g["cond"].device)
        return self.inverse_z(z, g)


def feeder_graph_tensors(feeder, device="cpu"):
    """Build the per-feeder conditioning dict g from a dataset.build_feeder dict."""
    from .encoding import rwpe

    ei = torch.as_tensor(feeder["edge_index"], dtype=torch.long)
    ea_raw = torch.as_tensor(feeder["edge_attr"], dtype=torch.float32)
    n = feeder["static"].shape[0]
    pe = rwpe(ei, ea_raw.double(), n).float()
    static = torch.as_tensor(feeder["static"], dtype=torch.float32)
    # normalize conditioning features to O(1): log-compress admittance & nameplate
    ea = torch.sign(ea_raw) * torch.log1p(ea_raw.abs()) / 10.0
    st = static.clone()
    st[:, :3] = torch.log1p(st[:, :3])
    cond = torch.cat([st, pe], dim=1).unsqueeze(0)          # (1, N, 6+16)
    mask = torch.as_tensor(feeder["mask"], dtype=torch.bool)
    return dict(cond=cond.to(device), mask=mask.to(device),
                edge_index=ei.to(device), edge_attr=ea.to(device))


D_COND = 22
