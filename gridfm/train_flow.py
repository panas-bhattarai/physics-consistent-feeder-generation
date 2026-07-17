"""Training driver for the graph-conditioned flow (M2/M3).

Usage:
  python -m gridfm.train_flow --tag v0 --epochs 600
  (M3 adds --physics-weight; M2 trains pure maximum likelihood.)

Trains on the TRAIN days of the train feeders (rural0, semiurb0), early-stops on
validation NLL, writes checkpoints/{tag}_best.pt + results/m2/history_{tag}.csv.
Deterministic per seed.
"""
import argparse
import csv
import time
from pathlib import Path

import numpy as np
import torch

from .dataset import TimePCA, build_feeder
from .flow import D_COND, GraphFlow, feeder_graph_tensors

ROOT = Path(__file__).resolve().parents[1]
TRAIN_FEEDERS = ["rural0", "semiurb0"]


def load_all(k, device):
    feeders = {}
    for name in TRAIN_FEEDERS:
        f = build_feeder(ROOT / "data" / "m1" / f"{name}_year.npz")
        f["g"] = feeder_graph_tensors(f, device)
        feeders[name] = f
    pca = TimePCA(k).fit([feeders[n]["days_pu"][feeders[n]["train_idx"]]
                          for n in TRAIN_FEEDERS])
    for f in feeders.values():
        co = pca.encode(f["days_pu"])
        f["coeff"] = torch.as_tensor(co, dtype=torch.float32)
    return feeders, pca


def run(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    feeders, pca = load_all(args.k, device)
    d_x = 6 * args.k
    heads = {}
    if args.physics_weight > 0:
        from .physics import PhysicsHead
        heads = {n: PhysicsHead(feeders[n], pca).to(device) for n in TRAIN_FEEDERS}
    model = GraphFlow(d_x, D_COND, n_layers=args.layers, hidden=args.hidden).to(device)
    n_par = sum(p.numel() for p in model.parameters())
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    ckdir = ROOT / "checkpoints"
    ckdir.mkdir(exist_ok=True)
    resdir = ROOT / "results" / args.hist
    resdir.mkdir(parents=True, exist_ok=True)
    pca.save(ckdir / f"timepca_k{args.k}.npz")

    dim_count = sum(feeders[n]["coeff"].shape[1] * d_x for n in TRAIN_FEEDERS) / len(TRAIN_FEEDERS)
    print(f"device={device} params={n_par} d_x={d_x} mean_dims/sample={dim_count:.0f}",
          flush=True)

    best = float("inf")
    hist = []
    t0 = time.time()
    for ep in range(args.epochs):
        model.train()
        tr_nll = 0.0
        n_b = 0
        for name in TRAIN_FEEDERS:
            f = feeders[name]
            idx = f["train_idx"][np.random.permutation(len(f["train_idx"]))]
            for i in range(0, len(idx), args.batch):
                x = f["coeff"][idx[i:i + args.batch]].to(device)
                x = x + args.noise * torch.randn_like(x)      # dequantization
                lp = model.log_prob(x, f["g"])
                loss = -lp.mean() / x.shape[1] / d_x          # NLL per dim
                if args.physics_weight > 0:
                    z = torch.randn(args.phys_batch, x.shape[1], d_x, device=device)
                    s = model.inverse_z(z, f["g"])
                    loss = loss + args.physics_weight * heads[name].loss(s)
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                tr_nll += float(loss)
                n_b += 1
        sched.step()

        model.eval()
        va_nll = 0.0
        n_v = 0
        with torch.no_grad():
            for name in TRAIN_FEEDERS:
                f = feeders[name]
                x = f["coeff"][f["val_idx"]].to(device)
                lp = model.log_prob(x, f["g"])
                va_nll += float(-lp.mean() / x.shape[1] / d_x)
                n_v += 1
        tr_nll /= max(n_b, 1)
        va_nll /= max(n_v, 1)
        hist.append((ep, tr_nll, va_nll))
        if va_nll < best:
            best = va_nll
            torch.save({"model": model.state_dict(), "args": vars(args),
                        "epoch": ep, "val_nll": va_nll},
                       ckdir / f"{args.tag}_best.pt")
        if ep % 20 == 0 or ep == args.epochs - 1:
            print(f"ep {ep:4d}  train {tr_nll:.4f}  val {va_nll:.4f}  "
                  f"best {best:.4f}  {time.time()-t0:.0f}s", flush=True)

    with open(resdir / f"history_{args.tag}.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["epoch", "train_nll_per_dim", "val_nll_per_dim"])
        w.writerows(hist)
    print(f"done: best val NLL/dim {best:.4f}  wall {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="v0")
    ap.add_argument("--epochs", type=int, default=600)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--layers", type=int, default=8)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--noise", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--physics-weight", type=float, default=0.0,
                    help="lambda on mean squared sample residual (MVA^2); 0 = off")
    ap.add_argument("--phys-batch", type=int, default=8)
    ap.add_argument("--hist", default="m2", help="results subdir for history CSV")
    run(ap.parse_args())
