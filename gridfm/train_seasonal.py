"""M5 seasonal experiment trainers: the graph-flow (v0 architecture) trained on

  A: rural0 SUMMER days only                (the data-poor DSO: 4 months, one feeder)
  B: rural0 SUMMER days + semiurb0 FULL year (pretrain-broad: another feeder's year)

Question: does B generate winter-like days for rural0 although it never saw a rural0
winter — i.e. does cross-feeder training transfer seasonal structure through the
shared weights? (v0 architecture used because its weights are feeder-shared; v1's
day basis is per-feeder and cannot pool.)
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
SUMMER = np.arange(121, 244)          # May–Aug of the SimBench study year (day-of-year)


def run(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pca = TimePCA.load(ROOT / "checkpoints" / "timepca_k16.npz")

    plans = {"A": [("rural0", SUMMER)],
             "B": [("rural0", SUMMER), ("semiurb0", None)]}
    feeders = []
    for name, days in plans[args.mode]:
        f = build_feeder(ROOT / "data" / "m1" / f"{name}_year.npz")
        f["g"] = feeder_graph_tensors(f, device)
        idx = days if days is not None else f["train_idx"]
        f["coeff"] = torch.as_tensor(pca.encode(f["days_pu"][idx]))
        f["label"] = name
        feeders.append(f)
        print(f"{args.mode}: {name} -> {len(idx)} train days", flush=True)

    d_x = 6 * pca.k
    model = GraphFlow(d_x, D_COND, n_layers=8, hidden=128).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    t0 = time.time()
    hist = []
    for ep in range(args.epochs):
        model.train()
        tr, nb = 0.0, 0
        for f in feeders:
            perm = torch.randperm(len(f["coeff"]))
            for i in range(0, len(perm), 32):
                x = f["coeff"][perm[i:i + 32]].to(device)
                x = x + 0.01 * torch.randn_like(x)
                loss = -model.log_prob(x, f["g"]).mean() / x.shape[1] / d_x
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                tr += float(loss); nb += 1
        sched.step()
        hist.append((ep, tr / nb))
        if ep % 40 == 0:
            print(f"ep {ep:4d} train {tr/nb:.4f} {time.time()-t0:.0f}s", flush=True)

    torch.save({"model": model.state_dict(), "epochs": args.epochs,
                "mode": args.mode, "k": pca.k},
               ROOT / "checkpoints" / f"seasonal_{args.mode}_best.pt")
    resdir = ROOT / "results" / "m5"
    resdir.mkdir(parents=True, exist_ok=True)
    with open(resdir / f"history_seasonal_{args.mode}.csv", "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["epoch", "train_nll_per_dim"]); w.writerows(hist)
    print(f"done {args.mode}: {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["A", "B"], required=True)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    run(ap.parse_args())
