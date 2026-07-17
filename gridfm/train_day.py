"""Train the v1 day-score flow (M4). Fast: 64-dim, ~1 min per seed on GPU.

python -m gridfm.train_day --seed 42
Writes checkpoints/v1_s{seed}_best.pt + results/m4/history_v1_s{seed}.csv and (once)
checkpoints/dayspace_{feeder}.npz.
"""
import argparse
import csv
from pathlib import Path

import numpy as np
import torch

from .dataset import TimePCA, build_feeder
from .dayspace import KD, DaySpace
from .flow_day import DayFlow, feeder_context

ROOT = Path(__file__).resolve().parents[1]
TRAIN_FEEDERS = ["rural0", "semiurb0"]


def load_scores(device):
    pca = TimePCA.load(ROOT / "checkpoints" / "timepca_k16.npz")
    data = {}
    for n in TRAIN_FEEDERS:
        f = build_feeder(ROOT / "data" / "m1" / f"{n}_year.npz")
        co = pca.encode(f["days_pu"])
        ds = DaySpace().fit(co[f["train_idx"]])
        np.savez(ROOT / "checkpoints" / f"dayspace_{n}.npz", mean=ds.mean,
                 comps=ds.comps, scale=ds.scale, kd=ds.kd, n=ds.n, dx=ds.dx,
                 explained=ds.explained)
        s_tr = torch.as_tensor(ds.encode(co[f["train_idx"]]))
        s_va = torch.as_tensor(ds.encode(co[f["val_idx"]]))
        ctx = feeder_context(f).unsqueeze(0).to(device)
        data[n] = dict(s_tr=s_tr, s_va=s_va, ctx=ctx, explained=ds.explained)
        print(f"{n}: day-space explains {ds.explained:.4f} of coeff variance", flush=True)
    return data


def run(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data = load_scores(device)
    d_ctx = 44
    model = DayFlow(KD, d_ctx, n_layers=args.layers, hidden=args.hidden,
                    seed=args.seed).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    resdir = ROOT / "results" / "m4"
    resdir.mkdir(parents=True, exist_ok=True)

    best = float("inf")
    hist = []
    for ep in range(args.epochs):
        model.train()
        tr = 0.0
        nb = 0
        for n in TRAIN_FEEDERS:
            d = data[n]
            idx = torch.randperm(len(d["s_tr"]))
            for i in range(0, len(idx), args.batch):
                x = d["s_tr"][idx[i:i + args.batch]].to(device)
                x = x + args.noise * torch.randn_like(x)
                loss = -model.log_prob(x, d["ctx"].expand(x.shape[0], -1)).mean() / KD
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                tr += float(loss)
                nb += 1
        sched.step()
        model.eval()
        va = 0.0
        with torch.no_grad():
            for n in TRAIN_FEEDERS:
                d = data[n]
                x = d["s_va"].to(device)
                va += float(-model.log_prob(x, d["ctx"].expand(x.shape[0], -1)).mean() / KD)
        va /= len(TRAIN_FEEDERS)
        hist.append((ep, tr / nb, va))
        if va < best:
            best = va
            torch.save({"model": model.state_dict(), "args": vars(args),
                        "epoch": ep, "val_nll": va},
                       ROOT / "checkpoints" / f"v1_s{args.seed}_best.pt")
        if ep % 50 == 0:
            print(f"ep {ep:4d} train {tr/nb:.4f} val {va:.4f} best {best:.4f}", flush=True)

    with open(resdir / f"history_v1_s{args.seed}.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["epoch", "train_nll_per_dim", "val_nll_per_dim"])
        w.writerows(hist)
    print(f"done seed {args.seed}: best val NLL/dim {best:.4f}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--layers", type=int, default=8)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--noise", type=float, default=0.03)
    ap.add_argument("--seed", type=int, default=42)
    run(ap.parse_args())
