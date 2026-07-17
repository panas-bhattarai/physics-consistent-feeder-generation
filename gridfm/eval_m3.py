"""M3 evaluation: physics-feasibility × statistical-fidelity for every run + baselines.

For each trained run (3 lambdas x 3 seeds) and each train feeder: sample 400 days,
compute the residual statistics (physics axis) and the M2 fidelity metrics
(statistics axis), plus the share of physically-out-of-range voltages. Baseline rows
(real float32, PCA ceiling, independent bootstrap, PCA-Gaussian) give the reference
frame. Output: results/m3/m3_runs.csv — the notebook only reads and plots.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from . import metrics as M
from .dataset import TimePCA, build_feeder, denormalize_days
from .flow import D_COND, GraphFlow, feeder_graph_tensors
from .physics import PhysicsHead, residual_stats

ROOT = Path(__file__).resolve().parents[1]
TRAIN_FEEDERS = ["rural0", "semiurb0"]
N_GEN = 400
LAMBDAS = {"lam0": 0.0, "lam1e3": 1e-3, "lam1e2": 1e-2}
SEEDS = [42, 43, 44]


def fidelity_row(real, gen, ref):
    sc = M.scorecard(real, gen, ref)
    keep = {k: sc[k] for k in ["W1_Pd", "W1_Vm", "ACF_head_rmse", "W1_head_ramps",
                               "XCorr_Pd_rmse", "revflow_abs_err_pct"]}
    return keep


def vm_out_of_range_pct(days, vm_lo, vm_hi):
    vm = days[:, :, 4, :]
    return 100.0 * float(((vm < vm_lo) | (vm > vm_hi)).mean())


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pca = TimePCA.load(ROOT / "checkpoints" / "timepca_k16.npz")
    feeders, heads, gs, reals, pools, vm_rng = {}, {}, {}, {}, {}, {}
    for n in TRAIN_FEEDERS:
        f = build_feeder(ROOT / "data" / "m1" / f"{n}_year.npz")
        feeders[n] = f
        heads[n] = PhysicsHead(f, pca).to(device)
        gs[n] = feeder_graph_tensors(f, device)
        reals[n] = denormalize_days(f["days_pu"][f["val_idx"]], f["static"])
        pools[n] = denormalize_days(f["days_pu"][f["train_idx"]], f["static"])
        vm_all = denormalize_days(f["days_pu"], f["static"])[:, :, 4, :]
        vm_rng[n] = (float(vm_all.min()), float(vm_all.max()))

    rows = []

    def add_row(label, kind, n, days, extra=None):
        r = dict(label=label, kind=kind, feeder=n)
        r.update(residual_stats(days, heads[n]))
        r.update(fidelity_row(reals[n], days, feeders[n]["ref_row"]))
        r["vm_oor_pct"] = vm_out_of_range_pct(days, *vm_rng[n])
        # diversity: across-sample std at fixed (bus, step), ratio to real —
        # pooled-marginal metrics cannot see generator collapse; this can.
        for c, cn in [(0, "pd"), (4, "vm")]:
            r[f"div_{cn}_ratio"] = float(days[:, :, c, :].std(axis=0).mean()
                                         / reals[n][:, :, c, :].std(axis=0).mean())
        r.update(extra or {})
        rows.append(r)
        print(f"{label:22s} {n:9s} resid_mean {r['resid_mean_mva']:9.4f}  "
              f"W1_Vm {r['W1_Vm']:.4f}  vm_oor {r['vm_oor_pct']:.2f}%", flush=True)

    rng = np.random.RandomState(7)
    for n in TRAIN_FEEDERS:                                   # baselines
        f = feeders[n]
        add_row("real_float32", "baseline", n, reals[n])
        rec = denormalize_days(pca.decode(pca.encode(f["days_pu"][f["val_idx"]])),
                               f["static"])
        add_row("pca_ceiling", "baseline", n, rec)
        add_row("independent", "baseline", n,
                M.sample_independent(pools[n], N_GEN, rng))
        co_tr = pca.encode(f["days_pu"][f["train_idx"]])
        d_tr = co_tr.reshape(len(co_tr), -1)
        mu_d = d_tr.mean(0)
        Xc = d_tr - mu_d
        U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
        kd = 64
        sc = Xc @ Vt[:kd].T
        zg = rng.multivariate_normal(sc.mean(0), np.cov(sc.T) + 1e-6 * np.eye(kd), N_GEN)
        co_g = (zg @ Vt[:kd] + mu_d).reshape(N_GEN, *co_tr.shape[1:]).astype(np.float32)
        add_row("gaussian", "baseline", n, denormalize_days(pca.decode(co_g), f["static"]))

    for lam_tag, lam in LAMBDAS.items():                      # trained runs
        for seed in SEEDS:
            tag = f"m3_{lam_tag}_s{seed}"
            ckp = ROOT / "checkpoints" / f"{tag}_best.pt"
            if not ckp.exists():
                print(f"SKIP missing {tag}", flush=True)
                continue
            ck = torch.load(ckp, map_location=device, weights_only=False)
            a = ck["args"]
            model = GraphFlow(6 * a["k"], D_COND, n_layers=a["layers"],
                              hidden=a["hidden"]).to(device)
            model.load_state_dict(ck["model"])
            model.eval()
            for n in TRAIN_FEEDERS:
                zs = []
                for i in range(0, N_GEN, 100):
                    zs.append(model.sample(min(100, N_GEN - i), gs[n]).cpu().numpy())
                days = denormalize_days(pca.decode(np.concatenate(zs)),
                                        feeders[n]["static"])
                add_row(tag, "flow", n, days,
                        {"lam": lam, "seed": seed, "val_nll": ck["val_nll"],
                         "best_epoch": ck["epoch"]})

    out = ROOT / "results" / "m3"
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / "m3_runs.csv", index=False)
    print("wrote", out / "m3_runs.csv", flush=True)


if __name__ == "__main__":
    main()
