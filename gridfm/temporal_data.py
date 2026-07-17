"""Temporal dataset generator for the physics-generative prototype (M1).

Builds the "real" dataset the generative model must imitate: for each SimBench MV
feeder, apply the feeder's OWN full-year 15-min profiles (load P/Q, DER P, storage)
step by step, solve AC power flow per step, and store the full state per bus.

One feeder-year = dyn array (T, N, 6): T ~ 35,136 steps, N buses,
features Pd, Qd, Pg, Qg, Vm, Va (MW / MVAr / pu / rad) + static nameplate per bus
+ directed edge list with series (g, b).

Reuses the companion topology study's conventions (gridfm/simbench_data.py):
  * -no_sw grids, topology via line in_service only -> exact bus<->solver bijection
  * storage folded into net demand, consumer sign (storage-folding lesson: caught by the
    physics gate, not by eye)
  * physics self-check on EVERY step: max |S_calc - S_inj| <= 1e-3 MVA, vectorized
    over the whole year after the solve loop

Deviation from the companion topology study: no random scenario sampling. Time correlation is the point
here -- the SimBench profiles carry the daily/seasonal structure the generator must
learn, so the dataset is the year's actual 15-min sequence, in order.
"""
import json
import time

import numpy as np
import pandapower as pp
import simbench as sb

from .simbench_data import FEEDERS, extract_features, load_feeder

DYN_COLS = ["Pd", "Qd", "Pg", "Qg", "Vm", "Va"]
STATIC_COLS = ["peak_Pd", "peak_Qd", "inst_DER", "has_storage", "PQ", "REF"]

# M1 feeder split (rationale in notebook 01): train on two residential/mixed
# feeders, hold out the commercial one -- for a generative model the transfer
# stressor is the daily SHAPE (commercial vs residential), not just the graph.
M1_FEEDERS = {"rural0": "train", "semiurb0": "train", "comm0": "heldout"}

STEPS_PER_DAY = 96
RESID_GATE_MVA = 1e-3


def _bus_rows(net, element_bus_series, pos):
    return np.array([pos[int(b)] for b in element_bus_series], dtype=np.int64)


def load_profiles(net):
    """SimBench absolute profiles for this net, columns aligned to element tables.

    Returns dict with keys 'load_p', 'load_q', 'sgen_p', 'storage_p' (missing
    elements -> None). Every returned frame has exactly the element table's index
    as columns, in order, so `.values` rows can be assigned straight into the net.
    """
    prof = sb.get_absolute_values(net, profiles_instead_of_study_cases=True)

    def aligned(key, table):
        if key not in prof or not len(table):
            return None
        df = prof[key]
        missing = set(table.index) - set(df.columns)
        if missing:
            raise ValueError(f"profile {key} misses elements {sorted(missing)}")
        return df[table.index]

    return {
        "load_p": aligned(("load", "p_mw"), net.load),
        "load_q": aligned(("load", "q_mvar"), net.load),
        "sgen_p": aligned(("sgen", "p_mw"), net.sgen),
        "storage_p": aligned(("storage", "p_mw"), net.storage),
    }


def static_nameplate(net, profiles, pos):
    """Per-bus static conditioning features (STATIC_COLS order).

    Nameplate is derived from the year's own profiles (peak per bus), i.e. what a
    DSO would read off its asset registry / billing peaks -- not from any solved
    operating point.
    """
    nb = len(pos)
    out = np.zeros((nb, len(STATIC_COLS)), dtype=np.float32)
    lrows = _bus_rows(net, net.load.bus, pos)
    if profiles["load_p"] is not None:
        # peak of the bus-aggregated load time series (coincident peak per bus)
        agg = np.zeros((len(profiles["load_p"]), nb), dtype=np.float32)
        for col, r in zip(profiles["load_p"].columns, lrows):
            agg[:, r] += profiles["load_p"][col].values
        out[:, 0] = agg.max(axis=0)
        aggq = np.zeros_like(agg)
        for col, r in zip(profiles["load_q"].columns, lrows):
            aggq[:, r] += profiles["load_q"][col].values
        out[:, 1] = aggq.max(axis=0)
    if profiles["sgen_p"] is not None and len(net.sgen):
        srows = _bus_rows(net, net.sgen.bus, pos)
        aggs = np.zeros((len(profiles["sgen_p"]), nb), dtype=np.float32)
        for col, r in zip(profiles["sgen_p"].columns, srows):
            aggs[:, r] += profiles["sgen_p"][col].values
        out[:, 2] = aggs.max(axis=0)
    if len(net.storage):
        strows = _bus_rows(net, net.storage.bus, pos)
        out[strows, 3] = 1.0
    ref = np.zeros(nb, dtype=np.float32)
    for b in net.ext_grid.bus:
        ref[pos[int(b)]] = 1.0
    out[:, 4] = 1.0 - ref    # PQ
    out[:, 5] = ref          # REF
    return out


def generate_feeder_year(feeder_key, out_path, t0=0, t1=None, progress_every=2000):
    """Solve the feeder's year of 15-min AC power flows and save one .npz.

    Saved arrays: dyn (T,N,6 float32), static (N,6), edge_index (2,2E),
    edge_attr (2E,2), bus_ids (N,), resid_mva (T,), meta (json string).
    Deterministic: no sampling anywhere; the profiles ARE the scenario sequence.
    """
    t_start = time.time()
    net = load_feeder(feeder_key)
    profiles = load_profiles(net)
    T_all = len(profiles["load_p"])
    t1 = T_all if t1 is None else min(t1, T_all)
    steps = range(t0, t1)
    T = len(steps)

    buses = net.bus.index.values
    nb = len(buses)
    pos = {int(b): i for i, b in enumerate(buses)}
    lrows = _bus_rows(net, net.load.bus, pos)
    srows = _bus_rows(net, net.sgen.bus, pos) if len(net.sgen) else None
    strows = _bus_rows(net, net.storage.bus, pos) if len(net.storage) else None
    grows = _bus_rows(net, net.ext_grid.bus, pos)

    static = static_nameplate(net, profiles, pos)
    # float64 throughout: the physics self-check needs the solver's full precision.
    # Casting the state to float32 alone raises the residual from ~1e-10 to ~1e-3 MVA
    # (measured, notebook 01) -- that float32 floor is recorded in meta because the
    # M3 physics loss will operate on float32 tensors and must be read against it.
    dyn = np.zeros((T, nb, len(DYN_COLS)), dtype=np.float64)
    n_fail = 0

    for k, t in enumerate(steps):
        net.load["p_mw"] = profiles["load_p"].values[t]
        net.load["q_mvar"] = profiles["load_q"].values[t]
        if srows is not None and profiles["sgen_p"] is not None:
            net.sgen["p_mw"] = profiles["sgen_p"].values[t]
        if strows is not None and profiles["storage_p"] is not None:
            net.storage["p_mw"] = profiles["storage_p"].values[t]
        try:
            pp.runpp(net, init="results" if k else "auto", numba=True)
            ok = bool(net.converged)
        except Exception:
            ok = False
        if not ok:
            # retry cold -- warm start can fail after a sharp profile jump
            try:
                pp.runpp(net, init="auto", numba=True)
                ok = bool(net.converged)
            except Exception:
                ok = False
        if not ok:
            n_fail += 1
            dyn[k, :, :] = np.nan
            continue

        pd_bus = np.zeros(nb)
        qd_bus = np.zeros(nb)
        np.add.at(pd_bus, lrows, net.load.p_mw.values)
        np.add.at(qd_bus, lrows, net.load.q_mvar.values)
        if strows is not None:
            # consumer sign: charging (>0) adds to demand, discharging subtracts
            np.add.at(pd_bus, strows, net.res_storage.p_mw.values)
            np.add.at(qd_bus, strows, net.res_storage.q_mvar.values)
        pg_bus = np.zeros(nb)
        qg_bus = np.zeros(nb)
        np.add.at(pg_bus, grows, net.res_ext_grid.p_mw.values)
        np.add.at(qg_bus, grows, net.res_ext_grid.q_mvar.values)
        if srows is not None:
            np.add.at(pg_bus, srows, net.res_sgen.p_mw.values)
            np.add.at(qg_bus, srows, net.res_sgen.q_mvar.values)

        dyn[k, :, 0] = pd_bus
        dyn[k, :, 1] = qd_bus
        dyn[k, :, 2] = pg_bus
        dyn[k, :, 3] = qg_bus
        dyn[k, :, 4] = net.res_bus.vm_pu.values
        dyn[k, :, 5] = np.deg2rad(net.res_bus.va_degree.values)

        if progress_every and (k + 1) % progress_every == 0:
            rate = (k + 1) / (time.time() - t_start)
            print(f"[{feeder_key}] {k+1}/{T} steps  ({rate:.0f} PF/s)", flush=True)

    # --- topology (constant across the year) + vectorized physics self-check ---
    node_df, edge_df, ybus_df, base_mva = extract_features(net, 0)
    Y = np.zeros((nb, nb), dtype=complex)
    for i, j, G, B in zip(ybus_df.i, ybus_df.j, ybus_df.G, ybus_df.B):
        Y[pos[int(i)], pos[int(j)]] = G + 1j * B
    def year_residual(state):
        V = state[:, :, 4] * np.exp(1j * state[:, :, 5])      # (T, N) complex
        s_calc = V * np.conj(V @ Y.T) * base_mva
        s_inj = (state[:, :, 2] - state[:, :, 0]) + 1j * (state[:, :, 3] - state[:, :, 1])
        return np.abs(s_calc - s_inj).max(axis=1)             # (T,) max over buses

    resid = year_residual(dyn)                                # float64: solver truth
    resid32 = year_residual(dyn.astype(np.float32).astype(np.float64))  # storage floor

    busidx = {int(b): i for i, b in enumerate(buses)}
    fi = [busidx[int(b)] for b in edge_df.from_bus]
    ti = [busidx[int(b)] for b in edge_df.to_bus]
    edge_index = np.array([fi + ti, ti + fi], dtype=np.int64)
    ga = edge_df[["g", "b"]].values.astype(np.float32)
    edge_attr = np.vstack([ga, ga])

    ok_mask = ~np.isnan(dyn[:, 0, 4])
    meta = {
        "feeder": feeder_key,
        "code": FEEDERS[feeder_key]["code"],
        "role": M1_FEEDERS.get(feeder_key, "extra"),
        "t0": int(t0), "t1": int(t1), "steps": int(T),
        "steps_per_day": STEPS_PER_DAY,
        "n_bus": int(nb), "n_edge_directed": int(edge_index.shape[1]),
        "base_mva": float(base_mva),
        "pf_failures": int(n_fail),
        "resid_gate_mva": RESID_GATE_MVA,
        "resid_max_mva": float(np.nanmax(resid[ok_mask])) if ok_mask.any() else None,
        "resid_float32_floor_mva": float(np.nanmax(resid32[ok_mask])) if ok_mask.any() else None,
        "gate_pass": bool(ok_mask.any() and np.nanmax(resid[ok_mask]) <= RESID_GATE_MVA),
        "wall_seconds": round(time.time() - t_start, 1),
        "dyn_cols": DYN_COLS, "static_cols": STATIC_COLS,
    }
    np.savez_compressed(
        out_path, dyn=dyn, static=static, edge_index=edge_index, edge_attr=edge_attr,
        bus_ids=buses.astype(np.int64), resid_mva=resid.astype(np.float64),
        meta=json.dumps(meta),
    )
    print(f"[{feeder_key}] done: {meta}", flush=True)
    return meta


if __name__ == "__main__":
    import argparse
    import pathlib

    ap = argparse.ArgumentParser()
    ap.add_argument("feeder", choices=list(FEEDERS))
    ap.add_argument("--out-dir", default="data/m1")
    ap.add_argument("--t0", type=int, default=0)
    ap.add_argument("--t1", type=int, default=None)
    a = ap.parse_args()
    outd = pathlib.Path(a.out_dir)
    outd.mkdir(parents=True, exist_ok=True)
    generate_feeder_year(a.feeder, outd / f"{a.feeder}_year.npz", t0=a.t0, t1=a.t1)
