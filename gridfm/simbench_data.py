"""Distribution scenario generator for radial SimBench MV feeders.

Retargets the GridFM thesis-recreation data pipeline (github.com/panas-bhattarai/gridfm-thesis-recreation)
from meshed transmission grids to **radial SimBench MV distribution feeders with
tie-switch reconfiguration**.

Pipeline per scenario:
    load a SimBench MV feeder  ->  apply a switch RECONFIGURATION (branch-exchange:
    bring a normally-open tie line into service, take a sectionalizing line out ->
    a different radial topology, same assets)  ->  scale loads and DER (sgen)
    injections  ->  AC power flow (Newton-Raphson; NO OPF -- distribution feeders
    have no dispatchable generation, power flows from the HV substation)  ->
    extract node/edge/Ybus features  ->  physics self-check.

Representation choice (see notebook 01 / MILESTONES M1 -- resolves a real pitfall):
we use the SimBench **`-no_sw`** grids and express topology purely through line
`in_service`. Every OPEN line switch is converted to an out-of-service line and all
switches are then closed, so the solved network has NO open switches. This matters
because pandapower models each *open* switch with an auxiliary "stub" bus, which
breaks the clean 1:1 bus<->solver mapping the feature extractor needs. With this
normalization the mapping is exact (ppc buses == real buses) and the physics
residual sits at ~1e-9 MVA, matching the transmission evidence base. Reconfiguration
then = "which lines are switched out", the framing a distribution engineer uses.

Key deviations from the transmission pipeline (ASSUMPTIONS, logged in M1):
  * No AC-OPF: distribution generation is non-dispatchable DER (pandapower `sgen`,
    fixed PQ). Bus types are REF (the HV ext_grid) + PQ everywhere else; no PV buses.
  * Topology perturbation = switch RECONFIGURATION (branch-exchange), not element
    outage -- the topology operation distribution grids actually perform.
  * Load and DER scale by INDEPENDENT factors (sunny midday = low load, high PV ->
    reverse power flow), a distribution phenomenon a single global scale can't express.
"""
import copy
import logging
import numpy as np
import pandas as pd
import networkx as nx
import pandapower as pp
import pandapower.topology as top
import simbench as sb

logging.getLogger("pandapower").setLevel(logging.CRITICAL)

# get_simbench_net(...) costs ~4 s; cache the pristine normalized net per code and
# hand out cheap deep copies. The cache is per-process, so it also works under the
# multiprocessing pool used for bulk generation (each worker loads each grid once).
_PRISTINE = {}

# ---------------------------------------------------------------------------
# Feeder registry. 'train' feeders are seen in pre-training; 'heldout' feeders
# are reserved for zero-shot topology transfer (never pre-trained on) -- the
# transmission evidence base held out case39/case300 the same way.
# NOTE: we deliberately use the -no_sw variants (see module docstring).
# ---------------------------------------------------------------------------
FEEDERS = {
    "rural0":   {"code": "1-MV-rural--0-no_sw",   "role": "train"},
    "rural1":   {"code": "1-MV-rural--1-no_sw",   "role": "train"},
    "semiurb0": {"code": "1-MV-semiurb--0-no_sw", "role": "train"},
    "urban0":   {"code": "1-MV-urban--0-no_sw",   "role": "heldout"},
    "comm0":    {"code": "1-MV-comm--0-no_sw",    "role": "heldout"},
}

# Scenario sampling ranges (ASSUMPTIONS -- the thesis scales only active load by a
# single yearly-profile factor; here load and DER move independently).
LOAD_SCALE_RANGE = (0.5, 1.1)   # fraction of SimBench peak load
DER_SCALE_RANGE  = (0.0, 1.0)   # fraction of installed DER (0 = night, 1 = full sun)


# ---------------------------------------------------------------------------
# Feeder loading, switch normalization, topology
# ---------------------------------------------------------------------------
def load_feeder(key_or_code):
    """Return a fresh (deep-copied) normalized net for a feeder key or SimBench code.
    The pristine normalized net is loaded once per code and cached; callers may mutate
    the returned copy freely."""
    code = FEEDERS[key_or_code]["code"] if key_or_code in FEEDERS else key_or_code
    if code not in _PRISTINE:
        _PRISTINE[code] = normalize_to_lines(sb.get_simbench_net(code))
    return copy.deepcopy(_PRISTINE[code])


def normalize_to_lines(net):
    """Convert every OPEN line-switch into an out-of-service line, then close all
    switches -> the solved net has no open switches (hence no auxiliary buses).
    Idempotent on an already-normalized net."""
    ls = net.switch[net.switch.et == "l"]
    for sw_id, row in ls.iterrows():
        if not row.closed:
            net.line.at[int(row.element), "in_service"] = False
        net.switch.at[sw_id, "closed"] = True
    return net


def base_out_lines(code):
    """The set of line ids initially out of service (the normally-open tie lines)."""
    net = load_feeder(code)
    return set(net.line.index[~net.line.in_service].tolist())


def loop_count(net):
    """(connected?, independent-cycle count) of the energised bus graph. A radially
    operated feeder fed by two parallel substation transformers has exactly one fixed
    cycle at the substation; a valid reconfiguration preserves that count."""
    g = top.create_nxgraph(net, respect_switches=True)
    ncomp = nx.number_connected_components(g)
    loops = g.number_of_edges() - g.number_of_nodes() + ncomp
    return ncomp == 1, loops


# ---------------------------------------------------------------------------
# Reconfiguration: enumerate valid branch-exchange topologies
# ---------------------------------------------------------------------------
def enumerate_reconfigurations(code, max_configs=8):
    """Return (configs, detail).

    A *config* is a frozenset of out-of-service line ids -> one radial topology.
    configs[0] is the base (the SimBench normally-open ties). Each further config is
    a single branch-exchange: bring one tie line into service, take one other line
    out, staying connected + radial (same cycle count as the base) with AC PF
    converging. Deterministic (ties/lines iterated in id order).
    """
    net = load_feeder(code)                             # load once, mutate in place
    _, base_loops = loop_count(net)
    base_out = set(net.line.index[~net.line.in_service].tolist())
    in_lines = list(net.line.index[net.line.in_service])

    configs = [frozenset(base_out)]
    detail = [{"kind": "base", "tie_in": None, "line_out": None,
               "out_lines": sorted(base_out)}]

    for tie in sorted(base_out):
        if len(configs) >= max_configs:
            break
        for line_out in in_lines:
            trial = (base_out - {tie}) | {line_out}     # tie in, line_out out
            net.line["in_service"] = True
            net.line.loc[list(trial), "in_service"] = False
            conn, loops = loop_count(net)
            if not (conn and loops == base_loops):
                continue
            try:
                pp.runpp(net)
                ok = bool(net.converged) and not net.res_bus.vm_pu.isna().any()
            except Exception:
                ok = False
            if ok:
                configs.append(frozenset(trial))
                detail.append({"kind": "branch_exchange", "tie_in": tie,
                               "line_out": line_out, "out_lines": sorted(trial)})
                break
    return configs, detail


def apply_config(net, config):
    """Set line in_service so that exactly `config` (a set of line ids) is out."""
    net.line["in_service"] = True
    net.line.loc[list(config), "in_service"] = False


# ---------------------------------------------------------------------------
# Scenario sampling
# ---------------------------------------------------------------------------
def sample_scales(seed):
    rng = np.random.RandomState(seed)
    return (float(rng.uniform(*LOAD_SCALE_RANGE)), float(rng.uniform(*DER_SCALE_RANGE)))


def apply_scales(net, load_scale, der_scale):
    net.load["p_mw"] *= load_scale
    net.load["q_mvar"] *= load_scale
    if len(net.sgen):
        net.sgen["p_mw"] *= der_scale
        net.sgen["q_mvar"] *= der_scale


# ---------------------------------------------------------------------------
# Feature extraction (mirrors thesis extract_features; distribution-adapted).
# With switch normalization the ppc<->pandapower bus mapping is 1:1, so this is a
# clean bijection just like the transmission pipeline.
# ---------------------------------------------------------------------------
NODE_COLS = ["Pd", "Qd", "Pg", "Qg", "Vm", "Va", "PQ", "PV", "REF"]   # 9, thesis layout


def extract_features(net, scenario_id):
    from pandapower.pypower.idx_brch import F_BUS, T_BUS, BR_R, BR_X, BR_STATUS

    base_mva = float(net._ppc["baseMVA"])
    buses = net.bus.index.values
    nb = len(buses)
    pos = {int(b): i for i, b in enumerate(buses)}

    pd_mw = np.zeros(nb); qd_mvar = np.zeros(nb)
    lmask = net.load.in_service.values
    for b, p, q in zip(net.load.loc[lmask, "bus"], net.load.loc[lmask, "p_mw"],
                       net.load.loc[lmask, "q_mvar"]):
        pd_mw[pos[int(b)]] += p; qd_mvar[pos[int(b)]] += q
    # Storage (batteries) use pandapower's consumer sign convention (p_mw > 0 =
    # charging = load-like). SimBench MV feeders include them; fold the SOLVED
    # storage power into net demand so the power-balance label stays exact.
    # ASSUMPTION: storage is represented as flexible net demand (Pd), not a separate
    # feature channel -- it keeps the 9-feature thesis layout; revisit if M3 wants
    # storage as its own controllable input.
    if len(net.storage):
        stmask = net.storage.in_service.values
        for b, p, q in zip(net.storage.loc[stmask, "bus"], net.res_storage.loc[stmask, "p_mw"],
                           net.res_storage.loc[stmask, "q_mvar"]):
            pd_mw[pos[int(b)]] += p; qd_mvar[pos[int(b)]] += q

    pg_mw = np.zeros(nb); qg_mvar = np.zeros(nb)
    for b, p, q in zip(net.ext_grid.bus, net.res_ext_grid.p_mw, net.res_ext_grid.q_mvar):
        pg_mw[pos[int(b)]] += p; qg_mvar[pos[int(b)]] += q
    if len(net.sgen):
        smask = net.sgen.in_service.values
        for b, p, q in zip(net.sgen.loc[smask, "bus"], net.res_sgen.loc[smask, "p_mw"],
                           net.res_sgen.loc[smask, "q_mvar"]):
            pg_mw[pos[int(b)]] += p; qg_mvar[pos[int(b)]] += q
    if len(net.gen):
        gmask = net.gen.in_service.values
        for b, p, q in zip(net.gen.loc[gmask, "bus"], net.res_gen.loc[gmask, "p_mw"],
                           net.res_gen.loc[gmask, "q_mvar"]):
            pg_mw[pos[int(b)]] += p; qg_mvar[pos[int(b)]] += q

    vm = net.res_bus.vm_pu.values.copy()
    va = np.deg2rad(net.res_bus.va_degree.values)

    is_ref = np.zeros(nb, dtype=int)
    for b in net.ext_grid.bus:
        is_ref[pos[int(b)]] = 1
    is_pv = np.zeros(nb, dtype=int)                 # no PV buses in distribution
    if len(net.gen):
        for b in net.gen.loc[net.gen.in_service, "bus"]:
            if not is_ref[pos[int(b)]]:
                is_pv[pos[int(b)]] = 1
    is_pq = ((is_ref == 0) & (is_pv == 0)).astype(int)

    node_df = pd.DataFrame({
        "scenario": scenario_id, "bus": buses,
        "Pd": pd_mw, "Qd": qd_mvar, "Pg": pg_mw, "Qg": qg_mvar,
        "Vm": vm, "Va": va, "PQ": is_pq, "PV": is_pv, "REF": is_ref,
    })

    br = net._ppc["branch"]
    lookup = net._pd2ppc_lookups["bus"]
    inv = {int(lookup[int(b)]): int(b) for b in buses}
    on = br[:, BR_STATUS].real > 0
    f = [inv[int(i.real)] for i in br[on, F_BUS]]
    t = [inv[int(i.real)] for i in br[on, T_BUS]]
    ys = 1.0 / (br[on, BR_R].real + 1j * br[on, BR_X].real)
    edge_df = pd.DataFrame({
        "scenario": scenario_id, "from_bus": f, "to_bus": t,
        "g": ys.real, "b": ys.imag,
    })

    Y = net._ppc["internal"]["Ybus"].tocoo()
    ybus_df = pd.DataFrame({
        "scenario": scenario_id,
        "i": [inv[int(r)] for r in Y.row], "j": [inv[int(c)] for c in Y.col],
        "G": Y.data.real, "B": Y.data.imag,
    })
    return node_df, edge_df, ybus_df, base_mva


def result_to_arrays(r):
    """Convert an 'ok' generate_one result into model-ready numpy arrays.

    Returns (x, edge_index, edge_attr, bus_ids):
      x           (n, 9)  node features in the NODE_COLS order (physical units)
      edge_index  (2, 2E) both branch orientations (source row, target row)
      edge_attr   (2E, 2) series (g, b) per directed edge
      bus_ids     (n,)    pandapower bus id of each row
    Row order follows the node table (== net.bus order).
    """
    nd, ed = r["node"], r["edge"]
    bus_ids = nd.bus.values.astype(np.int64)
    busidx = {int(b): i for i, b in enumerate(bus_ids)}
    x = nd[NODE_COLS].values.astype(np.float32)
    fi = [busidx[int(b)] for b in ed.from_bus]
    ti = [busidx[int(b)] for b in ed.to_bus]
    edge_index = np.array([fi + ti, ti + fi], dtype=np.int64)
    ga = ed[["g", "b"]].values.astype(np.float32)
    edge_attr = np.vstack([ga, ga])
    return x, edge_index, edge_attr, bus_ids


def physics_residual_mw(node_df, ybus_df, base_mva):
    """Max nodal apparent-power mismatch |S_bus(V) + S_load - S_gen| in MVA.
    ~1e-9 for a correctly stored Newton-Raphson solution (verified in notebook 01)."""
    nb = len(node_df)
    pos = {int(b): i for i, b in enumerate(node_df.bus.values)}
    V = node_df.Vm.values * np.exp(1j * node_df.Va.values)
    Y = np.zeros((nb, nb), dtype=complex)
    for i, j, G, B in zip(ybus_df.i, ybus_df.j, ybus_df.G, ybus_df.B):
        Y[pos[int(i)], pos[int(j)]] = G + 1j * B
    s_calc = V * np.conj(Y @ V) * base_mva
    s_inj = (node_df.Pg.values - node_df.Pd.values) + 1j * (node_df.Qg.values - node_df.Qd.values)
    return float(np.abs(s_calc - s_inj).max())


# ---------------------------------------------------------------------------
# One scenario (worker)
# ---------------------------------------------------------------------------
def generate_one(task):
    """task = (feeder_key, code, config, config_id, scenario_id, seed).
    `config` is a frozenset of out-of-service line ids from enumerate_reconfigurations.
    Returns a dict with status 'ok' (+ feature frames + meta) or a failure tag."""
    feeder_key, code, config, config_id, scenario_id, seed = task
    load_scale, der_scale = sample_scales(seed)
    net = load_feeder(code)
    apply_config(net, config)
    conn, _ = loop_count(net)
    if not conn:
        return {"status": "island", "feeder": feeder_key, "config": config_id, "scenario": scenario_id}
    apply_scales(net, load_scale, der_scale)
    try:
        pp.runpp(net)
        ok = bool(net.converged)
    except Exception:
        ok = False
    if not ok:
        return {"status": "pf_fail", "feeder": feeder_key, "config": config_id, "scenario": scenario_id}
    if net.res_bus.vm_pu.isna().any():
        return {"status": "nan_bus", "feeder": feeder_key, "config": config_id, "scenario": scenario_id}

    node_df, edge_df, ybus_df, base_mva = extract_features(net, scenario_id)
    resid = physics_residual_mw(node_df, ybus_df, base_mva)
    if resid > 1e-3 * base_mva:      # ~1e-3 MVA on the SimBench base_mva=1.0 scale

        return {"status": "physics_fail", "feeder": feeder_key, "config": config_id,
                "scenario": scenario_id, "residual_mva": resid}

    meta = pd.DataFrame([{
        "scenario": scenario_id, "feeder": feeder_key, "config": config_id,
        "load_scale": load_scale, "der_scale": der_scale,
        "vmin": float(net.res_bus.vm_pu.min()), "vmax": float(net.res_bus.vm_pu.max()),
        "losses_mw": float(net.res_line.pl_mw.sum()),
        "reverse_flow": bool(net.res_ext_grid.p_mw.iloc[0] < 0),
        "residual_mva": resid, "base_mva": base_mva,
    }])
    return {"status": "ok", "feeder": feeder_key, "config": config_id, "scenario": scenario_id,
            "node": node_df, "edge": edge_df, "ybus": ybus_df, "meta": meta}
