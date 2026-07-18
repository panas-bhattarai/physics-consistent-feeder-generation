"""Admittance-weighted random-walk positional encoding (RWPE), GridFM v0.2.

Reused verbatim (in spirit) from the thesis-recreation evidence base
(github.com/panas-bhattarai/gridfm-thesis-recreation, gridfm/encoding.py; 
This is the structural "fingerprint" whose invariance under distribution-feeder
reconfiguration later notebooks probe.

Idea: give every bus a fingerprint = the probability that a random walker
starting at bus i is back at bus i after 1, 2, ..., k steps, where at each step
the walker picks an adjacent branch with probability proportional to the magnitude
of its series admittance |y| = sqrt(g^2 + b^2). Strong (high-admittance) lines are
walked often, weak ties rarely -- so the fingerprint reflects *electrical*
structure, not just hop counts.

    w_ij  = sqrt(g_ij^2 + b_ij^2)          (0 if i-j not a branch)
    M_ij  = w_ij / sum_k w_ik              (row-normalized transition matrix)
    RWPE_i = [M_ii, (M^2)_ii, ..., (M^k)_ii]

k=16 carried over from the transmission evidence base (it saturated the return
probabilities up to 300-bus grids). Whether k=16 is still the right value on
*radial* feeders is an open question this prototype's M2 probe measures directly
-- do NOT assume it; test it. The encoding depends only on topology + admittances
(never on masked electrical features), so it is computed once per graph/config and
is never masked.
"""
import torch

K_STEPS = 16   # inherited from the transmission evidence base; RE-EXAMINED in M2 on radial feeders


def transition_matrix(edge_index, edge_attr, num_nodes):
    """Dense admittance-weighted random-walk matrix M.

    edge_index (2, E) holds both orientations of every branch; edge_attr (E, 2)
    holds the series (g, b) of each branch. Parallel branches accumulate.
    """
    w = torch.sqrt(edge_attr[:, 0] ** 2 + edge_attr[:, 1] ** 2)   # |y| per edge
    W = torch.zeros(num_nodes, num_nodes, dtype=edge_attr.dtype)
    W.index_put_((edge_index[0], edge_index[1]), w, accumulate=True)
    deg = W.sum(dim=1, keepdim=True)
    return W / deg.clamp(min=1e-30)


def rwpe(edge_index, edge_attr, num_nodes, k=K_STEPS, lazy=0.0):
    """Random-walk positional encoding, shape (num_nodes, k).

    Column t is diag(M^t): the t-step return probability of the admittance-
    weighted walk. Column 0 is identically zero on shunt-free graphs (no self-loop
    to stay put) -- kept anyway to match the thesis formula.

    `lazy` in (0,1): use a lazy walk M' = (1-lazy)*M + lazy*I. On a bipartite
    (tree) graph the plain walk (lazy=0) can only return on EVEN steps, so odd
    columns are identically zero (see M2 / notebook 03). A lazy walk adds a
    self-loop, so the walker can "stay put", making every step's return
    probability informative -- the encoding ablation for the radial regime.
    """
    M = transition_matrix(edge_index, edge_attr, num_nodes)
    if lazy > 0.0:
        M = (1.0 - lazy) * M + lazy * torch.eye(num_nodes, dtype=M.dtype)
    out = torch.empty(num_nodes, k, dtype=M.dtype)
    P = M.clone()
    out[:, 0] = P.diagonal()
    for t in range(1, k):
        P = P @ M
        out[:, t] = P.diagonal()
    return out


# ---------------------------------------------------------------------------
# Encoding variants for the M3 ablation (each tied to an M2 finding).
# All take (edge_index, edge_attr, num_nodes) and return (num_nodes, dim).
# ---------------------------------------------------------------------------
def rwpe_variant(edge_index, edge_attr, num_nodes, variant="rwpe16"):
    """Dispatch the positional encoding used by a v0.2 model.

    - "rwpe16"     : default, k=16 (odd columns ~empty on radial feeders)      -> 16 dims
    - "rwpe32"     : k=32 (tests the 'even steps still evolving past 16')       -> 32 dims
    - "even8"      : the 8 EVEN steps of a 16-step walk (drops the empty odds)  -> 8 dims
    - "selfloop16" : lazy walk (lazy=0.5), k=16, so odd steps also carry signal -> 16 dims
    """
    if variant == "rwpe16":
        return rwpe(edge_index, edge_attr, num_nodes, k=16)
    if variant == "rwpe32":
        return rwpe(edge_index, edge_attr, num_nodes, k=32)
    if variant == "even8":
        P = rwpe(edge_index, edge_attr, num_nodes, k=16)
        return P[:, 1::2]                       # steps 2,4,...,16
    if variant == "selfloop16":
        return rwpe(edge_index, edge_attr, num_nodes, k=16, lazy=0.5)
    raise ValueError(f"unknown encoding variant {variant!r}")


VARIANT_DIMS = {"rwpe16": 16, "rwpe32": 32, "even8": 8, "selfloop16": 16}
