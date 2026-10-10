"""
rerank.py
=========

Quality scoring and reranking for K-draw EGNN predictions.

Without ground truth, "the best of K draws" is not well-defined. But it
can be approximated by chemical plausibility heuristics:

  1. symmetry_break_value     -- are symmetric atoms collapsing?
  2. bond_length_deviation    -- from typical values by (element pair, bond type)
  3. steric_clash_count       -- non-bonded atoms closer than vdW sum

The score is a weighted sum; lower is better. Weights are hardcoded from
experience -- they should be tuned if the model's failure modes change.

This module is used by the inference server to implement
`visual_draw="best"` and `visual_draw="median"` modes. The `median`
option is purely geometric (no chemistry heuristics) and is the more
robust default when the ensemble is tight with occasional outliers.

No dependency on the training script. Imports only from `visualize`
(for element symbols) and `egnn_inference` (for build_triplets, which
is the same vectorized triplet enumeration the model uses).
"""

import torch

from Util.visualise import Z_TO_SYMBOL
from egnn_inference import build_triplets



# Reference values: typical bond lengths in organic molecules

# (symbol_a, symbol_b, bond_type_index) -> mean length in Angstrom
# bond_type_index: 0=single, 1=double, 2=triple, 3=aromatic
TYPICAL_BOND_LENGTHS = {
    ("C", "C", 0): 1.53,
    ("C", "C", 1): 1.34,
    ("C", "C", 2): 1.20,
    ("C", "C", 3): 1.39,
    ("C", "H", 0): 1.09,
    ("C", "N", 0): 1.47,
    ("C", "N", 1): 1.28,
    ("C", "N", 3): 1.34,
    ("C", "O", 0): 1.42,
    ("C", "O", 1): 1.22,
    ("C", "F", 0): 1.35,
    ("N", "H", 0): 1.02,
    ("N", "N", 0): 1.45,
    ("N", "N", 1): 1.25,
    ("N", "O", 0): 1.40,
    ("N", "O", 1): 1.21,
    ("O", "H", 0): 0.97,
    ("O", "O", 0): 1.48,
    ("O", "O", 1): 1.21,
    ("F", "F", 0): 1.42,
}

# Approximate van der Waals radii (Angstrom), for steric clash detection.
VDW_RADII = {"H": 1.20, "C": 1.70, "N": 1.55, "O": 1.52, "F": 1.47}


# ----------------------------------------------------------------------
# Individual heuristics
# ----------------------------------------------------------------------

def _typical_length_for(z_i, z_j, bt_idx):
    """Look up the typical length for a bond. Returns None if unknown."""
    sym_a = Z_TO_SYMBOL.get(int(z_i), "?")
    sym_b = Z_TO_SYMBOL.get(int(z_j), "?")
    key = (min(sym_a, sym_b), max(sym_a, sym_b), bt_idx)
    return TYPICAL_BOND_LENGTHS.get(key)


def bond_length_deviation(pred, z, edge_index, edge_attr):
    """
    Mean absolute deviation of predicted bonded distances from typical
    values. Uses only bonds we have reference values for.

    Iterates over one direction of each undirected bond only (a >= b is
    skipped) so each bond is counted once. For QM9-sized molecules this
    Python loop over edges is a non-issue; for much larger molecules it
    would need vectorization, but this module is not targeted at that.

    Returns (mean_deviation, count_used). If no bonds matched the table,
    returns (0.0, 0).
    """
    deviations = []
    src, dst = edge_index[0], edge_index[1]
    for k in range(edge_index.size(1)):
        a, b = int(src[k]), int(dst[k])
        if a >= b:
            continue
        bt_idx = int(edge_attr[k].argmax().item())
        typical = _typical_length_for(z[a], z[b], bt_idx)
        if typical is None:
            continue
        d = (pred[a] - pred[b]).norm().item()
        deviations.append(abs(d - typical))
    if not deviations:
        return 0.0, 0
    return sum(deviations) / len(deviations), len(deviations)


def steric_clash_count(pred, z, edge_index, tolerance=0.7):
    """
    Count non-bonded atom pairs whose distance is less than `tolerance`
    times the sum of their van der Waals radii.

    Bonded pairs are excluded. A high count indicates severe clashes
    (atoms overlapping or nearly overlapping). Real molecules have 0 or
    very few.

    O(N^2) Python loop. For QM9 (N <= 29), this is ~400 pairs per
    molecule, negligible at inference time.
    """
    N = pred.size(0)

    bonded = set()
    for k in range(edge_index.size(1)):
        a, b = int(edge_index[0, k]), int(edge_index[1, k])
        bonded.add((min(a, b), max(a, b)))

    symbols = [Z_TO_SYMBOL.get(int(zi), "?") for zi in z]
    count = 0
    for i in range(N):
        for j in range(i + 1, N):
            if (i, j) in bonded:
                continue
            ri = VDW_RADII.get(symbols[i], 1.5)
            rj = VDW_RADII.get(symbols[j], 1.5)
            threshold = tolerance * (ri + rj)
            d = (pred[i] - pred[j]).norm().item()
            if d < threshold:
                count += 1
    return count


def symmetry_break_value(pred, edge_index, min_dist=1.0):
    """
    Symmetry_break_loss value for a single structure. Uses the same
    triplet enumeration the model uses (build_triplets).
    """
    src, dst = edge_index
    idx_j, idx_k = build_triplets(edge_index, pred.size(0))
    if idx_j.numel() == 0:
        return 0.0
    src_j, src_k = src[idx_j], src[idx_k]
    d = (pred[src_j] - pred[src_k]).norm(dim=-1)
    return torch.relu(min_dist - d).mean().item()


# ----------------------------------------------------------------------
# Composite quality score
# ----------------------------------------------------------------------

def quality_score(pred, z, edge_index, edge_attr,
                  w_sym=1.0, w_bond=5.0, w_clash=0.5):
    """
    Scalar quality score for a single predicted structure. Lower is
    better. Combines three heuristics that don't require ground truth:

      - symmetry break value:  penalizes collapsed symmetric atoms
      - bond length deviation: penalizes wrong bond lengths
      - steric clash count:    penalizes overlapping non-bonded atoms

    Weights are chosen so the terms are roughly comparable in magnitude
    for typical failure modes:
      - a fully collapsed CH2 group contributes ~1.0 from the symmetry term
      - a bond length off by 0.2 A on average contributes ~1.0 from the bond term
      - a single serious clash contributes ~0.5

    Returns (score, breakdown_dict). The breakdown is useful for
    diagnostics via /predict_draws.
    """
    sym = symmetry_break_value(pred, edge_index, min_dist=1.0)
    bond_dev, n_bonds = bond_length_deviation(pred, z, edge_index, edge_attr)
    clash = steric_clash_count(pred, z, edge_index)

    score = w_sym * sym + w_bond * bond_dev + w_clash * clash
    return score, {
        "symmetry_break": sym,
        "bond_length_deviation": bond_dev,
        "n_bonds_used": n_bonds,
        "steric_clash_count": clash,
        "total": score,
    }


# ----------------------------------------------------------------------
# Reranking: pick from K aligned draws
# ----------------------------------------------------------------------

def select_best_draw(aligned_preds, z, edge_index, edge_attr):
    """
    Given K Kabsch-aligned predictions (all in the same reference frame),
    score each with quality_score and return the index of the best.

    Note: this uses chemistry heuristics with hardcoded weights. It is
    the more opinionated reranker. If the model's failure modes are
    different from what the weights assume, use select_median_draw.

    Returns (best_index, all_scores).
    """
    scores = []
    for pred in aligned_preds:
        score, _ = quality_score(pred, z, edge_index, edge_attr)
        scores.append(score)
    best = int(min(range(len(scores)), key=lambda i: scores[i]))
    return best, scores


def select_median_draw(aligned_preds):
    """
    Given K Kabsch-aligned predictions, pick the one with the smallest
    mean distance to the other draws. This is a purely geometric
    centrality criterion -- it doesn't use chemistry heuristics, and it
    is robust when the ensemble is tight but has occasional outliers.

    Returns the index of the median draw.
    """
    K = len(aligned_preds)
    if K == 1:
        return 0
    dists = []
    for i in range(K):
        d_i = 0.0
        for j in range(K):
            if i == j:
                continue
            d_ij = (aligned_preds[i] - aligned_preds[j]).norm(dim=-1).mean().item()
            d_i += d_ij
        dists.append(d_i / (K - 1))
    return int(min(range(K), key=lambda i: dists[i]))