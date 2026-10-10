"""
qm9_splits.py -- the ONE place that decides which QM9 molecules are in the
train / val / test sets.

Both the training script and the viewer/ranking scripts rely on it, so they
can never disagree about the split. The training script stores the final
index lists inside the checkpoint; everything downstream just reads them.

Two independent choices
-----------------------
1. THE POOL (which molecules are eligible at all)
     subset_size=None            -> the full QM9 (~130k molecules)
     subset_size=N, sampling="first"   -> the first N molecules (legacy)
     subset_size=N, sampling="random"  -> N molecules drawn at random from
                                          the full QM9 (seeded)

2. THE SPLIT (how the pool is divided 80/10/10)
     "contiguous" -> legacy: cut the pool in order, no shuffling
     "random"     -> seeded shuffle, then cut
     "stratified" -> seeded random split that keeps the train/val/test
                     distributions matched on the features in `stratify_by`

Stratification features (computed from z / edge_index only, no RDKit):
     heavy_atoms   number of non-hydrogen atoms
     num_atoms     total atoms, in bins of 4
     sym_groups    heavy atoms carrying >=2 H or >=2 F (CH2, CH3, NH2, CF3,
                   ...), capped at 2  -> 0 / 1 / 2+   (the groups behind
                   most of your symmetry-collapse errors)
     rings         number of rings (bonds - atoms + 1), capped at 3
     has_F, has_N, has_O   contains that element (0/1)

Put the most important feature FIRST in `stratify_by`. If a combination of
feature values has fewer than `min_stratum_size` molecules, those molecules
fall back to a coarser key (the last feature is dropped, then the next, ...),
so rare cells never break the split.

Run this file directly to print a balance report comparing the three split
methods on your own settings (edit CONFIG at the bottom).
"""

from collections import Counter, defaultdict

import numpy as np

FEATURE_NAMES = ["heavy_atoms", "num_atoms", "sym_groups", "rings", "has_F", "has_N", "has_O"]
DEFAULT_STRATIFY_BY = ["heavy_atoms", "sym_groups", "has_F"]


# ----------------------------------------------------------------------
# Features (pure numpy core, so it is easy to test)
# ----------------------------------------------------------------------

def features_from_arrays(z, src, dst, batch, num_graphs):
    """
    Per-molecule features for a batch of molecules, vectorised.

    z      [N]  atomic numbers of all atoms in the batch
    src,dst [E] directed edges (global atom indices within the batch);
                QM9 edges are bidirectional, so each bond appears twice
    batch  [N]  molecule id (0..num_graphs-1) of each atom
    Returns dict feature_name -> int array of length num_graphs.
    """
    n_atoms = z.shape[0]
    zd = z[dst]
    nH = np.bincount(src, weights=(zd == 1).astype(float), minlength=n_atoms)
    nF = np.bincount(src, weights=(zd == 9).astype(float), minlength=n_atoms)
    heavy = z > 1

    def per_graph(values):
        return np.bincount(batch, weights=values.astype(float), minlength=num_graphs)

    atoms = per_graph(np.ones(n_atoms))
    n_heavy = per_graph(heavy)
    n_sym = per_graph(heavy & ((nH >= 2) | (nF >= 2)))
    n_bonds = np.bincount(batch[src], minlength=num_graphs) / 2.0
    rings = np.maximum(n_bonds - atoms + 1, 0)   # assumes each molecule is connected

    return {
        "heavy_atoms": n_heavy.astype(int),
        "num_atoms": (atoms // 4).astype(int),
        "sym_groups": np.minimum(n_sym, 2).astype(int),
        "rings": np.minimum(rings, 3).astype(int),
        "has_F": (per_graph(z == 9) > 0).astype(int),
        "has_N": (per_graph(z == 7) > 0).astype(int),
        "has_O": (per_graph(z == 8) > 0).astype(int),
    }


def compute_features(full_dataset, pool, batch_size=2048):
    """Features for the molecules `pool` (global QM9 indices), in that order."""
    from torch_geometric.loader import DataLoader

    pool = [int(i) for i in pool]
    loader = DataLoader(full_dataset[pool], batch_size=batch_size, shuffle=False)
    out = defaultdict(list)
    done = 0
    for b in loader:
        ei = b.edge_index.numpy()
        f = features_from_arrays(b.z.numpy(), ei[0], ei[1], b.batch.numpy(), b.num_graphs)
        for k, v in f.items():
            out[k].append(v)
        done += b.num_graphs
        if len(pool) > 20000 and (done // batch_size) % 10 == 0:
            print(f"  computed split features for {done}/{len(pool)} molecules")
    return {k: np.concatenate(v) for k, v in out.items()}


# ----------------------------------------------------------------------
# Pool selection and splitting (pure numpy)
# ----------------------------------------------------------------------

def select_pool(total, subset_size, subset_sampling, seed):
    """Global QM9 indices (sorted) of the molecules eligible for splitting."""
    if subset_size is None or subset_size >= total:
        return np.arange(total)
    if subset_sampling == "first":
        return np.arange(subset_size)
    if subset_sampling == "random":
        rng = np.random.default_rng(seed)
        return np.sort(rng.choice(total, size=subset_size, replace=False))
    raise ValueError("subset_sampling must be 'first' or 'random'")


def _cut(order, train_frac, val_frac):
    n = len(order)
    n_train, n_val = int(train_frac * n), int(val_frac * n)
    return order[:n_train], order[n_train:n_train + n_val], order[n_train + n_val:]


def stratified_assignment(feature_matrix, rng, train_frac, val_frac, min_stratum_size):
    """
    feature_matrix [n, f] ints (columns in priority order). Returns
    (train_pos, val_pos, test_pos) as arrays of row positions.
    """
    n, f = feature_matrix.shape
    keys = [None] * n
    remaining = list(range(n))
    for level in range(f, -1, -1):
        sub = {i: tuple(feature_matrix[i, :level]) for i in remaining}
        counts = Counter(sub.values())
        still = []
        for i in remaining:
            if level == 0 or counts[sub[i]] >= min_stratum_size:
                keys[i] = (level, sub[i])
            else:
                still.append(i)
        remaining = still
        if not remaining:
            break

    groups = defaultdict(list)
    for i, k in enumerate(keys):
        groups[k].append(i)

    test_frac = 1.0 - train_frac - val_frac
    train, val, test = [], [], []
    carry_v = carry_t = 0.0   # carry rounding remainders so totals stay on target
    for k in sorted(groups):
        members = np.array(groups[k])
        rng.shuffle(members)
        m = len(members)
        tv = val_frac * m + carry_v
        n_val = int(round(tv)); carry_v = tv - n_val
        tt = test_frac * m + carry_t
        n_test = int(round(tt)); carry_t = tt - n_test
        n_val = max(0, min(n_val, m))
        n_test = max(0, min(n_test, m - n_val))
        val.extend(members[:n_val])
        test.extend(members[n_val:n_val + n_test])
        train.extend(members[n_val + n_test:])
    return np.array(train), np.array(val), np.array(test)


def resolve_splits(full_dataset, subset_size=10000, subset_sampling="first",
                   split_method="contiguous", split_seed=0, stratify_by=None,
                   train_frac=0.8, val_frac=0.1, min_stratum_size=10,
                   compute_report_features=True):
    """
    Decide the split. Returns (indices, info):
      indices: {"train": [...], "val": [...], "test": [...]} of GLOBAL QM9
               indices (sorted ascending)
      info:    {"pool": array, "features": dict or None} for balance reports
    """
    total = len(full_dataset)
    pool = select_pool(total, subset_size, subset_sampling, split_seed)
    rng = np.random.default_rng(split_seed)
    stratify_by = list(stratify_by) if stratify_by else list(DEFAULT_STRATIFY_BY)
    bad = [s for s in stratify_by if s not in FEATURE_NAMES]
    if bad:
        raise ValueError(f"Unknown stratify_by feature(s) {bad}. Options: {FEATURE_NAMES}")

    features = None
    if split_method == "contiguous":
        tr, va, te = _cut(np.arange(len(pool)), train_frac, val_frac)
    elif split_method == "random":
        tr, va, te = _cut(rng.permutation(len(pool)), train_frac, val_frac)
    elif split_method == "stratified":
        print(f"Computing stratification features for {len(pool)} molecules...")
        features = compute_features(full_dataset, pool)
        fm = np.stack([features[s] for s in stratify_by], axis=1)
        tr, va, te = stratified_assignment(fm, rng, train_frac, val_frac, min_stratum_size)
    else:
        raise ValueError("split_method must be 'contiguous', 'random' or 'stratified'")

    if features is None and compute_report_features:
        features = compute_features(full_dataset, pool)

    indices = {"train": np.sort(pool[tr]).tolist(),
               "val": np.sort(pool[va]).tolist(),
               "test": np.sort(pool[te]).tolist()}
    allidx = indices["train"] + indices["val"] + indices["test"]
    assert len(allidx) == len(set(allidx)) == len(pool), "splits must be disjoint and cover the pool"
    return indices, {"pool": pool, "features": features}



# Loading molecules and reporting
class LazySplit:
    """Sequence of QM9 molecules for a list of global indices. Molecules are
    fetched on access (fresh Data object each time) and carry `.qm9_index`,
    their true index in the full QM9 -- so a 130k split isn't held in RAM
    unless you actually touch it."""

    def __init__(self, full_dataset, indices):
        self.ds = full_dataset
        self.idx = [int(i) for i in indices]

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        gi = self.idx[i]
        d = self.ds[gi]
        d.qm9_index = gi
        return d


def load_split_data(full_dataset, indices):
    """Materialised list of Data objects (used by training)."""
    return list(LazySplit(full_dataset, indices))


def print_split_balance(indices, info, stratify_by=None):
    """Percentage of each split falling in each value of each feature."""
    feats = info.get("features")
    if feats is None:
        print("(no features available for a balance report)")
        return
    pos = {int(g): p for p, g in enumerate(info["pool"])}
    cols = {s: np.array([pos[g] for g in indices[s]]) for s in ("train", "val", "test")}
    stratified = set(stratify_by or [])
    print(f"\nSplit sizes: train={len(indices['train'])}  val={len(indices['val'])}  "
          f"test={len(indices['test'])}")
    print("Feature balance (% of each split):")
    for name in FEATURE_NAMES:
        mark = "  [stratified]" if name in stratified else ""
        print(f"  {name}{mark}")
        for v in sorted(set(feats[name].tolist())):
            row = [100.0 * np.mean(feats[name][cols[s]] == v) if len(cols[s]) else 0.0
                   for s in ("train", "val", "test")]
            if max(row) < 0.05:
                continue
            print(f"    {v:>3}:  train {row[0]:6.2f}   val {row[1]:6.2f}   test {row[2]:6.2f}")



# Standalone: compare split methods on your own data
CONFIG = {
    "root": "./data/QM9",
    "subset_size": 10000,          # None -> full QM9
    "subset_sampling": "random",   # "first" or "random"
    "split_seed": 42,
    "stratify_by": DEFAULT_STRATIFY_BY,
    "min_stratum_size": 10,
    "methods_to_compare": ["contiguous", "random", "stratified"],
}


def main():
    from torch_geometric.datasets import QM9
    full = QM9(root=CONFIG["root"])
    for method in CONFIG["methods_to_compare"]:
        print("\n" + "=" * 70 + f"\nsplit_method = {method}\n" + "=" * 70)
        idx, info = resolve_splits(
            full, subset_size=CONFIG["subset_size"],
            subset_sampling=CONFIG["subset_sampling"], split_method=method,
            split_seed=CONFIG["split_seed"], stratify_by=CONFIG["stratify_by"],
            min_stratum_size=CONFIG["min_stratum_size"])
        print_split_balance(idx, info, CONFIG["stratify_by"])


if __name__ == "__main__":
    main()
