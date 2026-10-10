"""
Find the most and least accurate predictions of a trained EGNN on QM9.

What it does
  1. Rebuilds the exact train/val/test split used in training.
  2. Runs the model on EVERY molecule in the splits you choose, using the
     same K-draw consensus + Kabsch alignment as training's validation loop
     (noise draws are seeded per molecule, so results are reproducible).
  3. Records per-molecule metrics: Kabsch RMSD, bond-length MAE, bond-angle
     MAE (deg), symmetry-break penalty, combined training loss, and the
     spread between the K draws.
  4. Prints the best N and worst N molecules, saves a full CSV, saves
     summary plots, and exports an interactive 3D HTML for each best/worst
     molecule (plus an index.html linking them all).

Needs interactive_prediction_viewer.py in the same folder (it provides the
model loading, evaluation and 3D figure code), plus: pip install plotly

Usage: edit CONFIG, then run this file directly.

A note on reading the rankings: RMSD grows with molecule size, so "worst by
RMSD" will lean toward larger molecules. Check the num_atoms column and the
RMSD-vs-size plot, or rank by bond_mae / angle_mae_deg instead, which are
less size-dependent.
"""

import csv
import html
import os
import time

import numpy as np
import torch

from Util.interactive_prediction_viewer_2 import (
    CONFIG as VIEWER_CONFIG,
    load_checkpoint, build_model, load_splits, evaluate_molecule, render_molecule,
)


CONFIG = {
    "checkpoint_path": "./egnn_checkpoint.pt",
    "output_dir": "./ranking_results_radius_chem_edge_trip",

    # Which splits to score: any of "train", "val", "test".
    # "train" is 8x bigger than the others, so it takes proportionally longer.
    "splits": ["train","val", "test"],

    # Metrics to print best/worst tables for. The FIRST one decides which
    # molecules get exported as interactive HTML.
    # Options: rmsd, bond_mae, angle_mae_deg, training_loss, sym_break, draw_spread,
    # plus un-averaged versions: single_draw_rmsd_mean, single_draw_bond_mae_mean,
    # single_draw_angle_mae_deg_mean, single_draw_sym_break_mean
    "rank_by": ["rmsd", "bond_mae", "angle_mae_deg"],
    "top_n": 10,

    "export_html": True,           # interactive 3D file per best/worst molecule
    "color_predicted_by_error": True,
    "visual_draw": "best",        # which single draw the 3D files show: "first", "best", or an int
    "save_plots": True,            # summary PNGs (needs matplotlib)

    "num_pos_init_draws_eval": None,   # None -> value stored in the checkpoint (8)
    "seed": 900119099,
    "device": None,                    # None -> cuda if available
    "max_molecules_per_split": 200,   # e.g. 200 for a quick test run
}

LOWER_IS_BETTER = ("rmsd", "bond_mae", "angle_mae_deg", "training_loss", "sym_break", "draw_spread")
CSV_COLUMNS = ["split", "split_index", "qm9_index", "name", "smiles", "num_atoms",
               "rmsd", "bond_mae", "angle_mae_deg", "sym_break", "training_loss",
               "single_draw_rmsd_mean", "single_draw_bond_mae_mean",
               "single_draw_angle_mae_deg_mean", "single_draw_sym_break_mean", "draw_spread"]


# Scoring
def score_split(model, molecules, split, train_config, K, device, seed, limit=None):
    rows = []
    n = len(molecules) if limit is None else min(limit, len(molecules))
    t0 = time.time()
    for i in range(n):
        data = molecules[i]
        r = evaluate_molecule(model, data, train_config, K, device, seed=seed + data.qm9_index)
        rows.append({
            "split": split, "split_index": i, "qm9_index": data.qm9_index,
            "name": getattr(data, "name", ""),
            "smiles": getattr(data, "smiles", ""),
            "num_atoms": r["num_atoms"],
            **{k: r[k] for k in ("rmsd", "bond_mae", "angle_mae_deg", "sym_break",
                                 "training_loss", "single_draw_rmsd_mean",
                                 "single_draw_bond_mae_mean", "single_draw_angle_mae_deg_mean",
                                 "single_draw_sym_break_mean", "draw_spread")},
        })
        if (i + 1) % 100 == 0 or i + 1 == n:
            print(f"  [{split}] {i + 1}/{n} molecules scored  ({time.time() - t0:.0f}s)")
    return rows


def save_csv(rows, path):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{r[k]:.6f}" if isinstance(r[k], float) else r[k])
                        for k in CSV_COLUMNS})


def print_table(title, rows):
    print(f"\n{title}")
    print(f"  {'split':<5} {'idx':>5} {'qm9':>6} {'atoms':>5} {'rmsd':>8} {'bondMAE':>8} "
          f"{'angMAE':>7} {'symbrk':>7} {'spread':>7}  smiles")
    for r in rows:
        print(f"  {r['split']:<5} {r['split_index']:>5} {r['qm9_index']:>6} {r['num_atoms']:>5} "
              f"{r['rmsd']:>8.4f} {r['bond_mae']:>8.4f} {r['angle_mae_deg']:>7.1f} "
              f"{r['sym_break']:>7.4f} {r['draw_spread']:>7.4f}  {r['smiles']}")


def print_summary(rows):
    print("\nOverall summary")
    for m in ("rmsd", "bond_mae", "angle_mae_deg", "sym_break", "draw_spread",
              "single_draw_bond_mae_mean", "single_draw_angle_mae_deg_mean",
              "single_draw_sym_break_mean"):
        v = np.array([r[m] for r in rows])
        print(f"  {m:<32} mean={v.mean():.4f}  median={np.median(v):.4f}  "
              f"p90={np.percentile(v, 90):.4f}  p99={np.percentile(v, 99):.4f}  max={v.max():.4f}")
    n_collapse = sum(r["sym_break"] > 0 for r in rows)
    print(f"  molecules with any symmetry-break penalty (possible collapsed atoms): "
          f"{n_collapse}/{len(rows)}")


def save_plots(rows, out_dir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed -- skipping summary plots.")
        return
    rmsd = np.array([r["rmsd"] for r in rows])
    atoms = np.array([r["num_atoms"] for r in rows])
    split = np.array([r["split"] for r in rows])

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    axes[0].hist(rmsd, bins=50, color="steelblue", edgecolor="white")
    axes[0].axvline(np.median(rmsd), color="k", ls="--", label=f"median {np.median(rmsd):.3f}")
    axes[0].set(xlabel="Kabsch RMSD (A)", ylabel="molecules", title="RMSD distribution")
    axes[0].legend()

    for s in sorted(set(split)):
        m = split == s
        axes[1].scatter(atoms[m], rmsd[m], s=10, alpha=0.5, label=s)
    axes[1].set(xlabel="number of atoms", ylabel="Kabsch RMSD (A)", title="RMSD vs molecule size")
    axes[1].legend()

    sym = np.array([r["sym_break"] for r in rows]) > 0
    axes[2].scatter([r["bond_mae"] for r in rows], [r["angle_mae_deg"] for r in rows],
                    c=np.where(sym, "crimson", "steelblue"), s=10, alpha=0.5)
    axes[2].set(xlabel="bond length MAE (A)", ylabel="bond angle MAE (deg)",
                title="Bond vs angle error (red = symmetry-break penalty > 0)")
    plt.tight_layout()
    path = os.path.join(out_dir, "summary_plots.png")
    plt.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}")


# ----------------------------------------------------------------------
# HTML export
# ----------------------------------------------------------------------

def export_html(model, splits, selections, train_config, K, device, out_dir,
                color_by_error, seed, visual_draw="first"):
    """selections: list of (label, rank, row). Returns index entries."""
    html_dir = os.path.join(out_dir, "molecules")
    entries = []
    for label, rank, row in selections:
        data = splits[row["split"]][row["split_index"]]
        name = f"{label}_{rank:02d}_{row['split']}_{row['split_index']}_qm9idx{row['qm9_index']}"
        _, path = render_molecule(model, data, name, train_config, K, device, html_dir,
                                  color_by_error=color_by_error, open_in_browser=False,
                                  seed=seed + data.qm9_index, visual_draw=visual_draw)
        entries.append((label, rank, row, os.path.relpath(path, out_dir)))
    return entries


def write_index(entries, metric, out_dir):
    parts = ["<html><head><meta charset='utf-8'><title>Prediction ranking</title>",
             "<style>body{font-family:sans-serif;margin:2em}table{border-collapse:collapse}"
             "td,th{border:1px solid #ccc;padding:4px 10px;text-align:right}"
             "td:nth-child(1),td:nth-child(2),td:last-child{text-align:left}</style></head><body>",
             f"<h2>Best and worst predictions (ranked by {html.escape(metric)})</h2>",
             "<table><tr><th>group</th><th>rank</th><th>split/idx</th><th>atoms</th>"
             "<th>RMSD</th><th>bond MAE</th><th>angle MAE</th><th>sym-break</th><th>SMILES</th>"
             "<th>3D view</th></tr>"]
    for label, rank, r, rel in entries:
        parts.append(
            f"<tr><td>{label}</td><td>{rank}</td><td>{r['split']} {r['split_index']}</td>"
            f"<td>{r['num_atoms']}</td><td>{r['rmsd']:.4f}</td><td>{r['bond_mae']:.4f}</td>"
            f"<td>{r['angle_mae_deg']:.1f}</td><td>{r['sym_break']:.4f}</td>"
            f"<td>{html.escape(str(r['smiles']))}</td>"
            f"<td><a href='{html.escape(rel)}' target='_blank'>open</a></td></tr>")
    parts.append("</table></body></html>")
    path = os.path.join(out_dir, "index.html")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(parts))
    return path


# ----------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------

def main():
    device = CONFIG["device"] or ("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = CONFIG["output_dir"]
    os.makedirs(out_dir, exist_ok=True)

    state_dict, train_config = load_checkpoint(CONFIG["checkpoint_path"],
                                               VIEWER_CONFIG["fallback_train_config"])
    model = build_model(train_config, state_dict, device)
    K = CONFIG["num_pos_init_draws_eval"] or train_config.get("num_pos_init_draws", 8)
    print(f"Loaded checkpoint ({sum(p.numel() for p in model.parameters()):,} parameters), "
          f"device={device}, K={K} draws per molecule.")

    splits = load_splits(train_config)
    rows = []
    for s in CONFIG["splits"]:
        print(f"\nScoring {s} split ({len(splits[s])} molecules)...")
        rows += score_split(model, splits[s], s, train_config, K, device,
                            CONFIG["seed"], CONFIG["max_molecules_per_split"])

    csv_path = os.path.join(out_dir, "all_molecule_metrics.csv")
    save_csv(rows, csv_path)
    print(f"\nSaved per-molecule metrics for {len(rows)} molecules to {csv_path}")
    print_summary(rows)

    n = min(CONFIG["top_n"], len(rows))
    primary_selections = []
    for mi, metric in enumerate(CONFIG["rank_by"]):
        ranked = sorted(rows, key=lambda r: r[metric])      # ascending: best first
        best, worst = ranked[:n], ranked[::-1][:n]
        print_table(f"BEST {n} by {metric} (lowest)", best)
        print_table(f"WORST {n} by {metric} (highest)", worst)
        if mi == 0:
            primary_metric = metric
            primary_selections = ([("best", i + 1, r) for i, r in enumerate(best)] +
                                  [("worst", i + 1, r) for i, r in enumerate(worst)])

    if CONFIG["save_plots"]:
        save_plots(rows, out_dir)

    if CONFIG["export_html"]:
        print(f"\nExporting interactive 3D files for best/worst {n} by {primary_metric}...")
        entries = export_html(model, splits, primary_selections, train_config, K, device,
                              out_dir, CONFIG["color_predicted_by_error"], CONFIG["seed"],
                              CONFIG["visual_draw"])
        idx = write_index(entries, primary_metric, out_dir)
        print(f"Open {idx} to browse them.")

    print(f"\nDone. Everything is in {out_dir}/")


if __name__ == "__main__":
    main()
