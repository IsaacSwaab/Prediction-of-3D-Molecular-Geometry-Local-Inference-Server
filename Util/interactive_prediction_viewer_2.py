"""
Interactive 3D viewer: predicted vs. actual QM9 geometry.

Opens each molecule as an interactive Plotly figure in your browser:
  - LEFT:   actual (DFT) geometry
  - MIDDLE: predicted geometry (Kabsch-aligned, optionally coloured by
            per-atom error)
  - RIGHT:  overlay -- actual atoms, predicted atoms, and a red line from
            every predicted atom to where it should be
All three panels share one camera: rotate/zoom/pan one and the others follow.

Mouse controls (Plotly 3D):
  - Left-drag   : rotate (orbit)
  - Scroll      : zoom
  - Right-drag  : pan
  - Hover       : atom index, element, position error (A)
  - Legend click: hide/show an element (double-click isolates it)
  - Camera-reset button in the top-right toolbar of the figure

SETUP
  pip install plotly
  Put this file in the SAME FOLDER as your training script and make sure the
  import below matches that script's filename (without .py).

Usage: edit CONFIG, then run this file directly.

This file is also imported by rank_predictions.py (the best/worst finder),
so keep the two files together.
"""

import os
import random
import webbrowser

import numpy as np
import torch
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from torch_geometric.datasets import QM9

# vvv EDIT THIS if your training script has a different module name vvv
from Util.egnn_qm9 import (
    EGNN, kabsch_align, bond_length_mae, bond_angle_loss, bond_angle_mae_degrees,
    symmetry_break_loss, add_noise_initial_coords,
)


CONFIG = {
    "checkpoint_path": "./egnn_checkpoint.pt",
    "output_dir": "./interactive_plots",

    # Which molecules to show. Either/both of:
    #   "picks":  explicit list of (split, index_within_split), where split is
    #             "train", "val" or "test", e.g. [("val", 12), ("test", 3)]
    #   "random": how many random molecules to draw from each split
    "picks": [],
    "random": {"train": 2, "val": 2, "test": 2},
    "seed": 900119,

    # K noise draws averaged into the "consensus" prediction. Training's
    # validation loop uses num_pos_init_draws (8), so default to that.
    # Set to None to use the value stored in the checkpoint's config.
    "num_pos_init_draws_eval": None,

    # Metrics (RMSD, bond/angle MAE, ...) always use the K-draw consensus, but
    # averaging blurs symmetric hydrogens together. So the 3D view shows ONE
    # real draw instead. Which one?
    #   "first" -> draw 0 (a typical, unselected draw)
    #   "best"  -> the draw with the lowest RMSD to the truth
    #   an int  -> that specific draw index (0 .. K-1)
    "visual_draw": "first",

    "color_predicted_by_error": True,   # colour predicted atoms by position error
    "open_in_browser": True,            # open each HTML file automatically
    "device": None,                     # None -> cuda if available, else cpu

    # Only used if the checkpoint has no attached config (older format).
    "fallback_train_config": {
        "root": "./data/QM9", "subset_size": 10000,
        "hidden_dim": 64, "num_layers": 8, "edge_attr_dim": 4,
        "max_disp": 5.0, "num_rbf": 16, "radius_cutoff": 6.0,
        "use_radius_graph": False, "use_angular": True, "use_attention": True,
        "noise_scale": 5, "num_pos_init_draws": 8, "bond_loss_weight": 0.5,
        "angle_loss_weight": 1.0, "symmetry_break_weight": 2.0,
        "symmetry_break_min_dist": 1.0,
    },
}

Z_TO_SYMBOL = {1: "H", 6: "C", 7: "N", 8: "O", 9: "F"}
ELEMENT_COLORS = {"H": "#d9d9d9", "C": "#303030", "N": "#3050F8", "O": "#FF0D0D", "F": "#2FB52F"}
ELEMENT_SIZES = {"H": 5, "C": 9, "N": 9, "O": 9, "F": 8}


# ----------------------------------------------------------------------
# Loading: checkpoint, model, dataset splits
# ----------------------------------------------------------------------

def load_checkpoint(checkpoint_path, fallback_config):
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        return ckpt["model_state_dict"], ckpt.get("config", fallback_config)
    print("WARNING: checkpoint has no attached config (older format). Using "
          "CONFIG['fallback_train_config'] -- make sure it matches training.")
    return ckpt, fallback_config


def build_model(train_config, state_dict, device):
    # A missing B.1/B.2/B.4 key means the checkpoint predates that feature,
    # so it was trained WITHOUT it -> default to False (not True).
    model = EGNN(
        num_layers=train_config["num_layers"], hidden_dim=train_config["hidden_dim"],
        edge_attr_dim=train_config.get("edge_attr_dim", 4),
        max_disp=train_config["max_disp"], num_rbf=train_config["num_rbf"],
        radius_cutoff=train_config["radius_cutoff"],
        use_radius_graph=train_config["use_radius_graph"],
        use_angular=train_config["use_angular"], use_attention=train_config["use_attention"],
        use_edge_conditioned_phi_x=train_config.get("use_edge_conditioned_phi_x", False),
        use_split_attention=train_config.get("use_split_attention", False),
        use_edge_embeddings=train_config.get("use_edge_embeddings", False),
    )
    model.load_state_dict(state_dict)
    return model.to(device).eval()


class LazySplit:
    """Sequence of QM9 molecules for a list of global QM9 indices. Molecules
    are fetched on access, so a split of ~100k molecules is not loaded into
    memory unless you actually touch it. Each molecule carries `.qm9_index`,
    its true index in the full QM9 dataset."""

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


def load_splits(train_config):
    """
    Reproduce the split the checkpoint was trained with.

    - New checkpoints store the exact train/val/test index lists
      (config["split_indices"]) -- these are used directly, so any split
      method (random / stratified / full QM9) is reproduced exactly.
    - Older checkpoints have no stored indices: fall back to the legacy
      rule (first subset_size molecules, 80/10/10 in order, no shuffling).
    """
    full = QM9(root=train_config["root"])
    stored = train_config.get("split_indices")
    if stored is not None:
        indices = {k: torch.as_tensor(stored[k]).tolist() for k in ("train", "val", "test")}
    else:
        n = len(full) if train_config["subset_size"] is None else min(train_config["subset_size"], len(full))
        n_train, n_val = int(0.8 * n), int(0.1 * n)
        indices = {"train": range(0, n_train),
                   "val": range(n_train, n_train + n_val),
                   "test": range(n_train + n_val, n)}
    return {k: LazySplit(full, v) for k, v in indices.items()}


# ----------------------------------------------------------------------
# Evaluation of one molecule
# ----------------------------------------------------------------------

@torch.no_grad()
def evaluate_molecule(model, data, train_config, K, device, seed=None, visual_draw="first"):
    """
    K noise draws -> Kabsch-align each to the truth -> average. Same
    consensus procedure as train()'s validation loop.

    `seed` makes the noise draws reproducible per molecule, so rankings
    and plots come out identical from run to run.

    METRICS (rmsd, bond_mae, angle_mae_deg, sym_break, training_loss) come
    from the K-draw consensus, exactly as in training validation. The
    geometry returned for DISPLAY ("pred_pos") is a single draw chosen by
    `visual_draw`, so symmetric hydrogens are not averaged together. The
    shown draw's own metrics are returned as shown_*; the mean over all
    single draws as single_draw_*_mean.

    Returns a dict with predicted coords, centred true coords, and metrics.
    """
    if seed is not None:
        torch.manual_seed(seed)

    data = data.to(device)
    z, pos = data.z, data.pos
    edge_index, edge_attr = data.edge_index, data.edge_attr
    batch = torch.zeros(z.size(0), dtype=torch.long, device=device)

    aligned_list, target_c = [], None
    for _ in range(K):
        pos_init = add_noise_initial_coords(pos, scale=train_config["noise_scale"])
        pred = model(z, pos_init, edge_index, edge_attr, batch=batch)
        aligned, target_c = kabsch_align(pred, pos, batch, num_graphs=1)
        aligned_list.append(aligned)

    stack = torch.stack(aligned_list, dim=0)            # [K, N, 3]
    aligned_mean = stack.mean(dim=0)

    def rmsd_of(p):
        return ((p - target_c) ** 2).sum(dim=-1).mean().sqrt().item()

    rmsd = rmsd_of(aligned_mean)
    bond_mae = bond_length_mae(aligned_mean, edge_index, pos).item()
    angle_mae = bond_angle_mae_degrees(aligned_mean, pos, edge_index)
    angle_loss = bond_angle_loss(aligned_mean, pos, edge_index).item()
    sym_break = symmetry_break_loss(
        aligned_mean, edge_index,
        min_dist=train_config.get("symmetry_break_min_dist", 1.0)).item()

    training_loss = (rmsd
                     + train_config["bond_loss_weight"] * bond_mae
                     + train_config.get("angle_loss_weight", 0.0) * angle_loss
                     + train_config.get("symmetry_break_weight", 0.0) * sym_break)

    # Per-draw metrics (no averaging, so no blurring of symmetric atoms)
    sb_min = train_config.get("symmetry_break_min_dist", 1.0)
    draw_rmsds, draw_bond, draw_angle, draw_sym = [], [], [], []
    for a in aligned_list:
        draw_rmsds.append(rmsd_of(a))
        draw_bond.append(bond_length_mae(a, edge_index, pos).item())
        draw_angle.append(bond_angle_mae_degrees(a, pos, edge_index))
        draw_sym.append(symmetry_break_loss(a, edge_index, min_dist=sb_min).item())

    # How much do the K draws disagree with each other? (high = unstable)
    if K > 1:
        pair = [((stack[i] - stack[j]) ** 2).sum(-1).mean().sqrt().item()
                for i in range(K) for j in range(i + 1, K)]
        draw_spread = float(np.mean(pair))
    else:
        draw_spread = 0.0

    # Choose the single draw to display
    if visual_draw == "first":
        k_show = 0
    elif visual_draw == "best":
        k_show = int(np.argmin(draw_rmsds))
    elif isinstance(visual_draw, int):
        k_show = visual_draw % K
    else:
        raise ValueError("visual_draw must be 'first', 'best', or an int draw index")
    shown = stack[k_show]
    per_atom_err = (shown - target_c).norm(dim=-1)

    return {
        "pred_pos": shown.cpu(), "target_pos": target_c.cpu(),
        "shown_draw": k_show, "num_draws": K,
        "shown_rmsd": draw_rmsds[k_show], "shown_bond_mae": draw_bond[k_show],
        "shown_angle_mae_deg": draw_angle[k_show],
        "per_atom_err": per_atom_err.cpu(),
        "rmsd": rmsd, "bond_mae": bond_mae, "angle_mae_deg": angle_mae,
        "sym_break": sym_break, "training_loss": training_loss,
        "single_draw_rmsd_mean": float(np.mean(draw_rmsds)),
        "single_draw_bond_mae_mean": float(np.mean(draw_bond)),
        "single_draw_angle_mae_deg_mean": float(np.mean(draw_angle)),
        "single_draw_sym_break_mean": float(np.mean(draw_sym)),
        "draw_spread": draw_spread,
        "num_atoms": int(z.size(0)),
    }


# ----------------------------------------------------------------------
# Plotly figure construction
# ----------------------------------------------------------------------

def _unique_bonds(edge_index):
    seen = set()
    for s, d in zip(edge_index[0].tolist(), edge_index[1].tolist()):
        seen.add((min(s, d), max(s, d)))
    return sorted(seen)


def _bond_trace(pos, bonds, color="rgba(120,120,120,0.9)", width=5, dash=None, name="bonds"):
    xs, ys, zs = [], [], []
    for a, b in bonds:
        xs += [pos[a, 0], pos[b, 0], None]
        ys += [pos[a, 1], pos[b, 1], None]
        zs += [pos[a, 2], pos[b, 2], None]
    return go.Scatter3d(x=xs, y=ys, z=zs, mode="lines", name=name, showlegend=False,
                        line=dict(color=color, width=width, dash=dash) if dash
                        else dict(color=color, width=width),
                        hoverinfo="skip")


def _atom_traces(pos, z, label, err=None, legend_group=None, show_legend=True,
                 color_by_error=False, err_max=None, show_colorbar=False):
    """One trace per element (so the legend can toggle elements), or a single
    error-coloured trace if color_by_error is set."""
    pos = np.asarray(pos)
    z = np.asarray(z)
    symbols = np.array([Z_TO_SYMBOL.get(int(zi), "?") for zi in z])
    traces = []

    def hover(idx):
        out = []
        for i in idx:
            t = f"{label} atom {i} ({symbols[i]})"
            if err is not None:
                t += f"<br>position error: {err[i]:.3f} A"
            out.append(t)
        return out

    if color_by_error and err is not None:
        idx = np.arange(len(z))
        traces.append(go.Scatter3d(
            x=pos[:, 0], y=pos[:, 1], z=pos[:, 2], mode="markers+text",
            text=[s if s != "H" else "" for s in symbols], textposition="top center",
            textfont=dict(size=9),
            name=label, showlegend=False, hovertext=hover(idx), hoverinfo="text",
            marker=dict(size=[ELEMENT_SIZES.get(s, 8) for s in symbols], color=err,
                        colorscale="Turbo", cmin=0, cmax=err_max, opacity=1.0,
                        line=dict(color="#222", width=1),
                        colorbar=dict(title="Position<br>error (A)", len=0.6, x=0.66, thickness=12)
                        if show_colorbar else None,
                        showscale=show_colorbar)))
        return traces

    for sym in sorted(set(symbols)):
        idx = np.where(symbols == sym)[0]
        traces.append(go.Scatter3d(
            x=pos[idx, 0], y=pos[idx, 1], z=pos[idx, 2], mode="markers",
            name=sym, legendgroup=sym + (legend_group or ""), showlegend=show_legend,
            hovertext=hover(idx), hoverinfo="text",
            marker=dict(size=ELEMENT_SIZES.get(sym, 8) * 1.0,
                        color=ELEMENT_COLORS.get(sym, "purple"),
                        line=dict(color="#222", width=1))))
    return traces


_SYNC_JS = """
var gd = document.getElementById('{plot_id}');
var scenes = ['scene', 'scene2', 'scene3'];
var syncing = false;
gd.on('plotly_relayout', function(ed) {
  if (syncing) return;
  var src = null;
  scenes.forEach(function(s) { if (ed[s + '.camera']) src = s; });
  if (!src) return;
  var upd = {};
  scenes.forEach(function(s) { if (s !== src) upd[s + '.camera'] = ed[src + '.camera']; });
  syncing = true;
  Plotly.relayout(gd, upd).then(function() { syncing = false; });
});
"""


def build_figure(result, data, title, color_by_error=True):
    pos_t = result["target_pos"].numpy()
    pos_p = result["pred_pos"].numpy()
    err = result["per_atom_err"].numpy()
    z = data.z.cpu().numpy()
    bonds = _unique_bonds(data.edge_index.cpu())
    err_max = max(float(err.max()), 1e-3)

    fig = make_subplots(
        rows=1, cols=3,
        specs=[[{"type": "scene"}, {"type": "scene"}, {"type": "scene"}]],
        subplot_titles=("Actual (DFT)",
                        f"Predicted (draw {result['shown_draw']} of {result['num_draws']})",
                        "Overlay + error vectors"),
        horizontal_spacing=0.01,
    )

    # Panel 1: actual
    fig.add_trace(_bond_trace(pos_t, bonds), row=1, col=1)
    for t in _atom_traces(pos_t, z, "Actual", legend_group="a"):
        fig.add_trace(t, row=1, col=1)

    # Panel 2: predicted
    fig.add_trace(_bond_trace(pos_p, bonds), row=1, col=2)
    for t in _atom_traces(pos_p, z, "Predicted", err=err, legend_group="p", show_legend=False,
                          color_by_error=color_by_error, err_max=err_max,
                          show_colorbar=color_by_error):
        fig.add_trace(t, row=1, col=2)

    # Panel 3: overlay -- true bonds grey, predicted bonds orange dashed,
    # error vectors red, true atoms by element, predicted atoms orange diamonds.
    fig.add_trace(_bond_trace(pos_t, bonds, color="rgba(120,120,120,0.7)"), row=1, col=3)
    fig.add_trace(_bond_trace(pos_p, bonds, color="rgba(255,140,0,0.8)", width=4, dash="dash"),
                  row=1, col=3)
    xs, ys, zs = [], [], []
    for i in range(len(z)):
        xs += [pos_t[i, 0], pos_p[i, 0], None]
        ys += [pos_t[i, 1], pos_p[i, 1], None]
        zs += [pos_t[i, 2], pos_p[i, 2], None]
    fig.add_trace(go.Scatter3d(x=xs, y=ys, z=zs, mode="lines", showlegend=False,
                               line=dict(color="red", width=3), hoverinfo="skip"),
                  row=1, col=3)
    for t in _atom_traces(pos_t, z, "Actual", legend_group="a", show_legend=False):
        fig.add_trace(t, row=1, col=3)
    fig.add_trace(go.Scatter3d(
        x=pos_p[:, 0], y=pos_p[:, 1], z=pos_p[:, 2], mode="markers", showlegend=False,
        hovertext=[f"Predicted atom {i} ({Z_TO_SYMBOL.get(int(z[i]), '?')})<br>"
                   f"position error: {err[i]:.3f} A" for i in range(len(z))],
        hoverinfo="text",
        marker=dict(size=6, color="orange", symbol="diamond", opacity=0.9,
                    line=dict(color="#7a4300", width=1))),
        row=1, col=3)

    axis = dict(visible=False)
    scene_kw = dict(xaxis=axis, yaxis=axis, zaxis=axis, aspectmode="data",
                    camera=dict(eye=dict(x=1.5, y=1.5, z=1.0)))
    fig.update_layout(
        scene=scene_kw, scene2=scene_kw, scene3=scene_kw,
        template="plotly_white",
        title=dict(
            text=(f"{title}  |  {result['num_atoms']} atoms<br>"
                  f"<sub>CONSENSUS of {result['num_draws']} draws: "
                  f"RMSD {result['rmsd']:.4f} A  |  bond MAE {result['bond_mae']:.4f} A  |  "
                  f"angle MAE {result['angle_mae_deg']:.1f} deg  |  "
                  f"training loss {result['training_loss']:.4f}</sub><br>"
                  f"<sub>SHOWN draw {result['shown_draw']}: "
                  f"RMSD {result['shown_rmsd']:.4f} A  |  bond MAE {result['shown_bond_mae']:.4f} A  |  "
                  f"angle MAE {result['shown_angle_mae_deg']:.1f} deg  |  "
                  f"max atom error {err.max():.3f} A</sub>"),
            x=0.5, xanchor="center", y=0.985, yanchor="top"),
        legend=dict(title="Element (click to toggle)", orientation="h", y=-0.02, x=0.5,
                    xanchor="center"),
        margin=dict(l=0, r=0, t=170, b=0),
        height=760,
    )
    fig.update_annotations(font_size=13)   # panel titles
    return fig


def save_figure(fig, path, open_in_browser=False):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.write_html(path, include_plotlyjs="cdn", post_script=_SYNC_JS,
                   config={"scrollZoom": True, "displaylogo": False})
    if open_in_browser:
        webbrowser.open("file://" + os.path.abspath(path))
    return path


def render_molecule(model, data, name, train_config, K, device, output_dir,
                    color_by_error=True, open_in_browser=False, seed=None,
                    visual_draw="first"):
    """Evaluate one molecule, write its interactive HTML, return (result, path)."""
    result = evaluate_molecule(model, data, train_config, K, device, seed=seed,
                               visual_draw=visual_draw)
    fig = build_figure(result, data, name, color_by_error=color_by_error)
    path = save_figure(fig, os.path.join(output_dir, f"{name}.html"), open_in_browser)
    return result, path


# ----------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------

def main():
    random.seed(CONFIG["seed"])
    device = CONFIG["device"] or ("cuda" if torch.cuda.is_available() else "cpu")

    state_dict, train_config = load_checkpoint(CONFIG["checkpoint_path"],
                                               CONFIG["fallback_train_config"])
    model = build_model(train_config, state_dict, device)
    K = CONFIG["num_pos_init_draws_eval"] or train_config.get("num_pos_init_draws", 8)
    print(f"Loaded checkpoint ({sum(p.numel() for p in model.parameters()):,} parameters). "
          f"Using K={K} noise draws per molecule.")

    splits = load_splits(train_config)
    print({k: len(v) for k, v in splits.items()}, "molecules per split")

    picks = list(CONFIG["picks"])
    for split, n in CONFIG["random"].items():
        pool = splits[split]
        for idx in random.sample(range(len(pool)), min(n, len(pool))):
            picks.append((split, idx))

    for split, idx in picks:
        data = splits[split][idx]
        name = f"{split}_{idx}_qm9idx{data.qm9_index}"
        result, path = render_molecule(
            model, data, name, train_config, K, device, CONFIG["output_dir"],
            color_by_error=CONFIG["color_predicted_by_error"],
            open_in_browser=CONFIG["open_in_browser"],
            seed=CONFIG["seed"] + data.qm9_index,
            visual_draw=CONFIG["visual_draw"])
        print(f"  {name}: rmsd={result['rmsd']:.4f}  bond_mae={result['bond_mae']:.4f}  "
              f"angle_mae={result['angle_mae_deg']:.1f}deg  -> {path}")

    print(f"\nDone. Interactive files are in {CONFIG['output_dir']}/")


if __name__ == "__main__":
    main()
