"""
visualize.py
============

Reusable Plotly 3D visualization for EGNN predictions.

Extracted from interactive_prediction_viewer_2.py so the same plotting
code can be used both as a standalone script and embedded in the
inference server. This module has no dependency on the training script,
the QM9 dataset, or the model architecture -- it just takes coordinates
and produces interactive HTML.

Two modes:
  - build_prediction_figure(pred_pos, z, edge_index, ...)
        Single-panel: prediction only. Use when you don't have ground
        truth (arbitrary SMILES queries).
  - build_prediction_figure(..., target_pos=...)
        Three-panel: actual / predicted / overlay with error vectors.
        Use when ground truth is available (QM9 verification).
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


Z_TO_SYMBOL = {1: "H", 6: "C", 7: "N", 8: "O", 9: "F"}
ELEMENT_COLORS = {"H": "#d9d9d9", "C": "#303030", "N": "#3050F8",
                  "O": "#FF0D0D", "F": "#2FB52F"}
ELEMENT_SIZES = {"H": 5, "C": 9, "N": 9, "O": 9, "F": 8}


# ----------------------------------------------------------------------
# Low-level trace builders
# ----------------------------------------------------------------------

def _unique_bonds(edge_index):
    """Undirected unique bond list from a    edge_index."""
    seen = set()
    for s, d in zip(edge_index[0].tolist(), edge_index[1].tolist()):
        seen.add((min(s, d), max(s, d)))
    return sorted(seen)


def _bond_trace(pos, bonds, color="rgba(120,120,120,0.9)", width=5,
                dash=None, name="bonds"):
    xs, ys, zs = [], [], []
    for a, b in bonds:
        xs += [pos[a, 0], pos[b, 0], None]
        ys += [pos[a, 1], pos[b, 1], None]
        zs += [pos[a, 2], pos[b, 2], None]
    line = dict(color=color, width=width)
    if dash:
        line["dash"] = dash
    return go.Scatter3d(x=xs, y=ys, z=zs, mode="lines", name=name,
                        showlegend=False, line=line, hoverinfo="skip")


def _atom_traces(pos, z, label, err=None, legend_group=None, show_legend=True,
                 color_by_error=False, err_max=None, show_colorbar=False):
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
            text=[s if s != "H" else "" for s in symbols],
            textposition="top center", textfont=dict(size=9),
            name=label, showlegend=False, hovertext=hover(idx), hoverinfo="text",
            marker=dict(
                size=[ELEMENT_SIZES.get(s, 8) for s in symbols],
                color=err, colorscale="Turbo", cmin=0, cmax=err_max,
                opacity=1.0, line=dict(color="#222", width=1),
                colorbar=dict(title="Position<br>error (A)", len=0.6,
                              x=0.66, thickness=12)
                if show_colorbar else None,
                showscale=show_colorbar)))
        return traces

    for sym in sorted(set(symbols)):
        idx = np.where(symbols == sym)[0]
        traces.append(go.Scatter3d(
            x=pos[idx, 0], y=pos[idx, 1], z=pos[idx, 2], mode="markers",
            name=sym, legendgroup=sym + (legend_group or ""),
            showlegend=show_legend, hovertext=hover(idx), hoverinfo="text",
            marker=dict(size=ELEMENT_SIZES.get(sym, 8),
                        color=ELEMENT_COLORS.get(sym, "purple"),
                        line=dict(color="#222", width=1))))
    return traces


# Camera-sync between the three subplots -- drag one, the others follow.
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


# ----------------------------------------------------------------------
# Public: build a figure (1-panel or 3-panel)
# ----------------------------------------------------------------------

def build_prediction_figure(pred_pos, z, edge_index, title,
                            target_pos=None, per_atom_err=None,
                            color_by_error=True, shown_draw=None,
                            num_draws=None):
    """
    Build an interactive Plotly figure.

    If target_pos is None: single-panel prediction-only view.
    If target_pos is given: three-panel comparison (actual / predicted /
    overlay with error vectors), with per-atom error colouring.

    pred_pos, target_pos: (N, 3) arrays or tensors (numpy-convertible).
    z: (N,) atomic numbers.
    edge_index: (2, E) bond edges (COO format, both directions ok).
    """
    pred_pos = np.asarray(pred_pos)
    z = np.asarray(z)
    bonds = _unique_bonds(edge_index)
    axis = dict(visible=False)
    cam = dict(eye=dict(x=1.5, y=1.5, z=1.0))

    if target_pos is None:
        fig = go.Figure()
        fig.add_trace(_bond_trace(pred_pos, bonds))
        for t in _atom_traces(pred_pos, z, "Predicted"):
            fig.add_trace(t)
        fig.update_layout(
            scene=dict(xaxis=axis, yaxis=axis, zaxis=axis,
                       aspectmode="data", camera=cam),
            template="plotly_white",
            title=dict(text=title, x=0.5, xanchor="center"),
            legend=dict(title="Element (click to toggle)", orientation="h",
                        y=-0.02, x=0.5, xanchor="center"),
            margin=dict(l=0, r=0, t=80, b=0),
            height=720,
        )
        return fig

    # Three-panel comparison
    target_pos = np.asarray(target_pos)
    if per_atom_err is None:
        per_atom_err = np.linalg.norm(pred_pos - target_pos, axis=1)
    err = np.asarray(per_atom_err)
    err_max = max(float(err.max()), 1e-3)

    subplot_titles = [
        "Actual (DFT)",
        f"Predicted (draw {shown_draw} of {num_draws})"
        if shown_draw is not None else "Predicted",
        "Overlay + error vectors",
    ]
    fig = make_subplots(
        rows=1, cols=3,
        specs=[[{"type": "scene"}, {"type": "scene"}, {"type": "scene"}]],
        subplot_titles=subplot_titles,
        horizontal_spacing=0.01,
    )

    # Panel 1: actual
    fig.add_trace(_bond_trace(target_pos, bonds), row=1, col=1)
    for t in _atom_traces(target_pos, z, "Actual", legend_group="a"):
        fig.add_trace(t, row=1, col=1)

    # Panel 2: predicted (optionally colored by error)
    fig.add_trace(_bond_trace(pred_pos, bonds), row=1, col=2)
    for t in _atom_traces(pred_pos, z, "Predicted", err=err, legend_group="p",
                          show_legend=False, color_by_error=color_by_error,
                          err_max=err_max, show_colorbar=color_by_error):
        fig.add_trace(t, row=1, col=2)

    # Panel 3: overlay
    fig.add_trace(_bond_trace(target_pos, bonds, color="rgba(120,120,120,0.7)"),
                  row=1, col=3)
    fig.add_trace(_bond_trace(pred_pos, bonds, color="rgba(255,140,0,0.8)",
                              width=4, dash="dash"), row=1, col=3)
    xs, ys, zs = [], [], []
    for i in range(len(z)):
        xs += [target_pos[i, 0], pred_pos[i, 0], None]
        ys += [target_pos[i, 1], pred_pos[i, 1], None]
        zs += [target_pos[i, 2], pred_pos[i, 2], None]
    fig.add_trace(go.Scatter3d(x=xs, y=ys, z=zs, mode="lines", showlegend=False,
                               line=dict(color="red", width=3),
                               hoverinfo="skip"), row=1, col=3)
    for t in _atom_traces(target_pos, z, "Actual", legend_group="a",
                          show_legend=False):
        fig.add_trace(t, row=1, col=3)
    fig.add_trace(go.Scatter3d(
        x=pred_pos[:, 0], y=pred_pos[:, 1], z=pred_pos[:, 2],
        mode="markers", showlegend=False,
        hovertext=[f"Predicted atom {i} ({Z_TO_SYMBOL.get(int(z[i]), '?')})<br>"
                   f"position error: {err[i]:.3f} A" for i in range(len(z))],
        hoverinfo="text",
        marker=dict(size=6, color="orange", symbol="diamond", opacity=0.9,
                    line=dict(color="#7a4300", width=1))), row=1, col=3)

    scene_kw = dict(xaxis=axis, yaxis=axis, zaxis=axis,
                    aspectmode="data", camera=cam)
    fig.update_layout(
        scene=scene_kw, scene2=scene_kw, scene3=scene_kw,
        template="plotly_white",
        title=dict(text=title, x=0.5, xanchor="center", y=0.985, yanchor="top"),
        legend=dict(title="Element (click to toggle)", orientation="h",
                    y=-0.02, x=0.5, xanchor="center"),
        margin=dict(l=0, r=0, t=170, b=0),
        height=760,
    )
    fig.update_annotations(font_size=13)
    return fig


def figure_to_html(fig, embed_plotlyjs="cdn"):
    """Serialize a figure to standalone HTML with camera-sync script."""
    return fig.to_html(
        include_plotlyjs=embed_plotlyjs,
        full_html=True,
        post_script=_SYNC_JS,
        config={"scrollZoom": True, "displaylogo": False},
    )