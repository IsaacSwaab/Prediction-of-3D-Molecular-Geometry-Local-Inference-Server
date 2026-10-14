"""
server.py
=========

Local FastAPI inference server for the trained EGNN model, with
interactive 3D visualization and best-of-K reranking.

Usage:
    python server.py

Endpoints:
    GET  /                              Web form for quick queries
    GET  /docs                          Auto-generated API docs (Swagger UI)
    GET  /health                        Load balancer probe
    GET  /info                          Model config summary
    POST /predict                       JSON prediction (coordinates + XYZ)
    POST /predict_visual                Interactive 3D HTML (any SMILES)
    POST /predict_draws                 JSON: all K draws with quality scores
    GET  /predict_visual_qm9/{index}    Three-panel comparison vs QM9 truth
                                        (requires EGNN_ENABLE_QM9=1)

Environment variables (all optional):
    EGNN_CHECKPOINT    path to checkpoint (default: ./egnn_checkpoint.pt)
    EGNN_DEVICE        'cuda' or 'cpu' (default: auto-detect)
    EGNN_HOST          bind address (default: 127.0.0.1, localhost only)
                       set to 0.0.0.0 to allow LAN access
    EGNN_PORT          port (default: 8000)
    EGNN_ENABLE_QM9    set to '1' to load QM9 (needed for /predict_visual_qm9)
    EGNN_QM9_ROOT      path to QM9 data (default: ./data/QM9)
"""

import os
import time

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from egnn_inference import load_model, kabsch_align_rows
from Util.smiles_to_graph import smiles_to_graph, GraphConversionError, ELEMENT_SYMBOLS
from Util.visualise import build_prediction_figure, figure_to_html
from Util.rerank import select_best_draw, select_median_draw, quality_score


# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------

CHECKPOINT_PATH = os.environ.get("EGNN_CHECKPOINT", "./egnn_checkpoint_5A_96_8_Full_1000.pt")
DEVICE = os.environ.get("EGNN_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
HOST = os.environ.get("EGNN_HOST", "127.0.0.1")
PORT = int(os.environ.get("EGNN_PORT", "8000"))
QM9_ROOT = os.environ.get("EGNN_QM9_ROOT", "./data/QM9")



# Load the model once
print("Loading EGNN inference server")
print(f"  Checkpoint: {CHECKPOINT_PATH}")
print(f"  Device: {DEVICE}")

model, cfg = load_model(CHECKPOINT_PATH, device=DEVICE, verbose=True)
NOISE_SCALE = cfg["noise_scale"]
# DEFAULT_NUM_DRAWS = cfg["num_pos_init_draws"]
DEFAULT_NUM_DRAWS = 16



# Optional QM9 dataset the prediction-only path has no QM9 dependency).
QM9_DATASET = None
if os.environ.get("EGNN_ENABLE_QM9") == "1":
    from torch_geometric.datasets import QM9
    print(f"Loading QM9 from {QM9_ROOT} for verification endpoint...")
    QM9_DATASET = QM9(root=QM9_ROOT)
    print(f"  QM9 loaded: {len(QM9_DATASET)} molecules.")



# Inference helpers


@torch.no_grad()
def _run_k_draws(z, edge_index, edge_attr, num_draws):
    """
    Run num_draws independent forward passes with fresh noise each time.

    Returns a list of [N, 3] prediction tensors, each in its own
    reference frame (they will have different rotations/translations).
    """
    z_dev = z.to(DEVICE)
    ei_dev = edge_index.to(DEVICE)
    ea_dev = edge_attr.to(DEVICE)
    batch = torch.zeros(z_dev.size(0), dtype=torch.long, device=DEVICE)

    preds = []
    for _ in range(num_draws):
        pos_init = torch.randn(z_dev.size(0), 3, device=DEVICE) * NOISE_SCALE
        pred = model(z_dev, pos_init, ei_dev, ea_dev, batch=batch)
        preds.append(pred)
    return preds


def _align_all(preds):
    """
    Kabsch-align all draws to draw 0 so they are in a common reference
    frame. Returns the aligned list. This is required before scoring,
    averaging, or computing ensemble centrality, since each draw is
    otherwise in an arbitrary frame.
    """
    centered = [p - p.mean(dim=0, keepdim=True) for p in preds]
    ref = centered[0]
    aligned = [ref]
    for p_c in centered[1:]:
        R = kabsch_align_rows(p_c, ref)
        aligned.append(p_c @ R.T)
    return aligned


def _consensus(preds):
    """Average of the K aligned draws."""
    return torch.stack(_align_all(preds)).mean(dim=0)


def _kabsch_align_single(pred, target):
    """Kabsch-align a single [N, 3] prediction to a single [N, 3] target."""
    pred_c = pred - pred.mean(0, keepdim=True)
    target_c = target - target.mean(0, keepdim=True)
    H = pred_c.T @ target_c
    U, S, Vt = torch.linalg.svd(H)
    d = torch.sign(torch.linalg.det(Vt.T @ U.T))
    D = torch.diag(torch.tensor([1.0, 1.0, d], device=pred.device, dtype=pred.dtype))
    R = Vt.T @ D @ U.T
    return pred_c @ R.T, target_c


def _coords_to_xyz(elements, coords):
    """Format a prediction as a standard XYZ file string."""
    lines = [str(len(elements)), "predicted by EGNN"]
    for elem, (x, y, z) in zip(elements, coords):
        lines.append(f"{elem:2s} {x: .6f} {y: .6f} {z: .6f}")
    return "\n".join(lines)


def _select_draw(aligned, visual_draw, z, edge_index, edge_attr):
    """
    Apply the requested draw selection policy to a list of aligned draws.
    Returns (shown_coords_tensor, label_string, diagnostics_dict_or_None).
    """
    K = len(aligned)

    if visual_draw == "consensus":
        shown = torch.stack(aligned).mean(dim=0)
        return shown, f"consensus of {K}", None

    if visual_draw == "last":
        return aligned[-1], f"draw {K - 1}", None

    if visual_draw == "median":
        idx = select_median_draw(aligned)
        return aligned[idx], f"median of {K} (draw {idx})", {"chosen": idx}

    if visual_draw == "best":
        idx, scores = select_best_draw(aligned, z, edge_index, edge_attr)
        return (aligned[idx],
                f"best of {K} (draw {idx})",
                {"chosen": idx, "scores": [round(float(s), 4) for s in scores]})

    # Default: "first"
    return aligned[0], "draw 0", None



# FastAPI app
app = FastAPI(
    title="EGNN QM9 Local Inference Server",
    description="Predict 3D molecular geometry from SMILES using a trained "
                "E(3)-equivariant GNN. Includes interactive visualization "
                "and best-of-K reranking.",
    version="1.2",
)


# Request/response schemas
class PredictRequest(BaseModel):
    smiles: str = Field(..., description="SMILES string (H, C, N, O, F only)")
    num_draws: int = Field(DEFAULT_NUM_DRAWS, ge=1, le=32,
                          description="Number of noise draws to average over")


class PredictResponse(BaseModel):
    smiles: str
    num_atoms: int
    elements: list[str]
    coordinates: list[list[float]]
    num_draws: int
    latency_ms: float
    xyz: str

class PredictVisualRequest(BaseModel):
    smiles: str = Field(..., description="SMILES (H, C, N, O, F only)")
    num_draws: int = Field(DEFAULT_NUM_DRAWS, ge=1, le=16)
    visual_draw: str = Field(
        "best",
        description=(
            "Which draw to display. Options: "
            "'first'     -- draw 0 (simple, unselected); "
            "'last'      -- draw K-1; "
            "'consensus' -- average of all draws (blurs symmetric atoms); "
            "'best'      -- rerank by chemistry heuristics (default); "
            "'median'    -- the draw closest to the ensemble mean."
        ),
    )



# JSON endpoints
@app.get("/health")
def health():
    return {"status": "ok", "device": DEVICE}


@app.get("/info")
def info():
    return {
        "model_params": sum(p.numel() for p in model.parameters()),
        "device": DEVICE,
        "hidden_dim": cfg["hidden_dim"],
        "num_layers": cfg["num_layers"],
        "noise_scale": NOISE_SCALE,
        "default_num_draws": DEFAULT_NUM_DRAWS,
        "qm9_loaded": QM9_DATASET is not None,
    }


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    """
    Predict 3D geometry for a SMILES string. Returns coordinates as JSON
    plus a formatted XYZ string. The returned coordinates are the K-draw
    consensus (each draw Kabsch-aligned to draw 0, then averaged).
    """
    t0 = time.time()
    try:
        z, edge_index, edge_attr, _ = smiles_to_graph(req.smiles)
    except GraphConversionError as e:
        raise HTTPException(status_code=400, detail=str(e))

    try:
        preds = _run_k_draws(z, edge_index, edge_attr, req.num_draws)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {e}")

    coords = _consensus(preds).cpu().numpy()
    elements = [ELEMENT_SYMBOLS.get(int(zi), "?") for zi in z]
    latency = (time.time() - t0) * 1000

    return PredictResponse(
        smiles=req.smiles,
        num_atoms=len(elements),
        elements=elements,
        coordinates=coords.tolist(),
        num_draws=req.num_draws,
        latency_ms=round(latency, 1),
        xyz=_coords_to_xyz(elements, coords),
    )


@app.post("/predict_draws")
def predict_draws(req: PredictVisualRequest):
    """
    Return the K draws as JSON, each with a per-draw quality score
    breakdown. Useful for understanding what the model is doing across
    draws -- e.g. to see whether one draw is clearly worse than the
    others, or whether the ensemble is uniform.

    Response fields per draw: symmetry_break, bond_length_deviation,
    n_bonds_used, steric_clash_count, total. Also reports which draw
    was chosen by the 'best' and 'median' selectors.
    """
    try:
        z, edge_index, edge_attr, _ = smiles_to_graph(req.smiles)
    except GraphConversionError as e:
        raise HTTPException(status_code=400, detail=str(e))

    preds = _run_k_draws(z, edge_index, edge_attr, req.num_draws)
    aligned = _align_all(preds)

    details = []
    for i, pred in enumerate(aligned):
        _, breakdown = quality_score(pred, z, edge_index, edge_attr)
        details.append({
            "draw": i,
            "symmetry_break": round(float(breakdown["symmetry_break"]), 4),
            "bond_length_deviation": round(float(breakdown["bond_length_deviation"]), 4),
            "n_bonds_used": int(breakdown["n_bonds_used"]),
            "steric_clash_count": int(breakdown["steric_clash_count"]),
            "total": round(float(breakdown["total"]), 4),
        })

    best_idx, _ = select_best_draw(aligned, z, edge_index, edge_attr)
    median_idx = select_median_draw(aligned)

    return {
        "smiles": req.smiles,
        "num_draws": req.num_draws,
        "per_draw_scores": details,
        "best_draw": best_idx,
        "median_draw": median_idx,
    }



# Visual endpoints (return standalone interactive HTML)
@app.post("/predict_visual", response_class=HTMLResponse)
def predict_visual(req: PredictVisualRequest):
    """
    Predict geometry for an arbitrary SMILES string and return a
    self-contained interactive 3D HTML view.

    The visual_draw parameter controls which of the K draws (or their
    consensus) is shown. Default is 'best' -- chemistry-heuristic
    reranking that selects the most plausible draw.
    """
    try:
        z, edge_index, edge_attr, _ = smiles_to_graph(req.smiles)
    except GraphConversionError as e:
        raise HTTPException(status_code=400, detail=str(e))

    try:
        preds = _run_k_draws(z, edge_index, edge_attr, req.num_draws)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {e}")

    aligned = _align_all(preds)
    shown, shown_label, _ = _select_draw(
        aligned, req.visual_draw, z, edge_index, edge_attr
    )
    coords = shown.cpu().numpy()

    fig = build_prediction_figure(
        pred_pos=coords,
        z=z.numpy(),
        edge_index=edge_index,
        title=f"{req.smiles}  |  {z.size(0)} atoms  |  showing {shown_label}",
    )
    return HTMLResponse(figure_to_html(fig))


@app.get("/predict_visual_qm9/{qm9_index}", response_class=HTMLResponse)
def predict_visual_qm9(qm9_index: int, num_draws: int = DEFAULT_NUM_DRAWS):
    """
    Predict geometry for a QM9 molecule and return a three-panel
    comparison: actual DFT geometry, predicted geometry, and overlay
    with per-atom error vectors.

    Requires QM9 to be loaded (set EGNN_ENABLE_QM9=1 when starting the
    server). The displayed draw is always draw 0 -- for ground-truth
    comparison we want a specific, unselected draw so the reported RMSD
    matches what is shown.
    """
    if QM9_DATASET is None:
        raise HTTPException(
            status_code=503,
            detail="QM9 dataset not loaded. Restart the server with "
                   "EGNN_ENABLE_QM9=1 to enable this endpoint.",
        )
    if qm9_index < 0 or qm9_index >= len(QM9_DATASET):
        raise HTTPException(
            status_code=400,
            detail=f"QM9 index out of range [0, {len(QM9_DATASET)})",
        )

    data = QM9_DATASET[qm9_index]
    z, pos = data.z, data.pos
    edge_index, edge_attr = data.edge_index, data.edge_attr

    z_dev, pos_dev = z.to(DEVICE), pos.to(DEVICE)
    ei_dev, ea_dev = edge_index.to(DEVICE), edge_attr.to(DEVICE)
    batch = torch.zeros(z_dev.size(0), dtype=torch.long, device=DEVICE)

    with torch.no_grad():
        aligned_preds = []
        target_c = None
        for _ in range(num_draws):
            pos_init = torch.randn(z_dev.size(0), 3, device=DEVICE) * NOISE_SCALE
            pred = model(z_dev, pos_init, ei_dev, ea_dev, batch=batch)
            aligned, target_c = _kabsch_align_single(pred, pos_dev)
            aligned_preds.append(aligned)

    shown = aligned_preds[0]
    per_atom_err = (shown - target_c).norm(dim=-1).cpu().numpy()
    rmsd = float(((shown - target_c) ** 2).sum(-1).mean().sqrt())

    fig = build_prediction_figure(
        pred_pos=shown.cpu().numpy(),
        z=z.numpy(),
        edge_index=edge_index,
        title=f"QM9 index {qm9_index}  |  {z.size(0)} atoms  |  "
              f"draw 0 RMSD = {rmsd:.4f} A",
        target_pos=target_c.cpu().numpy(),
        per_atom_err=per_atom_err,
        shown_draw=0,
        num_draws=num_draws,
        color_by_error=True,
    )
    return HTMLResponse(figure_to_html(fig))


# Minimal HTML front-end
INDEX_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>EGNN Inference</title>
    <style>
        body { font-family: -apple-system, sans-serif; max-width: 820px;
               margin: 40px auto; padding: 0 20px; color: #222; }
        h1 { font-weight: 600; margin-bottom: 4px; }
        .sub { color: #666; margin-top: 0; }
        .row { margin: 14px 0; }
        input[type=text] { width: 320px; padding: 8px; font-size: 14px;
                           border: 1px solid #ccc; border-radius: 4px; }
        select { padding: 8px; font-size: 14px; border: 1px solid #ccc;
                 border-radius: 4px; margin-left: 6px; }
        button { padding: 8px 16px; font-size: 14px; cursor: pointer;
                 background: #0066cc; color: white; border: none;
                 border-radius: 4px; margin-left: 6px; }
        button:hover { background: #0055aa; }
        pre { background: #f4f4f4; padding: 15px; border-radius: 4px;
              overflow: auto; max-height: 400px; font-size: 12px; }
        small { color: #666; display: block; margin-top: 6px; }
        #status { margin-left: 12px; color: #666; font-size: 13px; }
        .section { margin-top: 28px; }
        .section h3 { font-size: 15px; margin-bottom: 6px; }
        .links { font-size: 13px; color: #555; }
        .links a { color: #0066cc; text-decoration: none; }
        .links a:hover { text-decoration: underline; }
        code { background: #f0f0f0; padding: 1px 4px; border-radius: 3px; }
    </style>
</head>
<body>
    <h1>EGNN QM9 Inference</h1>
    <p class="sub">Predict 3D geometry from SMILES. Supported elements: H, C, N, O, F.</p>

    <div class="section">
        <h3>Predict (JSON)</h3>
        <div class="row">
            <input id="smiles" type="text" value="CCO" />
            <button onclick="predict()">Predict</button>
            <span id="status"></span>
        </div>
        <pre id="output" style="display:none"></pre>
        <small>Examples: CCO (ethanol), c1ccccc1 (benzene), CC(=O)O (acetic acid)</small>
    </div>

    <div class="section">
        <h3>Interactive 3D view</h3>
        <div class="row">
            <input id="smiles_viz" type="text" value="CC(C)O" />
            <select id="visual_draw">
                <option value="best">best of K (reranked)</option>
                <option value="median">median of K</option>
                <option value="first">first draw</option>
                <option value="consensus">consensus (blurs symmetric atoms)</option>
            </select>
            <button onclick="predictVisual()">Open 3D view</button>
        </div>
        <small>Opens the interactive Plotly view in a new tab.</small>
    </div>

    <div class="section links">
        <h3>Other endpoints</h3>
        <a href="/docs">/docs</a> &middot;
        <a href="/info">/info</a> &middot;
        <a href="/health">/health</a><br>
        <span style="color:#888">For QM9 verification views:
        <code>/predict_visual_qm9/0</code> (requires EGNN_ENABLE_QM9=1 at server start)</span>
    </div>

    <script>
    async function predict() {
        const smiles = document.getElementById('smiles').value;
        const status = document.getElementById('status');
        const output = document.getElementById('output');
        status.textContent = 'Predicting...';
        output.style.display = 'none';
        try {
            const r = await fetch('/predict', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({smiles: smiles}),
            });
            const data = await r.json();
            if (!r.ok) {
                status.textContent = 'Error';
                output.textContent = data.detail || JSON.stringify(data, null, 2);
            } else {
                status.textContent = `${data.num_atoms} atoms, ${data.latency_ms} ms`;
                output.textContent = data.xyz;
            }
            output.style.display = 'block';
        } catch (e) {
            status.textContent = 'Failed';
            output.textContent = String(e);
            output.style.display = 'block';
        }
    }

    function predictVisual() {
        const smiles = document.getElementById('smiles_viz').value;
        const mode = document.getElementById('visual_draw').value;
        fetch('/predict_visual', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({smiles: smiles, num_draws: 16, visual_draw: mode}),
        }).then(r => r.blob()).then(blob => {
            const url = URL.createObjectURL(blob);
            window.open(url, '_blank');
        }).catch(err => alert('Failed: ' + err));
    }
    </script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
def index():
    return INDEX_HTML



# Entry point
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=HOST, port=PORT)