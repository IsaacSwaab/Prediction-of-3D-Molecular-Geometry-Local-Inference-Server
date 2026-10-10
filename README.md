# EGNN QM9 Inference Server

A local, self-contained inference server for the trained EGNN model.
Predicts 3D molecular geometry from a SMILES string, with interactive
3D visualization and best-of-K reranking.

The model is a 3-million-parameter E(3)-equivariant graph neural network
trained on the full QM9 dataset (130,831 small organic molecules). It
predicts DFT-quality geometry from a molecular graph plus a random
coordinate initialization. Because the initialization is random, each
forward pass produces a slightly different result — the server runs K
draws per query and reranks them by chemical plausibility to return the
best one.

---

## Contents

### Files in this directory

| File | Purpose |
|------|---------|
| `egnn_inference.py`     | Model definition + checkpoint loader (no training code) |
| `smiles_to_graph.py`    | SMILES → PyG graph conversion |
| `visualize.py`          | Plotly 3D figure builders (single- and three-panel) |
| `rerank.py`             | Quality scoring and best-of-K reranking |
| `server.py`             | FastAPI server with REST endpoints and a web UI |
| `client_test.py`        | Python client for testing and QM9 verification |
| `egnn_checkpoint.pt`    | Your trained model (produced by the training script) |

### Quick start

```bash
pip install torch torch_geometric rdkit fastapi uvicorn requests numpy plotly
python server.py
```

Then open `http://localhost:8000` in a browser.

---

## Install

Requires Python 3.10+.

```bash
pip install torch torch_geometric rdkit fastapi uvicorn requests numpy plotly
```

For GPU inference, ensure you have a CUDA-enabled PyTorch install:

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

Should print `True`. If it prints `False` and you have a GPU, reinstall
PyTorch from the CUDA index:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

---

## Running the server

### Minimal (localhost only, no QM9)

```bash
python server.py
```

Loads the checkpoint and starts the server on `http://127.0.0.1:8000`.
Only the prediction and visualization endpoints are available; the QM9
verification endpoint returns 503.

### With QM9 verification enabled

**PowerShell (Windows):**

```powershell
$env:EGNN_ENABLE_QM9 = "1"
python server.py
```

**bash/zsh (macOS/Linux):**

```bash
EGNN_ENABLE_QM9=1 python server.py
```

This loads the QM9 dataset (~130,831 molecules, ~500 MB) at startup and
enables `/predict_visual_qm9/{index}`, which compares predictions against
DFT ground truth.

### Accessible on the local network

```powershell
$env:EGNN_HOST = "0.0.0.0"
$env:EGNN_ENABLE_QM9 = "1"
python server.py
```

Other devices on the same WiFi can query the server at your machine's
LAN IP (find it with `ipconfig` / `ifconfig`):

```
http://192.168.1.42:8000
```

**Windows firewall:** the first time you bind to `0.0.0.0`, Windows will
prompt to allow Python through the firewall. Click Allow. If no prompt
appears, add a rule manually (as Administrator):

```powershell
New-NetFirewallRule -DisplayName "EGNN Server" -Direction Inbound `
    -Protocol TCP -LocalPort 8000 -Action Allow
```

**Security:** `0.0.0.0` exposes the server to everyone on the same
network, with no authentication. Only do this on a trusted network
(home, university, lab). For untrusted networks, keep the default
`127.0.0.1` binding and use an SSH tunnel instead:

```bash
ssh -L 8000:localhost:8000 user@workstation
```

---

## Environment variables

All optional. Set before launching `python server.py`.

| Variable | Default | Purpose |
|----------|---------|---------|
| `EGNN_CHECKPOINT` | `./egnn_checkpoint.pt` | Path to the trained checkpoint |
| `EGNN_DEVICE` | auto (`cuda` if available, else `cpu`) | Force `cuda` or `cpu` |
| `EGNN_HOST` | `127.0.0.1` | Bind address (`0.0.0.0` for LAN access) |
| `EGNN_PORT` | `8000` | Port to listen on |
| `EGNN_ENABLE_QM9` | unset | Set to `1` to load QM9 for verification endpoint |
| `EGNN_QM9_ROOT` | `./data/QM9` | Path to QM9 dataset directory |

---

## Endpoints

### `GET /` — Web UI

Minimal HTML page with a JSON prediction form and an interactive 3D
view launcher. Simplest way to interact with the server.

### `GET /docs` — Interactive API documentation

Auto-generated Swagger UI. Every endpoint can be called from this page
with a form. Useful for exploring the API without writing any code.

### `GET /health` — Health check

```json
{"status": "ok", "device": "cuda"}
```

Use for load-balancer probes.

### `GET /info` — Model summary

```json
{
  "model_params": 3039624,
  "device": "cuda",
  "hidden_dim": 96,
  "num_layers": 8,
  "noise_scale": 2.0,
  "default_num_draws": 4,
  "qm9_loaded": true
}
```

### `POST /predict` — JSON prediction

**Request:**

```json
{"smiles": "CCO", "num_draws": 4}
```

**Response:**

```json
{
  "smiles": "CCO",
  "num_atoms": 9,
  "elements": ["C","C","O","H","H","H","H","H","H"],
  "coordinates": [[x,y,z], ...],
  "num_draws": 4,
  "latency_ms": 87.3,
  "xyz": "9\npredicted by EGNN\nC  1.234  0.567 -0.123\n..."
}
```

Returns the K-draw consensus (each draw Kabsch-aligned to draw 0, then
averaged). The `xyz` field is a standard XYZ file string that can be
saved and opened in any molecular viewer.

### `POST /predict_visual` — Interactive 3D HTML

**Request:**

```json
{
  "smiles": "CC(C)O",
  "num_draws": 8,
  "visual_draw": "best"
}
```

**Response:** a standalone HTML file containing the interactive Plotly
3D view. Open it in a browser to rotate, zoom, and inspect the structure.

`visual_draw` options:

| Value | What it shows |
|-------|---------------|
| `"best"` (default) | The best of K draws, ranked by chemistry heuristics |
| `"median"` | The draw closest to the ensemble's geometric center |
| `"first"` | Draw 0 (simple, unselected) |
| `"last"` | Draw K-1 |
| `"consensus"` | Average of all draws (blurs symmetric atoms) |

The default `"best"` is recommended for normal use. `"median"` is the
weight-free alternative when you want to avoid the heuristic scoring.
`"consensus"` gives the lowest RMSD but can collapse symmetric hydrogens
onto each other.

### `POST /predict_draws` — Per-draw diagnostics

**Request:**

```json
{"smiles": "CC(C)O", "num_draws": 8}
```

**Response:**

```json
{
  "smiles": "CC(C)O",
  "num_draws": 8,
  "per_draw_scores": [
    {
      "draw": 0,
      "symmetry_break": 0.0023,
      "bond_length_deviation": 0.0412,
      "n_bonds_used": 10,
      "steric_clash_count": 0,
      "total": 0.208
    }
  ],
  "best_draw": 3,
  "median_draw": 5
}
```

Shows each of the K draws' individual quality scores. Useful for
understanding whether the ensemble is uniform (all draws similar) or
has outliers (one draw clearly worse). The `best_draw` and `median_draw`
fields report which draw would be selected by each policy.

### `GET /predict_visual_qm9/{index}` — Three-panel verification

**Example:** `http://localhost:8000/predict_visual_qm9/0`

Returns an interactive three-panel comparison: actual DFT geometry from
QM9, predicted geometry, and an overlay with red error vectors from
every predicted atom to where it should be. Hover any atom to see its
index, element, and per-atom error.

Requires `EGNN_ENABLE_QM9=1` at server start. Use this endpoint to
visually verify that the model produces correct predictions on data it
was trained on.

---

## Verification

Before trusting the server's output, verify that the SMILES → graph
conversion matches what QM9 provides. A mismatch means the model is
seeing inputs it was never trained on, and outputs will be silently
wrong.

```bash
python -c "from smiles_to_graph import verify_against_qm9; verify_against_qm9('CCO', 0)"
```

Should print `PASS: structural properties match`. If it doesn't, that's
the bug to fix before anything else.

You can also visually verify by opening
`http://localhost:8000/predict_visual_qm9/0` in a browser (with QM9
enabled). A healthy model gives Kabsch RMSD well under 1 Å for most
QM9 molecules.

---

## How the model works

### Inputs

The model takes three things:

1. **A molecular graph** — atomic numbers `z`, bonded edges `edge_index`,
   and bond types `edge_attr` (one-hot over single/double/triple/aromatic).
2. **A random coordinate initialization** — `pos_init`, an `[N, 3]` tensor
   drawn from `N(0, noise_scale²)`. The model was trained with `noise_scale=2.0`.
3. **A batch index** — needed only for batched inference; a single molecule
   uses a zeros tensor.

### Outputs

Predicted 3D coordinates, `[N, 3]`, in a reference frame that depends on
the input noise.

### Why the output is non-deterministic

The model was trained to *refine* a noisy initial guess, not to produce
a fixed output from a graph. Because `pos_init` is random, each forward
pass produces a slightly different output. Different draws can end up in
different local minima of the loss landscape, so the K predictions are
not identical.

The server mitigates this in two ways:

- **Averaging (consensus)** — Kabsch-align all K draws to draw 0, then
  average. This is the metric the training validation loop reports. It
  is the "smoothest" answer, but can blur symmetric atoms (like CH₂
  hydrogens) toward their midpoint.
- **Reranking (best)** — score each draw with a chemistry heuristic and
  return the top one. Preserves individual molecular geometry (no
  blurring), at the cost of a heuristic that may not always pick the
  ideal draw.

The default for visual output is `"best"`. For numeric metrics (RMSD to
ground truth, JSON coordinates), the consensus is used.

### Reranking heuristics

The `quality_score` in `rerank.py` combines three ground-truth-free
signals:

1. **Symmetry break value** — penalizes pairs of atoms bonded to the
   same center that are too close together. Directly targets the
   symmetric-hydrogen collapse failure mode.
2. **Bond length deviation** — mean absolute deviation from typical
   bond lengths for each (element pair, bond type). Uses a hardcoded
   table of ~20 typical values.
3. **Steric clash count** — non-bonded atom pairs closer than 0.7 ×
   sum of van der Waals radii.

Each term is scaled so a "typical" failure contributes roughly 1.0 to
the total score. Lower score = more chemically plausible draw.

The `"median"` policy is a weight-free alternative: pick the draw
closest to the ensemble's geometric center. This is more robust when
the heuristics' weights don't match the actual failure modes.

---

## Querying the server

### From a browser

- `http://localhost:8000` — Web UI
- `http://localhost:8000/docs` — Swagger UI with interactive forms
- `http://localhost:8000/predict_visual_qm9/0` — QM9 verification view
  (requires QM9 enabled)

### From the command line (PowerShell)

**JSON prediction:**

```powershell
$body = @{ smiles = "CCO"; num_draws = 4 } | ConvertTo-Json
$r = Invoke-RestMethod -Uri http://localhost:8000/predict -Method Post `
    -ContentType "application/json" -Body $body
$r.xyz
```

**Interactive 3D view:**

```powershell
$body = @{ smiles = "CC(C)O"; num_draws = 8; visual_draw = "best" } | ConvertTo-Json
Invoke-WebRequest -Uri http://localhost:8000/predict_visual -Method Post `
    -ContentType "application/json" -Body $body -OutFile isopropanol.html
Start-Process isopropanol.html
```

**Per-draw diagnostics:**

```powershell
$body = @{ smiles = "CC(C)O"; num_draws = 8 } | ConvertTo-Json
$r = Invoke-RestMethod -Uri http://localhost:8000/predict_draws -Method Post `
    -ContentType "application/json" -Body $body
$r.per_draw_scores | Format-Table
```

### From the command line (bash)

```bash
curl -X POST http://localhost:8000/predict \
     -H "Content-Type: application/json" \
     -d '{"smiles": "CCO", "num_draws": 4}' | python -m json.tool
```

### From Python

```python
import requests

# JSON prediction
r = requests.post("http://localhost:8000/predict",
                  json={"smiles": "CCO", "num_draws": 4})
print(r.json()["xyz"])

# Interactive HTML (save to file)
r = requests.post("http://localhost:8000/predict_visual",
                  json={"smiles": "CC(C)O", "num_draws": 8, "visual_draw": "best"})
with open("isopropanol.html", "w") as f:
    f.write(r.text)
```

---

## Performance notes

- **Latency per request** is dominated by K forward passes. On a GPU
  (RTX 2070-class), a QM9-sized molecule takes ~100–400 ms for K=4–8
  draws. On CPU, expect 1–3 seconds per draw.
- **Memory** is small; the model is ~36 MB of weights, and each forward
  pass fits comfortably in VRAM even on modest GPUs.
- **Throughput** is one request at a time by default (single uvicorn
  worker). For concurrent traffic, run multiple uvicorn workers or add
  request batching.
- **Plotly HTML files** are ~20 KB each (loading Plotly JS from CDN).
  First load in a browser fetches the JS (~3 MB), subsequent views are
  cached.

---

## Troubleshooting

**Server won't start, "Address already in use":** another process is
using port 8000. Kill it (`netstat -ano | findstr :8000` on Windows,
`lsof -i :8000` on Linux) or set `EGNN_PORT=8001`.

**Predictions look wrong:** run `verify_against_qm9` first. If that
fails, the SMILES → graph conversion doesn't match QM9's format and
needs to be fixed. If it passes, the checkpoint may need more training
or the molecule may be outside QM9's chemical scope (e.g. elements
beyond H, C, N, O, F).

**CUDA out of memory:** reduce `num_draws` in the request, or set
`EGNN_DEVICE=cpu` to run on CPU.

**Slow inference (>1 s per draw):** running on CPU. Check
`EGNN_DEVICE=cuda` and that CUDA-enabled PyTorch is installed.

**Symmetry-collapsed hydrogens in the visualization:** try
`visual_draw="best"` or `"median"` instead of `"consensus"`. The
consensus averages K draws, which blurs symmetric atoms together.

**All K draws look identical:** the model has converged to a stable
solution for that molecule. This is normal and desirable.

**All K draws look completely different:** the model is unstable on
that molecule. Try more draws (`num_draws=16`) and use the `best`
selector. If it persists across many molecules, the model may need
retraining.

**QM9 endpoint returns 503:** QM9 wasn't loaded at startup. Restart with
`EGNN_ENABLE_QM9=1`.

**LAN access doesn't work:** check (a) the server is bound to `0.0.0.0`
via `EGNN_HOST`, (b) the firewall allows port 8000, (c) both machines
are on the same network, (d) you're using the LAN IP not `localhost`.

---

## Scientific background

The model is an E(3)-equivariant graph neural network (EGNN, Satorras
et al. 2021). "Equivariant" means its outputs transform correctly under
rotation and translation of the input — rotate the input coordinates
and the output rotates the same way, without the model needing to
re-learn orientations. This is the natural symmetry for molecular
geometry prediction.

Training used a K-draw noise pool to prevent the model from memorizing
the specific random initialization for each training molecule: each
molecule stored K independent noise draws, one of which was sampled per
batch. This forced the model to learn a noise-invariant mapping from
graph structure to geometry.

The loss combined:

- **Kabsch-aligned RMSD** — rotation- and translation-invariant global
  error.
- **Bond length MAE** — local chemistry (correct bond lengths).
- **Bond angle MAE** — local chemistry (correct angles).
- **Symmetry break barrier** — prevents symmetric atoms from collapsing
  onto each other, a failure mode specific to equivariant models trained
  with noise input.

The checkpoint included in this directory is the result of training on
the full QM9 dataset (~130k molecules) for multiple hundred epochs,
with a cosine LR schedule.

---

## License / attribution

This is a personal research project. The EGNN architecture is from
Satorras et al. (2021), "E(n) Equivariant Graph Neural Networks". The
QM9 dataset is from Ramakrishnan et al. (2014). Implementation,
training, and the inference server are original work.