"""
EGNN for QM9 Geometric Structure Prediction
=============================================

Implements:
  - EGNNLayer / EGNN model (Satorras et al., 2021 - E(n) Equivariant GNNs)
  - Kabsch-aligned RMSD loss (rotation/translation-invariant)
  - An empirical equivariance unit test (run this before trusting anything else)
  - Noise-based initial coordinates (default) -- see add_noise_initial_coords
  - A basic training loop

CHANGES IN THIS REVISION:
  - Added symmetry_break_loss: a distance-based barrier that penalizes any
    two atoms bonded to the same center being closer than min_dist
    (default 1.0 A). This directly targets the symmetry-collapse failure
    mode observed on CH2/CH3 groups, where symmetric hydrogens are
    predicted at the same coordinate.

    Why a distance barrier and not just more angle_loss_weight:
      bond_angle_loss compares cos(predicted angle) to cos(true angle).
      At exact collapse (both atoms at the same point) the angle is 0,
      cos(angle)=1, and cos is stationary there -- its gradient w.r.t.
      either atom's position is exactly zero. So the angle loss has a
      HIGH VALUE at collapse but ZERO GRADIENT there: gradient descent
      cannot escape, regardless of how large angle_loss_weight is set.
      This matches what was observed empirically.

      The distance barrier's gradient does NOT vanish at collapse:
      for two atoms at distance d < min_dist, the gradient of
      relu(min_dist - d) has magnitude 1 (a unit vector pointing away
      from the other atom), independent of how small d is. So it can
      push atoms apart from the collapsed configuration, after which
      the angle and RMSD losses (which have non-zero gradients away
      from exact collapse) can take over.

    Wired through CONFIG (symmetry_break_weight, symmetry_break_min_dist),
    the training and validation loops, checkpoint save/resume, and history.

  Previous revision's changes (all retained):
  - New CONFIG flag "gpu_resident_dataset" for GPU-resident training data.
  B.1 -- phi_x edge conditioning.
  B.2 -- Split attention for coordinate vs. feature updates.
  B.4 -- Persistent edge embeddings (chemical bonds only).
  Each of B.1/B.2/B.4 is toggleable via CONFIG.
  - K-draw pos_init_pool with node-axis-first layout [N_atoms, K, 3].
  - Validation averages predictions over all K draws.
  - Identity-model baseline; noise-draw variance diagnostic.
  - RBF distance features; dynamic radius graph; angular (3-body)
    messages; attention; auxiliary bond-length and bond-angle losses.
  - Checkpoint save/resume with optimizer state and history.

  Earlier fixes still in place:
  - Kabsch SVD ridge (1e-6) to prevent NaN gradients on degenerate
    covariances.
  - Non-finite gradient guard: skip optimizer.step() instead of
    poisoning weights.
  - Per-layer coordinate displacement clamp (max_disp, equivariant).
  - Per-epoch parameter finiteness check.
  - phi_x final-layer gain = 1.0 (was 0.001, which starved gradients).

NOTE ON INITIAL COORDINATES: pos_init is independent random noise --
NOT a chemically informed guess. This is intentional: it's nearly free,
removes a class of preprocessing bugs, and validates whether the EGNN
can learn geometry from the graph alone.

Dependencies: torch, torch_geometric
    pip install torch torch_geometric

Usage:
    Edit the CONFIG dict below, then just run this file directly.
"""

import math
import os
import time

import torch
import torch.nn as nn
from torch_geometric.utils import scatter, to_dense_batch
from torch_geometric.utils import softmax as pyg_softmax
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader
from torch_geometric.datasets import QM9
from Util.qm9_splits import resolve_splits, load_split_data, print_split_balance


# ----------------------------------------------------------------------
# 0. CONFIG — edit these and just hit "Run" 
# ----------------------------------------------------------------------

CONFIG = {
    "run_tests": False,
    "run_train": True,

    "root": "./data/QM9",
    "epochs": 100,
    "batch_size": 256,
    "lr": 1e-3,
    "hidden_dim": 64,
    "num_layers": 8,
    # --- Data pool and split (see qm9_splits.py) ---
    # subset_size: None -> use the FULL QM9 (~130k molecules); an int -> that many.
    #   (the full set makes every epoch ~13x longer than 10,000 molecules)
    # subset_sampling: "random" draws the subset from the whole of QM9 (seeded);
    #   "first" takes the first N molecules (legacy, chemically biased).
    # split_method: "stratified" | "random" | "contiguous" (legacy, in-order cut).
    # stratify_by: features to match across train/val/test, most important
    #   first. Options: heavy_atoms, num_atoms, sym_groups, rings, has_F, has_N, has_O
    "subset_size": 1000,
    "subset_sampling": "first",
    "split_method": "stratified",
    "split_seed": 421,
    "stratify_by": ["heavy_atoms", "sym_groups", "has_F"],
    "min_stratum_size": 10,
    "noise_scale": 2.0,
    "num_pos_init_draws": 8,

    "max_disp": 5.0,

    "num_rbf": 16,
    "radius_cutoff": 4.0,
    "use_radius_graph": True,
    "use_angular": True,
    "use_attention": True,
    "bond_loss_weight": 0.6,
    "angle_loss_weight": 1.0,

    # ---: symmetry-break barrier loss ---
    # Penalizes any two atoms bonded to the same center being closer
    # than symmetry_break_min_dist. Directly targets the CH2/CH3
    # collapse failure mode (see module docstring).
    #
    # The barrier has non-vanishing gradient at collapse, unlike
    # bond_angle_loss whose gradient is exactly zero there. Start with
    # weight=1.0 (comparable to angle_loss_weight); the loss is logged
    # per-epoch as "sym_break" so you can tune based on observed values.
    "symmetry_break_weight": 2.0,
    "symmetry_break_min_dist": 1.0,  # Angstrom

    "use_edge_conditioned_phi_x": True,   # B.1
    "use_split_attention": True,          # B.2
    "use_edge_embeddings": True,          # B.4

    "gpu_resident_dataset": False,

    "checkpoint_path": "./egnn_checkpoint.pt",
    "resume_from_checkpoint": False,
    "checkpoint_every_n_epochs": 10,
}



# 1. EGNN LAYER

class MLP(nn.Module):
    """Small 2-layer MLP used inside phi_e / phi_x / phi_h / attention heads."""

    def __init__(self, in_dim, out_dim, hidden_dim=None, final_gain=1.0):
        super().__init__()
        hidden_dim = hidden_dim or max(in_dim, out_dim)
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, out_dim),
        )
        nn.init.xavier_uniform_(self.net[-1].weight, gain=final_gain)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x):
        return self.net(x)


class GaussianRBF(nn.Module):
    """Expands a scalar distance into a vector of Gaussian RBF features."""

    def __init__(self, num_basis=16, cutoff=10.0):
        super().__init__()
        centers = torch.linspace(0, cutoff, num_basis)
        self.register_buffer("centers", centers)
        self.width = cutoff / num_basis

    def forward(self, dist):
        return torch.exp(-((dist - self.centers) ** 2) / (2 * self.width ** 2))


def safe_scatter_mean(src, index, dim_size):
    """Mean aggregation via sum/count, with a clamp + nan_to_num safety net."""
    summed = scatter(src, index, dim=0, dim_size=dim_size, reduce="sum")
    ones = torch.ones(src.size(0), 1, device=src.device, dtype=src.dtype)
    count = scatter(ones, index, dim=0, dim_size=dim_size, reduce="sum").clamp(min=1)
    return torch.nan_to_num(summed / count, nan=0.0, posinf=0.0, neginf=0.0)


# def build_radius_edges(x, batch, cutoff, exclude_edge_index=None):
#     """Distance-cutoff edges within each molecule, excluding chemical bonds."""
#     N = x.size(0)
#     same_mol = batch.unsqueeze(0) == batch.unsqueeze(1)
#     dist = torch.cdist(x, x)
#     within_cutoff = dist <= cutoff
#     not_self = ~torch.eye(N, dtype=torch.bool, device=x.device)
#     candidate = same_mol & within_cutoff & not_self

#     if exclude_edge_index is not None and exclude_edge_index.numel() > 0:
#         exclude_mask = torch.zeros(N, N, dtype=torch.bool, device=x.device)
#         exclude_mask[exclude_edge_index[0], exclude_edge_index[1]] = True
#         candidate = candidate & ~exclude_mask

#     src, dst = candidate.nonzero(as_tuple=True)
#     return torch.stack([src, dst], dim=0)


@torch.no_grad()
def build_radius_edges(x, batch, cutoff, exclude_edge_index=None, max_neighbors=8):
    N = x.size(0)
    xd, mask = to_dense_batch(x.detach(), batch)             # [G,M,3], [G,M]
    G, M, _ = xd.shape
    counts = mask.sum(1)
    local = torch.arange(N, device=x.device) - (torch.cumsum(counts, 0) - counts)[batch]
    gidx = torch.full((G, M), -1, dtype=torch.long, device=x.device)
    gidx[batch, local] = torch.arange(N, device=x.device)

    d = torch.cdist(xd, xd)
    ok = mask.unsqueeze(1) & mask.unsqueeze(2) & (d <= cutoff)
    ok &= ~torch.eye(M, dtype=torch.bool, device=x.device)
    if exclude_edge_index is not None and exclude_edge_index.numel() > 0:
        s, t = exclude_edge_index
        ok[batch[s], local[s], local[t]] = False
    if max_neighbors is not None:
        top = d.masked_fill(~ok, float("inf")).topk(min(max_neighbors, M), dim=-1, largest=False)
        keep = torch.zeros_like(ok)
        keep.scatter_(-1, top.indices, torch.isfinite(top.values))
        ok = keep
    g, i, j = ok.nonzero(as_tuple=True)
    return torch.stack([gidx[g, j], gidx[g, i]])             # src = neighbour, dst = receiver

def build_triplets(edge_index, num_nodes):
    """Fully vectorized enumeration of unordered neighbor pairs per center."""
    dst = edge_index[1]
    E = dst.size(0)
    if E == 0:
        empty = torch.empty(0, dtype=torch.long, device=edge_index.device)
        return empty, empty

    perm = torch.argsort(dst)
    dst_sorted = dst[perm]

    _, counts = torch.unique_consecutive(dst_sorted, return_counts=True)
    offsets = torch.cumsum(counts, dim=0) - counts
    num_groups = counts.size(0)

    counts_sq = counts * counts
    total_pairs = int(counts_sq.sum().item())
    if total_pairs == 0:
        empty = torch.empty(0, dtype=torch.long, device=edge_index.device)
        return empty, empty

    block_of_pair = torch.repeat_interleave(torch.arange(num_groups, device=edge_index.device), counts_sq)
    block_offset = torch.cumsum(counts_sq, dim=0) - counts_sq
    p_within_block = torch.arange(total_pairs, device=edge_index.device) - block_offset[block_of_pair]
    group_count = counts[block_of_pair]

    row_local = p_within_block // group_count
    col_local = p_within_block % group_count

    keep = row_local < col_local
    row_local, col_local, block_of_pair = row_local[keep], col_local[keep], block_of_pair[keep]

    row_sorted_idx = offsets[block_of_pair] + row_local
    col_sorted_idx = offsets[block_of_pair] + col_local

    idx_j = perm[row_sorted_idx]
    idx_k = perm[col_sorted_idx]
    return idx_j, idx_k


class EGNNLayer(nn.Module):
    """One equivariant message-passing layer with optional methods from config."""

    def __init__(self, hidden_dim, chem_edge_attr_dim=4, max_disp=5.0,
                 num_rbf=16, radius_cutoff=6.0,
                 use_radius_graph=True, use_angular=True, use_attention=True,
                 use_edge_conditioned_phi_x=True, use_split_attention=True,
                 use_edge_embeddings=True):
        super().__init__()
        self.rbf = GaussianRBF(num_basis=num_rbf, cutoff=max(radius_cutoff, 10.0))
        combined_edge_attr_dim = chem_edge_attr_dim + 1

        self.use_edge_conditioned_phi_x = use_edge_conditioned_phi_x
        self.use_split_attention = use_split_attention
        self.use_edge_embeddings = use_edge_embeddings

        edge_in_dim = 2 * hidden_dim + num_rbf + combined_edge_attr_dim
        if use_edge_embeddings:
            edge_in_dim += hidden_dim
        self.phi_e = MLP(edge_in_dim, hidden_dim)

        if use_edge_conditioned_phi_x:
            phi_x_in_dim = hidden_dim + num_rbf + combined_edge_attr_dim
        else:
            phi_x_in_dim = hidden_dim
        self.phi_x = MLP(phi_x_in_dim, 1, hidden_dim=hidden_dim, final_gain=1.0)

        self.phi_h = MLP(3 * hidden_dim, hidden_dim)

        self.phi_att_coord = MLP(hidden_dim, 1, hidden_dim=hidden_dim)
        if use_split_attention:
            self.phi_att_feat = MLP(hidden_dim, 1, hidden_dim=hidden_dim)
            nn.init.xavier_uniform_(self.phi_att_feat.net[-1].weight, gain=0.9)

        self.phi_angle = MLP(2 * hidden_dim + 3, hidden_dim)

        if use_edge_embeddings:
            self.phi_edge_update = MLP(2 * hidden_dim, hidden_dim)
            self.norm_e = nn.LayerNorm(hidden_dim)

        self.norm_h = nn.LayerNorm(hidden_dim)

        self.max_disp = max_disp
        self.radius_cutoff = radius_cutoff
        self.use_radius_graph = use_radius_graph
        self.use_angular = use_angular
        self.use_attention = use_attention

    def forward(self, h, x, chem_edge_index, chem_edge_attr, batch, e_chem=None):
        E_chem = chem_edge_index.size(1)
        chem_attr_aug = torch.cat(
            [chem_edge_attr, torch.ones(E_chem, 1, device=x.device, dtype=x.dtype)], dim=-1
        )

        if self.use_radius_graph:
            radius_edge_index = build_radius_edges(
                x, batch, self.radius_cutoff, exclude_edge_index=chem_edge_index
            )
            E_rad = radius_edge_index.size(1)
            radius_attr = torch.zeros(E_rad, chem_attr_aug.size(-1), device=x.device, dtype=x.dtype)
            edge_index = torch.cat([chem_edge_index, radius_edge_index], dim=1)
            edge_attr_combined = torch.cat([chem_attr_aug, radius_attr], dim=0)
        else:
            edge_index = chem_edge_index
            edge_attr_combined = chem_attr_aug

        src, dst = edge_index
        rel_pos = x[src] - x[dst]
        dist = rel_pos.norm(dim=-1, keepdim=True).clamp(max=self.rbf.centers[-1].item())
        rbf_feat = self.rbf(dist)

        if self.use_edge_embeddings:
            if e_chem is None:
                e_chem = torch.zeros(E_chem, self.phi_e.net[0].in_features - 2 * h.size(1)
                                     - self.rbf.centers.numel()
                                     - edge_attr_combined.size(-1),
                                     device=x.device, dtype=x.dtype)
            E_rad = edge_attr_combined.size(0) - E_chem
            if E_rad > 0:
                e_rad = torch.zeros(E_rad, e_chem.size(1), device=x.device, dtype=x.dtype)
                e_all = torch.cat([e_chem, e_rad], dim=0)
            else:
                e_all = e_chem
            phi_e_input = torch.cat([h[src], h[dst], e_all, rbf_feat, edge_attr_combined], dim=-1)
        else:
            phi_e_input = torch.cat([h[src], h[dst], rbf_feat, edge_attr_combined], dim=-1)

        m_ij = self.phi_e(phi_e_input)

        if self.use_attention:
            att_coord = pyg_softmax(self.phi_att_coord(m_ij), dst, num_nodes=x.size(0))
            if self.use_split_attention:
                att_feat = pyg_softmax(self.phi_att_feat(m_ij), dst, num_nodes=x.size(0))
            else:
                att_feat = att_coord
        else:
            att_coord = None
            att_feat = None

        if self.use_edge_conditioned_phi_x:
            phi_x_input = torch.cat([m_ij, rbf_feat, edge_attr_combined], dim=-1)
        else:
            phi_x_input = m_ij
        x_scale = self.phi_x(phi_x_input)
        coord_msg = rel_pos * x_scale
        if att_coord is not None:
            agg_coord = scatter(coord_msg * att_coord, dst, dim=0, dim_size=x.size(0), reduce="sum")
        else:
            agg_coord = safe_scatter_mean(coord_msg, dst, dim_size=x.size(0))
        x_new = x + agg_coord

        disp = x_new - x
        disp_norm = disp.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        scale = (self.max_disp / disp_norm).clamp(max=1.0)
        x_new = x + disp * scale

        if att_feat is not None:
            agg_m = scatter(m_ij * att_feat, dst, dim=0, dim_size=h.size(0), reduce="sum")
        else:
            agg_m = safe_scatter_mean(m_ij, dst, dim_size=h.size(0))

        if self.use_angular:
            # idx_j, idx_k = build_triplets(edge_index, num_nodes=x.size(0)) # builds triplets between ALL 
            idx_j, idx_k = build_triplets(chem_edge_index, num_nodes=x.size(0)) # build triplets between only bonded atoms
            if idx_j.numel() > 0:
                src_j, src_k = src[idx_j], src[idx_k]
                center = dst[idx_j]
                rel_j, rel_k = rel_pos[idx_j], rel_pos[idx_k]
                dist_j, dist_k = dist[idx_j].squeeze(-1), dist[idx_k].squeeze(-1)
                cos_angle = (rel_j * rel_k).sum(-1) / (dist_j * dist_k).clamp(min=1e-6)
                cos_angle = cos_angle.clamp(-1.0, 1.0)
                triplet_in = torch.cat(
                    [h[src_j], h[src_k], cos_angle.unsqueeze(-1),
                     dist_j.unsqueeze(-1), dist_k.unsqueeze(-1)], dim=-1
                )
                triplet_msg = self.phi_angle(triplet_in)
                agg_angle = safe_scatter_mean(triplet_msg, center, dim_size=h.size(0))
            else:
                agg_angle = torch.zeros_like(h)
        else:
            agg_angle = torch.zeros_like(h)

        h_new = self.phi_h(torch.cat([h, agg_m, agg_angle], dim=-1))
        h_new = self.norm_h(h + h_new)

        e_chem_new = None
        if self.use_edge_embeddings:
            m_chem = m_ij[:E_chem]
            delta_e = self.phi_edge_update(torch.cat([e_chem, m_chem], dim=-1))
            e_chem_new = self.norm_e(e_chem + delta_e)

        return h_new, x_new, e_chem_new


class EGNN(nn.Module):
    """Stack of EGNN layers."""

    def __init__(self, num_layers=4, hidden_dim=64, edge_attr_dim=4, max_z=10,
                 max_disp=5.0, num_rbf=16, radius_cutoff=6.0,
                 use_radius_graph=True, use_angular=True, use_attention=True,
                 use_edge_conditioned_phi_x=True, use_split_attention=True,
                 use_edge_embeddings=True):
        super().__init__()
        self.embed = nn.Embedding(max_z, hidden_dim)
        self.use_edge_embeddings = use_edge_embeddings

        if use_edge_embeddings:
            self.edge_init = nn.Linear(edge_attr_dim, hidden_dim)

        self.layers = nn.ModuleList([
            EGNNLayer(hidden_dim, chem_edge_attr_dim=edge_attr_dim, max_disp=max_disp,
                     num_rbf=num_rbf, radius_cutoff=radius_cutoff,
                     use_radius_graph=use_radius_graph, use_angular=use_angular,
                     use_attention=use_attention,
                     use_edge_conditioned_phi_x=use_edge_conditioned_phi_x,
                     use_split_attention=use_split_attention,
                     use_edge_embeddings=use_edge_embeddings)
            for _ in range(num_layers)
        ])

    def forward(self, z, x0, edge_index, edge_attr=None, batch=None):
        if batch is None:
            batch = torch.zeros(z.size(0), dtype=torch.long, device=z.device)
        if edge_attr is None:
            edge_attr = torch.zeros(edge_index.size(1), 4, device=z.device, dtype=x0.dtype)

        h = self.embed(z)
        x = x0

        e_chem = None
        if self.use_edge_embeddings and edge_index.size(1) > 0:
            e_chem = self.edge_init(edge_attr.to(x.dtype))

        for layer in self.layers:
            h, x, e_chem = layer(h, x, edge_index, edge_attr, batch, e_chem)
        return x


# ----------------------------------------------------------------------
# 2. KABSCH-ALIGNED RMSD LOSS
# ----------------------------------------------------------------------

def kabsch_align(pred, target, batch, num_graphs=None):
    """Optimal per-molecule rotation via batched SVD."""
    if num_graphs is None:
        num_graphs = int(batch.max().item()) + 1

    pred_mean = scatter(pred, batch, dim=0, dim_size=num_graphs, reduce="mean")
    target_mean = scatter(target, batch, dim=0, dim_size=num_graphs, reduce="mean")
    pred_c = pred - pred_mean[batch]
    target_c = target - target_mean[batch]

    outer = torch.einsum("ni,nj->nij", pred_c, target_c)
    H = scatter(outer, batch, dim=0, dim_size=num_graphs, reduce="sum")

    eye = torch.eye(3, device=H.device, dtype=H.dtype).expand_as(H)
    H = H + 1e-6 * eye

    U, S, Vt = torch.linalg.svd(H)

    d = torch.sign(torch.det(torch.matmul(Vt.transpose(-2, -1), U.transpose(-2, -1))))
    correction = torch.zeros(num_graphs, 3, 3, device=pred.device, dtype=pred.dtype)
    correction[:, 0, 0] = 1.0
    correction[:, 1, 1] = 1.0
    correction[:, 2, 2] = d
    R = torch.matmul(torch.matmul(Vt.transpose(-2, -1), correction), U.transpose(-2, -1))

    pred_aligned = torch.einsum("nij,nj->ni", R[batch], pred_c)
    return pred_aligned, target_c


def kabsch_rmsd_loss(pred, target, batch):
    num_graphs = int(batch.max().item()) + 1
    pred_aligned, target_c = kabsch_align(pred, target, batch, num_graphs=num_graphs)
    sq_dist = ((pred_aligned - target_c) ** 2).sum(dim=-1)
    per_graph_mse = scatter(sq_dist, batch, dim=0, dim_size=num_graphs, reduce="mean")
    return per_graph_mse.sqrt().mean()



# 2b. ADDITIONAL METRICS AND LOSS TERMS
@torch.no_grad()
def compute_baselines(loader_or_list, device):
    total_centroid, total_random, n = 0.0, 0.0, 0
    for batch_data in loader_or_list:
        target = batch_data.pos
        batch = batch_data.batch
        G = int(batch.max().item()) + 1

        centroid_pred = scatter(target, batch, dim=0, dim_size=G, reduce="mean")[batch]
        random_pred = torch.randn_like(target) * target.std()

        for pred, acc in [(centroid_pred, "c"), (random_pred, "r")]:
            loss = kabsch_rmsd_loss(pred, target, batch)
            if acc == "c":
                total_centroid += loss.item() * batch_data.num_graphs
            else:
                total_random += loss.item() * batch_data.num_graphs
        n += batch_data.num_graphs
    return total_centroid / n, total_random / n


@torch.no_grad()
def compute_identity_baseline(loader_or_list, device, draw_index=0):
    total, n = 0.0, 0
    for batch_data in loader_or_list:
        pos_init = batch_data.pos_init_pool[:, draw_index]
        loss = kabsch_rmsd_loss(pos_init, batch_data.pos, batch_data.batch)
        total += loss.item() * batch_data.num_graphs
        n += batch_data.num_graphs
    return total / n


def bond_angle_loss(pred, target, edge_index):
    """Cosine-difference angle loss. Note: gradient is exactly zero at
    exact collapse (cos is stationary at angle=0), so this cannot by
    itself prevent symmetric atoms from collapsing onto each other. See
    symmetry_break_loss below for the term that can."""
    src, dst = edge_index
    idx_j, idx_k = build_triplets(edge_index, pred.size(0))
    if idx_j.numel() == 0:
        return torch.zeros((), device=pred.device, dtype=pred.dtype)
    center = dst[idx_j]
    src_j, src_k = src[idx_j], src[idx_k]

    def cos_angles(pos):
        rel_j = pos[src_j] - pos[center]
        rel_k = pos[src_k] - pos[center]
        dist_j = rel_j.norm(dim=-1).clamp(min=1e-6)
        dist_k = rel_k.norm(dim=-1).clamp(min=1e-6)
        cos = (rel_j * rel_k).sum(-1) / (dist_j * dist_k)
        return cos.clamp(-1.0, 1.0)

    return (cos_angles(pred) - cos_angles(target)).abs().mean()


@torch.no_grad()
def bond_angle_mae_degrees(pred, target, edge_index):
    src, dst = edge_index
    idx_j, idx_k = build_triplets(edge_index, pred.size(0))
    if idx_j.numel() == 0:
        return 0.0
    center = dst[idx_j]
    src_j, src_k = src[idx_j], src[idx_k]

    def angles_deg(pos):
        rel_j = pos[src_j] - pos[center]
        rel_k = pos[src_k] - pos[center]
        dist_j = rel_j.norm(dim=-1).clamp(min=1e-6)
        dist_k = rel_k.norm(dim=-1).clamp(min=1e-6)
        cos = ((rel_j * rel_k).sum(-1) / (dist_j * dist_k)).clamp(-1.0, 1.0)
        return torch.rad2deg(torch.acos(cos))

    return (angles_deg(pred) - angles_deg(target)).abs().mean().item()


def bond_length_loss(pred, target, edge_index):
    src, dst = edge_index
    pred_dist = (pred[src] - pred[dst]).norm(dim=-1)
    target_dist = (target[src] - target[dst]).norm(dim=-1)
    return (pred_dist - target_dist).abs().mean()


@torch.no_grad()
def bond_length_mae(pred, edge_index, target):
    src, dst = edge_index
    pred_dist = (pred[src] - pred[dst]).norm(dim=-1)
    true_dist = (target[src] - target[dst]).norm(dim=-1)
    return (pred_dist - true_dist).abs().mean()


def symmetry_break_loss(pred, edge_index, min_dist=1.0):
    """
    Distance-based barrier: penalize any two atoms bonded to the same
    center being closer than `min_dist`. Directly targets the symmetry
    collapse failure mode (CH2/CH3 hydrogens predicted at the same point).

    Threshold choice:
      Real H-H in a CH2 is ~1.8 A; real C-C-C in cyclopropane is ~1.5 A.
      min_dist=1.0 A is a loose safety threshold that catches true
      collapse without fighting the other losses for correctly-spaced
      atoms. In QM9 rings, any two atoms bonded to the same center are
      >1 A apart in the true geometry, so the barrier does not activate
      on correctly-predicted ring systems.

    The loss is averaged per-triplet, so its scale is invariant to
    batch size and molecule count.
    """
    src, dst = edge_index
    idx_j, idx_k = build_triplets(edge_index, pred.size(0))
    if idx_j.numel() == 0:
        return torch.zeros((), device=pred.device, dtype=pred.dtype)
    src_j, src_k = src[idx_j], src[idx_k]
    d = (pred[src_j] - pred[src_k]).norm(dim=-1)
    return torch.relu(min_dist - d).mean()



# 3. EQUIVARIANCE + KABSCH UNIT TESTS
def random_rotation_matrix(device, dtype):
    A = torch.randn(3, 3, device=device, dtype=dtype)
    Q, R = torch.linalg.qr(A)
    Q = Q * torch.sign(torch.diagonal(R)).unsqueeze(0)
    if torch.det(Q) < 0:
        Q[:, 0] *= -1
    return Q


def test_equivariance(num_nodes=8, hidden_dim=16, num_layers=3, edge_attr_dim=4,
                      max_disp=5.0, seed=0):
    torch.manual_seed(seed)
    dtype = torch.float64

    model = EGNN(num_layers=num_layers, hidden_dim=hidden_dim,
                 edge_attr_dim=edge_attr_dim, max_disp=max_disp,
                 use_edge_conditioned_phi_x=True,
                 use_split_attention=True,
                 use_edge_embeddings=True).to(dtype)

    z = torch.randint(1, 9, (num_nodes,))
    x0 = torch.randn(num_nodes, 3, dtype=dtype)

    src = torch.arange(num_nodes).repeat_interleave(num_nodes)
    dst = torch.arange(num_nodes).repeat(num_nodes)
    keep = src != dst
    edge_index = torch.stack([src[keep], dst[keep]])
    edge_attr = torch.randn(edge_index.size(1), edge_attr_dim, dtype=dtype)

    out1 = model(z, x0, edge_index, edge_attr)

    R = random_rotation_matrix(device=x0.device, dtype=dtype)
    t = torch.randn(1, 3, dtype=dtype)
    x0_transformed = x0 @ R.T + t

    out2 = model(z, x0_transformed, edge_index, edge_attr)
    out1_transformed = out1 @ R.T + t

    max_error = (out2 - out1_transformed).abs().max().item()
    print(f"Equivariance test max error: {max_error:.2e} (should be ~1e-10 or smaller)")
    assert max_error < 1e-6, "EGNN layer is NOT equivariant."
    print("PASSED: model is equivariant to rotation + translation.")


def test_kabsch_loss(num_nodes=10, seed=0):
    torch.manual_seed(seed)
    dtype = torch.float64

    target = torch.randn(num_nodes, 3, dtype=dtype)
    R = random_rotation_matrix(device=target.device, dtype=dtype)
    t = torch.randn(1, 3, dtype=dtype)
    pred = target @ R.T + t
    batch = torch.zeros(num_nodes, dtype=torch.long)
    loss = kabsch_rmsd_loss(pred, target, batch)
    print(f"Kabsch RMSD (single molecule): {loss.item():.2e} (should be ~1e-6 or smaller)")
    assert loss.item() < 1e-4, "Kabsch alignment is not working correctly."

    sizes = [6, 10, 15]
    targets, preds, batch_idx = [], [], []
    for g, n in enumerate(sizes):
        tg = torch.randn(n, 3, dtype=dtype)
        Rg = random_rotation_matrix(device=tg.device, dtype=dtype)
        tg_t = torch.randn(1, 3, dtype=dtype)
        pd = tg @ Rg.T + tg_t
        targets.append(tg)
        preds.append(pd)
        batch_idx.append(torch.full((n,), g, dtype=torch.long))
    targets = torch.cat(targets, dim=0)
    preds = torch.cat(preds, dim=0)
    batch_idx = torch.cat(batch_idx, dim=0)

    batch_loss = kabsch_rmsd_loss(preds, targets, batch_idx)
    print(f"Kabsch RMSD (batch of {len(sizes)} molecules): "
          f"{batch_loss.item():.2e} (should be ~1e-6 or smaller)")
    assert batch_loss.item() < 1e-4, "Batched Kabsch alignment is not working correctly."
    print("PASSED: Kabsch loss correctly recognizes rotated copies as perfect matches.")



# 4. INITIAL COORDINATE GUESS
def add_noise_initial_coords(pos, scale=1.5):
    return torch.randn_like(pos) * scale





# 5. TRAINING LOOP
def train(root="./data/QM9", epochs=300, batch_size=32, lr=1e-3,
          hidden_dim=128, num_layers=6, subset_size=10000, device=None,
          checkpoint_path=None, noise_scale=1.5, num_pos_init_draws=4,
          max_disp=5.0, num_rbf=16, radius_cutoff=6.0,
          use_radius_graph=True, use_angular=True, use_attention=True,
          bond_loss_weight=0.5, angle_loss_weight=1.0,
          symmetry_break_weight=1.0, symmetry_break_min_dist=1.0,
          use_edge_conditioned_phi_x=True, use_split_attention=True,
          use_edge_embeddings=True,
          gpu_resident_dataset=True,
          resume_from_checkpoint=True, checkpoint_every_n_epochs=1,
          subset_sampling="first", split_method="contiguous", split_seed=0,
          stratify_by=None, min_stratum_size=10):
    """
    Train EGNN on QM9.

    Loss = rmsd_loss
         + bond_loss_weight * bond_length_loss
         + angle_loss_weight * bond_angle_loss
         + symmetry_break_weight * symmetry_break_loss


    """
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Using device: {device}")

    print("Loading QM9...")
    full_dataset = QM9(root=root)
    print(f"QM9 contains {len(full_dataset)} molecules in total.")

    # --- Decide the split --------------------------------------------------
    # When resuming, the split MUST NOT change (a new split would leak old
    # validation molecules into training), so reuse what the checkpoint has.
    saved_split = None
    if resume_from_checkpoint and checkpoint_path is not None and os.path.exists(checkpoint_path):
        _ck = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        _cfg = _ck.get("config", {})
        del _ck
        if _cfg.get("split_indices") is not None:
            saved_split = _cfg["split_indices"]
            subset_size = _cfg.get("subset_size", subset_size)
            subset_sampling = _cfg.get("subset_sampling", subset_sampling)
            split_method = _cfg.get("split_method", split_method)
            split_seed = _cfg.get("split_seed", split_seed)
            stratify_by = _cfg.get("stratify_by", stratify_by)
            print(f"Resuming: reusing the train/val/test split stored in the checkpoint "
                  f"(method={split_method}).")
        else:
            if split_method != "contiguous" or subset_sampling != "first":
                print("Resuming a checkpoint that predates stored splits: keeping its "
                      "original contiguous split (first subset_size molecules, 80/10/10 in "
                      "order) so val/test stay unseen. To train under the new split settings, "
                      "set resume_from_checkpoint=False or use a new checkpoint_path.")
            subset_size = _cfg.get("subset_size", subset_size)
            subset_sampling, split_method = "first", "contiguous"

    if saved_split is not None:
        split_indices = {k: [int(i) for i in torch.as_tensor(saved_split[k]).tolist()]
                         for k in ("train", "val", "test")}
        split_info = None
    else:
        split_indices, split_info = resolve_splits(
            full_dataset, subset_size=subset_size, subset_sampling=subset_sampling,
            split_method=split_method, split_seed=split_seed, stratify_by=stratify_by,
            min_stratum_size=min_stratum_size)
        print_split_balance(split_indices, split_info, stratify_by)
    split_index_tensors = {k: torch.tensor(v, dtype=torch.long) for k, v in split_indices.items()}

    train_set = load_split_data(full_dataset, split_indices["train"])
    val_set = load_split_data(full_dataset, split_indices["val"])
    test_set = load_split_data(full_dataset, split_indices["test"])
    print(f"Split: {len(train_set)} train / {len(val_set)} val / {len(test_set)} test molecules.")

    K = num_pos_init_draws
    for d in train_set + val_set + test_set:
        d.pos_init_pool = torch.stack(
            [add_noise_initial_coords(d.pos, scale=noise_scale) for _ in range(K)],
            dim=1,
        )
    print(f"Built pos_init_pool: K={K} noise draws per molecule.")

    # --- GPU-resident dataset (optional, per CONFIG) ---
    use_gpu_resident = gpu_resident_dataset and device.type == "cuda"
    if gpu_resident_dataset and device.type != "cuda":
        print("NOTE: gpu_resident_dataset=True but device is not CUDA. "
              "Falling back to standard DataLoader path.")

    if use_gpu_resident:
        print(f"Moving train ({len(train_set)}) and val ({len(val_set)}) splits "
              f"to {device}...")
        try:
            for d in train_set:
                d.to(device)
            for d in val_set:
                d.to(device)
            print(f"  OK -- dataset resident on GPU.")
        except RuntimeError as e:
            raise RuntimeError(
                f"Failed to move dataset to GPU (likely VRAM exhausted): {e}\n"
                f"Set CONFIG['gpu_resident_dataset'] = False to fall back to "
                f"the DataLoader path, or reduce subset_size / num_pos_init_draws."
            ) from e

        def gpu_batches(ds, bs, shuffle=True):
            n_local = len(ds)
            idx = torch.randperm(n_local).tolist() if shuffle else list(range(n_local))
            for start in range(0, n_local, bs):
                batch_idx = idx[start:start + bs]
                yield Batch.from_data_list([ds[i] for i in batch_idx])

        val_batches = list(gpu_batches(val_set, batch_size, shuffle=False))
        train_loader = None
        val_loader = val_batches
    else:
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                                  num_workers=0, pin_memory=True)
        val_loader = DataLoader(val_set, batch_size=batch_size,
                                num_workers=0, pin_memory=True)

    # --- Resume ---
    start_epoch = 1
    optimizer_state = None
    ckpt = None
    history = []
    if resume_from_checkpoint and checkpoint_path is not None and os.path.exists(checkpoint_path):
        print(f"Found existing checkpoint at {checkpoint_path} -- resuming training.")
        ckpt = torch.load(checkpoint_path, map_location=device)
        ckpt_cfg = ckpt.get("config", {})

        arch_keys = ["hidden_dim", "num_layers", "max_disp", "num_rbf", "radius_cutoff",
                     "use_radius_graph", "use_angular", "use_attention",
                     "use_edge_conditioned_phi_x", "use_split_attention",
                     "use_edge_embeddings"]
        overrides = {k: ckpt_cfg[k] for k in arch_keys if k in ckpt_cfg}
        changed = {k: v for k, v in overrides.items() if locals().get(k) != v}
        if changed:
            print(f"  Using checkpoint's saved architecture: {changed}")
        hidden_dim = overrides.get("hidden_dim", hidden_dim)
        num_layers = overrides.get("num_layers", num_layers)
        max_disp = overrides.get("max_disp", max_disp)
        num_rbf = overrides.get("num_rbf", num_rbf)
        radius_cutoff = overrides.get("radius_cutoff", radius_cutoff)
        use_radius_graph = overrides.get("use_radius_graph", use_radius_graph)
        use_angular = overrides.get("use_angular", use_angular)
        use_attention = overrides.get("use_attention", use_attention)
        use_edge_conditioned_phi_x = overrides.get("use_edge_conditioned_phi_x", use_edge_conditioned_phi_x)
        use_split_attention = overrides.get("use_split_attention", use_split_attention)
        use_edge_embeddings = overrides.get("use_edge_embeddings", use_edge_embeddings)

        start_epoch = ckpt.get("epoch", 0) + 1
        optimizer_state = ckpt.get("optimizer_state_dict")
        history = ckpt.get("history", [])
        if start_epoch > epochs:
            print(f"  WARNING: checkpoint at epoch {ckpt.get('epoch', 0)} >= requested "
                  f"epochs={epochs}. Raise CONFIG['epochs'] to keep training.")

    model = EGNN(num_layers=num_layers, hidden_dim=hidden_dim, edge_attr_dim=4,
                 max_disp=max_disp, num_rbf=num_rbf, radius_cutoff=radius_cutoff,
                 use_radius_graph=use_radius_graph, use_angular=use_angular,
                 use_attention=use_attention,
                 use_edge_conditioned_phi_x=use_edge_conditioned_phi_x,
                 use_split_attention=use_split_attention,
                 use_edge_embeddings=use_edge_embeddings).to(device)

    if ckpt is not None:
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"  Loaded model weights from checkpoint (resuming at epoch {start_epoch}).")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    if optimizer_state is not None:
        optimizer.load_state_dict(optimizer_state)
        print(f"  Loaded optimizer state from checkpoint.")

    print(f"Model: {sum(p.numel() for p in model.parameters()):,} parameters. "
          f"radius_graph={use_radius_graph} (cutoff={radius_cutoff}A)  "
          f"angular={use_angular}  attention={use_attention}  "
          f"edge_phi_x={use_edge_conditioned_phi_x}  split_attn={use_split_attention}  "
          f"edge_emb={use_edge_embeddings}  "
          f"gpu_resident_dataset={use_gpu_resident}")
    print(f"Loss weights: bond={bond_loss_weight}  angle={angle_loss_weight}  "
          f"symmetry_break={symmetry_break_weight} (min_dist={symmetry_break_min_dist}A)")

    def lr_at(epoch, peak_lr=lr, floor_lr=1e-5):
        progress = epoch / max(epochs - 1, 1)
        return floor_lr + 0.5 * (peak_lr - floor_lr) * (1 + math.cos(math.pi * progress))

    def check_embedding_range(batch_data, split_name):
        z_min, z_max = int(batch_data.z.min()), int(batch_data.z.max())
        if z_min < 0 or z_max >= model.embed.num_embeddings:
            raise ValueError(
                f"Atomic number out of embedding range in {split_name} batch: "
                f"z ranges [{z_min}, {z_max}], embedding supports "
                f"[0, {model.embed.num_embeddings - 1}]."
            )

    def sample_pos_init(batch_data):
        pool = batch_data.pos_init_pool
        G = int(batch_data.batch.max().item()) + 1
        per_mol_idx = torch.randint(0, K, (G,), device=pool.device)
        per_atom_idx = per_mol_idx[batch_data.batch]
        return pool[torch.arange(pool.size(0), device=pool.device), per_atom_idx]

    # --- Baselines ---
    print("\nComputing trivial baselines...")
    centroid_rmsd, random_rmsd = compute_baselines(val_loader, device)
    print(f"  Baseline RMSD -- centroid:                {centroid_rmsd:.4f}")
    print(f"  Baseline RMSD -- random guess:            {random_rmsd:.4f}")
    identity_rmsd = compute_identity_baseline(val_loader, device, draw_index=0)
    print(f"  Baseline RMSD -- identity pos_init draw:  {identity_rmsd:.4f}\n")

    for epoch in range(start_epoch, epochs + 1):
        current_lr = lr_at(epoch - 1)
        for g in optimizer.param_groups:
            g["lr"] = current_lr

        model.train()
        total_train_loss = 0.0
        total_train_bond_mae = 0.0
        total_train_angle_mae = 0.0
        total_train_sym_break = 0.0
        n_train_seen = 0

        if use_gpu_resident:
            train_iter = gpu_batches(train_set, batch_size, shuffle=True)
        else:
            train_iter = train_loader

        for batch_data in train_iter:
            batch_data = batch_data.to(device)
            check_embedding_range(batch_data, "training")

            pos_init = sample_pos_init(batch_data)

            optimizer.zero_grad()
            pred = model(batch_data.z, pos_init, batch_data.edge_index,
                        batch_data.edge_attr, batch=batch_data.batch)

            if not torch.isfinite(pred).all():
                bad_graphs = batch_data.batch[~torch.isfinite(pred).any(dim=-1)].unique()
                print(f"  WARNING: non-finite coords for {len(bad_graphs)} molecule(s); skipping batch")
                optimizer.zero_grad()
                continue

            rmsd_loss = kabsch_rmsd_loss(pred, batch_data.pos, batch_data.batch)
            bl_loss = bond_length_loss(pred, batch_data.pos, batch_data.edge_index)
            angle_loss = bond_angle_loss(pred, batch_data.pos, batch_data.edge_index)
            sym_loss = symmetry_break_loss(pred, batch_data.edge_index,
                                           min_dist=symmetry_break_min_dist)
            loss = (rmsd_loss
                    + bond_loss_weight * bl_loss
                    + angle_loss_weight * angle_loss
                    + symmetry_break_weight * sym_loss)
            if not torch.isfinite(loss):
                print(f"  WARNING: non-finite loss ({loss.item()}); skipping batch")
                optimizer.zero_grad()
                continue

            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

            if not torch.isfinite(grad_norm):
                print(f"  WARNING: non-finite grad_norm ({grad_norm.item()}); skipping step")
                optimizer.zero_grad()
                continue

            if grad_norm > 100.0:
                print(f"  LARGE GRAD: norm={grad_norm.item():.1f}  graphs={batch_data.num_graphs}")

            optimizer.step()

            total_train_loss += rmsd_loss.item() * batch_data.num_graphs
            total_train_bond_mae += (
                bond_length_mae(pred.detach(), batch_data.edge_index, batch_data.pos).item()
                * batch_data.num_graphs
            )
            total_train_angle_mae += (
                bond_angle_mae_degrees(pred.detach(), batch_data.pos, batch_data.edge_index)
                * batch_data.num_graphs
            )
            total_train_sym_break += (
                sym_loss.item() * batch_data.num_graphs
            )
            n_train_seen += batch_data.num_graphs

        avg_train_loss = total_train_loss / max(n_train_seen, 1)
        avg_train_bond_mae = total_train_bond_mae / max(n_train_seen, 1)
        avg_train_angle_mae = total_train_angle_mae / max(n_train_seen, 1)
        avg_train_sym_break = total_train_sym_break / max(n_train_seen, 1)

        # --- Validation ---
        model.eval()
        total_val_loss = 0.0
        total_val_bond_mae = 0.0
        total_val_angle_mae = 0.0
        total_val_sym_break = 0.0
        total_val_draw_var = 0.0
        n_val_seen = 0

        with torch.no_grad():
            for batch_data in val_loader:
                batch_data = batch_data.to(device)
                check_embedding_range(batch_data, "validation")

                pool = batch_data.pos_init_pool
                K_batch = pool.size(1)
                batch = batch_data.batch
                G = int(batch.max().item()) + 1

                aligned_sum = torch.zeros_like(batch_data.pos)
                individual_aligned = []
                target_c = None

                for k in range(K_batch):
                    pos_init_k = pool[:, k]
                    pred_k = model(batch_data.z, pos_init_k,
                                   batch_data.edge_index, batch_data.edge_attr,
                                   batch=batch_data.batch)
                    aligned_k, target_c = kabsch_align(
                        pred_k, batch_data.pos, batch, num_graphs=G
                    )
                    aligned_sum += aligned_k
                    individual_aligned.append(aligned_k)

                aligned_mean = aligned_sum / K_batch
                sq_dist = ((aligned_mean - target_c) ** 2).sum(dim=-1)
                per_graph_mse = scatter(sq_dist, batch, dim=0, dim_size=G, reduce="mean")
                loss = per_graph_mse.sqrt().mean()

                total_val_bond_mae += (
                    bond_length_mae(aligned_mean, batch_data.edge_index,
                                    batch_data.pos).item() * batch_data.num_graphs
                )
                total_val_angle_mae += (
                    bond_angle_mae_degrees(aligned_mean, batch_data.pos,
                                           batch_data.edge_index) * batch_data.num_graphs
                )
                total_val_sym_break += (
                    symmetry_break_loss(aligned_mean, batch_data.edge_index,
                                        min_dist=symmetry_break_min_dist).item()
                    * batch_data.num_graphs
                )

                if K_batch > 1:
                    stack = torch.stack(individual_aligned, dim=0)
                    pair_sum = 0.0
                    pair_count = 0
                    for i in range(K_batch):
                        for j in range(i + 1, K_batch):
                            d_ij = (stack[i] - stack[j]).norm(dim=-1)
                            per_mol_rmsd = scatter(
                                d_ij ** 2, batch, dim=0, dim_size=G, reduce="mean"
                            ).sqrt()
                            pair_sum += per_mol_rmsd.mean().item()
                            pair_count += 1
                    total_val_draw_var += (pair_sum / pair_count) * batch_data.num_graphs

                total_val_loss += loss.item() * batch_data.num_graphs
                n_val_seen += batch_data.num_graphs

        avg_val_loss = total_val_loss / max(n_val_seen, 1)
        avg_val_bond_mae = total_val_bond_mae / max(n_val_seen, 1)
        avg_val_angle_mae = total_val_angle_mae / max(n_val_seen, 1)
        avg_val_sym_break = total_val_sym_break / max(n_val_seen, 1)
        avg_val_draw_var = total_val_draw_var / max(n_val_seen, 1)

        print(f"Epoch {epoch:3d}/{epochs}  lr={current_lr:.2e}  "
              f"train_rmsd={avg_train_loss:.4f}  val_rmsd={avg_val_loss:.4f}  "
              f"train_bond_mae={avg_train_bond_mae:.4f}  val_bond_mae={avg_val_bond_mae:.4f}  "
              f"train_angle_mae={avg_train_angle_mae:.1f}deg  val_angle_mae={avg_val_angle_mae:.1f}deg  "
              f"train_sym_break={avg_train_sym_break:.4f}  val_sym_break={avg_val_sym_break:.4f}  "
              f"val_draw_var={avg_val_draw_var:.4f}")

        history.append({
            "epoch": epoch, "lr": current_lr,
            "train_rmsd": avg_train_loss, "val_rmsd": avg_val_loss,
            "train_bond_mae": avg_train_bond_mae, "val_bond_mae": avg_val_bond_mae,
            "train_angle_mae": avg_train_angle_mae, "val_angle_mae": avg_val_angle_mae,
            "train_sym_break": avg_train_sym_break, "val_sym_break": avg_val_sym_break,
            "val_draw_var": avg_val_draw_var,
        })

        for name, p in model.named_parameters():
            if not torch.isfinite(p).all():
                raise RuntimeError(
                    f"Parameter '{name}' is non-finite after epoch {epoch}. "
                    f"Training has been poisoned; reload from the last good "
                    f"checkpoint or reduce lr / max_disp."
                )

        def save_checkpoint(completed_epoch):
            if checkpoint_path is None:
                return
            torch.save({
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": completed_epoch,
                "history": history,
                "config": {
                    "root": root, "subset_size": subset_size,
                    "subset_sampling": subset_sampling, "split_method": split_method,
                    "split_seed": split_seed, "stratify_by": stratify_by,
                    "min_stratum_size": min_stratum_size,
                    "split_indices": split_index_tensors,
                    "hidden_dim": hidden_dim, "num_layers": num_layers, "edge_attr_dim": 4,
                    "max_disp": max_disp, "num_rbf": num_rbf, "radius_cutoff": radius_cutoff,
                    "use_radius_graph": use_radius_graph, "use_angular": use_angular,
                    "use_attention": use_attention, "noise_scale": noise_scale,
                    "num_pos_init_draws": num_pos_init_draws,
                    "bond_loss_weight": bond_loss_weight,
                    "angle_loss_weight": angle_loss_weight,
                    "symmetry_break_weight": symmetry_break_weight,
                    "symmetry_break_min_dist": symmetry_break_min_dist,
                    "use_edge_conditioned_phi_x": use_edge_conditioned_phi_x,
                    "use_split_attention": use_split_attention,
                    "use_edge_embeddings": use_edge_embeddings,
                },
            }, checkpoint_path)

        if checkpoint_every_n_epochs and epoch % checkpoint_every_n_epochs == 0:
            save_checkpoint(epoch)
            print(f"  Saved checkpoint at epoch {epoch} to {checkpoint_path}")

    if checkpoint_path is not None and epochs >= start_epoch:
        save_checkpoint(epochs)
        print(f"Saved final model weights + optimizer state to {checkpoint_path}")

    return model, (train_set, val_set, test_set), history



# 6. ENTRY POINT
def main():
    if CONFIG["run_tests"]:
        test_equivariance(max_disp=CONFIG["max_disp"])
        test_kabsch_loss()

    if CONFIG["run_train"]:
        train(root=CONFIG["root"], epochs=CONFIG["epochs"],
              batch_size=CONFIG["batch_size"], lr=CONFIG["lr"],
              hidden_dim=CONFIG["hidden_dim"], num_layers=CONFIG["num_layers"],
              subset_size=CONFIG["subset_size"], checkpoint_path=CONFIG["checkpoint_path"],
              subset_sampling=CONFIG["subset_sampling"], split_method=CONFIG["split_method"],
              split_seed=CONFIG["split_seed"], stratify_by=CONFIG["stratify_by"],
              min_stratum_size=CONFIG["min_stratum_size"],
              noise_scale=CONFIG["noise_scale"],
              num_pos_init_draws=CONFIG["num_pos_init_draws"],
              max_disp=CONFIG["max_disp"],
              num_rbf=CONFIG["num_rbf"], radius_cutoff=CONFIG["radius_cutoff"],
              use_radius_graph=CONFIG["use_radius_graph"], use_angular=CONFIG["use_angular"],
              use_attention=CONFIG["use_attention"], bond_loss_weight=CONFIG["bond_loss_weight"],
              angle_loss_weight=CONFIG["angle_loss_weight"],
              symmetry_break_weight=CONFIG["symmetry_break_weight"],
              symmetry_break_min_dist=CONFIG["symmetry_break_min_dist"],
              use_edge_conditioned_phi_x=CONFIG["use_edge_conditioned_phi_x"],
              use_split_attention=CONFIG["use_split_attention"],
              use_edge_embeddings=CONFIG["use_edge_embeddings"],
              gpu_resident_dataset=CONFIG["gpu_resident_dataset"],
              resume_from_checkpoint=CONFIG["resume_from_checkpoint"],
              checkpoint_every_n_epochs=CONFIG["checkpoint_every_n_epochs"])


if __name__ == "__main__":
    main()