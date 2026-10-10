"""
egnn_inference.py
=================

Self-contained EGNN model definition for inference.

This is a clean extraction of the model architecture from the training
script. It has no dependency on the training loop, the optimizer, the
QM9 dataset, or the loss functions -- only what's needed to define the
model and run a forward pass.

If the model architecture changes in the training script, update this
file to match. Keeping it separate means the inference server doesn't
break when training code is refactored.
"""

import torch
import torch.nn as nn
from torch_geometric.utils import scatter, to_dense_batch
from torch_geometric.utils import softmax as pyg_softmax


# ----------------------------------------------------------------------
# Model building blocks
# ----------------------------------------------------------------------

class MLP(nn.Module):
    """Two-layer MLP used inside phi_e / phi_x / phi_h / attention heads."""

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
    """Expands a scalar distance into Gaussian radial basis features."""

    def __init__(self, num_basis=16, cutoff=10.0):
        super().__init__()
        centers = torch.linspace(0, cutoff, num_basis)
        self.register_buffer("centers", centers)
        self.width = cutoff / num_basis

    def forward(self, dist):
        return torch.exp(-((dist - self.centers) ** 2) / (2 * self.width ** 2))


def safe_scatter_mean(src, index, dim_size):
    """Mean aggregation with clamp + nan_to_num safety net."""
    summed = scatter(src, index, dim=0, dim_size=dim_size, reduce="sum")
    ones = torch.ones(src.size(0), 1, device=src.device, dtype=src.dtype)
    count = scatter(ones, index, dim=0, dim_size=dim_size, reduce="sum").clamp(min=1)
    return torch.nan_to_num(summed / count, nan=0.0, posinf=0.0, neginf=0.0)


@torch.no_grad()
def build_radius_edges(x, batch, cutoff, exclude_edge_index=None, max_neighbors=6):
    """Build non-bonded edges within `cutoff`, capped at max_neighbors per node."""
    N = x.size(0)
    xd, mask = to_dense_batch(x.detach(), batch)
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
    return torch.stack([gidx[g, j], gidx[g, i]])


def build_triplets(edge_index, num_nodes):
    """Vectorized enumeration of unordered neighbor pairs per center atom."""
    dst = edge_index[1]
    E = dst.size(0)
    if E == 0:
        empty = torch.empty(0, dtype=torch.long, device=edge_index.device)
        return empty, empty

    perm = torch.argsort(dst, stable=True)
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

    return perm[row_sorted_idx], perm[col_sorted_idx]


# ----------------------------------------------------------------------
# EGNN layer and full model
# ----------------------------------------------------------------------

class EGNNLayer(nn.Module):
    """One equivariant message-passing layer. See training script for details."""

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

        phi_x_in_dim = (hidden_dim + num_rbf + combined_edge_attr_dim
                        if use_edge_conditioned_phi_x else hidden_dim)
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
                e_dim = self.phi_e.net[0].in_features - 2 * h.size(1) \
                        - self.rbf.centers.numel() - edge_attr_combined.size(-1)
                e_chem = torch.zeros(E_chem, e_dim, device=x.device, dtype=x.dtype)
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
            att_feat = (pyg_softmax(self.phi_att_feat(m_ij), dst, num_nodes=x.size(0))
                        if self.use_split_attention else att_coord)
        else:
            att_coord = att_feat = None

        phi_x_input = (torch.cat([m_ij, rbf_feat, edge_attr_combined], dim=-1)
                       if self.use_edge_conditioned_phi_x else m_ij)
        x_scale = self.phi_x(phi_x_input)
        coord_msg = rel_pos * x_scale
        agg_coord = (scatter(coord_msg * att_coord, dst, dim=0, dim_size=x.size(0), reduce="sum")
                     if att_coord is not None
                     else safe_scatter_mean(coord_msg, dst, dim_size=x.size(0)))
        x_new = x + agg_coord

        disp = x_new - x
        disp_norm = disp.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        scale = (self.max_disp / disp_norm).clamp(max=1.0)
        x_new = x + disp * scale

        agg_m = (scatter(m_ij * att_feat, dst, dim=0, dim_size=h.size(0), reduce="sum")
                 if att_feat is not None
                 else safe_scatter_mean(m_ij, dst, dim_size=h.size(0)))

        if self.use_angular:
            idx_j, idx_k = build_triplets(chem_edge_index, num_nodes=x.size(0))
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
    """Stack of EGNN layers: graph + noise init -> refined coordinates."""

    def __init__(self, num_layers=8, hidden_dim=96, edge_attr_dim=4, max_z=10,
                 max_disp=5.0, num_rbf=16, radius_cutoff=4.0,
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
        e_chem = (self.edge_init(edge_attr.to(x.dtype))
                  if self.use_edge_embeddings and edge_index.size(1) > 0 else None)

        for layer in self.layers:
            h, x, e_chem = layer(h, x, edge_index, edge_attr, batch, e_chem)
        return x


# ----------------------------------------------------------------------
# Kabsch alignment (for averaging noise draws at inference)
# ----------------------------------------------------------------------

def kabsch_align_rows(pred, target):
    """
    Rotate pred onto target via Kabsch. Both are [N, 3] tensors.
    No centering is done inside this function -- caller should center.
    Returns the rotation matrix R such that pred @ R.T is aligned to target.
    """
    H = pred.T @ target
    U, S, Vt = torch.linalg.svd(H)
    d = torch.sign(torch.det(Vt.T @ U.T))
    D = torch.diag(torch.tensor([1.0, 1.0, d], device=pred.device, dtype=pred.dtype))
    return Vt.T @ D @ U.T


# ----------------------------------------------------------------------
# Model loading
# ----------------------------------------------------------------------

def load_model(checkpoint_path, device="cuda", verbose=True):
    """
    Load a trained EGNN from a training checkpoint.

    The checkpoint is expected to be the standard output of the training
    script: a dict with keys 'model_state_dict' and 'config'. Only the
    architecture-relevant parts of config are used; training-only settings
    (optimizer, history, split indices) are ignored.
    """
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]

    model = EGNN(
        num_layers=cfg["num_layers"],
        hidden_dim=cfg["hidden_dim"],
        edge_attr_dim=cfg["edge_attr_dim"],
        max_disp=cfg["max_disp"],
        num_rbf=cfg["num_rbf"],
        radius_cutoff=cfg["radius_cutoff"],
        use_radius_graph=cfg["use_radius_graph"],
        use_angular=cfg["use_angular"],
        use_attention=cfg["use_attention"],
        use_edge_conditioned_phi_x=cfg["use_edge_conditioned_phi_x"],
        use_split_attention=cfg["use_split_attention"],
        use_edge_embeddings=cfg["use_edge_embeddings"],
    ).to(device)

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    if verbose:
        n_params = sum(p.numel() for p in model.parameters())
        print(f"  Loaded EGNN: {n_params:,} parameters")
        print(f"  Config: hidden_dim={cfg['hidden_dim']}, num_layers={cfg['num_layers']}, "
              f"radius_cutoff={cfg['radius_cutoff']}, radius_graph={cfg['use_radius_graph']}")
        print(f"  Inference settings: noise_scale={cfg['noise_scale']}, "
              f"num_pos_init_draws={cfg['num_pos_init_draws']}")

    return model, cfg