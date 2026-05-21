"""DSF-Direct: Neighbor-Preserving Repulsion (NPR), k-NN Preservation Loss (KPL),
and Reciprocal Neighbor Consistency Loss (RNCL).

Module A — NPR: modulates repulsion gradient by k-NN proximity instead of exposure.
  "Don't push apart items that are semantic neighbors."

Module B — KPL: direct InfoNCE loss with k-NN neighbors as positives.
  Complementary to QuaSID InfoNCE (co-occurrence positives) — preserves semantic topology.

Module C — RNCL: gentle cosine-distance loss on reciprocal k-NN pairs only.
  "Actively pull together the strongest, mutually-closest semantic pairs."
  Unlike KPL: reciprocal only (not all k-NN), cosine distance (not InfoNCE), 10x smaller λ.
"""

import logging
from typing import Optional, Tuple

import torch
import torch.nn.functional as F

log = logging.getLogger(__name__)


def load_knn_g0(filepath: str, num_items: int, device: torch.device) -> torch.Tensor:
    """Load G0 k-NN index tensor, slice to num_items, move to device.

    Returns (num_items, K) LongTensor.
    """
    knn = torch.load(filepath, map_location="cpu", weights_only=True)
    knn = knn[:num_items].to(device).long()
    log.info("Loaded G0 k-NN indices: shape=%s from %s", tuple(knn.shape), filepath)
    return knn


def build_npr_mask(
    knn_g0_indices: torch.Tensor,
    item_ids: torch.Tensor,
    npr_alpha_min: float = 0.01,
) -> torch.Tensor:
    """Build alpha_eff mask for Neighbor-Preserving Repulsion.

    For each collision pair (i, j):
      - If j ∈ kNN_G0(i): j is i's semantic neighbor → protect (alpha = npr_alpha_min)
      - If j ∉ kNN_G0(i): j is a distant item → safe to repel fully (alpha = 1.0)

    Args:
        knn_g0_indices: (N, K) precomputed G0 k-NN indices (item indices, global).
        item_ids: (B,) global item IDs for the current batch.
        npr_alpha_min: minimum alpha for protected pairs (default 0.01).

    Returns:
        alpha_matrix: (B, B) float tensor. alpha_matrix[i, j] = gradient scale
                      applied to side-i when repelling pair (i, j).
    """
    B = item_ids.shape[0]
    device = item_ids.device

    # Map global item IDs → batch indices
    # knn_g0_indices[item_ids[i]] gives the global k-NN item IDs for batch item i
    iids = item_ids.long()
    knn_global = knn_g0_indices[iids]  # (B, K) — global item IDs of G0 k-NN

    # For each pair (i, j): is j's global ID among i's G0 k-NN?
    # Expand: compare (B, 1, K) vs (1, B, 1) → (B, B, K) → any(K)
    knn_global_expanded = knn_global.unsqueeze(1)  # (B, 1, K)
    j_global = iids.unsqueeze(0).unsqueeze(-1)  # (1, B, 1)
    is_neighbor = (knn_global_expanded == j_global).any(dim=-1)  # (B, B) bool

    alpha = torch.full((B, B), 1.0, device=device)
    alpha[is_neighbor] = npr_alpha_min

    # Never protect self-pairs (should already be excluded by CVPM, but safety first)
    alpha.fill_diagonal_(1.0)

    return alpha


def build_reciprocal_mask(
    knn_g0_indices: torch.Tensor,
    item_ids: torch.Tensor,
) -> torch.Tensor:
    """Build boolean mask for reciprocal k-NN pairs.

    A pair (i, j) is reciprocal if j ∈ kNN(i) AND i ∈ kNN(j).
    Reciprocal pairs are the strongest, most reliable semantic relationships —
    only these pairs receive RNCL attraction (unlike KPL which pulls all k-NN).

    Args:
        knn_g0_indices: (N, K) precomputed k-NN indices (item indices, global).
        item_ids: (B,) global item IDs for the current batch.

    Returns:
        reciprocal: (B, B) bool tensor. reciprocal[i, j] = True iff pair is reciprocal.
    """
    B = item_ids.shape[0]
    device = item_ids.device

    iids = item_ids.long()
    knn_global = knn_g0_indices[iids]  # (B, K)

    # j ∈ kNN(i)?
    knn_global_expanded = knn_global.unsqueeze(1)  # (B, 1, K)
    j_global = iids.unsqueeze(0).unsqueeze(-1)  # (1, B, 1)
    j_in_knn_i = (knn_global_expanded == j_global).any(dim=-1)  # (B, B) bool

    # Reciprocal: j ∈ kNN(i) AND i ∈ kNN(j)
    reciprocal = j_in_knn_i & j_in_knn_i.T
    reciprocal.fill_diagonal_(False)

    return reciprocal


def compute_kpl_loss(
    z_online: torch.Tensor,
    z_target: torch.Tensor,
    knn_g0_indices: torch.Tensor,
    item_ids: torch.Tensor,
    tau: float = 0.5,
    cluster_ids: Optional[torch.Tensor] = None,
    collision_exclude_radius: int = 0,
) -> torch.Tensor:
    """k-NN Preservation Loss — InfoNCE with G0 k-NN neighbors as positives.

    For each anchor i with at least one G0 k-NN neighbor in the batch:
        L_i = -log( Σ_{j∈P_i} exp(S_{i,j}) / Σ_{k: k≠i} exp(S_{i,k}) )
    where P_i = {j: item_j ∈ kNN_G0(item_i), j in current batch}.

    Collision-aware mode (cluster_ids provided, collision_exclude_radius ≥ 0):
      Excludes positives j where Hamming(i, j) ≤ collision_exclude_radius.
      This prevents KPL from pulling together items that VCF is actively repelling,
      eliminating the gradient conflict that destabilizes the codebook.

      - collision_exclude_radius=0: exclude only full collisions (H=0, same SID)
      - collision_exclude_radius=2: exclude all collision pairs (H ≤ hamming_radius)

    Args:
        z_online: (B, D) online encoder outputs (gradients flow through here).
        z_target: (B, D) target/anchored encoder outputs (no gradients — dual-tower).
        knn_g0_indices: (N, K) precomputed G0 k-NN global item indices.
        item_ids: (B,) global item IDs for the current batch.
        tau: temperature (default 0.5, matching QuaSID InfoNCE).
        cluster_ids: (B, L) current SID token assignments (optional, for collision-aware).
        collision_exclude_radius: exclude KPL positives where H ≤ this value.

    Returns:
        scalar InfoNCE loss averaged over anchors with ≥1 positive.
    """
    B = z_online.shape[0]
    if B < 2:
        return torch.tensor(0.0, device=z_online.device, dtype=torch.float32)

    device = z_online.device
    iids = item_ids.long()

    # — Build positive pair matrix from G0 k-NN —
    knn_global = knn_g0_indices[iids]  # (B, K)
    knn_expanded = knn_global.unsqueeze(1)  # (B, 1, K)
    j_global = iids.unsqueeze(0).unsqueeze(-1)  # (1, B, 1)
    pos_mask = (knn_expanded == j_global).any(dim=-1)  # (B, B), True if j ∈ kNN_G0(i)

    # — Collision-aware filtering: exclude positives where VCF is already active —
    if cluster_ids is not None and collision_exclude_radius >= 0:
        # Hamming distance over SID tokens: H[i,j] = count of differing layers
        H = (cluster_ids.unsqueeze(0) != cluster_ids.unsqueeze(1)).sum(dim=-1)  # (B, B)
        collision_mask = H <= collision_exclude_radius  # VCF-active pairs
        pos_mask = pos_mask & ~collision_mask            # Exclude them from KPL

    # — Compute logits —
    z_online_norm = F.normalize(z_online, dim=1)
    z_target_norm = F.normalize(z_target, dim=1)
    S = torch.mm(z_online_norm, z_target_norm.t()) / tau  # (B, B)

    # — Mask: exclude self and same-ID items (same as QuaSID InfoNCE) —
    valid_mask = torch.ones(B, B, dtype=torch.bool, device=device)
    valid_mask.fill_diagonal_(False)
    # same-ID false-negative exclusion
    same_id = iids.unsqueeze(0) == iids.unsqueeze(1)
    valid_mask = valid_mask & ~same_id

    # — Per-anchor loss —
    losses = []
    for i in range(B):
        positives = pos_mask[i] & valid_mask[i]
        n_pos = positives.sum().item()
        if n_pos == 0:
            continue
        neg_mask = valid_mask[i]  # log-sum-exp over all valid items
        # Stability: subtract max before exp
        s_max = S[i, neg_mask].max()
        s_pos = (S[i, positives] - s_max).exp().sum()
        s_all = (S[i, neg_mask] - s_max).exp().sum()
        L_i = -torch.log(s_pos / s_all.clamp(min=1e-12))
        losses.append(L_i)

    if not losses:
        return torch.tensor(0.0, device=device, dtype=torch.float32)

    return torch.stack(losses).mean()


def compute_rncl_loss(
    z_online: torch.Tensor,
    z_target: torch.Tensor,
    reciprocal_mask: torch.Tensor,
) -> torch.Tensor:
    """Reciprocal Neighbor Consistency Loss — gentle cosine-distance on reciprocal pairs.

    Unlike KPL which uses InfoNCE (softmax over all batch items creates global
    competition), RNCL uses a simple pairwise cosine distance averaged over
    reciprocal pairs only.  This is a pure attractive force with no denominator —
    it pulls reciprocal neighbors together without pushing anything apart.

    Args:
        z_online: (B, D) online encoder outputs (gradients flow through here).
        z_target: (B, D) target encoder outputs (no gradients — dual-tower).
        reciprocal_mask: (B, B) bool, True for reciprocal k-NN pairs.

    Returns:
        scalar loss = mean(1 - cos_sim) over reciprocal pairs.
    """
    B = z_online.shape[0]
    if B < 2 or not reciprocal_mask.any():
        return torch.tensor(0.0, device=z_online.device, dtype=torch.float32)

    z_i_norm = F.normalize(z_online, dim=1)
    z_j_norm = F.normalize(z_target, dim=1)
    S = torch.mm(z_i_norm, z_j_norm.t())  # (B, B) cosine similarity

    n_pairs = reciprocal_mask.sum()
    loss = (1.0 - S[reciprocal_mask]).sum() / n_pairs.clamp(min=1)
    return loss
