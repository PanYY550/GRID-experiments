#!/usr/bin/env python3
"""
Analyze SID collisions from merged_predictions_tensor.pt.

Usage:
  python scripts/analyze_sid_collisions.py --path <merged_predictions_tensor.pt>
  python scripts/analyze_sid_collisions.py --path <path> --mc_pairs 300000 --hamming_radius 2
"""

import argparse
import torch
import numpy as np
from collections import Counter


def analyze_collisions(path: str, mc_pairs: int = 200000, hamming_radius: int = 2):
    t = torch.load(path, map_location="cpu", weights_only=False)
    # Shape: (D, N) where D = num_layers + 1 (last row = dedup indicator)
    dedup = t[-1]
    tokens = t[:-1].t().contiguous()  # (N, L)
    N, L = tokens.shape
    K = 256  # codebook_width

    # ── Full collision stats ──
    sid_tuples = [tuple(tokens[i].tolist()) for i in range(N)]
    sid_counts = Counter(sid_tuples)
    unique_sids = len(sid_counts)
    colliding_groups = sum(1 for c in sid_counts.values() if c > 1)
    colliding_items = sum(c for c in sid_counts.values() if c > 1)
    max_group_size = max(sid_counts.values())

    print("== Full collision stats ==")
    print(f"num_items: {N}")
    print(f"num_layers: {L}")
    print(f"num_unique_sids: {unique_sids}")
    print(f"frac_unique_sids: {unique_sids / N}")
    print(f"colliding_groups: {colliding_groups}")
    print(f"colliding_items: {colliding_items}")
    print(f"colliding_items_frac: {colliding_items / N}")
    print(f"extra_items_due_to_collisions: {N - unique_sids}")
    print(f"max_collision_group_size: {max_group_size}")

    # ── Dedup indicator ──
    dedup_nz = (dedup > 0).sum().item()
    print(f"\n== Dedup indicator (from inference post-processing) ==")
    print(f"dedup_indicator_nonzero_frac: {dedup_nz / N}")
    print(f"dedup_indicator_max: {dedup.max().item()}")

    # ── Per-layer token utilization ──
    print(f"\n== Per-layer token utilization ==")
    for l in range(L):
        layer_tokens = tokens[:, l]
        unique_vals = layer_tokens.unique()
        coverage = len(unique_vals) / K
        token_counts = Counter(layer_tokens.tolist())
        max_frac = max(token_counts.values()) / N
        freqs = torch.zeros(K)
        for t_idx, c in token_counts.items():
            freqs[t_idx] = c
        probs = freqs / freqs.sum()
        entropy = -(probs[probs > 0] * torch.log(probs[probs > 0])).sum().item()
        entropy_norm = entropy / np.log(K)
        print(f"layer{l}: unique_tokens={len(unique_vals)} coverage={coverage:.4f} entropy_norm={entropy_norm:.4f} max_token_frac={max_frac:.4f}")

    # ── Partial collision (MC estimate) ──
    print(f"\n== Partial collision (MC estimate) ==")
    i_idx = torch.randint(0, N, (mc_pairs,))
    j_idx = torch.randint(0, N, (mc_pairs,))
    mask = i_idx != j_idx
    i_idx, j_idx = i_idx[mask], j_idx[mask]
    t0 = tokens[i_idx]
    t1 = tokens[j_idx]
    ham = (t0 != t1).sum(dim=1)
    full_rate = (ham == 0).float().mean().item()
    partial_rate = ((ham > 0) & (ham <= hamming_radius)).float().mean().item()
    print(f"num_pairs: {len(i_idx)}")
    print(f"full_collision_rate(H=0): {full_rate:.6f}")
    print(f"partial_collision_rate(0<H<=R, R={hamming_radius}): {partial_rate:.6f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", required=True, help="Path to merged_predictions_tensor.pt")
    parser.add_argument("--mc_pairs", type=int, default=200000)
    parser.add_argument("--hamming_radius", type=int, default=2)
    args = parser.parse_args()
    analyze_collisions(args.path, args.mc_pairs, args.hamming_radius)
