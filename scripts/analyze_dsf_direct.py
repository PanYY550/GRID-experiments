#!/usr/bin/env python3
"""
DSF (Drift Semantic Fidelity) analysis: k-NN preservation rate vs multiple references.

Reports two DSF metrics for each checkpoint:
  - DSF_G0:     encoder k-NN vs G0 (pre-trained semantic ground truth)
  - DSF_online: encoder k-NN vs online EMA k-NN (from checkpoint, if available)

Usage:
  python scripts/analyze_dsf_direct.py \\
    --g0_ckpt <G0 checkpoint> \\
    --ckpts name1:path1 name2:path2 ... \\
    --k 50
"""

import argparse
import os
import sys
import numpy as np
import torch
import torch.nn as nn
from collections import defaultdict

# Ensure the project root is on sys.path
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from scripts.analyze_ecrd import load_checkpoint, compute_encoder_outputs, EMBEDDING_PATH, EXPOSURE_PATH

K = 50
BIN_EDGES = [0, 1, 3, 7, 20, 400]
BIN_LABELS = ["0 (never)", "1-2", "3-6", "7-19", "20+"]


def compute_knn(z, k):
    zn = nn.functional.normalize(z, p=2, dim=-1)
    sim = torch.mm(zn, zn.t())
    sim.fill_diagonal_(-float("inf"))
    return torch.topk(sim, k, dim=-1).indices


def compute_dsf(gknn, rknn):
    dsf = np.zeros(gknn.shape[0], dtype=np.float32)
    for i in range(len(dsf)):
        dsf[i] = len(set(gknn[i].tolist()) & set(rknn[i].tolist())) / K
    return dsf


def extract_online_knn(checkpoint_path: str) -> torch.Tensor | None:
    """Extract the online k-NN buffer from a checkpoint, if present."""
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    sd = ckpt.get("state_dict", ckpt)
    for key in ("online_knn_indices", "_online_knn_indices"):
        if key in sd:
            buf = sd[key]
            if buf.abs().sum() > 0:
                return buf
    return None


def bin_items(vals, exp):
    bv = defaultdict(list)
    for i in range(len(vals)):
        e = int(exp[i].item())
        for j in range(len(BIN_EDGES) - 1):
            if j == len(BIN_EDGES) - 2:
                if e >= BIN_EDGES[j]:
                    bv[BIN_LABELS[j]].append(float(vals[i]))
                    break
            elif BIN_EDGES[j] <= e < BIN_EDGES[j + 1]:
                bv[BIN_LABELS[j]].append(float(vals[i]))
                break
    return dict(bv)


def print_dsf_table(title, groups, all_dsf, ref_label):
    print(f"\n{'=' * 80}")
    print(f"{title} (k={K}, ref={ref_label})")
    print(f"{'=' * 80}")
    print(f"{'Group':<34s} {'Mean':>8s} {'Std':>8s} {'Median':>8s}")
    print("-" * 58)
    for name in groups:
        d = all_dsf[name]
        print(f"  {name:<32s}: {d.mean():.4f}  {d.std():.4f}  {np.median(d):.4f}")


def print_per_bin(title, groups, all_bin, bc, ref_label):
    print(f"\n{'=' * 80}")
    print(f"{title} (per exposure bin, ref={ref_label})")
    print(f"{'=' * 80}")
    header = f"{'Bin':<16s} {'Count':>6s}"
    for g in groups:
        header += f" {g:>24s}"
    print(header)
    print("-" * len(header))
    for label in BIN_LABELS:
        if bc[label] == 0:
            continue
        row = f"{label:<16s} {bc[label]:>6d}"
        for g in groups:
            row += f" {np.mean(all_bin[g][label]):>24.4f}"
        print(row)


def print_head_tail(title, groups, all_bin, ref_label):
    print(f"\n{'=' * 80}")
    print(f"{title} (ref={ref_label})")
    print(f"{'=' * 80}")
    for g in groups:
        hm = np.mean(all_bin[g]["20+"])
        tm = np.mean(all_bin[g]["3-6"])
        print(f"  {g:<32s}: head={hm:.4f}  tail={tm:.4f}  tail/head={tm / hm:.4f}")


def print_comparison(groups, dsf_g0, dsf_online, has_online):
    """Side-by-side summary of both DSF metrics."""
    print(f"\n{'=' * 80}")
    print("DSF Comparison: G0 vs Online")
    print(f"{'=' * 80}")
    print(f"{'Group':<34s} {'DSF_G0':>10s} {'DSF_online':>12s} {'Delta':>10s}")
    print("-" * 68)
    for g in groups:
        g0_val = dsf_g0[g].mean()
        if has_online[g]:
            on_val = dsf_online[g].mean()
            delta = on_val - g0_val
            print(f"  {g:<32s}: {g0_val:10.4f} {on_val:12.4f} {delta:+10.4f}")
        else:
            print(f"  {g:<32s}: {g0_val:10.4f} {'N/A':>12s} {'--':>10s}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--g0_ckpt", required=True, help="Path to G0 (no-VCF) checkpoint")
    parser.add_argument("--ckpts", nargs="+", required=True,
                        help="Other checkpoints as name:path pairs")
    parser.add_argument("--k", type=int, default=50)
    parser.add_argument("--embedding_path", default=EMBEDDING_PATH)
    parser.add_argument("--exposure_path", default=EXPOSURE_PATH)
    parser.add_argument("--output", default=None,
                        help="If set, save per-item DSF arrays and summary to this .npz file")
    args = parser.parse_args()

    # Parse checkpoint entries
    ckpts = {}
    for entry in args.ckpts:
        name, path = entry.split(":", 1)
        ckpts[name] = path
    groups = list(ckpts.keys())

    print("Loading data...")
    emb = torch.load(args.embedding_path, map_location="cpu", weights_only=False)
    exp_t = torch.load(args.exposure_path, map_location="cpu", weights_only=False)
    n = len(exp_t)
    emb = emb[:n]
    print(f"  Items: {n}")

    # Bin counts
    bc = defaultdict(int)
    for i in range(n):
        e = int(exp_t[i].item())
        for j in range(len(BIN_EDGES) - 1):
            if j == len(BIN_EDGES) - 2:
                if e >= BIN_EDGES[j]:
                    bc[BIN_LABELS[j]] += 1
                    break
            elif BIN_EDGES[j] <= e < BIN_EDGES[j + 1]:
                bc[BIN_LABELS[j]] += 1
                break

    # Load G0
    print("\nLoading G0 checkpoint...")
    g0_model = load_checkpoint(args.g0_ckpt)
    z_g0 = compute_encoder_outputs(emb, g0_model)
    knn_g0 = compute_knn(z_g0, args.k)

    # Load comparison checkpoints and extract online k-NN buffers
    all_z = {}
    online_knn = {}
    has_online = {}
    for name, path in ckpts.items():
        print(f"  Loading {name}...")
        all_z[name] = compute_encoder_outputs(emb, load_checkpoint(path))
        online_knn[name] = extract_online_knn(path)
        has_online[name] = online_knn[name] is not None
        if has_online[name]:
            print(f"    -> found online k-NN buffer ({online_knn[name].shape})")
        else:
            print(f"    -> no online k-NN buffer (legacy checkpoint)")

    # Compute encoder k-NN for each group
    all_knn = {}
    for name in groups:
        print(f"  Computing k-NN for {name}...")
        all_knn[name] = compute_knn(all_z[name], args.k)

    # ── DSF vs G0 ──
    dsf_g0 = {}
    for name in groups:
        dsf_g0[name] = compute_dsf(all_knn[name], knn_g0)
    print_dsf_table("DSF_G0 — k-NN Preservation vs G0", groups, dsf_g0, "G0")
    all_bin_g0 = {name: bin_items(dsf_g0[name], exp_t) for name in groups}
    print_per_bin("DSF_G0 Per-Bin", groups, all_bin_g0, bc, "G0")
    print_head_tail("DSF_G0 Head vs Tail", groups, all_bin_g0, "G0")

    # ── DSF vs Online k-NN ──
    dsf_online = {}
    online_groups = [g for g in groups if has_online[g]]
    if online_groups:
        for name in online_groups:
            dsf_online[name] = compute_dsf(all_knn[name], online_knn[name])
        print_dsf_table("DSF_online — k-NN Preservation vs Online EMA k-NN",
                        online_groups, dsf_online, "online EMA")
        all_bin_on = {name: bin_items(dsf_online[name], exp_t) for name in online_groups}
        print_per_bin("DSF_online Per-Bin", online_groups, all_bin_on, bc, "online EMA")
        print_head_tail("DSF_online Head vs Tail", online_groups, all_bin_on, "online EMA")
    else:
        print("\n[DSF_online] No checkpoints with online k-NN buffer found.")

    # ── Side-by-side comparison ──
    print_comparison(groups, dsf_g0, dsf_online, has_online)

    # ── Save per-item DSF if requested ──
    if args.output:
        save_data = {}
        for name in groups:
            save_data[f"{name}_dsf_g0"] = dsf_g0[name]
            save_data[f"{name}_dsf_g0_mean"] = np.array([dsf_g0[name].mean()])
            if has_online[name]:
                save_data[f"{name}_dsf_online"] = dsf_online[name]
                save_data[f"{name}_dsf_online_mean"] = np.array([dsf_online[name].mean()])
        save_data["exposure_times"] = exp_t.numpy()
        save_data["groups"] = np.array(groups)
        save_data["k"] = np.array([args.k])
        np.savez(args.output, **save_data)
        print(f"\nPer-item DSF saved to: {args.output}")

    print("\nDone.")


if __name__ == "__main__":
    main()
