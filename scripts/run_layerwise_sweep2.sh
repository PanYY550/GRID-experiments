#!/usr/bin/env bash
# ===========================================================================
# run_layerwise_sweep2.sh — Direction 2: 第二轮参数扫描
#
# 基于第一轮发现:
#   - LW_dfl (uniform m=0.67, α=0.01) 最佳
#   - LW_exp (m_L2=1.0, α_L2=0.05) 过强, unique SID 下降
#   - L2 coverage 全部提升 (0.29-0.41 vs 0.20 baseline)
#
# 本轮探索:
#   Round 4: m_L2=1.1/1.2 (更高 margin) + α_L2 精细 (0.015/0.02)
#   Round 5: L1 强化 (m_L1=0.9) + λ_L2=0.20
#   Round 6: dfl 变体 (统一 α=0.005, 渐进 margin)
# ===========================================================================

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

DATA_DIR="${DATA_DIR:-/home/pyy/GRID/src/data/amazon_data/beauty}"
EMB_PATH="${EMB_PATH:-/home/pyy/GRID/embeddings/beauty/pickle/merged_predictions_tensor.pt}"
EXPOSURE_COUNTS="${EXPOSURE_COUNTS:-/home/pyy/GRID/src/data/amazon_data/beauty/exposure_counts.pt}"

SID_STEPS="${SID_STEPS:-400}"
GPU1="${GPU1:-1}"
GPU2="${GPU2:-2}"

RUN_TAG="${RUN_TAG:-layerwise_sweep2_$(date +%Y%m%d_%H%M%S)}"
OUT_ROOT="${OUT_ROOT:-outputs/layerwise_sweep/${RUN_TAG}}"
LOG_ROOT="${LOG_ROOT:-logs/layerwise_sweep/${RUN_TAG}}"

SID_COMMON_BASE=(
    experiment=rqvae_vcf_online_knn_train_flat
    data_dir="${DATA_DIR}"
    embedding_path="${EMB_PATH}"
    embedding_dim=768
    num_hierarchies=3
    codebook_width=256
    "data_loading.datamodule.train_dataloader_config.batch_size_per_device=__BATCH_PLACEHOLDER__"
    data_loading.datamodule.train_dataloader_config.num_workers=12
    data_loading.datamodule.train_dataloader_config.pin_memory=true
    data_loading.datamodule.train_dataloader_config.persistent_workers=true
    seed=42
    trainer.max_epochs=null
    "trainer.max_steps=${SID_STEPS}"
    "callbacks.model_checkpoint.every_n_train_steps=${SID_STEPS}"
    model.use_vcf=true
    model.use_cvpm=true
    model.cvpm_temperature=0.15
    model.alpha=0.5
    model.exposure_counts_path="${EXPOSURE_COUNTS}"
    model.use_time_decay=false
    model.use_dynamic_margin=false
    model.use_asymmetric_repulsion=false
    model.use_npr=__NPR_PLACEHOLDER__
    model.use_online_knn=__ONLINE_KNN_PLACEHOLDER__
    model.use_kpl=false
    model.use_rncl=false
    model.use_tcl=true
    model.use_dual_tower=true
    "model.quasid_cl_tau=__TAU_PLACEHOLDER__"
    model.repulsion_warmup_steps=0
    model.cl_ramp_steps=0
    model.margin_ramp_steps=0
    model.hamming_radius=1
    optim.optimizer._target_=torch.optim.Adam
    optim.optimizer.lr=0.0003
    optim.optimizer.weight_decay=0.00001
    optim.scheduler.warmup_steps=0
    ++should_skip_retry=true
)

BATCH="256"
TAU="0.5"
CL="0.1"

# ══════════════════════════════════════════════════════════════════════
# Round 4: m_L2=1.1/1.2 + α_L2 fine (0.015/0.02)
# L0 (m=0.6,α=0.005,λ=0.15)  L1 (m=0.8,α=0.01,λ=0.10)
# ══════════════════════════════════════════════════════════════════════

# G7: m_L2=1.2, α_L2=0.01 (LW_dfl 的 α, 更高 margin)
G7_NAME="LW_m12_a01"
G7_DESC="m_L2=1.2 α_L2=0.01 (high margin, dfl alpha)"
G7_OVERRIDES=(
    model.use_layerwise_repulsion=true
    model.use_npr=true
    model.use_online_knn=true
    model.online_knn_k=50
    model.m_rep_L0=0.6
    model.m_rep_L1=0.8
    model.m_rep_L2=1.2
    "model.lambda_rep_L0=0.15"
    "model.lambda_rep_L1=0.10"
    "model.lambda_rep_L2=0.15"
    model.npr_alpha_L0=0.005
    model.npr_alpha_L1=0.01
    model.npr_alpha_L2=0.01
)

# G8: m_L2=1.1, α_L2=0.015
G8_NAME="LW_m11_a015"
G8_DESC="m_L2=1.1 α_L2=0.015 (moderate)"
G8_OVERRIDES=(
    model.use_layerwise_repulsion=true
    model.use_npr=true
    model.use_online_knn=true
    model.online_knn_k=50
    model.m_rep_L0=0.6
    model.m_rep_L1=0.8
    model.m_rep_L2=1.1
    "model.lambda_rep_L0=0.15"
    "model.lambda_rep_L1=0.10"
    "model.lambda_rep_L2=0.15"
    model.npr_alpha_L0=0.005
    model.npr_alpha_L1=0.01
    model.npr_alpha_L2=0.015
)

# ══════════════════════════════════════════════════════════════════════
# Round 5: L1 强化 (m_L1=0.9) + λ_L2=0.20
# ══════════════════════════════════════════════════════════════════════

# G9: L1 强化 m_L1=0.9, L2 标准 (m=1.0, α=0.01)
G9_NAME="LW_L1m9"
G9_DESC="L1 m=0.9 (stronger L1), m_L2=1.0 α_L2=0.01"
G9_OVERRIDES=(
    model.use_layerwise_repulsion=true
    model.use_npr=true
    model.use_online_knn=true
    model.online_knn_k=50
    model.m_rep_L0=0.6
    model.m_rep_L1=0.9
    model.m_rep_L2=1.0
    "model.lambda_rep_L0=0.15"
    "model.lambda_rep_L1=0.10"
    "model.lambda_rep_L2=0.15"
    model.npr_alpha_L0=0.005
    model.npr_alpha_L1=0.01
    model.npr_alpha_L2=0.01
)

# G10: λ_L2=0.20, m_L2=1.0, α_L2=0.01
G10_NAME="LW_l20"
G10_DESC="λ_L2=0.20 (higher L2 weight)"
G10_OVERRIDES=(
    model.use_layerwise_repulsion=true
    model.use_npr=true
    model.use_online_knn=true
    model.online_knn_k=50
    model.m_rep_L0=0.6
    model.m_rep_L1=0.8
    model.m_rep_L2=1.0
    "model.lambda_rep_L0=0.15"
    "model.lambda_rep_L1=0.10"
    "model.lambda_rep_L2=0.20"
    model.npr_alpha_L0=0.005
    model.npr_alpha_L1=0.01
    model.npr_alpha_L2=0.01
)

# ══════════════════════════════════════════════════════════════════════
# Round 6: dfl 变体 — 统一 α=0.005 + 渐进 margin
# ══════════════════════════════════════════════════════════════════════

# G11: 统一 α=0.005 (更弱 NPR 保护), 渐进 margin m:0.5/0.67/0.85
G11_NAME="LW_lowNPR"
G11_DESC="α=0.005 all (weak NPR), m:0.5/0.67/0.85"
G11_OVERRIDES=(
    model.use_layerwise_repulsion=true
    model.use_npr=true
    model.use_online_knn=true
    model.online_knn_k=50
    model.m_rep_L0=0.5
    model.m_rep_L1=0.67
    model.m_rep_L2=0.85
    "model.lambda_rep_L0=0.15"
    "model.lambda_rep_L1=0.10"
    "model.lambda_rep_L2=0.15"
    model.npr_alpha_L0=0.005
    model.npr_alpha_L1=0.005
    model.npr_alpha_L2=0.005
)

# G12: α=0.02 all (中等 NPR), m:0.5/0.67/0.85
G12_NAME="LW_midNPR"
G12_DESC="α=0.02 all (mid NPR), m:0.5/0.67/0.85"
G12_OVERRIDES=(
    model.use_layerwise_repulsion=true
    model.use_npr=true
    model.use_online_knn=true
    model.online_knn_k=50
    model.m_rep_L0=0.5
    model.m_rep_L1=0.67
    model.m_rep_L2=0.85
    "model.lambda_rep_L0=0.15"
    "model.lambda_rep_L1=0.10"
    "model.lambda_rep_L2=0.15"
    model.npr_alpha_L0=0.02
    model.npr_alpha_L1=0.02
    model.npr_alpha_L2=0.02
)

# ══════════════════════════════════════════════════════════════════════
# 脚本逻辑
# ══════════════════════════════════════════════════════════════════════

if [[ -f "/root/miniconda3/etc/profile.d/conda.sh" ]]; then
    source /root/miniconda3/etc/profile.d/conda.sh
    conda activate grid
else
    echo "ERROR: conda.sh not found" >&2; exit 1
fi

mkdir -p "$OUT_ROOT" "$LOG_ROOT"
RUN_ALL_LOG="${LOG_ROOT}/run_all.log"

_latest_ckpt() { ls -t "${1}"/checkpoints/*.ckpt 2>/dev/null | head -n 1; }
_die() { echo "ERROR: $*" >&2; return 1; }

run_group() {
    local gpu="$1" name="$2" desc="$3" batch="$4" tau="$5" cl="$6"
    shift 6
    local override_args=("$@")
    local run_root="${OUT_ROOT}/${name}"
    local log_pref="${LOG_ROOT}/${name}"
    mkdir -p "${run_root}/01_sid_train" "${run_root}/02_sid_inference"

    echo "── [${name}] ${desc} (GPU${gpu}, ${SID_STEPS}steps, batch=${batch}) ──" | tee -a "$RUN_ALL_LOG"

    local args=("${SID_COMMON_BASE[@]/__BATCH_PLACEHOLDER__/${batch}}")
    args=("${args[@]/__TAU_PLACEHOLDER__/${tau}}")
    args=("${args[@]/__NPR_PLACEHOLDER__/false}")
    args=("${args[@]/__ONLINE_KNN_PLACEHOLDER__/false}")

    CUDA_VISIBLE_DEVICES="${gpu}" python -m src.train \
        "${args[@]}" \
        "${override_args[@]}" \
        "model.lambda_cl=${cl}" \
        "hydra.run.dir=${run_root}/01_sid_train" \
        > "${log_pref}_01_sid_train.log" 2>&1

    local ckpt; ckpt="$(_latest_ckpt "${run_root}/01_sid_train")"
    if [[ -z "$ckpt" || ! -f "$ckpt" ]]; then
        _die "[${name}] SID checkpoint 未生成"; return 1
    fi
    echo "[${name}] SID ckpt: ${ckpt}" | tee -a "$RUN_ALL_LOG"

    echo "── [${name}] SID推理 (GPU${gpu}) ──" | tee -a "$RUN_ALL_LOG"
    CUDA_VISIBLE_DEVICES="${gpu}" python -m src.inference \
        experiment=rqvae_vcf_inference_flat \
        data_dir="${DATA_DIR}" \
        embedding_path="${EMB_PATH}" \
        embedding_dim=768 \
        num_hierarchies=3 \
        codebook_width=256 \
        "ckpt_path=${ckpt}" \
        seed=42 \
        "hydra.run.dir=${run_root}/02_sid_inference" \
        > "${log_pref}_02_sid_inference.log" 2>&1

    local sid_pt="${run_root}/02_sid_inference/pickle/merged_predictions_tensor.pt"
    if [[ ! -f "$sid_pt" ]]; then
        _die "[${name}] SID tensor 未生成"; return 1
    fi
    echo "[${name}] SID OK: ${sid_pt}" | tee -a "$RUN_ALL_LOG"

    echo "── [${name}] 碰撞分析 ──" | tee -a "$RUN_ALL_LOG"
    python scripts/analyze_sid_collisions.py --path "${sid_pt}" \
        | tee -a "$RUN_ALL_LOG" "${log_pref}_collision.txt"
    return 0
}

# ══════════════════════════════════════════════════════════════════════
{
    echo "╔══════════════════════════════════════════════════════════╗"
    echo "║  Direction 2: Layer-wise Repulsion — 第二轮扫描         ║"
    echo "╠══════════════════════════════════════════════════════════╣"
    printf "║  RUN_TAG: %-46s ║\n" "$RUN_TAG"
    printf "║  SID steps=%-4s  batch=%-3s  TCL: λ=%-5s τ=%-5s     ║\n" \
        "$SID_STEPS" "$BATCH" "$CL" "$TAU"
    echo "╠══════════════════════════════════════════════════════════╣"
    echo "║  Round 4: m_L2=1.1/1.2 + α_L2 fine (0.015/0.02)       ║"
    printf "║    GPU%-2s→ %-16s %-35s ║\n" "$GPU1" "$G7_NAME" "$G7_DESC"
    printf "║    GPU%-2s→ %-16s %-35s ║\n" "$GPU2" "$G8_NAME" "$G8_DESC"
    echo "╠══════════════════════════════════════════════════════════╣"
    echo "║  Round 5: L1 强化 (m_L1=0.9) + λ_L2=0.20              ║"
    printf "║    GPU%-2s→ %-16s %-35s ║\n" "$GPU1" "$G9_NAME" "$G9_DESC"
    printf "║    GPU%-2s→ %-16s %-35s ║\n" "$GPU2" "$G10_NAME" "$G10_DESC"
    echo "╠══════════════════════════════════════════════════════════╣"
    echo "║  Round 6: dfl 变体 — 统一 α + 渐进 margin             ║"
    printf "║    GPU%-2s→ %-16s %-35s ║\n" "$GPU1" "$G11_NAME" "$G11_DESC"
    printf "║    GPU%-2s→ %-16s %-35s ║\n" "$GPU2" "$G12_NAME" "$G12_DESC"
    echo "╚══════════════════════════════════════════════════════════╝"
} | tee "$RUN_ALL_LOG"

# ══════════════════════════════════════════════════════════════════════
# Round 4: m_L2=1.1/1.2 + α_L2 fine
# ══════════════════════════════════════════════════════════════════════
echo "" | tee -a "$RUN_ALL_LOG"
echo "════════ Round 4: High m_L2 + fine α ════════" | tee -a "$RUN_ALL_LOG"
echo "[main] 启动 ${G7_NAME} (GPU ${GPU1}) — ${G7_DESC}" | tee -a "$RUN_ALL_LOG"
(run_group "$GPU1" "$G7_NAME" "$G7_DESC" "$BATCH" "$TAU" "$CL" "${G7_OVERRIDES[@]}") & PID7=$!
echo "[main] 启动 ${G8_NAME} (GPU ${GPU2}) — ${G8_DESC}" | tee -a "$RUN_ALL_LOG"
(run_group "$GPU2" "$G8_NAME" "$G8_DESC" "$BATCH" "$TAU" "$CL" "${G8_OVERRIDES[@]}") & PID8=$!
ok7=0 ok8=0
wait "$PID7" && ok7=1 && echo "[main] ${G7_NAME} ✓" | tee -a "$RUN_ALL_LOG" || echo "[main] ${G7_NAME} ✗" | tee -a "$RUN_ALL_LOG"
wait "$PID8" && ok8=1 && echo "[main] ${G8_NAME} ✓" | tee -a "$RUN_ALL_LOG" || echo "[main] ${G8_NAME} ✗" | tee -a "$RUN_ALL_LOG"

# ══════════════════════════════════════════════════════════════════════
# Round 5: L1 强化 + λ_L2=0.20
# ══════════════════════════════════════════════════════════════════════
echo "" | tee -a "$RUN_ALL_LOG"
echo "════════ Round 5: L1+λ_L2 ════════" | tee -a "$RUN_ALL_LOG"
echo "[main] 启动 ${G9_NAME} (GPU ${GPU1}) — ${G9_DESC}" | tee -a "$RUN_ALL_LOG"
(run_group "$GPU1" "$G9_NAME" "$G9_DESC" "$BATCH" "$TAU" "$CL" "${G9_OVERRIDES[@]}") & PID9=$!
echo "[main] 启动 ${G10_NAME} (GPU ${GPU2}) — ${G10_DESC}" | tee -a "$RUN_ALL_LOG"
(run_group "$GPU2" "$G10_NAME" "$G10_DESC" "$BATCH" "$TAU" "$CL" "${G10_OVERRIDES[@]}") & PID10=$!
ok9=0 ok10=0
wait "$PID9" && ok9=1 && echo "[main] ${G9_NAME} ✓" | tee -a "$RUN_ALL_LOG" || echo "[main] ${G9_NAME} ✗" | tee -a "$RUN_ALL_LOG"
wait "$PID10" && ok10=1 && echo "[main] ${G10_NAME} ✓" | tee -a "$RUN_ALL_LOG" || echo "[main] ${G10_NAME} ✗" | tee -a "$RUN_ALL_LOG"

# ══════════════════════════════════════════════════════════════════════
# Round 6: dfl 变体
# ══════════════════════════════════════════════════════════════════════
echo "" | tee -a "$RUN_ALL_LOG"
echo "════════ Round 6: dfl variants ════════" | tee -a "$RUN_ALL_LOG"
echo "[main] 启动 ${G11_NAME} (GPU ${GPU1}) — ${G11_DESC}" | tee -a "$RUN_ALL_LOG"
(run_group "$GPU1" "$G11_NAME" "$G11_DESC" "$BATCH" "$TAU" "$CL" "${G11_OVERRIDES[@]}") & PID11=$!
echo "[main] 启动 ${G12_NAME} (GPU ${GPU2}) — ${G12_DESC}" | tee -a "$RUN_ALL_LOG"
(run_group "$GPU2" "$G12_NAME" "$G12_DESC" "$BATCH" "$TAU" "$CL" "${G12_OVERRIDES[@]}") & PID12=$!
ok11=0 ok12=0
wait "$PID11" && ok11=1 && echo "[main] ${G11_NAME} ✓" | tee -a "$RUN_ALL_LOG" || echo "[main] ${G11_NAME} ✗" | tee -a "$RUN_ALL_LOG"
wait "$PID12" && ok12=1 && echo "[main] ${G12_NAME} ✓" | tee -a "$RUN_ALL_LOG" || echo "[main] ${G12_NAME} ✗" | tee -a "$RUN_ALL_LOG"

# ══════════════════════════════════════════════════════════════════════
echo "" | tee -a "$RUN_ALL_LOG"
echo "══════════════ 汇总 ══════════════" | tee -a "$RUN_ALL_LOG"
for i in 7 8 9 10 11 12; do
    varname="G${i}_NAME"; descvar="G${i}_DESC"; okvar="ok${i}"
    name="${!varname}"; desc="${!descvar}"; ok="${!okvar}"
    printf "  %-16s : %s  (%s)\n" "$name" "$([ "$ok" -eq 1 ] && echo "OK" || echo "FAIL")" "$desc" | tee -a "$RUN_ALL_LOG"
done
echo "" | tee -a "$RUN_ALL_LOG"
echo "Outputs: ${OUT_ROOT}/" | tee -a "$RUN_ALL_LOG"
echo "Logs:    ${LOG_ROOT}/" | tee -a "$RUN_ALL_LOG"
all_ok=$(( ok7 & ok8 & ok9 & ok10 & ok11 & ok12 ))
[[ "$all_ok" -eq 1 ]] && echo "All OK." | tee -a "$RUN_ALL_LOG" || echo "Some FAILED." | tee -a "$RUN_ALL_LOG"
exit $(( 1 - all_ok ))
