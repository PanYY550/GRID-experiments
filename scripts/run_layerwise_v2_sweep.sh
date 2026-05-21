#!/usr/bin/env bash
# ===========================================================================
# run_layerwise_v2_sweep.sh — Direction 2 V2: per-layer repulsion + Hamming filter
#                               码本稳定性参数扫描
#
# Round 1: 原始 VCF, BL_R1 (H≤1) vs BL_R0 (H=0)
# Rounds 2&3: Layerwise repulsion with H≤1 (layerwise_hamming_radius=1)
#       每层独立 margin/λ/α，H≤1 pair 仅在被碰撞的层排斥
#
# 用法:
#   bash scripts/run_layerwise_v2_sweep.sh
#
# 扫描维度:
#   Round 1: BL_R1 (H≤1) + BL_R0 (H=0) — original VCF, no layerwise
#   Round 2: V2_dfl + V2_m8 — layerwise margin sweep (H≤1)
#   Round 3: V2_mL2_12 + V2_aL2_005 — L2 margin/NPR (H≤1)
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

RUN_TAG="${RUN_TAG:-layerwise_v2_sweep_$(date +%Y%m%d_%H%M%S)}"
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
# Round 1: BL_baseline (R=1) + BL_R0 (R=0) — 原始 VCF，仅 Hamming 阈值不同
# ══════════════════════════════════════════════════════════════════════

G1_NAME="BL_R1"
G1_DESC="Baseline R=1 (H≤1 repelled, no layerwise)"
G1_OVERRIDES=(
    model.use_layerwise_repulsion=false
    model.use_npr=true
    model.use_online_knn=true
    model.npr_alpha_min=0.01
    model.online_knn_k=50
    model.lambda_full=0.2
    model.lambda_partial=0.1
    model.m_full=0.8
    model.m_partial=0.5
    model.m_full_start=0.8
    model.m_partial_start=0.5
    model.hamming_radius=1
)

G2_NAME="BL_R0"
G2_DESC="Baseline R=0 (only H=0 repelled, no layerwise)"
G2_OVERRIDES=(
    model.use_layerwise_repulsion=false
    model.use_npr=true
    model.use_online_knn=true
    model.npr_alpha_min=0.01
    model.online_knn_k=50
    model.lambda_full=0.2
    model.lambda_partial=0.1
    model.m_full=0.8
    model.m_partial=0.5
    model.m_full_start=0.8
    model.m_partial_start=0.5
    model.hamming_radius=0
)

# ══════════════════════════════════════════════════════════════════════
# Round 2: V2 layerwise — margin sweep
# ══════════════════════════════════════════════════════════════════════

# V2_dfl: uniform m=0.67 (same as BL_R0 but with per-layer structure)
G3_NAME="V2_dfl"
G3_DESC="V2 m=0.67 all α=0.01 (LW, uniform)"
G3_OVERRIDES=(
    model.use_layerwise_repulsion=true
    model.use_npr=true
    model.use_online_knn=true
    model.online_knn_k=50
    model.m_rep_L0=0.67
    model.m_rep_L1=0.67
    model.m_rep_L2=0.67
    "model.lambda_rep_L0=0.20"
    "model.lambda_rep_L1=0.10"
    "model.lambda_rep_L2=0.10"
    model.npr_alpha_L0=0.01
    model.npr_alpha_L1=0.01
    model.npr_alpha_L2=0.01
    model.layerwise_hamming_radius=1
)

# V2_m8: uniform m=0.8 (比 dfl 的 0.67 更高)
G4_NAME="V2_m8"
G4_DESC="V2 m=0.8 all α=0.01 (higher uniform margin)"
G4_OVERRIDES=(
    model.use_layerwise_repulsion=true
    model.use_npr=true
    model.use_online_knn=true
    model.online_knn_k=50
    model.m_rep_L0=0.8
    model.m_rep_L1=0.8
    model.m_rep_L2=0.8
    "model.lambda_rep_L0=0.20"
    "model.lambda_rep_L1=0.10"
    "model.lambda_rep_L2=0.10"
    model.npr_alpha_L0=0.01
    model.npr_alpha_L1=0.01
    model.npr_alpha_L2=0.01
    model.layerwise_hamming_radius=1
)

# ══════════════════════════════════════════════════════════════════════
# Round 3: L2 专项 — margin + NPR
# ══════════════════════════════════════════════════════════════════════

# V2_mL2_12: 仅提升 L2 margin 到 1.2
G5_NAME="V2_mL2_12"
G5_DESC="V2 m:0.67/0.67/1.2 α=0.01 (only L2 high margin)"
G5_OVERRIDES=(
    model.use_layerwise_repulsion=true
    model.use_npr=true
    model.use_online_knn=true
    model.online_knn_k=50
    model.m_rep_L0=0.67
    model.m_rep_L1=0.67
    model.m_rep_L2=1.2
    "model.lambda_rep_L0=0.20"
    "model.lambda_rep_L1=0.10"
    "model.lambda_rep_L2=0.15"
    model.npr_alpha_L0=0.01
    model.npr_alpha_L1=0.01
    model.npr_alpha_L2=0.01
    model.layerwise_hamming_radius=1
)

# V2_aL2_005: 弱 L2 NPR 保护 (搭配 m_L2=1.2)
G6_NAME="V2_aL2_005"
G6_DESC="V2 m:0.67/0.67/1.2 α_L2=0.005 (weaker L2 NPR)"
G6_OVERRIDES=(
    model.use_layerwise_repulsion=true
    model.use_npr=true
    model.use_online_knn=true
    model.online_knn_k=50
    model.m_rep_L0=0.67
    model.m_rep_L1=0.67
    model.m_rep_L2=1.2
    "model.lambda_rep_L0=0.20"
    "model.lambda_rep_L1=0.10"
    "model.lambda_rep_L2=0.15"
    model.npr_alpha_L0=0.01
    model.npr_alpha_L1=0.01
    model.npr_alpha_L2=0.005
    model.layerwise_hamming_radius=1
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
    echo "║  Direction 2 V2: Per-layer Repulsion + Hamming Filter  ║"
    echo "╠══════════════════════════════════════════════════════════╣"
    printf "║  RUN_TAG: %-46s ║\n" "$RUN_TAG"
    printf "║  SID steps=%-4s  batch=%-3s  TCL: λ=%-5s τ=%-5s     ║\n" \
        "$SID_STEPS" "$BATCH" "$CL" "$TAU"
    echo "║  Round 1: BL_R1(H≤1) vs BL_R0(H=0) — original VCF    ║"
    echo "║  Round 2&3: layerwise H≤1 (layerwise_hamming_radius=1) ║"
    echo "╠══════════════════════════════════════════════════════════╣"
    echo "║  Round 1: BL_R1 (H≤1) vs BL_R0 (H=0) — original VCF ║"
    printf "║    GPU%-2s→ %-16s %-35s ║\n" "$GPU1" "$G1_NAME" "$G1_DESC"
    printf "║    GPU%-2s→ %-16s %-35s ║\n" "$GPU2" "$G2_NAME" "$G2_DESC"
    echo "╠══════════════════════════════════════════════════════════╣"
    echo "║  Round 2: V2 layerwise — margin sweep                  ║"
    printf "║    GPU%-2s→ %-16s %-35s ║\n" "$GPU1" "$G3_NAME" "$G3_DESC"
    printf "║    GPU%-2s→ %-16s %-35s ║\n" "$GPU2" "$G4_NAME" "$G4_DESC"
    echo "╠══════════════════════════════════════════════════════════╣"
    echo "║  Round 3: V2 layerwise — L2 margin + NPR               ║"
    printf "║    GPU%-2s→ %-16s %-35s ║\n" "$GPU1" "$G5_NAME" "$G5_DESC"
    printf "║    GPU%-2s→ %-16s %-35s ║\n" "$GPU2" "$G6_NAME" "$G6_DESC"
    echo "╚══════════════════════════════════════════════════════════╝"
} | tee "$RUN_ALL_LOG"

# ══════════════════════════════════════════════════════════════════════
# Round 1
# ══════════════════════════════════════════════════════════════════════
echo "" | tee -a "$RUN_ALL_LOG"
echo "════════ Round 1: BL_R1 vs BL_R0 ════════" | tee -a "$RUN_ALL_LOG"
echo "[main] 启动 ${G1_NAME} (GPU ${GPU1}) — ${G1_DESC}" | tee -a "$RUN_ALL_LOG"
(run_group "$GPU1" "$G1_NAME" "$G1_DESC" "$BATCH" "$TAU" "$CL" "${G1_OVERRIDES[@]}") & PID1=$!
echo "[main] 启动 ${G2_NAME} (GPU ${GPU2}) — ${G2_DESC}" | tee -a "$RUN_ALL_LOG"
(run_group "$GPU2" "$G2_NAME" "$G2_DESC" "$BATCH" "$TAU" "$CL" "${G2_OVERRIDES[@]}") & PID2=$!
ok1=0 ok2=0
wait "$PID1" && ok1=1 && echo "[main] ${G1_NAME} ✓" | tee -a "$RUN_ALL_LOG" || echo "[main] ${G1_NAME} ✗" | tee -a "$RUN_ALL_LOG"
wait "$PID2" && ok2=1 && echo "[main] ${G2_NAME} ✓" | tee -a "$RUN_ALL_LOG" || echo "[main] ${G2_NAME} ✗" | tee -a "$RUN_ALL_LOG"

# ══════════════════════════════════════════════════════════════════════
# Round 2
# ══════════════════════════════════════════════════════════════════════
echo "" | tee -a "$RUN_ALL_LOG"
echo "════════ Round 2: V2 margin sweep ════════" | tee -a "$RUN_ALL_LOG"
echo "[main] 启动 ${G3_NAME} (GPU ${GPU1}) — ${G3_DESC}" | tee -a "$RUN_ALL_LOG"
(run_group "$GPU1" "$G3_NAME" "$G3_DESC" "$BATCH" "$TAU" "$CL" "${G3_OVERRIDES[@]}") & PID3=$!
echo "[main] 启动 ${G4_NAME} (GPU ${GPU2}) — ${G4_DESC}" | tee -a "$RUN_ALL_LOG"
(run_group "$GPU2" "$G4_NAME" "$G4_DESC" "$BATCH" "$TAU" "$CL" "${G4_OVERRIDES[@]}") & PID4=$!
ok3=0 ok4=0
wait "$PID3" && ok3=1 && echo "[main] ${G3_NAME} ✓" | tee -a "$RUN_ALL_LOG" || echo "[main] ${G3_NAME} ✗" | tee -a "$RUN_ALL_LOG"
wait "$PID4" && ok4=1 && echo "[main] ${G4_NAME} ✓" | tee -a "$RUN_ALL_LOG" || echo "[main] ${G4_NAME} ✗" | tee -a "$RUN_ALL_LOG"

# ══════════════════════════════════════════════════════════════════════
# Round 3
# ══════════════════════════════════════════════════════════════════════
echo "" | tee -a "$RUN_ALL_LOG"
echo "════════ Round 3: L2 margin + NPR ════════" | tee -a "$RUN_ALL_LOG"
echo "[main] 启动 ${G5_NAME} (GPU ${GPU1}) — ${G5_DESC}" | tee -a "$RUN_ALL_LOG"
(run_group "$GPU1" "$G5_NAME" "$G5_DESC" "$BATCH" "$TAU" "$CL" "${G5_OVERRIDES[@]}") & PID5=$!
echo "[main] 启动 ${G6_NAME} (GPU ${GPU2}) — ${G6_DESC}" | tee -a "$RUN_ALL_LOG"
(run_group "$GPU2" "$G6_NAME" "$G6_DESC" "$BATCH" "$TAU" "$CL" "${G6_OVERRIDES[@]}") & PID6=$!
ok5=0 ok6=0
wait "$PID5" && ok5=1 && echo "[main] ${G5_NAME} ✓" | tee -a "$RUN_ALL_LOG" || echo "[main] ${G5_NAME} ✗" | tee -a "$RUN_ALL_LOG"
wait "$PID6" && ok6=1 && echo "[main] ${G6_NAME} ✓" | tee -a "$RUN_ALL_LOG" || echo "[main] ${G6_NAME} ✗" | tee -a "$RUN_ALL_LOG"

# ══════════════════════════════════════════════════════════════════════
echo "" | tee -a "$RUN_ALL_LOG"
echo "══════════════ 汇总 ══════════════" | tee -a "$RUN_ALL_LOG"
for i in 1 2 3 4 5 6; do
    varname="G${i}_NAME"; descvar="G${i}_DESC"; okvar="ok${i}"
    name="${!varname}"; desc="${!descvar}"; ok="${!okvar}"
    printf "  %-16s : %s  (%s)\n" "$name" "$([ "$ok" -eq 1 ] && echo "OK" || echo "FAIL")" "$desc" | tee -a "$RUN_ALL_LOG"
done
echo "" | tee -a "$RUN_ALL_LOG"
echo "Outputs: ${OUT_ROOT}/" | tee -a "$RUN_ALL_LOG"
echo "Logs:    ${LOG_ROOT}/" | tee -a "$RUN_ALL_LOG"
all_ok=$(( ok1 & ok2 & ok3 & ok4 & ok5 & ok6 ))
[[ "$all_ok" -eq 1 ]] && echo "All OK." | tee -a "$RUN_ALL_LOG" || echo "Some FAILED." | tee -a "$RUN_ALL_LOG"
exit $(( 1 - all_ok ))
