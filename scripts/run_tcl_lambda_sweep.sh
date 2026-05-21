#!/usr/bin/env bash
# ===========================================================================
# run_tcl_lambda_sweep.sh — TCL_only λ_cl sweep 找稳定码本参数
#
# 用法:
#   bash scripts/run_tcl_lambda_sweep.sh
# ===========================================================================

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

# ╔══════════════════════════════════════════════════════════════════════╗
# ║                      控制台                                        ║
# ╚══════════════════════════════════════════════════════════════════════╝

DATA_DIR="${DATA_DIR:-/home/pyy/GRID/src/data/amazon_data/beauty}"
EMB_PATH="${EMB_PATH:-/home/pyy/GRID/embeddings/beauty/pickle/merged_predictions_tensor.pt}"
EXPOSURE_COUNTS="${EXPOSURE_COUNTS:-/home/pyy/GRID/src/data/amazon_data/beauty/exposure_counts.pt}"

SID_STEPS="${SID_STEPS:-400}"
WARMUP=$(( SID_STEPS / 3 ))           # ~133
MARGIN_RAMP=$(( 2000 * SID_STEPS / 3000 ))  # ~267
CL_RAMP=$(( 1000 * SID_STEPS / 3000 ))      # ~133

GPU1="${GPU1:-1}"
GPU2="${GPU2:-2}"

RUN_TAG="${RUN_TAG:-tcl_sweep_$(date +%Y%m%d_%H%M%S)}"
OUT_ROOT="${OUT_ROOT:-outputs/tcl_sweep/${RUN_TAG}}"
LOG_ROOT="${LOG_ROOT:-logs/tcl_sweep/${RUN_TAG}}"

# ── 实验组 ─────────────────────────────────────────────────────────────

# 组1: NPR+TCL npr_alpha_min=0.05 (中保护)
G1_NAME="NPR_TCL_a005"
G1_CL="0.1"
G1_BATCH="256"
G1_TAU="0.5"
G1_DESC="NPR+TCL, α_min=0.05"

# 组2: NPR+TCL npr_alpha_min=1.0 (无保护=对称)
G2_NAME="NPR_TCL_a10"
G2_CL="0.1"
G2_BATCH="256"
G2_TAU="0.5"
G2_DESC="NPR+TCL, α_min=1.0 (对称)"

# ╔══════════════════════════════════════════════════════════════════════╗
# ║                      脚本逻辑                                      ║
# ╚══════════════════════════════════════════════════════════════════════╝

if [[ -f "/root/miniconda3/etc/profile.d/conda.sh" ]]; then
    source /root/miniconda3/etc/profile.d/conda.sh
    conda activate grid
else
    echo "ERROR: conda.sh not found" >&2; exit 1
fi

mkdir -p "$OUT_ROOT" "$LOG_ROOT"
RUN_ALL_LOG="${LOG_ROOT}/run_all.log"

# 共用基础参数
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
    "optim.scheduler.warmup_steps=100"
    model.use_vcf=true
    model.use_cvpm=true
    model.cvpm_temperature=0.15
    model.alpha=0.5
    model.exposure_counts_path="${EXPOSURE_COUNTS}"
    model.use_time_decay=false
    model.use_dynamic_margin=false
    model.use_asymmetric_repulsion=false
    model.use_npr=false
    model.use_online_knn=false
    model.use_kpl=false
    model.use_tcl=true
    model.use_dual_tower=true
    "model.quasid_cl_tau=__TAU_PLACEHOLDER__"
    ++should_skip_retry=true
)

# G1 override: QuaSID + NPR (α_min=0.05, medium protection)
G1_OVERRIDES=(
    model.hamming_radius=1
    model.lambda_full=0.2
    model.lambda_partial=0.1
    model.m_full=0.8
    model.m_partial=0.5
    model.m_full_start=0.8
    model.m_partial_start=0.5
    model.margin_ramp_steps=0
    model.repulsion_warmup_steps=0
    model.cl_ramp_steps=0
    optim.scheduler.warmup_steps=0
    optim.optimizer._target_=torch.optim.Adam
    optim.optimizer.lr=0.0003
    optim.optimizer.weight_decay=0.00001
    model.use_npr=true
    model.use_online_knn=true
    model.npr_alpha_min=0.05
    model.online_knn_k=50
)

# G2 override: QuaSID + NPR (α_min=1.0, symmetric baseline)
G2_OVERRIDES=(
    model.hamming_radius=1
    model.lambda_full=0.2
    model.lambda_partial=0.1
    model.m_full=0.8
    model.m_partial=0.5
    model.m_full_start=0.8
    model.m_partial_start=0.5
    model.margin_ramp_steps=0
    model.repulsion_warmup_steps=0
    model.cl_ramp_steps=0
    optim.scheduler.warmup_steps=0
    optim.optimizer._target_=torch.optim.Adam
    optim.optimizer.lr=0.0003
    optim.optimizer.weight_decay=0.00001
    model.use_npr=true
    model.use_online_knn=true
    model.npr_alpha_min=1.0
    model.online_knn_k=50
)

_latest_ckpt() { ls -t "${1}"/checkpoints/*.ckpt 2>/dev/null | head -n 1; }
_die() { echo "ERROR: $*" >&2; return 1; }

run_group() {
    local gpu="$1" name="$2" lambda_cl="$3" batch="$4" tau="$5"
    shift 5
    local override_args=("$@")
    local run_root="${OUT_ROOT}/${name}"
    local log_pref="${LOG_ROOT}/${name}"
    mkdir -p "${run_root}/01_sid_train" "${run_root}/02_sid_inference"

    # ── SID 训练 ──────────────────────────────────────────────────
    echo "── [${name}] SID训练 (GPU${gpu}, ${SID_STEPS}steps, batch=${batch}, λ_cl=${lambda_cl}, τ=${tau}) ──" | tee -a "$RUN_ALL_LOG"

    local args=("${SID_COMMON_BASE[@]/__BATCH_PLACEHOLDER__/${batch}}")
    args=("${args[@]/__TAU_PLACEHOLDER__/${tau}}")

    CUDA_VISIBLE_DEVICES="${gpu}" python -m src.train \
        "${args[@]}" \
        "${override_args[@]}" \
        "model.lambda_cl=${lambda_cl}" \
        "hydra.run.dir=${run_root}/01_sid_train" \
        > "${log_pref}_01_sid_train.log" 2>&1

    local ckpt; ckpt="$(_latest_ckpt "${run_root}/01_sid_train")"
    if [[ -z "$ckpt" || ! -f "$ckpt" ]]; then
        _die "[${name}] SID checkpoint 未生成"; return 1
    fi
    echo "[${name}] SID ckpt: ${ckpt}" | tee -a "$RUN_ALL_LOG"

    # ── SID 推理 ──────────────────────────────────────────────────
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

    # ── 碰撞分析 ──────────────────────────────────────────────────
    echo "── [${name}] 碰撞分析 ──" | tee -a "$RUN_ALL_LOG"
    python scripts/analyze_sid_collisions.py --path "${sid_pt}" \
        | tee -a "$RUN_ALL_LOG" "${log_pref}_collision.txt"

    return 0
}

# ══════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════

{
    echo "╔══════════════════════════════════════════════════════════╗"
    echo "║     TCL_only λ_cl Sweep — 找稳定码本参数                ║"
    echo "╠══════════════════════════════════════════════════════════╣"
    printf "║  RUN_TAG: %-46s ║\n" "$RUN_TAG"
    printf "║  steps=%-3s  batch=4096  warmup=%-4s  ramp=%-4s       ║\n" \
        "$SID_STEPS" "$WARMUP" "$MARGIN_RAMP"
    echo "║  底座: QuaSID (R=1,λ=0.2/0.1,m=0.8/0.5 static,Adam)  ║"
    echo "║  NPR=ON  dynM=OFF  TCL=ON  dual_tower=ON              ║"
    echo "╠══════════════════════════════════════════════════════════╣"
    printf "║  GPU%-2s→ %-16s batch=%-5s λ_cl=%-6s τ=%-5s %-30s ║\n" \
        "$GPU1" "$G1_NAME" "$G1_BATCH" "$G1_CL" "$G1_TAU" "$G1_DESC"
    printf "║  GPU%-2s→ %-16s batch=%-5s λ_cl=%-6s τ=%-5s %-30s ║\n" \
        "$GPU2" "$G2_NAME" "$G2_BATCH" "$G2_CL" "$G2_TAU" "$G2_DESC"
    echo "╚══════════════════════════════════════════════════════════╝"
} | tee "$RUN_ALL_LOG"

echo "[main] 启动 ${G1_NAME} (GPU ${GPU1}) — ${G1_DESC}" | tee -a "$RUN_ALL_LOG"
(run_group "$GPU1" "$G1_NAME" "$G1_CL" "$G1_BATCH" "$G1_TAU" "${G1_OVERRIDES[@]}") &
PID1=$!

echo "[main] 启动 ${G2_NAME} (GPU ${GPU2}) — ${G2_DESC}" | tee -a "$RUN_ALL_LOG"
(run_group "$GPU2" "$G2_NAME" "$G2_CL" "$G2_BATCH" "$G2_TAU" "${G2_OVERRIDES[@]}") &
PID2=$!

ok1=0 ok2=0
wait "$PID1" && ok1=1 && echo "[main] ${G1_NAME} ✓" | tee -a "$RUN_ALL_LOG" \
    || echo "[main] ${G1_NAME} ✗" | tee -a "$RUN_ALL_LOG"
wait "$PID2" && ok2=1 && echo "[main] ${G2_NAME} ✓" | tee -a "$RUN_ALL_LOG" \
    || echo "[main] ${G2_NAME} ✗" | tee -a "$RUN_ALL_LOG"

echo "" | tee -a "$RUN_ALL_LOG"
echo "══════════════ 汇总 ══════════════" | tee -a "$RUN_ALL_LOG"
printf "  %-12s : %s\n" "$G1_NAME" "$([ "$ok1" -eq 1 ] && echo "OK" || echo "FAIL")" | tee -a "$RUN_ALL_LOG"
printf "  %-12s : %s\n" "$G2_NAME" "$([ "$ok2" -eq 1 ] && echo "OK" || echo "FAIL")" | tee -a "$RUN_ALL_LOG"
echo "" | tee -a "$RUN_ALL_LOG"
echo "Outputs: ${OUT_ROOT}/" | tee -a "$RUN_ALL_LOG"
echo "Logs:    ${LOG_ROOT}/" | tee -a "$RUN_ALL_LOG"
[[ "$ok1" -eq 1 && "$ok2" -eq 1 ]] && echo "All OK." | tee -a "$RUN_ALL_LOG" \
    || echo "Some FAILED — check logs." | tee -a "$RUN_ALL_LOG"

exit $(( 1 - (ok1 & ok2) ))
