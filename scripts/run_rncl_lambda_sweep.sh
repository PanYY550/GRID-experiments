#!/usr/bin/env bash
# ===========================================================================
# run_rncl_lambda_sweep.sh — RNCL λ sweep 多组参数扫描 (400步)
#
# 底座: NPR+TCL (完全对齐 4g 实验) + RNCL
# 扫描: 4 组 lambda_rncl，分 2 轮在 2 GPU 上运行
#
# 用法:
#   ROUND=0 bash scripts/run_rncl_lambda_sweep.sh   # Round 0: λ=1e-6, 5e-6
#   ROUND=1 bash scripts/run_rncl_lambda_sweep.sh   # Round 1: λ=1e-5, 3e-5
#   ROUND=2 bash scripts/run_rncl_lambda_sweep.sh   # Round 2: λ=1e-4, 3e-4
#   GPU1=0 GPU2=1 bash scripts/run_rncl_lambda_sweep.sh
# ===========================================================================

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

# ╔══════════════════════════════════════════════════════════════════════╗
# ║                      控制台                                        ║
# ╚══════════════════════════════════════════════════════════════════════╝

ROUND="${ROUND:-1}"

DATA_DIR="${DATA_DIR:-/home/pyy/GRID/src/data/amazon_data/beauty}"
EMB_PATH="${EMB_PATH:-/home/pyy/GRID/embeddings/beauty/pickle/merged_predictions_tensor.pt}"
EXPOSURE_COUNTS="${EXPOSURE_COUNTS:-/home/pyy/GRID/src/data/amazon_data/beauty/exposure_counts.pt}"

SID_STEPS="${SID_STEPS:-400}"
GPU1="${GPU1:-1}"
GPU2="${GPU2:-2}"

RUN_TAG="${RUN_TAG:-rncl_sweep4_$(date +%Y%m%d_%H%M%S)}"
OUT_ROOT="${OUT_ROOT:-outputs/rncl_sweep4/${RUN_TAG}}"
LOG_ROOT="${LOG_ROOT:-logs/rncl_sweep4/${RUN_TAG}}"

# ── 所有实验组定义 ─────────────────────────────────────────────────────
# Round 1: 超小 λ (安全性验证)
# Round 2: 默认及中等 λ (效果验证)

declare -A G_LAMBDA G_DESC

if [[ "$ROUND" == "0" ]]; then
    G1_NAME="rncl_L1e-6"
    G1_LAMBDA="0.000001"
    G1_DESC="RNCL λ=1e-6 (micro)"

    G2_NAME="rncl_L5e-6"
    G2_LAMBDA="0.000005"
    G2_DESC="RNCL λ=5e-6 (tiny)"
elif [[ "$ROUND" == "1" ]]; then
    G1_NAME="rncl_L1e-5"
    G1_LAMBDA="0.00001"
    G1_DESC="RNCL λ=1e-5 (ultra-safe)"

    G2_NAME="rncl_L3e-5"
    G2_LAMBDA="0.00003"
    G2_DESC="RNCL λ=3e-5 (safe)"
elif [[ "$ROUND" == "2" ]]; then
    G1_NAME="rncl_L1e-4"
    G1_LAMBDA="0.0001"
    G1_DESC="RNCL λ=1e-4 (default)"

    G2_NAME="rncl_L3e-4"
    G2_LAMBDA="0.0003"
    G2_DESC="RNCL λ=3e-4 (moderate)"
else
    echo "ERROR: ROUND must be 0, 1, or 2, got $ROUND" >&2
    exit 1
fi

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
RUN_ALL_LOG="${LOG_ROOT}/run_all_round${ROUND}.log"

# ── 共用底座参数 (完全对齐 4g NPR+TCL) ──────────────────────────────────
SID_COMMON_BASE=(
    experiment=rqvae_vcf_online_knn_train_flat
    data_dir="${DATA_DIR}"
    embedding_path="${EMB_PATH}"
    embedding_dim=768
    num_hierarchies=3
    codebook_width=256
    "data_loading.datamodule.train_dataloader_config.batch_size_per_device=256"
    data_loading.datamodule.train_dataloader_config.num_workers=12
    data_loading.datamodule.train_dataloader_config.pin_memory=true
    data_loading.datamodule.train_dataloader_config.persistent_workers=true
    seed=42
    trainer.max_epochs=null
    "trainer.max_steps=${SID_STEPS}"
    "callbacks.model_checkpoint.every_n_train_steps=${SID_STEPS}"
    optim.scheduler.warmup_steps=0
    optim.optimizer._target_=torch.optim.Adam
    optim.optimizer.lr=0.0003
    optim.optimizer.weight_decay=0.00001
    # ── VCF/QuaSID (对齐 4g) ──
    model.use_vcf=true
    model.use_cvpm=true
    model.cvpm_temperature=0.15
    model.alpha=0.5
    model.exposure_counts_path="${EXPOSURE_COUNTS}"
    model.use_time_decay=false
    model.use_dynamic_margin=false
    model.use_asymmetric_repulsion=false
    model.hamming_radius=1
    model.lambda_full=0.2
    model.lambda_partial=0.1
    model.m_full=0.8
    model.m_partial=0.5
    model.m_full_start=0.8
    model.m_partial_start=0.5
    model.margin_ramp_steps=0
    model.repulsion_warmup_steps=0
    # ── TCL (对齐 4g) ──
    model.use_tcl=true
    model.lambda_cl=0.1
    model.quasid_cl_tau=0.5
    model.cl_ramp_steps=0
    # ── NPR + online k-NN (对齐 4g) ──
    model.use_npr=true
    model.npr_alpha_min=0.01
    model.use_online_knn=true
    model.online_knn_k=50
    model.online_knn_ema_momentum=0.99
    model.online_knn_update_interval=100
    # ── KPL OFF ──
    model.use_kpl=false
    # ── Dual tower ──
    model.use_dual_tower=true
    # ── Codebook (对齐 4g) ──
    model.codebook_entropy_weight=0.1
    model.codebook_reset_interval=10
    model.codebook_reset_decay=0.9
    # ── Misc ──
    ++should_skip_retry=true
)

_latest_ckpt() { ls -t "${1}"/checkpoints/*.ckpt 2>/dev/null | head -n 1; }
_die() { echo "ERROR: $*" >&2; return 1; }

run_group() {
    local gpu="$1" name="$2" lambda_rncl="$3"
    local run_root="${OUT_ROOT}/${name}"
    local log_pref="${LOG_ROOT}/${name}"
    mkdir -p "${run_root}/01_sid_train" "${run_root}/02_sid_inference"

    # ── SID 训练 ──────────────────────────────────────────────────
    echo "── [${name}] SID训练 (GPU${gpu}, ${SID_STEPS}steps, λ_rncl=${lambda_rncl}) ──" | tee -a "$RUN_ALL_LOG"

    CUDA_VISIBLE_DEVICES="${gpu}" python -m src.train \
        "${SID_COMMON_BASE[@]}" \
        model.use_rncl=true \
        "model.lambda_rncl=${lambda_rncl}" \
        model.rncl_ramp_steps=500 \
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
    echo "║     RNCL λ Sweep — 多组参数扫描 (400步)                ║"
    echo "╠══════════════════════════════════════════════════════════╣"
    printf "║  RUN_TAG: %-46s ║\n" "$RUN_TAG"
    printf "║  ROUND=%-2s  steps=%-3s  batch=256                     ║\n" "$ROUND" "$SID_STEPS"
    echo "║  底座: NPR+TCL (完全对齐4g, Adam, lr=3e-4)             ║"
    echo "╠══════════════════════════════════════════════════════════╣"
    printf "║  GPU%-2s→ %-16s λ_rncl=%-8s %-24s ║\n" \
        "$GPU1" "$G1_NAME" "$G1_LAMBDA" "$G1_DESC"
    printf "║  GPU%-2s→ %-16s λ_rncl=%-8s %-24s ║\n" \
        "$GPU2" "$G2_NAME" "$G2_LAMBDA" "$G2_DESC"
    echo "╚══════════════════════════════════════════════════════════╝"
} | tee "$RUN_ALL_LOG"

echo "[main] 启动 ${G1_NAME} (GPU ${GPU1}) — ${G1_DESC}" | tee -a "$RUN_ALL_LOG"
(run_group "$GPU1" "$G1_NAME" "$G1_LAMBDA") &
PID1=$!

echo "[main] 启动 ${G2_NAME} (GPU ${GPU2}) — ${G2_DESC}" | tee -a "$RUN_ALL_LOG"
(run_group "$GPU2" "$G2_NAME" "$G2_LAMBDA") &
PID2=$!

ok1=0 ok2=0
wait "$PID1" && ok1=1 && echo "[main] ${G1_NAME} ✓" | tee -a "$RUN_ALL_LOG" \
    || echo "[main] ${G1_NAME} ✗" | tee -a "$RUN_ALL_LOG"
wait "$PID2" && ok2=1 && echo "[main] ${G2_NAME} ✓" | tee -a "$RUN_ALL_LOG" \
    || echo "[main] ${G2_NAME} ✗" | tee -a "$RUN_ALL_LOG"

echo "" | tee -a "$RUN_ALL_LOG"
echo "══════════════ 汇总 (Round ${ROUND}) ══════════════" | tee -a "$RUN_ALL_LOG"
printf "  %-12s : %s\n" "$G1_NAME" "$([ "$ok1" -eq 1 ] && echo "OK" || echo "FAIL")" | tee -a "$RUN_ALL_LOG"
printf "  %-12s : %s\n" "$G2_NAME" "$([ "$ok2" -eq 1 ] && echo "OK" || echo "FAIL")" | tee -a "$RUN_ALL_LOG"
echo "" | tee -a "$RUN_ALL_LOG"
echo "Outputs: ${OUT_ROOT}/" | tee -a "$RUN_ALL_LOG"
echo "Logs:    ${LOG_ROOT}/" | tee -a "$RUN_ALL_LOG"
[[ "$ok1" -eq 1 && "$ok2" -eq 1 ]] && echo "All OK." | tee -a "$RUN_ALL_LOG" \
    || echo "Some FAILED — check logs." | tee -a "$RUN_ALL_LOG"

exit $(( 1 - (ok1 & ok2) ))
