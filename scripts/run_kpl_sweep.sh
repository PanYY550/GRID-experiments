#!/usr/bin/env bash
# ===========================================================================
# run_kpl_sweep.sh — NPR+TCL + KPL 码本稳定性参数扫描
#
# 用法:
#   bash scripts/run_kpl_sweep.sh
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

GPU1="${GPU1:-1}"
GPU2="${GPU2:-2}"

RUN_TAG="${RUN_TAG:-kpl_sweep_$(date +%Y%m%d_%H%M%S)}"
OUT_ROOT="${OUT_ROOT:-outputs/kpl_sweep/${RUN_TAG}}"
LOG_ROOT="${LOG_ROOT:-logs/kpl_sweep/${RUN_TAG}}"

# ── 实验组 ─────────────────────────────────────────────────────────────
# 编辑以下数组切换要扫的参数

# 组1
G1_NAME="KPL_l1m3_r2"
G1_LAMBDA_KPL="0.001"
G1_EXCLUDE_R="2"
G1_DESC="λ_kpl=0.001, excl_r=2 (best λ)"

# 组2
G2_NAME="KPL_l1m2_r2"
G2_LAMBDA_KPL="0.01"
G2_EXCLUDE_R="2"
G2_DESC="λ_kpl=0.01, excl_r=2 (rescue?)"

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

# 共用基础参数 — QuaSID base + NPR(α_min=0.01) + TCL(λ_cl=0.1, τ=0.5)
SID_COMMON_BASE=(
    experiment=rqvae_vcf_online_knn_train_flat
    data_dir="${DATA_DIR}"
    embedding_path="${EMB_PATH}"
    embedding_dim=768
    num_hierarchies=3
    codebook_width=256
    data_loading.datamodule.train_dataloader_config.batch_size_per_device=256
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
    model.use_npr=true
    model.use_online_knn=true
    model.use_kpl=true
    model.use_tcl=true
    model.use_dual_tower=true
    "model.quasid_cl_tau=0.5"
    ++should_skip_retry=true
)

# 共用底座 overrides: QuaSID params + NPR(α_min=0.01) + TCL
COMMON_OVERRIDES=(
    # QuaSID VCF
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
    # Optimizer
    optim.optimizer._target_=torch.optim.Adam
    optim.optimizer.lr=0.0003
    optim.optimizer.weight_decay=0.00001
    # NPR
    model.npr_alpha_min=0.01
    model.online_knn_k=50
    # TCL + KPL ramp (step 0 即满权)
    model.kpl_ramp_steps=0
)

_latest_ckpt() { ls -t "${1}"/checkpoints/*.ckpt 2>/dev/null | head -n 1; }
_die() { echo "ERROR: $*" >&2; return 1; }

run_group() {
    local gpu="$1" name="$2" lambda_kpl="$3" excl_r="$4"
    shift 4
    local extra_args=("$@")
    local run_root="${OUT_ROOT}/${name}"
    local log_pref="${LOG_ROOT}/${name}"
    mkdir -p "${run_root}/01_sid_train" "${run_root}/02_sid_inference"

    # ── SID 训练 ──────────────────────────────────────────────────
    echo "── [${name}] SID训练 (GPU${gpu}, ${SID_STEPS}steps, λ_kpl=${lambda_kpl}, excl_r=${excl_r}) ──" | tee -a "$RUN_ALL_LOG"

    CUDA_VISIBLE_DEVICES="${gpu}" python -m src.train \
        "${SID_COMMON_BASE[@]}" \
        "${COMMON_OVERRIDES[@]}" \
        "model.lambda_kpl=${lambda_kpl}" \
        "model.kpl_collision_exclude_radius=${excl_r}" \
        "${extra_args[@]}" \
        "model.lambda_cl=0.1" \
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
    echo "║   KPL Sweep — NPR+TCL + KPL 码本稳定性                  ║"
    echo "╠══════════════════════════════════════════════════════════╣"
    printf "║  RUN_TAG: %-46s ║\n" "$RUN_TAG"
    printf "║  steps=%-3s  batch=256  base=NPR(α=0.01)+TCL(λ=0.1)  ║\n" "$SID_STEPS"
    echo "║  底座: QuaSID(R=1,λ=0.2/0.1,m=0.8/0.5 static,Adam)   ║"
    echo "║  KPL ramp_steps=0 (step 0 满权)                        ║"
    echo "╠══════════════════════════════════════════════════════════╣"
    printf "║  GPU%-2s→ %-16s λ_kpl=%-6s excl_r=%-2s %-25s ║\n" \
        "$GPU1" "$G1_NAME" "$G1_LAMBDA_KPL" "$G1_EXCLUDE_R" "$G1_DESC"
    printf "║  GPU%-2s→ %-16s λ_kpl=%-6s excl_r=%-2s %-25s ║\n" \
        "$GPU2" "$G2_NAME" "$G2_LAMBDA_KPL" "$G2_EXCLUDE_R" "$G2_DESC"
    echo "╚══════════════════════════════════════════════════════════╝"
} | tee "$RUN_ALL_LOG"

echo "[main] 启动 ${G1_NAME} (GPU ${GPU1}) — ${G1_DESC}" | tee -a "$RUN_ALL_LOG"
(run_group "$GPU1" "$G1_NAME" "$G1_LAMBDA_KPL" "$G1_EXCLUDE_R") &
PID1=$!

echo "[main] 启动 ${G2_NAME} (GPU ${GPU2}) — ${G2_DESC}" | tee -a "$RUN_ALL_LOG"
(run_group "$GPU2" "$G2_NAME" "$G2_LAMBDA_KPL" "$G2_EXCLUDE_R") &
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
