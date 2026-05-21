#!/usr/bin/env bash
# ===========================================================================
# run_expr_online_npr_compare.sh
#
# Online NPR 对照实验：三组并行，共同底座 = G0 RQ-VAE + batch=4096 + steps=3000
#
#   E1 (GPU0): Online NPR + HaMR + CVPM
#     — HaMR+CVPM 底座 + Online k-NN + NPR
#     — 对比基线: C2ctrl* (本脚本), G0 (0.03256)
#
#   E2 (GPU1): Online NPR, 仅相同SID推开, 无HaMR, 无CVPM
#     — 最简 VCF (hamming_radius=0) + Online k-NN + NPR
#     — 对比基线: G0 (0.03256)
#
#   C2ctrl* (GPU2): HaMR + CVPM (无 NPR, 无 CL)
#     — 修正了旧 C2ctrl 的 margin_ramp_steps=2000 问题 → 真正静态 margin
#     — 对比基线: G0 (0.03256), 旧 C2ctrl (0.03284)
#
# 用法:
#   bash scripts/run_expr_online_npr_compare.sh
#   SID_MAX_STEPS=3000 bash scripts/run_expr_online_npr_compare.sh
# ===========================================================================

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

# ─── conda 环境 ─────────────────────────────────────────────────────────────
if [[ -f "/root/miniconda3/etc/profile.d/conda.sh" ]]; then
    source /root/miniconda3/etc/profile.d/conda.sh
    conda activate grid
    python - <<'PY'
import sys
try:
    import hydra  # noqa
except Exception as e:
    print(f"ERROR: python env missing hydra: {e}", file=sys.stderr); raise
print("[OK] python env ready")
PY
else
    echo "ERROR: conda.sh not found" >&2; exit 1
fi

# ─── 数据 / 嵌入路径 ─────────────────────────────────────────────────────────
DATA_DIR="${DATA_DIR:-/home/pyy/GRID/src/data/amazon_data/beauty}"
EMB_PATH="${EMB_PATH:-/home/pyy/GRID/embeddings/beauty/pickle/merged_predictions_tensor.pt}"
EMB_DIM="${EMB_DIM:-768}"
EXPOSURE_COUNTS="${EXPOSURE_COUNTS:-/home/pyy/GRID/src/data/amazon_data/beauty/exposure_counts.pt}"

# ─── 通用超参（对齐 G0/C2ctrl）──────────────────────────────────────────────
SID_NUM_HIER="${SID_NUM_HIER:-3}"
TIGER_NUM_HIER="${TIGER_NUM_HIER:-4}"
CODEBOOK="${CODEBOOK:-256}"
SEED="${SEED:-42}"

SID_MAX_STEPS="${SID_MAX_STEPS:-3000}"
SID_BATCH_PER_DEVICE="${SID_BATCH_PER_DEVICE:-4096}"
SID_NUM_WORKERS="${SID_NUM_WORKERS:-16}"
TIGER_MAX_STEPS="${TIGER_MAX_STEPS:-10000}"

# ─── VCF 共享超参 ───────────────────────────────────────────────────────────
HAMMING_R=2
M_FULL=0.5
M_PARTIAL=0.3
LAMBDA_FULL=0.2
LAMBDA_PARTIAL=0.1
REPULSION_WARMUP=2000
CVPM_TEMP=0.15

# ─── Online k-NN + NPR 共享超参 ─────────────────────────────────────────────
NPR_ALPHA_MIN=0.01
ONLINE_KNN_K=50
ONLINE_KNN_EMA=0.99
ONLINE_KNN_INTERVAL=100

# ─── GPU 分配 ───────────────────────────────────────────────────────────────
GPU_E1="${GPU_E1:-0}"
GPU_E2="${GPU_E2:-1}"
GPU_CTRL="${GPU_CTRL:-2}"

# ─── 输出/日志目录 ──────────────────────────────────────────────────────────
RUN_TAG="${RUN_TAG:-online_npr_compare_$(date +%Y%m%d_%H%M%S)}"
OUT_ROOT="outputs/online_npr_compare/${RUN_TAG}"
LOG_ROOT="logs/online_npr_compare/${RUN_TAG}"
mkdir -p "$OUT_ROOT" "$LOG_ROOT"
RUN_ALL_LOG="${LOG_ROOT}/run_all.log"

# ─── 辅助函数 ───────────────────────────────────────────────────────────────
latest_ckpt() { ls -t "${1}"/checkpoints/*.ckpt 2>/dev/null | head -n 1; }

require_ckpt() {
    local ckpt="$1" who="$2"
    if [[ -z "$ckpt" || ! -f "$ckpt" ]]; then
        echo "ERROR: [${who}] checkpoint not found." >&2; exit 2
    fi
}

assert_sid_ok() {
    local pt="$1" hier="$2"
    python - <<PY
import sys, torch, os
p = ${pt@Q}; h = int(${hier@Q})
if not os.path.exists(p):
    print(f"ERROR: SID tensor not found: {p}", file=sys.stderr); sys.exit(2)
t = torch.load(p, map_location="cpu")
if t.ndim != 2 or t.shape[0] != h:
    print(f"ERROR: shape mismatch: expected ({h}, N) got {tuple(t.shape)}", file=sys.stderr); sys.exit(2)
print(f"[OK] SID tensor: shape={tuple(t.shape)} dtype={t.dtype}", file=sys.stderr)
PY
}

collision_check() {
    local sid_pt="$1" who="$2" out_txt="$3"
    echo "== Collision check: ${who} ==" | tee -a "$RUN_ALL_LOG"
    python scripts/analyze_sid_collisions.py --path "${sid_pt}" | tee "${out_txt}" | tee -a "$RUN_ALL_LOG"
}

print_tiger_metrics() {
    local log="$1" who="$2"
    echo "── ${who} TIGER best metrics ──"
    if [[ -f "$log" ]]; then
        grep -E "ndcg@10|recall@10|ndcg@5|recall@5|ndcg_plus_hr" "$log" | tail -20 \
            || echo "  (metrics not found in log)"
    else
        echo "  (log not found: $log)"
    fi
}

# ══════════════════════════════════════════════════════════════════════════════
# SID 训练 — 共用参数底座
# ══════════════════════════════════════════════════════════════════════════════
SID_COMMON_ARGS=(
    experiment=rqvae_vcf_online_knn_train_flat
    data_dir="${DATA_DIR}"
    embedding_path="${EMB_PATH}"
    embedding_dim="${EMB_DIM}"
    num_hierarchies="${SID_NUM_HIER}"
    codebook_width="${CODEBOOK}"
    "data_loading.datamodule.train_dataloader_config.batch_size_per_device=${SID_BATCH_PER_DEVICE}"
    "data_loading.datamodule.train_dataloader_config.num_workers=${SID_NUM_WORKERS}"
    data_loading.datamodule.train_dataloader_config.pin_memory=true
    data_loading.datamodule.train_dataloader_config.persistent_workers=true
    seed="${SEED}"
    trainer.max_epochs=null
    "trainer.max_steps=${SID_MAX_STEPS}"
    model.use_time_decay=false
    model.use_dynamic_margin=false
    model.use_asymmetric_repulsion=false
    model.use_tcl=false
    model.use_kpl=false
    model.use_dual_tower=false
    "model.margin_ramp_steps=0"
    "model.online_knn_ema_momentum=${ONLINE_KNN_EMA}"
    "model.online_knn_update_interval=${ONLINE_KNN_INTERVAL}"
    "model.online_knn_k=${ONLINE_KNN_K}"
    "model.npr_alpha_min=${NPR_ALPHA_MIN}"
    callbacks.model_checkpoint.every_n_train_steps="${SID_MAX_STEPS}"
    "optim.scheduler.warmup_steps=1000"
    ++should_skip_retry=true
)

# ══════════════════════════════════════════════════════════════════════════════
# SID 训练 runner
# ══════════════════════════════════════════════════════════════════════════════
run_sid_train() {
    local gpu="$1" name="$2" run_dir="$3" log_file="$4"
    shift 4
    local extra_args=("$@")

    echo "[${name}] (GPU ${gpu}) SID Training (${SID_MAX_STEPS} steps)..." | tee -a "$RUN_ALL_LOG" >&2

    CUDA_VISIBLE_DEVICES="${gpu}" python -m src.train \
        "${SID_COMMON_ARGS[@]}" \
        "${extra_args[@]}" \
        "hydra.run.dir=${run_dir}" \
        > "${log_file}" 2>&1

    local ckpt; ckpt="$(latest_ckpt "${run_dir}")"
    require_ckpt "${ckpt}" "${name}"
    echo "[${name}] Checkpoint: ${ckpt}" | tee -a "$RUN_ALL_LOG" >&2
    echo "${ckpt}"
}

# ══════════════════════════════════════════════════════════════════════════════
# SID 推理 runner
# ══════════════════════════════════════════════════════════════════════════════
run_sid_infer() {
    local gpu="$1" name="$2" ckpt="$3" run_dir="$4" log_file="$5"

    echo "[${name}] (GPU ${gpu}) SID Inference..." | tee -a "$RUN_ALL_LOG" >&2

    CUDA_VISIBLE_DEVICES="${gpu}" python -m src.inference \
        experiment=rqvae_vcf_inference_flat \
        data_dir="${DATA_DIR}" \
        embedding_path="${EMB_PATH}" \
        embedding_dim="${EMB_DIM}" \
        num_hierarchies="${SID_NUM_HIER}" \
        codebook_width="${CODEBOOK}" \
        "ckpt_path=${ckpt}" \
        seed="${SEED}" \
        "hydra.run.dir=${run_dir}" \
        > "${log_file}" 2>&1

    local sid_pt="${ROOT_DIR}/${run_dir}/pickle/merged_predictions_tensor.pt"
    assert_sid_ok "${sid_pt}" "${TIGER_NUM_HIER}"
    echo "[${name}] SID OK: ${sid_pt}" | tee -a "$RUN_ALL_LOG" >&2
    echo "${sid_pt}"
}

# ══════════════════════════════════════════════════════════════════════════════
# TIGER 训练 runner
# ══════════════════════════════════════════════════════════════════════════════
run_tiger_train() {
    local gpu="$1" name="$2" sid_pt="$3" run_dir="$4" log_file="$5"

    echo "[${name}] (GPU ${gpu}) TIGER Training..." | tee -a "$RUN_ALL_LOG" >&2

    local -a tiger_args=(
        experiment=tiger_train_flat
        data_dir="${DATA_DIR}"
        num_hierarchies="${TIGER_NUM_HIER}"
        "semantic_id_path=${sid_pt}"
        seed="${SEED}"
        "hydra.run.dir=${run_dir}"
    )
    [[ -n "${TIGER_MAX_STEPS}" ]] && tiger_args+=( "trainer.max_steps=${TIGER_MAX_STEPS}" )

    CUDA_VISIBLE_DEVICES="${gpu}" python -m src.train \
        "${tiger_args[@]}" \
        > "${log_file}" 2>&1

    echo "[${name}] Done." | tee -a "$RUN_ALL_LOG" >&2
}

# ══════════════════════════════════════════════════════════════════════════════
# 单组三阶段流水线
# ══════════════════════════════════════════════════════════════════════════════
run_pipeline() {
    local gpu="$1" name="$2"
    shift 2
    local extra_args=("$@")

    local run_root="${OUT_ROOT}/${name}"
    local log_prefix="${LOG_ROOT}/${name}"
    mkdir -p "${run_root}/01_sid_train" "${run_root}/02_sid_inference" "${run_root}/03_tiger_train"

    local ckpt
    ckpt="$(run_sid_train  "${gpu}" "${name}" \
                           "${run_root}/01_sid_train" \
                           "${log_prefix}_01_sid_train.log" \
                           "${extra_args[@]}")"

    local sid_pt
    sid_pt="$(run_sid_infer   "${gpu}" "${name}" "${ckpt}" \
                               "${run_root}/02_sid_inference" \
                               "${log_prefix}_02_sid_inference.log")"

    run_tiger_train "${gpu}" "${name}" "${sid_pt}" \
                    "${run_root}/03_tiger_train" \
                    "${log_prefix}_03_tiger_train.log"
}

# ══════════════════════════════════════════════════════════════════════════════
# E1: Online NPR + HaMR + CVPM  (GPU 0)
# ══════════════════════════════════════════════════════════════════════════════
E1_NAME="E1_online_npr_hamr_cvpm"
E1_ARGS=(
    model.use_vcf=true
    model.use_cvpm=true
    "model.hamming_radius=${HAMMING_R}"
    "model.lambda_full=${LAMBDA_FULL}"
    "model.lambda_partial=${LAMBDA_PARTIAL}"
    "model.m_full=${M_FULL}"
    "model.m_partial=${M_PARTIAL}"
    "model.alpha=0.5"
    "model.exposure_counts_path=${EXPOSURE_COUNTS}"
    "model.repulsion_warmup_steps=${REPULSION_WARMUP}"
    "model.cvpm_temperature=${CVPM_TEMP}"
    model.use_online_knn=true
    model.use_npr=true
)

# ══════════════════════════════════════════════════════════════════════════════
# E2: Online NPR, 仅相同SID推开, 无HaMR, 无CVPM  (GPU 1)
# ══════════════════════════════════════════════════════════════════════════════
E2_NAME="E2_online_npr_minimal"
E2_ARGS=(
    model.use_vcf=true
    model.use_cvpm=false
    "model.hamming_radius=0"
    "model.lambda_full=${LAMBDA_FULL}"
    "model.m_full=${M_FULL}"
    "model.alpha=0.0"
    "model.repulsion_warmup_steps=${REPULSION_WARMUP}"
    model.use_online_knn=true
    model.use_npr=true
)

# ══════════════════════════════════════════════════════════════════════════════
# C2ctrl*: HaMR + CVPM (修正版，静态 margin，无 NPR，无 CL)  (GPU 2)
# ══════════════════════════════════════════════════════════════════════════════
CTRL_NAME="C2ctrl_corrected"
CTRL_ARGS=(
    model.use_vcf=true
    model.use_cvpm=true
    "model.hamming_radius=${HAMMING_R}"
    "model.lambda_full=${LAMBDA_FULL}"
    "model.lambda_partial=${LAMBDA_PARTIAL}"
    "model.m_full=${M_FULL}"
    "model.m_partial=${M_PARTIAL}"
    "model.alpha=0.5"
    "model.exposure_counts_path=${EXPOSURE_COUNTS}"
    "model.repulsion_warmup_steps=${REPULSION_WARMUP}"
    "model.cvpm_temperature=${CVPM_TEMP}"
    model.use_online_knn=false
    model.use_npr=false
)

# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════
main() {
    {
        echo "============================================================"
        echo "Online NPR Compare — 三组对照实验"
        echo "RUN_TAG: ${RUN_TAG}"
        echo "------------------------------------------------------------"
        echo "共同底座: batch=4096, steps=${SID_MAX_STEPS}, seed=${SEED}"
        echo "共同关闭: time_decay, dynamic_margin, CL, dual_tower, asymmetric"
        echo "共同设定: margin_ramp_steps=0 (静态 margin)"
        echo "------------------------------------------------------------"
        echo "GPU 分配:"
        echo "  GPU${GPU_E1} → ${E1_NAME}: VCF(HaMR)+CVPM + Online-kNN+NPR"
        echo "  GPU${GPU_E2} → ${E2_NAME}: VCF(same-SID only) + Online-kNN+NPR"
        echo "  GPU${GPU_CTRL} → ${CTRL_NAME}: VCF(HaMR)+CVPM (修正静态 margin)"
        echo "------------------------------------------------------------"
        echo "VCF 参数:"
        echo "  λ_full=${LAMBDA_FULL}  λ_partial=${LAMBDA_PARTIAL}"
        echo "  m_full=${M_FULL}  m_partial=${M_PARTIAL}  hamming_radius=${HAMMING_R}"
        echo "  repulsion_warmup=${REPULSION_WARMUP}  cvpm_temp=${CVPM_TEMP}"
        echo "------------------------------------------------------------"
        echo "Online k-NN + NPR:"
        echo "  k=${ONLINE_KNN_K}  ema_momentum=${ONLINE_KNN_EMA}  update_interval=${ONLINE_KNN_INTERVAL}"
        echo "  npr_alpha_min=${NPR_ALPHA_MIN}"
        echo "------------------------------------------------------------"
        echo "输出根目录: ${OUT_ROOT}"
        echo "日志根目录: ${LOG_ROOT}"
        echo "============================================================"
    } | tee "$RUN_ALL_LOG"

    (run_pipeline "${GPU_E1}"   "${E1_NAME}"   "${E1_ARGS[@]}")   & PID_E1=$!
    (run_pipeline "${GPU_E2}"   "${E2_NAME}"   "${E2_ARGS[@]}")   & PID_E2=$!
    (run_pipeline "${GPU_CTRL}" "${CTRL_NAME}" "${CTRL_ARGS[@]}") & PID_CTRL=$!

    echo "[main] E1 pid=${PID_E1} | E2 pid=${PID_E2} | C2ctrl* pid=${PID_CTRL}" | tee -a "$RUN_ALL_LOG"

    local ok_e1=0 ok_e2=0 ok_ctrl=0
    wait "$PID_E1"   && ok_e1=1   && echo "[main] E1 done"   | tee -a "$RUN_ALL_LOG" \
        || echo "[main] E1 FAILED"   | tee -a "$RUN_ALL_LOG"
    wait "$PID_E2"   && ok_e2=1   && echo "[main] E2 done"   | tee -a "$RUN_ALL_LOG" \
        || echo "[main] E2 FAILED"   | tee -a "$RUN_ALL_LOG"
    wait "$PID_CTRL" && ok_ctrl=1 && echo "[main] C2ctrl* done" | tee -a "$RUN_ALL_LOG" \
        || echo "[main] C2ctrl* FAILED" | tee -a "$RUN_ALL_LOG"

    echo "" | tee -a "$RUN_ALL_LOG"
    echo "═══════════ SID 碰撞分析 ═══════════" | tee -a "$RUN_ALL_LOG"
    local e1_pt="${ROOT_DIR}/${OUT_ROOT}/${E1_NAME}/02_sid_inference/pickle/merged_predictions_tensor.pt"
    local e2_pt="${ROOT_DIR}/${OUT_ROOT}/${E2_NAME}/02_sid_inference/pickle/merged_predictions_tensor.pt"
    local ctrl_pt="${ROOT_DIR}/${OUT_ROOT}/${CTRL_NAME}/02_sid_inference/pickle/merged_predictions_tensor.pt"

    [[ "$ok_e1"   -eq 1 ]] && collision_check "${e1_pt}"   "${E1_NAME} (Online NPR+HaMR+CVPM)"   "${LOG_ROOT}/${E1_NAME}_collision.txt"
    [[ "$ok_e2"   -eq 1 ]] && collision_check "${e2_pt}"   "${E2_NAME} (Online NPR, minimal VCF)" "${LOG_ROOT}/${E2_NAME}_collision.txt"
    [[ "$ok_ctrl" -eq 1 ]] && collision_check "${ctrl_pt}" "${CTRL_NAME} (HaMR+CVPM only)"       "${LOG_ROOT}/${CTRL_NAME}_collision.txt"

    echo "" | tee -a "$RUN_ALL_LOG"
    echo "══════════════ TIGER 最终指标汇总 ══════════════" | tee -a "$RUN_ALL_LOG"

    if [[ "$ok_e1" -eq 1 ]]; then
        print_tiger_metrics "${LOG_ROOT}/${E1_NAME}_03_tiger_train.log" "${E1_NAME} (Online NPR+HaMR+CVPM)"
    fi
    if [[ "$ok_e2" -eq 1 ]]; then
        print_tiger_metrics "${LOG_ROOT}/${E2_NAME}_03_tiger_train.log" "${E2_NAME} (Online NPR, minimal VCF)"
    fi
    if [[ "$ok_ctrl" -eq 1 ]]; then
        print_tiger_metrics "${LOG_ROOT}/${CTRL_NAME}_03_tiger_train.log" "${CTRL_NAME} (HaMR+CVPM only, static margin)"
    fi

    echo "" | tee -a "$RUN_ALL_LOG"
    echo "=== All pipelines finished ===" | tee -a "$RUN_ALL_LOG"
    echo "E1       : $([ $ok_e1 -eq 1 ] && echo 'OK' || echo 'FAILED')" | tee -a "$RUN_ALL_LOG"
    echo "E2       : $([ $ok_e2 -eq 1 ] && echo 'OK' || echo 'FAILED')" | tee -a "$RUN_ALL_LOG"
    echo "C2ctrl*  : $([ $ok_ctrl -eq 1 ] && echo 'OK' || echo 'FAILED')" | tee -a "$RUN_ALL_LOG"

    local all_ok=0
    [[ $ok_e1 -eq 1 && $ok_e2 -eq 1 && $ok_ctrl -eq 1 ]] && all_ok=1

    echo ""
    echo "Outputs: ${OUT_ROOT}/"
    echo "Logs:    ${LOG_ROOT}/"

    return $(( 1 - all_ok ))
}

main "$@"
