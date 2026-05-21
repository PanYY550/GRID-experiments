#!/usr/bin/env bash
# ===========================================================================
# run_sid_tiger_pipeline.sh — SID训练 → SID推理 → TIGER训练
#
# 用法:
#   bash scripts/run_sid_tiger_pipeline.sh
#   N_GROUPS=2 bash scripts/run_sid_tiger_pipeline.sh  # 环境变量覆盖
#
# 流水线:
#   01 SID训练 → 02 SID推理 → 03 TIGER训练
# ===========================================================================

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                        ██  控制台 (CONSOLE)  ██                            ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

# ── 1. 实验组数 & GPU 分配 ──────────────────────────────────────────────────
N_GROUPS="${N_GROUPS:-3}"
GPU_LIST=(${GPU_LIST:-0 1 2})
# 用法: GPU_LIST="1 2" N_GROUPS=2 bash run_sid_tiger_pipeline.sh

# ── 2. 数据路径 ─────────────────────────────────────────────────────────────
DATA_DIR="${DATA_DIR:-/home/pyy/GRID/src/data/amazon_data/beauty}"
EMB_PATH="${EMB_PATH:-/home/pyy/GRID/embeddings/beauty/pickle/merged_predictions_tensor.pt}"
EMB_DIM="${EMB_DIM:-768}"
EXPOSURE_COUNTS="${EXPOSURE_COUNTS:-/home/pyy/GRID/src/data/amazon_data/beauty/exposure_counts.pt}"

# ── 3. SID 模型结构 & 训练 ──────────────────────────────────────────────────
SID_NUM_HIER="${SID_NUM_HIER:-3}"
CODEBOOK="${CODEBOOK:-256}"
SEED="${SEED:-42}"
SID_MAX_STEPS="${SID_MAX_STEPS:-3000}"
SID_BATCH_SIZE="${SID_BATCH_SIZE:-4096}"
SID_NUM_WORKERS="${SID_NUM_WORKERS:-12}"
SCHEDULER_WARMUP="${SCHEDULER_WARMUP:-1000}"
SID_OPTIMIZER="${SID_OPTIMIZER:-torch.optim.Adagrad}"
SID_LR="${SID_LR:-0.001}"
SID_WEIGHT_DECAY="${SID_WEIGHT_DECAY:-0.0}"

# ── 4. VCF 通用底座 ─────────────────────────────────────────────────────────
USE_VCF="${USE_VCF:-true}"
USE_CVPM="${USE_CVPM:-true}"
HAMMING_R="${HAMMING_R:-2}"
LAMBDA_FULL="${LAMBDA_FULL:-1.0}"
LAMBDA_PARTIAL="${LAMBDA_PARTIAL:-0.5}"
M_FULL="${M_FULL:-0.3}"
M_PARTIAL="${M_PARTIAL:-0.15}"
M_FULL_START="${M_FULL_START:-0.05}"
M_PARTIAL_START="${M_PARTIAL_START:-0.02}"
MARGIN_RAMP="${MARGIN_RAMP:-2000}"
REPULSION_WARMUP="${REPULSION_WARMUP:-1000}"
CVPM_TEMP="${CVPM_TEMP:-0.15}"
ALPHA="${ALPHA:-0.5}"

# ── 5. 增强开关 ─────────────────────────────────────────────────────────────
USE_TIME_DECAY="${USE_TIME_DECAY:-false}"
USE_DYNAMIC_MARGIN="${USE_DYNAMIC_MARGIN:-false}"
USE_ASYMMETRIC="${USE_ASYMMETRIC:-false}"
USE_TCL="${USE_TCL:-false}"
USE_DUAL_TOWER="${USE_DUAL_TOWER:-false}"
USE_KPL="${USE_KPL:-false}"
KPL_COLLISION_EXCLUDE_RADIUS="${KPL_COLLISION_EXCLUDE_RADIUS:-0}"
KPL_LAMBDA="${KPL_LAMBDA:-0.001}"
KPL_RAMP="${KPL_RAMP:-1000}"

# ── 6. TCL 参数 ─────────────────────────────────────────────────────────────
LAMBDA_CL="${LAMBDA_CL:-0.001}"
QUASID_CL_TAU="${QUASID_CL_TAU:-0.5}"
CL_RAMP_STEPS="${CL_RAMP_STEPS:-1000}"

# ── 7. Online k-NN + NPR ────────────────────────────────────────────────────
USE_ONLINE_KNN="${USE_ONLINE_KNN:-true}"
USE_NPR="${USE_NPR:-true}"
NPR_ALPHA_MIN="${NPR_ALPHA_MIN:-0.01}"
ONLINE_KNN_K="${ONLINE_KNN_K:-50}"
ONLINE_KNN_EMA="${ONLINE_KNN_EMA:-0.99}"
ONLINE_KNN_INTERVAL="${ONLINE_KNN_INTERVAL:-100}"

# ── 8. TIGER 训练参数 ───────────────────────────────────────────────────────
TIGER_NUM_HIER="${TIGER_NUM_HIER:-4}"                # SID层数+1 (含item_id)
TIGER_MAX_STEPS="${TIGER_MAX_STEPS:-320000}"
TIGER_BATCH_SIZE="${TIGER_BATCH_SIZE:-32}"
TIGER_LR="${TIGER_LR:-0.001}"
TIGER_WEIGHT_DECAY="${TIGER_WEIGHT_DECAY:-0.0001}"
TIGER_VAL_INTERVAL="${TIGER_VAL_INTERVAL:-1600}"
TIGER_GRAD_ACCUM="${TIGER_GRAD_ACCUM:-16}"
TIGER_LOG_INTERVAL="${TIGER_LOG_INTERVAL:-100}"
SEQUENCE_LENGTH="${SEQUENCE_LENGTH:-120}"
TIGER_CHECKPOINT_EVERY="${TIGER_CHECKPOINT_EVERY:-null}"

# ── 10. 各组特殊参数 ────────────────────────────────────────────────────────
GROUP_NAME[0]="${G1_NAME:-TCL_only}"
GROUP_SPECS[0]="${G1_SPECS:-model.use_tcl=true model.use_npr=false model.use_dual_tower=true model.use_online_knn=false}"
GROUP_DESC[0]="${G1_DESC:-TCL-only: VCF+CVPM+TCL (no NPR)}"

GROUP_NAME[1]="${G2_NAME:-NPR_only}"
GROUP_SPECS[1]="${G2_SPECS:-model.use_tcl=false model.use_npr=true}"
GROUP_DESC[1]="${G2_DESC:-NPR-only: VCF+CVPM+NPR (no TCL)}"

GROUP_NAME[2]="${G3_NAME:-NPR_TCL}"
GROUP_SPECS[2]="${G3_SPECS:-model.use_tcl=true model.use_npr=true model.use_dual_tower=true}"
GROUP_DESC[2]="${G3_DESC:-NPR+TCL: VCF+CVPM+TCL+NPR}"

# ── 11. RUN_TAG & 输出/日志路径 ─────────────────────────────────────────────
RUN_TAG="${RUN_TAG:-sid_tiger_$(date +%Y%m%d_%H%M%S)}"
OUT_ROOT="${OUT_ROOT:-outputs/sid_tiger/${RUN_TAG}}"
LOG_ROOT="${LOG_ROOT:-logs/sid_tiger/${RUN_TAG}}"

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                         ██  脚本逻辑  ██                                   ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

# ── 环境检查 ─────────────────────────────────────────────────────────────────
if [[ -f "/root/miniconda3/etc/profile.d/conda.sh" ]]; then
    source /root/miniconda3/etc/profile.d/conda.sh
    conda activate grid
else
    echo "ERROR: conda.sh not found" >&2; exit 1
fi

mkdir -p "$OUT_ROOT" "$LOG_ROOT"
RUN_ALL_LOG="${LOG_ROOT}/run_all.log"

# ── SID 训练参数底座 ─────────────────────────────────────────────────────────
SID_BASE_ARGS=(
    experiment=rqvae_vcf_online_knn_train_flat
    data_dir="${DATA_DIR}"
    embedding_path="${EMB_PATH}"
    embedding_dim="${EMB_DIM}"
    num_hierarchies="${SID_NUM_HIER}"
    codebook_width="${CODEBOOK}"
    "data_loading.datamodule.train_dataloader_config.batch_size_per_device=${SID_BATCH_SIZE}"
    "data_loading.datamodule.train_dataloader_config.num_workers=${SID_NUM_WORKERS}"
    data_loading.datamodule.train_dataloader_config.pin_memory=true
    data_loading.datamodule.train_dataloader_config.persistent_workers=true
    seed="${SEED}"
    trainer.max_epochs=null
    "trainer.max_steps=${SID_MAX_STEPS}"
    "model.use_vcf=${USE_VCF}"
    "model.use_cvpm=${USE_CVPM}"
    "model.hamming_radius=${HAMMING_R}"
    "model.lambda_full=${LAMBDA_FULL}"
    "model.lambda_partial=${LAMBDA_PARTIAL}"
    "model.m_full=${M_FULL}"
    "model.m_partial=${M_PARTIAL}"
    "model.m_full_start=${M_FULL_START}"
    "model.m_partial_start=${M_PARTIAL_START}"
    "model.margin_ramp_steps=${MARGIN_RAMP}"
    "model.alpha=${ALPHA}"
    "model.exposure_counts_path=${EXPOSURE_COUNTS}"
    "model.repulsion_warmup_steps=${REPULSION_WARMUP}"
    "model.cvpm_temperature=${CVPM_TEMP}"
    "model.use_time_decay=${USE_TIME_DECAY}"
    "model.use_dynamic_margin=${USE_DYNAMIC_MARGIN}"
    "model.use_asymmetric_repulsion=${USE_ASYMMETRIC}"
    "model.use_tcl=${USE_TCL}"
    "model.use_dual_tower=${USE_DUAL_TOWER}"
    "model.use_kpl=${USE_KPL}"
    "model.lambda_kpl=${KPL_LAMBDA}"
    "model.kpl_ramp_steps=${KPL_RAMP}"
    "model.kpl_collision_exclude_radius=${KPL_COLLISION_EXCLUDE_RADIUS}"
    "model.lambda_cl=${LAMBDA_CL}"
    "model.quasid_cl_tau=${QUASID_CL_TAU}"
    "model.cl_ramp_steps=${CL_RAMP_STEPS}"
    "model.use_online_knn=${USE_ONLINE_KNN}"
    "model.use_npr=${USE_NPR}"
    "model.npr_alpha_min=${NPR_ALPHA_MIN}"
    "model.online_knn_k=${ONLINE_KNN_K}"
    "model.online_knn_ema_momentum=${ONLINE_KNN_EMA}"
    "model.online_knn_update_interval=${ONLINE_KNN_INTERVAL}"
    "callbacks.model_checkpoint.every_n_train_steps=${SID_MAX_STEPS}"
    "optim.scheduler.warmup_steps=${SCHEDULER_WARMUP}"
    "optim.optimizer._target_=${SID_OPTIMIZER}"
    "optim.optimizer.lr=${SID_LR}"
    "optim.optimizer.weight_decay=${SID_WEIGHT_DECAY}"
    ++should_skip_retry=true
)

# ── TIGER 训练参数底座 ──────────────────────────────────────────────────────
TIGER_BASE_ARGS=(
    experiment=tiger_train_flat
    data_dir="${DATA_DIR}"
    seed="${SEED}"
    "num_hierarchies=${TIGER_NUM_HIER}"
    "sequence_length=${SEQUENCE_LENGTH}"
    "data_loading.train_dataloader_config.dataloader.batch_size_per_device=${TIGER_BATCH_SIZE}"
    "trainer.max_steps=${TIGER_MAX_STEPS}"
    "trainer.max_epochs=null"
    "trainer.val_check_interval=${TIGER_VAL_INTERVAL}"
    "trainer.log_every_n_steps=${TIGER_LOG_INTERVAL}"
    "trainer.accumulate_grad_batches=${TIGER_GRAD_ACCUM}"
    "optim.optimizer.lr=${TIGER_LR}"
    "optim.optimizer.weight_decay=${TIGER_WEIGHT_DECAY}"
    "callbacks.model_checkpoint.every_n_train_steps=${TIGER_CHECKPOINT_EVERY}"
    ++should_skip_retry=true
)

# ── 辅助函数 ─────────────────────────────────────────────────────────────────
_latest_ckpt() { ls -t "${1}"/checkpoints/*.ckpt 2>/dev/null | head -n 1; }
_die() { echo "ERROR: $*" >&2; return 1; }

# ── 01 SID 训练 ──────────────────────────────────────────────────────────────
run_sid_train() {
    local gpu="$1" name="$2" run_dir="$3" log_file="$4"
    shift 4
    local extra_args=("$@")

    echo "── [${name}] 01 SID训练 (GPU ${gpu}, ${SID_MAX_STEPS} steps) ──" >> "$RUN_ALL_LOG"
    echo "── [${name}] 01 SID训练 (GPU ${gpu}, ${SID_MAX_STEPS} steps) ──" >&2
    echo "    输出: ${run_dir}" >> "$RUN_ALL_LOG"

    CUDA_VISIBLE_DEVICES="${gpu}" python -m src.train \
        "${SID_BASE_ARGS[@]}" \
        "${extra_args[@]}" \
        "hydra.run.dir=${run_dir}" \
        > "${log_file}" 2>&1

    local ckpt; ckpt="$(_latest_ckpt "${run_dir}")"
    if [[ -z "$ckpt" || ! -f "$ckpt" ]]; then _die "[${name}] SID checkpoint 未生成"; return 1; fi
    echo "[${name}] SID ckpt: ${ckpt}" >> "$RUN_ALL_LOG"
    echo "[${name}] SID ckpt: ${ckpt}" >&2
    echo "${ckpt}"
}

# ── 02 SID 推理 ──────────────────────────────────────────────────────────────
run_sid_infer() {
    local gpu="$1" name="$2" ckpt="$3" run_dir="$4" log_file="$5"

    echo "── [${name}] 02 SID推理 (GPU ${gpu}) ──" >> "$RUN_ALL_LOG"
    echo "── [${name}] 02 SID推理 (GPU ${gpu}) ──" >&2

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

    local sid_pt="${run_dir}/pickle/merged_predictions_tensor.pt"
    if [[ ! -f "$sid_pt" ]]; then _die "[${name}] SID tensor 未生成: ${sid_pt}"; return 1; fi

    python -c "
import torch; t=torch.load('${sid_pt}',map_location='cpu',weights_only=True)
assert t.ndim==2 and t.shape[0]==$((SID_NUM_HIER + 1)), f'shape err: {t.shape}'
print(f'SID OK: {t.shape}')" >&2

    echo "[${name}] SID OK: ${sid_pt}" >> "$RUN_ALL_LOG"
    echo "[${name}] SID OK: ${sid_pt}" >&2
    echo "${sid_pt}"
}

# ── 03 TIGER 训练 ────────────────────────────────────────────────────────────
run_tiger_train() {
    local gpu="$1" name="$2" sid_pt="$3" run_dir="$4" log_file="$5"

    echo "── [${name}] 03 TIGER训练 (GPU ${gpu}, ${TIGER_MAX_STEPS} steps) ──" >> "$RUN_ALL_LOG"
    echo "── [${name}] 03 TIGER训练 (GPU ${gpu}, ${TIGER_MAX_STEPS} steps) ──" >&2
    echo "    SID: ${sid_pt}" >> "$RUN_ALL_LOG"

    CUDA_VISIBLE_DEVICES="${gpu}" python -m src.train \
        "${TIGER_BASE_ARGS[@]}" \
        "semantic_id_path=${sid_pt}" \
        "hydra.run.dir=${run_dir}" \
        > "${log_file}" 2>&1

    local ckpt; ckpt="$(_latest_ckpt "${run_dir}")"
    if [[ -z "$ckpt" || ! -f "$ckpt" ]]; then
        echo "[${name}] TIGER warning: no checkpoint (val_check_interval > steps?)" >> "$RUN_ALL_LOG"
        echo "[${name}] TIGER warning: no checkpoint (val_check_interval > steps?)" >&2
        echo ""
    else
        echo "[${name}] TIGER ckpt: ${ckpt}" >> "$RUN_ALL_LOG"
        echo "[${name}] TIGER ckpt: ${ckpt}" >&2
        echo "${ckpt}"
    fi
}

# ── 碰撞分析 ─────────────────────────────────────────────────────────────────
run_collision_check() {
    local sid_pt="$1" name="$2" out_txt="$3"
    echo "── [${name}] 碰撞分析 ──" >> "$RUN_ALL_LOG"
    echo "── [${name}] 碰撞分析 ──" >&2
    python scripts/analyze_sid_collisions.py --path "${sid_pt}" \
        > "${out_txt}" 2>&1
    cat "${out_txt}" >> "$RUN_ALL_LOG"
    cat "${out_txt}" >&2
}

# ── 单组流水线 ──────────────────────────────────────────────────────────────
run_group() {
    local gpu="$1" name="$2"
    shift 2
    local extra_args=("$@")

    local run_root="${OUT_ROOT}/${name}"
    mkdir -p "${run_root}/01_sid_train" "${run_root}/02_sid_inference" \
             "${run_root}/03_tiger_train"

    # 01 — SID 训练
    local sid_ckpt; sid_ckpt="$(run_sid_train "${gpu}" "${name}" \
        "${run_root}/01_sid_train" \
        "${LOG_ROOT}/${name}_01_sid_train.log" \
        "${extra_args[@]}")" || return 1

    # 02 — SID 推理
    local sid_pt; sid_pt="$(run_sid_infer "${gpu}" "${name}" "${sid_ckpt}" \
        "${run_root}/02_sid_inference" \
        "${LOG_ROOT}/${name}_02_sid_inference.log")" || return 1

    run_collision_check "${sid_pt}" "${name}" "${LOG_ROOT}/${name}_collision.txt"

    # 03 — TIGER 训练
    local tiger_ckpt; tiger_ckpt="$(run_tiger_train "${gpu}" "${name}" "${sid_pt}" \
        "${run_root}/03_tiger_train" \
        "${LOG_ROOT}/${name}_03_tiger_train.log")" || return 1

    return 0
}

# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

# ── 控制台摘要 ──────────────────────────────────────────────────────────────
{
    echo "╔══════════════════════════════════════════════════════════╗"
    echo "║    SID+TIGER Pipeline — 训练+推理全流程                 ║"
    echo "╠══════════════════════════════════════════════════════════╣"
    printf "║  RUN_TAG: %-46s ║\n" "$RUN_TAG"
    printf "║  实验组数: %-43s ║\n" "$N_GROUPS"
    echo "╠══════════════════════════════════════════════════════════╣"
    printf "║  SID: batch=%-4s steps=%-6s warmup=%-5s             ║\n" \
        "$SID_BATCH_SIZE" "$SID_MAX_STEPS" "$REPULSION_WARMUP"
    printf "║  TCL=%-5s NPR=%-5s online_knn=%-5s                 ║\n" \
        "$USE_TCL" "$USE_NPR" "$USE_ONLINE_KNN"
    printf "║  Optim=%-12s lr=%-6s wd=%-8s            ║\n" \
        "${SID_OPTIMIZER##*.}" "$SID_LR" "$SID_WEIGHT_DECAY"
    printf "║  TIGER: batch=%-3s steps=%-6s val_int=%-5s accum=%-3s ║\n" \
        "$TIGER_BATCH_SIZE" "$TIGER_MAX_STEPS" "$TIGER_VAL_INTERVAL" "$TIGER_GRAD_ACCUM"
    echo "╠══════════════════════════════════════════════════════════╣"
    for ((i=0; i<N_GROUPS; i++)); do
        printf "║  GPU%-2s → %-47s ║\n" "${GPU_LIST[$i]}" "${GROUP_NAME[$i]}"
    done
    echo "╠══════════════════════════════════════════════════════════╣"
    printf "║  输出: %-49s ║\n" "$OUT_ROOT"
    printf "║  日志: %-49s ║\n" "$LOG_ROOT"
    echo "╚══════════════════════════════════════════════════════════╝"
} | tee "$RUN_ALL_LOG"

# ── 启动各组 ─────────────────────────────────────────────────────────────────
declare -a GROUP_PIDS=()
declare -a GROUP_STATUS=()

for ((i=0; i<N_GROUPS; i++)); do
    gpu="${GPU_LIST[$i]}"
    name="${GROUP_NAME[$i]}"
    specs="${GROUP_SPECS[$i]:-}"
    desc="${GROUP_DESC[$i]:-}"

    echo "[main] 启动 ${name} (GPU ${gpu}) — ${desc}" | tee -a "$RUN_ALL_LOG"

    read -ra extra <<< "$specs"

    (run_group "$gpu" "$name" "${extra[@]}") &
    GROUP_PIDS+=($!)
    GROUP_STATUS+=(0)
done

# ── 等待完成 ─────────────────────────────────────────────────────────────────
for ((i=0; i<N_GROUPS; i++)); do
    wait "${GROUP_PIDS[$i]}" && GROUP_STATUS[$i]=1 && \
        echo "[main] ${GROUP_NAME[$i]} ✓ 完成" | tee -a "$RUN_ALL_LOG" || \
        echo "[main] ${GROUP_NAME[$i]} ✗ 失败" | tee -a "$RUN_ALL_LOG"
done

# ── 汇总 ─────────────────────────────────────────────────────────────────────
echo "" | tee -a "$RUN_ALL_LOG"
echo "══════════════ 汇总 ══════════════" | tee -a "$RUN_ALL_LOG"
all_ok=1
for ((i=0; i<N_GROUPS; i++)); do
    status=$([ "${GROUP_STATUS[$i]}" -eq 1 ] && echo "OK" || echo "FAIL")
    printf "  %-30s : %s\n" "${GROUP_NAME[$i]}" "$status" | tee -a "$RUN_ALL_LOG"
    [[ "${GROUP_STATUS[$i]}" -ne 1 ]] && all_ok=0
done

echo "" | tee -a "$RUN_ALL_LOG"
echo "Outputs: ${OUT_ROOT}/" | tee -a "$RUN_ALL_LOG"
echo "Logs:    ${LOG_ROOT}/" | tee -a "$RUN_ALL_LOG"

[[ "$all_ok" -eq 1 ]] && echo "All groups OK." | tee -a "$RUN_ALL_LOG" \
    || echo "Some groups FAILED — check logs." | tee -a "$RUN_ALL_LOG"

exit $(( 1 - all_ok ))
