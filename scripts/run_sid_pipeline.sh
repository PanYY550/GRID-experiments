#!/usr/bin/env bash
# ===========================================================================
# run_sid_pipeline.sh — 通用 SID 训练 + SID 推理流水线
#
# 用法:
#   bash scripts/run_sid_pipeline.sh          # 使用默认控制台参数运行
#   N_GROUPS=2 bash scripts/run_sid_pipeline.sh  # 环境变量覆盖
#
# 设计:
#   控制台 (10-115) → 参数解析 → SID训练 → SID推理 → 碰撞分析
# ===========================================================================

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                        ██  控制台 (CONSOLE)  ██                            ║
# ║                    所有可调参数集中在此，修改后直接运行                        ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

# ── 1. 实验组数 & GPU 分配 ──────────────────────────────────────────────────
N_GROUPS="${N_GROUPS:-3}"                         # 实验组数 (1/2/3/...)
GPU_LIST=(${GPU_LIST:-0 1 2})                     # GPU 分配, 默认: 组1→GPU0, 组2→GPU1, 组3→GPU2
# 用法示例: GPU_LIST="1 2" N_GROUPS=2 bash run_sid_pipeline.sh  # 两组都用GPU1/2

# ── 2. 通用参数 (所有实验组共享) ─────────────────────────────────────────────

# 数据路径
DATA_DIR="${DATA_DIR:-/home/pyy/GRID/src/data/amazon_data/beauty}"
EMB_PATH="${EMB_PATH:-/home/pyy/GRID/embeddings/beauty/pickle/merged_predictions_tensor.pt}"
EMB_DIM="${EMB_DIM:-768}"
EXPOSURE_COUNTS="${EXPOSURE_COUNTS:-/home/pyy/GRID/src/data/amazon_data/beauty/exposure_counts.pt}"

# 模型结构
SID_NUM_HIER="${SID_NUM_HIER:-3}"
CODEBOOK="${CODEBOOK:-256}"

# 训练控制
SEED="${SEED:-42}"
SID_MAX_STEPS="${SID_MAX_STEPS:-3000}"
BATCH_SIZE="${BATCH_SIZE:-2048}"
NUM_WORKERS="${NUM_WORKERS:-12}"
SCHEDULER_WARMUP="${SCHEDULER_WARMUP:-1000}"

# VCF 通用底座
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
REPULSION_WARMUP="${REPULSION_WARMUP:-200}"
CVPM_TEMP="${CVPM_TEMP:-0.15}"
ALPHA="${ALPHA:-0.5}"

# 增强开关 (默认全关，每组按需开启)
USE_TIME_DECAY="${USE_TIME_DECAY:-false}"
USE_DYNAMIC_MARGIN="${USE_DYNAMIC_MARGIN:-false}"
USE_ASYMMETRIC="${USE_ASYMMETRIC:-false}"
USE_TCL="${USE_TCL:-false}"
USE_DUAL_TOWER="${USE_DUAL_TOWER:-false}"
USE_KPL="${USE_KPL:-false}"
KPL_COLLISION_EXCLUDE_RADIUS="${KPL_COLLISION_EXCLUDE_RADIUS:-0}"
KPL_LAMBDA="${KPL_LAMBDA:-0.001}"
KPL_RAMP="${KPL_RAMP:-1000}"

# TCL 参数
LAMBDA_CL="${LAMBDA_CL:-0.001}"
QUASID_CL_TAU="${QUASID_CL_TAU:-0.5}"
CL_RAMP_STEPS="${CL_RAMP_STEPS:-1000}"

# Online k-NN + NPR
USE_ONLINE_KNN="${USE_ONLINE_KNN:-true}"
USE_NPR="${USE_NPR:-true}"
NPR_ALPHA_MIN="${NPR_ALPHA_MIN:-0.01}"
ONLINE_KNN_K="${ONLINE_KNN_K:-50}"
ONLINE_KNN_EMA="${ONLINE_KNN_EMA:-0.99}"
ONLINE_KNN_INTERVAL="${ONLINE_KNN_INTERVAL:-100}"

# ── 3. 各组特殊参数 (覆盖通用参数) ────────────────────────────────────────
# 格式: GROUP_SPECS[i]="参数名1=值1 参数名2=值2 ..."
# 留空则完全使用通用参数
#
#  可用变量 (用 Bash 间接引用):
#    GROUP_NAME[i]    — 实验组名称
#    GROUP_SPECS[i]   — 额外 CLI 参数 (空格分隔)
#    GROUP_DESC[i]    — 实验描述 (打印日志用)

GROUP_NAME[0]="${G1_NAME:-TCL_only}"
GROUP_SPECS[0]="${G1_SPECS:-model.use_tcl=true model.use_npr=false model.use_dual_tower=true model.use_online_knn=false}"
GROUP_DESC[0]="${G1_DESC:-TCL-only: VCF+CVPM+TCL (no NPR)}"

GROUP_NAME[1]="${G2_NAME:-NPR_only}"
GROUP_SPECS[1]="${G2_SPECS:-model.use_tcl=false model.use_npr=true}"
GROUP_DESC[1]="${G2_DESC:-NPR-only: VCF+CVPM+NPR (no TCL)}"

GROUP_NAME[2]="${G3_NAME:-NPR_TCL}"
GROUP_SPECS[2]="${G3_SPECS:-model.use_tcl=true model.use_npr=true model.use_dual_tower=true}"
GROUP_DESC[2]="${G3_DESC:-NPR+TCL: VCF+CVPM+TCL+NPR}"

# ── 4. RUN_TAG & 输出/日志路径 ──────────────────────────────────────────────
RUN_TAG="${RUN_TAG:-sid_pipeline_$(date +%Y%m%d_%H%M%S)}"
OUT_ROOT="${OUT_ROOT:-outputs/sid_pipeline/${RUN_TAG}}"
LOG_ROOT="${LOG_ROOT:-logs/sid_pipeline/${RUN_TAG}}"

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                         ██  脚本逻辑 (勿修改)  ██                          ║
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

# ── 构建 SID 训练通用参数底座 ────────────────────────────────────────────────
SID_BASE_ARGS=(
    experiment=rqvae_vcf_online_knn_train_flat
    data_dir="${DATA_DIR}"
    embedding_path="${EMB_PATH}"
    embedding_dim="${EMB_DIM}"
    num_hierarchies="${SID_NUM_HIER}"
    codebook_width="${CODEBOOK}"
    "data_loading.datamodule.train_dataloader_config.batch_size_per_device=${BATCH_SIZE}"
    "data_loading.datamodule.train_dataloader_config.num_workers=${NUM_WORKERS}"
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
    ++should_skip_retry=true
)

# ── 辅助函数 ─────────────────────────────────────────────────────────────────
_latest_ckpt() { ls -t "${1}"/checkpoints/*.ckpt 2>/dev/null | head -n 1; }

_die() { echo "ERROR: $*" >&2; return 1; }

# ── SID 训练 ─────────────────────────────────────────────────────────────────
run_sid_train() {
    local gpu="$1" name="$2" run_dir="$3" log_file="$4"
    shift 4
    local extra_args=("$@")

    echo "── [${name}] SID 训练 (GPU ${gpu}, ${SID_MAX_STEPS} steps) ──" >> "$RUN_ALL_LOG"
    echo "── [${name}] SID 训练 (GPU ${gpu}, ${SID_MAX_STEPS} steps) ──" >&2
    echo "    输出: ${run_dir}" >> "$RUN_ALL_LOG"
    echo "    输出: ${run_dir}" >&2

    CUDA_VISIBLE_DEVICES="${gpu}" python -m src.train \
        "${SID_BASE_ARGS[@]}" \
        "${extra_args[@]}" \
        "hydra.run.dir=${run_dir}" \
        > "${log_file}" 2>&1

    local ckpt; ckpt="$(_latest_ckpt "${run_dir}")"
    if [[ -z "$ckpt" || ! -f "$ckpt" ]]; then _die "[${name}] checkpoint 未生成"; return 1; fi
    echo "[${name}] Checkpoint: ${ckpt}" >> "$RUN_ALL_LOG"
    echo "[${name}] Checkpoint: ${ckpt}" >&2
    echo "${ckpt}"
}

# ── SID 推理 ─────────────────────────────────────────────────────────────────
run_sid_infer() {
    local gpu="$1" name="$2" ckpt="$3" run_dir="$4" log_file="$5"

    echo "── [${name}] SID 推理 (GPU ${gpu}) ──" >> "$RUN_ALL_LOG"
    echo "── [${name}] SID 推理 (GPU ${gpu}) ──" >&2

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

    # 快速校验
    python -c "
import torch; t=torch.load('${sid_pt}',map_location='cpu',weights_only=True)
assert t.ndim==2 and t.shape[0]==$((SID_NUM_HIER + 1)), f'shape err: {t.shape}'
print(f'SID OK: {t.shape}')" >&2

    echo "[${name}] SID OK: ${sid_pt}" >> "$RUN_ALL_LOG"
    echo "[${name}] SID OK: ${sid_pt}" >&2
    echo "${sid_pt}"
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

# ── 单组流水线 (SID训练 → SID推理) ──────────────────────────────────────────
run_group() {
    local gpu="$1" name="$2"
    shift 2
    local extra_args=("$@")

    local run_root="${OUT_ROOT}/${name}"
    mkdir -p "${run_root}/01_sid_train" "${run_root}/02_sid_inference"

    local ckpt sid_pt
    ckpt="$(run_sid_train "${gpu}" "${name}" \
                         "${run_root}/01_sid_train" \
                         "${LOG_ROOT}/${name}_01_sid_train.log" \
                         "${extra_args[@]}")" || return 1

    sid_pt="$(run_sid_infer "${gpu}" "${name}" "${ckpt}" \
                             "${run_root}/02_sid_inference" \
                             "${LOG_ROOT}/${name}_02_sid_inference.log")" || return 1

    run_collision_check "${sid_pt}" "${name}" "${LOG_ROOT}/${name}_collision.txt"
    return 0
}

# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

# ── 打印控制台摘要 ──────────────────────────────────────────────────────────
{
    echo "╔══════════════════════════════════════════════════════════╗"
    echo "║        SID Pipeline — 通用训练+推理流水线               ║"
    echo "╠══════════════════════════════════════════════════════════╣"
    printf "║  RUN_TAG: %-46s ║\n" "$RUN_TAG"
    printf "║  实验组数: %-43s ║\n" "$N_GROUPS"
    echo "╠══════════════════════════════════════════════════════════╣"
    printf "║  batch=%-6s  workers=%-4s  steps=%-6s  warmup=%-5s ║\n" \
        "$BATCH_SIZE" "$NUM_WORKERS" "$SID_MAX_STEPS" "$REPULSION_WARMUP"
    printf "║  λ_full=%-5s λ_partial=%-6s m_full=%-5s m_partial=%-5s ║\n" \
        "$LAMBDA_FULL" "$LAMBDA_PARTIAL" "$M_FULL" "$M_PARTIAL"
    printf "║  time_decay=%-5s dynamic_m=%-5s tcl=%-5s npr=%-5s ║\n" \
        "$USE_TIME_DECAY" "$USE_DYNAMIC_MARGIN" "$USE_TCL" "$USE_NPR"
    echo "╠══════════════════════════════════════════════════════════╣"
    for ((i=0; i<N_GROUPS; i++)); do
        printf "║  GPU%-2s → %-47s ║\n" "${GPU_LIST[$i]}" "${GROUP_NAME[$i]}"
    done
    echo "╠══════════════════════════════════════════════════════════╣"
    printf "║  输出: %-49s ║\n" "$OUT_ROOT"
    printf "║  日志: %-49s ║\n" "$LOG_ROOT"
    echo "╚══════════════════════════════════════════════════════════╝"
} | tee "$RUN_ALL_LOG"

# ── 启动各组 (后台并行) ─────────────────────────────────────────────────────
declare -a GROUP_PIDS=()
declare -a GROUP_STATUS=()

for ((i=0; i<N_GROUPS; i++)); do
    gpu="${GPU_LIST[$i]}"
    name="${GROUP_NAME[$i]}"
    specs="${GROUP_SPECS[$i]:-}"
    desc="${GROUP_DESC[$i]:-}"

    echo "[main] 启动 ${name} (GPU ${gpu}) — ${desc}" | tee -a "$RUN_ALL_LOG"

    # 将 specs 字符串拆分为数组
    read -ra extra <<< "$specs"

    (run_group "$gpu" "$name" "${extra[@]}") &
    GROUP_PIDS+=($!)
    GROUP_STATUS+=(0)
done

# ── 等待所有组完成 ──────────────────────────────────────────────────────────
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
