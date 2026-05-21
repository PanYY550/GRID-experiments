#!/usr/bin/env bash
# ===========================================================================
# run_npr_tcl_dynm.sh — NPR+TCL+dynamic_margin 消融实验 (SID→TIGER 全流程)
#
# 用法:
#   bash scripts/run_npr_tcl_dynm.sh
#   SID_MAX_STEPS=100 TIGER_MAX_STEPS=100 bash scripts/run_npr_tcl_dynm.sh  # smoke
#
# 流水线:
#   01 SID训练 → 02 SID推理 → 碰撞分析 → 03 TIGER训练
# ===========================================================================
#
# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║              ██  实验: dynamic_margin 消融验证  ██                          ║
# ╠══════════════════════════════════════════════════════════════════════════════╣
# ║                                                                              ║
# ║  目标: 验证 dynamic_margin 在 NPR+TCL 底座上的增量效果                        ║
# ║                                                                              ║
# ║  ┌──────────────────────────────────────────────────────────────────────┐   ║
# ║  │ 实验组 (1 组, GPU 1)                                                  │   ║
# ║  │                                                                       │   ║
# ║  │  NPR_TCL_dynM: NPR+TCL+dynamic_margin ★                               │   ║
# ║  │                                                                       │   ║
# ║  │  SID 底座 (对标 NPR+TCL 3g):                                           │   ║
# ║  │    batch=4096   warmup=1000   λ=1.0/0.5                               │   ║
# ║  │    m_full=0.3   m_partial=0.15   m_start=0.05/0.02                     │   ║
# ║  │    margin_ramp=2000   hamming_radius=2   alpha=0.5                     │   ║
# ║  │    use_vcf=true   use_cvpm=true   cvpm_temp=0.15                       │   ║
# ║  │    use_tcl=true   lambda_cl=0.001   quasid_cl_tau=0.5   cl_ramp=1000   │   ║
# ║  │    use_dual_tower=true                                                 │   ║
# ║  │    use_npr=true   npr_alpha_min=0.01                                   │   ║
# ║  │    use_online_knn=true   online_knn_k=50   ema=0.99   interval=100     │   ║
# ║  │    use_time_decay=false   use_asymmetric=false   use_kpl=false         │   ║
# ║  │                                                                       │   ║
# ║  │  dynamic_margin 专项:                                                  │   ║
# ║  │    use_dynamic_margin=true   ★ 唯一差异                                │   ║
# ║  │    model.m0=0.8              ★ 对齐 QuaSID 原始 m_full=0.8             │   ║
# ║  │    margin_dissim_weight=0.3  (yaml 默认)                               │   ║
# ║  │    severity_beta=0.5         (yaml 默认)                               │   ║
# ║  │    min_margin=0.3            (yaml 默认)                               │   ║
# ║  │                                                                       │   ║
# ║  │  dynamic_margin 公式:                                                  │   ║
# ║  │    margin_base = m0 + margin_dissim_weight * (1-cos_sim)               │   ║
# ║  │    margin = margin_base * severity                                     │   ║
# ║  │    severity_full = 1 + beta = 1.5                                      │   ║
# ║  │    severity_partial = clamp(1+beta*(1-H/R), min=1.1)                   │   ║
# ║  │                                                                       │   ║
# ║  │  有效范围 (m0=0.8):                                                    │   ║
# ║  │    语义相似 (dissim≈0), 全碰撞: margin = 0.8 * 1.5 = 1.20              │   ║
# ║  │    语义迥异 (dissim≈0.8), 全碰撞: margin = 1.04 * 1.5 = 1.56           │   ║
# ║  │    语义相似 (dissim≈0), 部分碰撞: margin = 0.8 * 1.1 = 0.88            │   ║
# ║  │                                                                       │   ║
# ║  │  Online NPR 对照 (m0=0.5): margin_full ∈ [0.75, 1.20]                  │   ║
# ║  │  新脚本 (m0=0.8): margin_full ∈ [1.20, 1.56]                           │   ║
# ║  │  静态 3g 对照: m_full=0.3 (全碰撞), m_partial=0.15 (部分碰撞)           │   ║
# ║  │  QuaSID 原始: m_full=0.8 (全碰撞), m_partial=0.5 (部分碰撞)             │   ║
# ║  └──────────────────────────────────────────────────────────────────────┘   ║
# ║                                                                              ║
# ║  ┌──────────────────────────────────────────────────────────────────────┐   ║
# ║  │ 对标基线: NPR+TCL (3g) from expr_3g_sid3000_tiger                     │   ║
# ║  │                                                                       │   ║
# ║  │  NPR+TCL: 同上所有 SID 参数，仅 use_dynamic_margin=false  ★            │   ║
# ║  │    SID: L0=100% (256/256)  partial_collision(R≤2)=2.43%               │   ║
# ║  │    TIGER: test/ndcg@10=0.0335  test/recall@10=0.0604                  │   ║
# ║  │                                                                       │   ║
# ║  │  TIGER 训练配置 (与 3g 完全相同):                                       │   ║
# ║  │    num_hierarchies=4   max_steps=320000   batch=32                     │   ║
# ║  │    grad_accum=16   val_interval=1600   log_interval=100                │   ║
# ║  │    lr=0.001   weight_decay=0.0001                                      │   ║
# ║  └──────────────────────────────────────────────────────────────────────┘   ║
# ║                                                                              ║
# ║  假设: dynamic_margin = 按内容相似度差异化 margin                              ║
# ║        语义相似碰撞 → 低 margin (轻推，保护邻域)                              ║
# ║        语义迥异碰撞 → 高 margin (重推，强化区分)                              ║
# ║        → 更精准的排斥力分配 → partial_collision↓ + NDCG↑                      ║
# ║                                                                              ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                        ██  控制台 (CONSOLE)  ██                            ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

# ── 1. GPU ────────────────────────────────────────────────────────────────────
GPU="${GPU:-1}"

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
BATCH_SIZE="${BATCH_SIZE:-4096}"
NUM_WORKERS="${NUM_WORKERS:-12}"
SCHEDULER_WARMUP="${SCHEDULER_WARMUP:-1000}"

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
USE_DYNAMIC_MARGIN="${USE_DYNAMIC_MARGIN:-true}"
USE_ASYMMETRIC="${USE_ASYMMETRIC:-false}"
USE_TCL="${USE_TCL:-true}"
USE_DUAL_TOWER="${USE_DUAL_TOWER:-true}"
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

# ── 8. dynamic_margin 专项 (覆盖 yaml 默认 m0=0.5 → 对齐 m_full=0.3) ─────
DYN_M0="${DYN_M0:-0.8}"
DYN_DISSIM_WEIGHT="${DYN_DISSIM_WEIGHT:-0.3}"

# ── 9. TIGER 训练参数 (对标 3g: num_hier=4, 320K steps) ─────────────────────
TIGER_NUM_HIER="${TIGER_NUM_HIER:-4}"
TIGER_MAX_STEPS="${TIGER_MAX_STEPS:-320000}"
TIGER_BATCH_SIZE="${TIGER_BATCH_SIZE:-32}"
TIGER_LR="${TIGER_LR:-0.001}"
TIGER_WEIGHT_DECAY="${TIGER_WEIGHT_DECAY:-0.0001}"
TIGER_VAL_INTERVAL="${TIGER_VAL_INTERVAL:-1600}"
TIGER_GRAD_ACCUM="${TIGER_GRAD_ACCUM:-16}"
TIGER_LOG_INTERVAL="${TIGER_LOG_INTERVAL:-100}"
SEQUENCE_LENGTH="${SEQUENCE_LENGTH:-120}"
TIGER_CHECKPOINT_EVERY="${TIGER_CHECKPOINT_EVERY:-null}"

# ── 10. 实验命名 & 输出路径 ──────────────────────────────────────────────────
RUN_TAG="${RUN_TAG:-npr_tcl_dynm_$(date +%Y%m%d_%H%M%S)}"
OUT_ROOT="${OUT_ROOT:-outputs/sid_pipeline/${RUN_TAG}}"
LOG_ROOT="${LOG_ROOT:-logs/sid_pipeline/${RUN_TAG}}"

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

# ── 打印控制台摘要 ──────────────────────────────────────────────────────────
{
    echo "╔══════════════════════════════════════════════════════════╗"
    echo "║   NPR+TCL+dynamic_margin — SID+TIGER 消融实验         ║"
    echo "╠══════════════════════════════════════════════════════════╣"
    printf "║  RUN_TAG: %-46s ║\n" "$RUN_TAG"
    printf "║  GPU: %-51s ║\n" "$GPU"
    echo "╠══════════════════════════════════════════════════════════╣"
    printf "║  SID: batch=%-4s steps=%-5s warmup=%-5s             ║\n" \
        "$BATCH_SIZE" "$SID_MAX_STEPS" "$REPULSION_WARMUP"
    printf "║  λ_full=%-4s λ_partial=%-4s m_full=%-4s m_partial=%-4s ║\n" \
        "$LAMBDA_FULL" "$LAMBDA_PARTIAL" "$M_FULL" "$M_PARTIAL"
    printf "║  TCL=%-5s  NPR=%-5s  dynamic_margin=%-5s            ║\n" \
        "$USE_TCL" "$USE_NPR" "$USE_DYNAMIC_MARGIN"
    printf "║  dyn_m0=%-4s dyn_dissim_w=%-4s  (QuaSID m0=0.8)   ║\n" \
        "$DYN_M0" "$DYN_DISSIM_WEIGHT"
    echo "╠══════════════════════════════════════════════════════════╣"
    printf "║  TIGER: batch=%-3s steps=%-6s val_int=%-5s accum=%-3s ║\n" \
        "$TIGER_BATCH_SIZE" "$TIGER_MAX_STEPS" "$TIGER_VAL_INTERVAL" "$TIGER_GRAD_ACCUM"
    echo "╠══════════════════════════════════════════════════════════╣"
    printf "║  对标: NPR+TCL (3g) use_dynamic_margin=false           ║\n"
    printf "║        NDCG=0.0335  L0=100%%  pcol=2.43%%               ║\n"
    echo "╠══════════════════════════════════════════════════════════╣"
    printf "║  输出: %-49s ║\n" "$OUT_ROOT"
    printf "║  日志: %-49s ║\n" "$LOG_ROOT"
    echo "╚══════════════════════════════════════════════════════════╝"
}

# ── SID 训练参数底座 ─────────────────────────────────────────────────────────
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
    "model.m0=${DYN_M0}"
    "model.margin_dissim_weight=${DYN_DISSIM_WEIGHT}"
    "callbacks.model_checkpoint.every_n_train_steps=${SID_MAX_STEPS}"
    "optim.scheduler.warmup_steps=${SCHEDULER_WARMUP}"
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

# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

NAME="NPR_TCL_dynM"
RUN_DIR="${OUT_ROOT}/${NAME}"

# ── 01 SID 训练 ──────────────────────────────────────────────────────────────
echo "── [${NAME}] 01 SID训练 (GPU ${GPU}, ${SID_MAX_STEPS} steps) ──" >&2

mkdir -p "${RUN_DIR}/01_sid_train"

CUDA_VISIBLE_DEVICES="${GPU}" python -m src.train \
    "${SID_BASE_ARGS[@]}" \
    "hydra.run.dir=${RUN_DIR}/01_sid_train" \
    > "${LOG_ROOT}/${NAME}_01_sid_train.log" 2>&1

SID_CKPT="$(_latest_ckpt "${RUN_DIR}/01_sid_train")"
if [[ -z "$SID_CKPT" || ! -f "$SID_CKPT" ]]; then
    _die "[${NAME}] SID checkpoint 未生成"
    exit 1
fi
echo "[${NAME}] SID ckpt: ${SID_CKPT}" >&2

# ── 02 SID 推理 ──────────────────────────────────────────────────────────────
echo "── [${NAME}] 02 SID推理 (GPU ${GPU}) ──" >&2

mkdir -p "${RUN_DIR}/02_sid_inference"

CUDA_VISIBLE_DEVICES="${GPU}" python -m src.inference \
    experiment=rqvae_vcf_inference_flat \
    data_dir="${DATA_DIR}" \
    embedding_path="${EMB_PATH}" \
    embedding_dim="${EMB_DIM}" \
    num_hierarchies="${SID_NUM_HIER}" \
    codebook_width="${CODEBOOK}" \
    "ckpt_path=${SID_CKPT}" \
    seed="${SEED}" \
    "hydra.run.dir=${RUN_DIR}/02_sid_inference" \
    > "${LOG_ROOT}/${NAME}_02_sid_inference.log" 2>&1

SID_PT="${RUN_DIR}/02_sid_inference/pickle/merged_predictions_tensor.pt"
if [[ ! -f "$SID_PT" ]]; then
    _die "[${NAME}] SID tensor 未生成: ${SID_PT}"
    exit 1
fi

python -c "
import torch; t=torch.load('${SID_PT}',map_location='cpu',weights_only=True)
assert t.ndim==2 and t.shape[0]==$((SID_NUM_HIER + 1)), f'shape err: {t.shape}'
print(f'SID OK: {t.shape}')" >&2

echo "[${NAME}] SID OK: ${SID_PT}" >&2

# ── 碰撞分析 ─────────────────────────────────────────────────────────────────
echo "── [${NAME}] 碰撞分析 ──" >&2
python scripts/analyze_sid_collisions.py --path "${SID_PT}" \
    | tee "${LOG_ROOT}/${NAME}_collision.txt"

# ── 03 TIGER 训练 ────────────────────────────────────────────────────────────
echo "── [${NAME}] 03 TIGER训练 (GPU ${GPU}, ${TIGER_MAX_STEPS} steps) ──" >&2

mkdir -p "${RUN_DIR}/03_tiger_train"

CUDA_VISIBLE_DEVICES="${GPU}" python -m src.train \
    "${TIGER_BASE_ARGS[@]}" \
    "semantic_id_path=${SID_PT}" \
    "hydra.run.dir=${RUN_DIR}/03_tiger_train" \
    > "${LOG_ROOT}/${NAME}_03_tiger_train.log" 2>&1

TIGER_CKPT="$(_latest_ckpt "${RUN_DIR}/03_tiger_train")"
if [[ -z "$TIGER_CKPT" || ! -f "$TIGER_CKPT" ]]; then
    _die "[${NAME}] TIGER checkpoint 未生成"
    exit 1
fi
echo "[${NAME}] TIGER ckpt: ${TIGER_CKPT}" >&2

# ── 完成 ─────────────────────────────────────────────────────────────────────
echo "" >&2
echo "══════════════ 完成 ══════════════" >&2
echo "SID ckpt:    ${SID_CKPT}" >&2
echo "SID tensor:  ${SID_PT}" >&2
echo "TIGER ckpt:  ${TIGER_CKPT}" >&2
echo "Logs:        ${LOG_ROOT}/" >&2
echo "Outputs:     ${RUN_DIR}/" >&2
