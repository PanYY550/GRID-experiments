#!/usr/bin/env bash
# ===========================================================================
# run_final_4groups.sh — 最优参数 4 组全流程对比
#
# Phase 1 (GPU 0,1,2 并行):
#   Group 1 (GPU0): TCL_only      — QuaSID base + TCL
#   Group 2 (GPU1): NPR+TCL       — QuaSID base + NPR(α=0.01) + TCL
#   Group 3 (GPU2): NPR+TCL+KPL   — QuaSID base + NPR(α=0.01) + TCL + KPL(λ=0.001)
#
# Phase 2 (Phase 1 完成后, GPU0):
#   Group 4 (GPU0): G0            — 纯 RQ-VAE, 参数对齐 (batch=256, Adam)
#
# 用法:
#   bash scripts/run_final_4groups.sh
# ===========================================================================

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ -f "/root/miniconda3/etc/profile.d/conda.sh" ]]; then
    source /root/miniconda3/etc/profile.d/conda.sh
    conda activate grid
else
    echo "ERROR: conda.sh not found" >&2; exit 1
fi

RUN_TAG="${RUN_TAG:-final_4g_$(date +%Y%m%d_%H%M%S)}"

# ===========================================================================
# Phase 1: 3 组 QuaSID-base 实验 (GPU 0,1,2)
# ===========================================================================
echo "╔══════════════════════════════════════════════════════════╗"
echo "║  Phase 1: TCL_only / NPR+TCL / NPR+TCL+KPL (GPU 0,1,2) ║"
echo "╚══════════════════════════════════════════════════════════╝"

# ── 共用 QuaSID 底座参数 ───────────────────────────────────────────────
export DATA_DIR=/home/pyy/GRID/src/data/amazon_data/beauty
export EMB_PATH=/home/pyy/GRID/embeddings/beauty/pickle/merged_predictions_tensor.pt
export EMB_DIM=768
export EXPOSURE_COUNTS=/home/pyy/GRID/src/data/amazon_data/beauty/exposure_counts.pt

export SID_NUM_HIER=3
export CODEBOOK=256
export SEED=42
export SID_MAX_STEPS="${SID_MAX_STEPS:-3000}"
export SID_BATCH_SIZE=256
export SID_NUM_WORKERS=12

# QuaSID VCF params
export HAMMING_R=1
export LAMBDA_FULL=0.2 LAMBDA_PARTIAL=0.1
export M_FULL=0.8 M_PARTIAL=0.5
export M_FULL_START=0.8 M_PARTIAL_START=0.5
export MARGIN_RAMP=0
export REPULSION_WARMUP=0
export SCHEDULER_WARMUP=0
export ALPHA=0.5
export CVPM_TEMP=0.15

# 增强开关默认值
export USE_VCF=true
export USE_CVPM=true
export USE_TIME_DECAY=false
export USE_DYNAMIC_MARGIN=false
export USE_ASYMMETRIC=false
export USE_DUAL_TOWER=true
export USE_TCL=true
export USE_KPL=false
export KPL_LAMBDA=0.001
export KPL_RAMP=0
export KPL_COLLISION_EXCLUDE_RADIUS=0

# TCL params
export LAMBDA_CL=0.1
export QUASID_CL_TAU=0.5
export CL_RAMP_STEPS=0

# Online k-NN + NPR (default off)
export USE_ONLINE_KNN=false
export USE_NPR=false
export NPR_ALPHA_MIN=0.01
export ONLINE_KNN_K=50
export ONLINE_KNN_EMA=0.99
export ONLINE_KNN_INTERVAL=100

# Optimizer (QuaSID: Adam)
export SID_OPTIMIZER=torch.optim.Adam
export SID_LR=0.0003
export SID_WEIGHT_DECAY=0.00001

# TIGER params
export TIGER_NUM_HIER=4
export TIGER_MAX_STEPS="${TIGER_MAX_STEPS:-320000}"
export TIGER_BATCH_SIZE=32
export TIGER_LR=0.001
export TIGER_WEIGHT_DECAY=0.0001
export TIGER_VAL_INTERVAL=1600
export TIGER_GRAD_ACCUM=16
export TIGER_LOG_INTERVAL=100
export SEQUENCE_LENGTH=120

export N_GROUPS=3
export GPU_LIST="0 1 2"

# Group 1: TCL_only
export G1_NAME="TCL_only"
export G1_SPECS="model.use_tcl=true model.use_npr=false model.use_online_knn=false model.use_kpl=false model.use_dual_tower=true"
export G1_DESC="TCL_only: QuaSID(λ_cl=0.1,τ=0.5)"

# Group 2: NPR+TCL
export G2_NAME="NPR_TCL"
export G2_SPECS="model.use_tcl=true model.use_npr=true model.npr_alpha_min=0.01 model.use_online_knn=true model.online_knn_k=50 model.use_kpl=false model.use_dual_tower=true"
export G2_DESC="NPR+TCL: QuaSID+NPR(α=0.01)+TCL(λ_cl=0.1)"

# Group 3: NPR+TCL+KPL
export G3_NAME="NPR_TCL_KPL"
export G3_SPECS="model.use_tcl=true model.use_npr=true model.npr_alpha_min=0.01 model.use_online_knn=true model.online_knn_k=50 model.use_kpl=true model.lambda_kpl=0.001 model.kpl_collision_exclude_radius=0 model.use_dual_tower=true"
export G3_DESC="NPR+TCL+KPL: QuaSID+NPR(α=0.01)+TCL(λ_cl=0.1)+KPL(λ=0.001)"

export RUN_TAG="${RUN_TAG}"
export OUT_ROOT="outputs/final_4g/${RUN_TAG}"
export LOG_ROOT="logs/final_4g/${RUN_TAG}"

set +e  # Phase 1 may have individual group failures, don't abort
bash scripts/run_sid_tiger_pipeline.sh
PHASE1_EXIT=$?
set -e

if [[ $PHASE1_EXIT -ne 0 ]]; then
    echo "ERROR: Phase 1 有失败组，检查日志。继续执行 Phase 2..."
fi

# ===========================================================================
# Phase 2: G0 纯 RQ-VAE (GPU 0, 参数对齐)
# ===========================================================================
echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║  Phase 2: G0 纯 RQ-VAE (GPU 0, 参数对齐)               ║"
echo "╚══════════════════════════════════════════════════════════╝"

G0_OUT_ROOT="outputs/final_4g/${RUN_TAG}"
G0_LOG_ROOT="logs/final_4g/${RUN_TAG}"
mkdir -p "${G0_OUT_ROOT}/G0/01_sid_train" "${G0_OUT_ROOT}/G0/02_sid_inference" \
         "${G0_OUT_ROOT}/G0/03_tiger_train"

# ── 01 G0 SID 训练 ───────────────────────────────────────────────────
echo "── [G0] 01 SID训练 (GPU 0, ${SID_MAX_STEPS} steps, batch=256, Adam) ──"

CUDA_VISIBLE_DEVICES=0 python -m src.train \
    experiment=rqvae_vcf_online_knn_train_flat \
    data_dir="${DATA_DIR}" \
    embedding_path="${EMB_PATH}" \
    embedding_dim="${EMB_DIM}" \
    num_hierarchies="${SID_NUM_HIER}" \
    codebook_width="${CODEBOOK}" \
    "data_loading.datamodule.train_dataloader_config.batch_size_per_device=256" \
    data_loading.datamodule.train_dataloader_config.num_workers=12 \
    data_loading.datamodule.train_dataloader_config.pin_memory=true \
    data_loading.datamodule.train_dataloader_config.persistent_workers=true \
    seed=42 \
    trainer.max_epochs=null \
    "trainer.max_steps=${SID_MAX_STEPS}" \
    "callbacks.model_checkpoint.every_n_train_steps=${SID_MAX_STEPS}" \
    "optim.scheduler.warmup_steps=0" \
    "optim.optimizer._target_=torch.optim.Adam" \
    "optim.optimizer.lr=0.0003" \
    "optim.optimizer.weight_decay=0.00001" \
    "model.use_vcf=false" \
    "model.use_cvpm=false" \
    "model.use_tcl=false" \
    "model.use_npr=false" \
    "model.use_kpl=false" \
    "model.use_dual_tower=false" \
    "model.use_online_knn=false" \
    "model.repulsion_warmup_steps=0" \
    ++should_skip_retry=true \
    "hydra.run.dir=${G0_OUT_ROOT}/G0/01_sid_train" \
    > "${G0_LOG_ROOT}/G0_01_sid_train.log" 2>&1

G0_CKPT=$(ls -t "${G0_OUT_ROOT}/G0/01_sid_train/checkpoints/"*.ckpt 2>/dev/null | head -n 1)
if [[ -z "$G0_CKPT" || ! -f "$G0_CKPT" ]]; then
    echo "ERROR: [G0] SID checkpoint 未生成" >&2
    exit 1
fi
echo "[G0] SID ckpt: ${G0_CKPT}"

# ── 02 G0 SID 推理 ───────────────────────────────────────────────────
echo "── [G0] 02 SID推理 (GPU 0) ──"

CUDA_VISIBLE_DEVICES=0 python -m src.inference \
    experiment=rqvae_vcf_inference_flat \
    data_dir="${DATA_DIR}" \
    embedding_path="${EMB_PATH}" \
    embedding_dim="${EMB_DIM}" \
    num_hierarchies="${SID_NUM_HIER}" \
    codebook_width="${CODEBOOK}" \
    "ckpt_path=${G0_CKPT}" \
    seed=42 \
    "hydra.run.dir=${G0_OUT_ROOT}/G0/02_sid_inference" \
    > "${G0_LOG_ROOT}/G0_02_sid_inference.log" 2>&1

G0_SID_PT="${G0_OUT_ROOT}/G0/02_sid_inference/pickle/merged_predictions_tensor.pt"
if [[ ! -f "$G0_SID_PT" ]]; then
    echo "ERROR: [G0] SID tensor 未生成: ${G0_SID_PT}" >&2
    exit 1
fi
echo "[G0] SID OK: ${G0_SID_PT}"

# ── G0 碰撞分析 ──────────────────────────────────────────────────────
echo "── [G0] 碰撞分析 ──"
python scripts/analyze_sid_collisions.py --path "${G0_SID_PT}" \
    > "${G0_LOG_ROOT}/G0_collision.txt" 2>&1
cat "${G0_LOG_ROOT}/G0_collision.txt"

# ── 03 G0 TIGER 训练 ─────────────────────────────────────────────────
echo "── [G0] 03 TIGER训练 (GPU 0, ${TIGER_MAX_STEPS} steps) ──"

CUDA_VISIBLE_DEVICES=0 python -m src.train \
    experiment=tiger_train_flat \
    data_dir="${DATA_DIR}" \
    "semantic_id_path=${G0_SID_PT}" \
    seed=42 \
    num_hierarchies=4 \
    sequence_length=120 \
    "data_loading.train_dataloader_config.dataloader.batch_size_per_device=32" \
    "trainer.max_steps=${TIGER_MAX_STEPS}" \
    "trainer.val_check_interval=${TIGER_VAL_INTERVAL}" \
    "trainer.log_every_n_steps=${TIGER_LOG_INTERVAL}" \
    "trainer.accumulate_grad_batches=${TIGER_GRAD_ACCUM}" \
    "optim.optimizer.lr=${TIGER_LR}" \
    "optim.optimizer.weight_decay=${TIGER_WEIGHT_DECAY}" \
    "hydra.run.dir=${G0_OUT_ROOT}/G0/03_tiger_train" \
    > "${G0_LOG_ROOT}/G0_03_tiger_train.log" 2>&1

G0_TIGER_CKPT=$(ls -t "${G0_OUT_ROOT}/G0/03_tiger_train/checkpoints/"*.ckpt 2>/dev/null | head -n 1)
if [[ -z "$G0_TIGER_CKPT" || ! -f "$G0_TIGER_CKPT" ]]; then
    echo "[G0] TIGER warning: no checkpoint (val_check_interval > steps?)" >&2
else
    echo "[G0] TIGER ckpt: ${G0_TIGER_CKPT}"
fi

echo ""
echo "══════════════ 全部完成 ══════════════"
echo "Outputs: ${G0_OUT_ROOT}/"
echo "Logs:    ${G0_LOG_ROOT}/"
