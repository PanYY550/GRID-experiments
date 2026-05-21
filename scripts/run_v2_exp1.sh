#!/usr/bin/env bash
# ===========================================================================
# run_v2_exp1.sh — 实验1: V2_dfl vs BL_R0 vs BL_R1 全流程对比
#
# 3 组并行 (GPU 0,1,2)，每组: SID训练 → SID推理 → TIGER训练
#
# Group 1 (GPU0): V2_dfl — Layerwise H≤1, m=0.67 uniform, α=0.01
# Group 2 (GPU1): BL_R0 — Original VCF, H=0 only
# Group 3 (GPU2): BL_R1 — Original VCF, H≤1
#
# 用法:
#   bash scripts/run_v2_exp1.sh
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

RUN_TAG="${RUN_TAG:-v2_exp1_$(date +%Y%m%d_%H%M%S)}"

# ===========================================================================
# 共用参数 (所有 3 组)
# ===========================================================================

# 数据
export DATA_DIR=/home/pyy/GRID/src/data/amazon_data/beauty
export EMB_PATH=/home/pyy/GRID/embeddings/beauty/pickle/merged_predictions_tensor.pt
export EMB_DIM=768
export EXPOSURE_COUNTS=/home/pyy/GRID/src/data/amazon_data/beauty/exposure_counts.pt

# SID
export SID_NUM_HIER=3
export CODEBOOK=256
export SEED=42
export SID_MAX_STEPS="${SID_MAX_STEPS:-3000}"
export SID_BATCH_SIZE=256
export SID_NUM_WORKERS=12

# Optimizer (Adam, aligned with sweep)
export SID_OPTIMIZER=torch.optim.Adam
export SID_LR=0.0003
export SID_WEIGHT_DECAY=0.00001
export SCHEDULER_WARMUP=0

# VCF 底座 (for BL_R0 / BL_R1; ignored when use_layerwise_repulsion=true)
export USE_VCF=true
export USE_CVPM=true
export HAMMING_R=1
export LAMBDA_FULL=0.2
export LAMBDA_PARTIAL=0.1
export M_FULL=0.8
export M_PARTIAL=0.5
export M_FULL_START=0.8
export M_PARTIAL_START=0.5
export MARGIN_RAMP=0
export REPULSION_WARMUP=0
export CVPM_TEMP=0.15
export ALPHA=0.5

# 增强开关
export USE_TCL=true
export USE_DUAL_TOWER=true
export USE_KPL=false
export USE_RNCL=false
export USE_TIME_DECAY=false
export USE_DYNAMIC_MARGIN=false
export USE_ASYMMETRIC=false

# TCL
export LAMBDA_CL=0.1
export QUASID_CL_TAU=0.5
export CL_RAMP_STEPS=0

# NPR + Online k-NN
export USE_ONLINE_KNN=true
export USE_NPR=true
export NPR_ALPHA_MIN=0.01
export ONLINE_KNN_K=50
export ONLINE_KNN_EMA=0.99
export ONLINE_KNN_INTERVAL=100

# TIGER
export TIGER_NUM_HIER=4
export TIGER_MAX_STEPS="${TIGER_MAX_STEPS:-320000}"
export TIGER_BATCH_SIZE=32
export TIGER_LR=0.001
export TIGER_WEIGHT_DECAY=0.0001
export TIGER_VAL_INTERVAL=1600
export TIGER_GRAD_ACCUM=16
export TIGER_LOG_INTERVAL=100
export SEQUENCE_LENGTH=120

# Pipeline
export N_GROUPS=3
export GPU_LIST="0 1 2"

# ===========================================================================
# Group 1: V2_dfl — Layerwise H≤1, uniform m=0.67, α=0.01
# ===========================================================================
export G1_NAME="V2_dfl"
export G1_DESC="V2_dfl: Layerwise H≤1 m=0.67 α=0.01"
export G1_SPECS="model.use_layerwise_repulsion=true \
model.layerwise_hamming_radius=1 \
model.m_rep_L0=0.67 \
model.m_rep_L1=0.67 \
model.m_rep_L2=0.67 \
model.lambda_rep_L0=0.20 \
model.lambda_rep_L1=0.10 \
model.lambda_rep_L2=0.10 \
model.npr_alpha_L0=0.01 \
model.npr_alpha_L1=0.01 \
model.npr_alpha_L2=0.01 \
model.use_npr=true \
model.use_online_knn=true \
model.online_knn_k=50 \
model.use_tcl=true \
model.use_dual_tower=true"

# ===========================================================================
# Group 2: BL_R0 — Original VCF, H=0 only
# ===========================================================================
export G2_NAME="BL_R0"
export G2_DESC="BL_R0: VCF H=0 only"
export G2_SPECS="model.use_layerwise_repulsion=false \
model.use_npr=true \
model.npr_alpha_min=0.01 \
model.use_online_knn=true \
model.online_knn_k=50 \
model.use_tcl=true \
model.use_dual_tower=true \
model.hamming_radius=0"

# ===========================================================================
# Group 3: BL_R1 — Original VCF, H≤1
# ===========================================================================
export G3_NAME="BL_R1"
export G3_DESC="BL_R1: VCF H≤1"
export G3_SPECS="model.use_layerwise_repulsion=false \
model.use_npr=true \
model.npr_alpha_min=0.01 \
model.use_online_knn=true \
model.online_knn_k=50 \
model.use_tcl=true \
model.use_dual_tower=true \
model.hamming_radius=1"

export RUN_TAG="${RUN_TAG}"
export OUT_ROOT="outputs/layerwise_full/${RUN_TAG}"
export LOG_ROOT="logs/layerwise_full/${RUN_TAG}"

set +e
bash scripts/run_sid_tiger_pipeline.sh
EXIT_CODE=$?
set -e

echo ""
echo "══════════════ 全部完成 ══════════════"
echo "Outputs: ${OUT_ROOT}/"
echo "Logs:    ${LOG_ROOT}/"
exit $EXIT_CODE
