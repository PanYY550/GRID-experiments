#!/usr/bin/env bash
# ===========================================================================
# run_rncl_full_experiment.sh — NPR+TCL+RNCL (λ=5e-6) 完整 1000 步实验
#
# 流程: SID训练 → SID推理 → 碰撞分析 → TIGER训练 → TIGER评估
# 对比基线: NPR+TCL (test/ndcg@10=0.0355)
#
# 用法:
#   bash scripts/run_rncl_full_experiment.sh   # GPU 2
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

GPU="${GPU:-2}"
RUN_TAG="${RUN_TAG:-rncl_full_$(date +%Y%m%d_%H%M%S)}"

# ── 数据路径 ──
export DATA_DIR=/home/pyy/GRID/src/data/amazon_data/beauty
export EMB_PATH=/home/pyy/GRID/embeddings/beauty/pickle/merged_predictions_tensor.pt
export EMB_DIM=768
export EXPOSURE_COUNTS=/home/pyy/GRID/src/data/amazon_data/beauty/exposure_counts.pt

# ── SID params (对齐 4g) ──
export SID_NUM_HIER=3
export CODEBOOK=256
export SEED=42
export SID_MAX_STEPS="${SID_MAX_STEPS:-1000}"
export SID_BATCH_SIZE="${SID_BATCH_SIZE:-256}"
export SID_NUM_WORKERS="${SID_NUM_WORKERS:-12}"
export SCHEDULER_WARMUP="${SCHEDULER_WARMUP:-0}"
export SID_OPTIMIZER="${SID_OPTIMIZER:-torch.optim.Adam}"
export SID_LR="${SID_LR:-0.0003}"
export SID_WEIGHT_DECAY="${SID_WEIGHT_DECAY:-0.00001}"

# ── VCF 通用底座 (对齐 4g) ──
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

# ── 增强开关 (对齐 4g) ──
export USE_TIME_DECAY=false
export USE_DYNAMIC_MARGIN=false
export USE_ASYMMETRIC=false
export USE_DUAL_TOWER=true
export USE_TCL=true
export USE_KPL=false

# ── TCL params (对齐 4g) ──
export LAMBDA_CL=0.1
export QUASID_CL_TAU=0.5
export CL_RAMP_STEPS=0

# ── Online k-NN + NPR (对齐 4g) ──
export USE_ONLINE_KNN=true
export USE_NPR=true
export NPR_ALPHA_MIN=0.01
export ONLINE_KNN_K=50
export ONLINE_KNN_EMA=0.99
export ONLINE_KNN_INTERVAL=100

# ── TIGER params ──
export TIGER_NUM_HIER=4
export TIGER_MAX_STEPS="${TIGER_MAX_STEPS:-320000}"
export TIGER_BATCH_SIZE="${TIGER_BATCH_SIZE:-32}"
export TIGER_LR="${TIGER_LR:-0.001}"
export TIGER_WEIGHT_DECAY="${TIGER_WEIGHT_DECAY:-0.0001}"
export TIGER_VAL_INTERVAL="${TIGER_VAL_INTERVAL:-1600}"
export TIGER_GRAD_ACCUM="${TIGER_GRAD_ACCUM:-16}"
export TIGER_LOG_INTERVAL="${TIGER_LOG_INTERVAL:-100}"
export SEQUENCE_LENGTH=120

# ── Single group: NPR+TCL+RNCL ──
export N_GROUPS=1
export GPU_LIST="$GPU"
export G1_NAME="NPR_TCL_RNCL"
export G1_SPECS="model.use_tcl=true model.use_npr=true model.npr_alpha_min=0.01 model.use_online_knn=true model.online_knn_k=50 model.use_kpl=false model.use_dual_tower=true model.use_rncl=true model.lambda_rncl=0.000005 model.rncl_ramp_steps=500"
export G1_DESC="NPR+TCL+RNCL: 4g+NPR(α=0.01)+TCL(λ=0.1)+RNCL(λ=5e-6)"

export RUN_TAG="${RUN_TAG}"
export OUT_ROOT="outputs/rncl_full/${RUN_TAG}"
export LOG_ROOT="logs/rncl_full/${RUN_TAG}"

echo "╔══════════════════════════════════════════════════════════╗"
echo "║  RNCL Full Experiment: NPR+TCL+RNCL (λ=5e-6)           ║"
echo "╠══════════════════════════════════════════════════════════╣"
printf "║  RUN_TAG: %-46s ║\n" "$RUN_TAG"
echo "║  SID: 1000 steps, batch=256, Adam lr=3e-4              ║"
echo "║  Baseline: NPR+TCL test/ndcg@10=0.0355                 ║"
echo "╚══════════════════════════════════════════════════════════╝"

bash scripts/run_sid_tiger_pipeline.sh
