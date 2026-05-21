#!/usr/bin/env bash
# ===========================================================================
# run_g0_adagrad.sh — G0 纯 RQ-VAE (Adagrad) 全流程
#
# 对齐: 4g 实验 (batch=256, 1000步, warmup=0, seed=42)
# 差异: Adagrad 替代 Adam
# codebook_entropy/reset 继承 YAML 默认值 (与 4g 三组一致)
#
# 用法:
#   bash scripts/run_g0_adagrad.sh
#   SID_MAX_STEPS=200 bash scripts/run_g0_adagrad.sh   # smoke test
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

RUN_TAG="${RUN_TAG:-adagrad_$(date +%Y%m%d_%H%M%S)}"
GPU="${GPU:-1}"

# ── 与 4g 对齐的共用参数 ──────────────────────────────────────────────────
DATA_DIR=/home/pyy/GRID/src/data/amazon_data/beauty
EMB_PATH=/home/pyy/GRID/embeddings/beauty/pickle/merged_predictions_tensor.pt
EMB_DIM=768
SID_NUM_HIER=3
CODEBOOK=256
SEED=42
ENTROPY_WEIGHT="${ENTROPY_WEIGHT:-0.1}"
RESET_INTERVAL="${RESET_INTERVAL:-10}"
SID_MAX_STEPS="${SID_MAX_STEPS:-1000}"
SID_BATCH_SIZE=256
SID_NUM_WORKERS=12

# TIGER params
TIGER_MAX_STEPS="${TIGER_MAX_STEPS:-320000}"
TIGER_VAL_INTERVAL=1600
TIGER_GRAD_ACCUM=16
TIGER_LR=0.001
TIGER_WEIGHT_DECAY=0.0001

OUT_ROOT="outputs/final_4g/${RUN_TAG}"
LOG_ROOT="logs/final_4g/${RUN_TAG}"
mkdir -p "${OUT_ROOT}/01_sid_train" "${OUT_ROOT}/02_sid_inference" \
         "${OUT_ROOT}/03_tiger_train" "$LOG_ROOT"

echo "═══════════════════════════════════════════════════════"
echo "  G0 纯 RQ-VAE (Adagrad) 全流程"
echo "  Steps:  ${SID_MAX_STEPS}  Batch: ${SID_BATCH_SIZE}"
echo "  GPU:    ${GPU}  Entropy: ${ENTROPY_WEIGHT}  ResetInt: ${RESET_INTERVAL}"
echo "  Output: ${OUT_ROOT}"
echo "  Log:    ${LOG_ROOT}"
echo "═══════════════════════════════════════════════════════"

# ── 01 SID 训练 ──────────────────────────────────────────────────────────
echo "── [G0-Adagrad] 01 SID训练 (GPU ${GPU}, ${SID_MAX_STEPS} steps) ──"

CUDA_VISIBLE_DEVICES="${GPU}" python -m src.train \
    experiment=rqvae_vcf_online_knn_train_flat \
    data_dir="${DATA_DIR}" \
    embedding_path="${EMB_PATH}" \
    embedding_dim="${EMB_DIM}" \
    num_hierarchies="${SID_NUM_HIER}" \
    codebook_width="${CODEBOOK}" \
    "data_loading.datamodule.train_dataloader_config.batch_size_per_device=${SID_BATCH_SIZE}" \
    data_loading.datamodule.train_dataloader_config.num_workers="${SID_NUM_WORKERS}" \
    data_loading.datamodule.train_dataloader_config.pin_memory=true \
    data_loading.datamodule.train_dataloader_config.persistent_workers=true \
    seed="${SEED}" \
    trainer.max_epochs=null \
    "trainer.max_steps=${SID_MAX_STEPS}" \
    "callbacks.model_checkpoint.every_n_train_steps=${SID_MAX_STEPS}" \
    "optim.scheduler.warmup_steps=0" \
    "optim.optimizer._target_=torch.optim.Adagrad" \
    "optim.optimizer.lr=0.001" \
    "optim.optimizer.weight_decay=0.0" \
    "model.use_vcf=false" \
    "model.use_cvpm=false" \
    "model.use_tcl=false" \
    "model.use_npr=false" \
    "model.use_kpl=false" \
    "model.use_dual_tower=false" \
    "model.use_online_knn=false" \
    "model.repulsion_warmup_steps=0" \
    "++model.codebook_entropy_weight=${ENTROPY_WEIGHT}" \
    "model.codebook_reset_interval=${RESET_INTERVAL}" \
    ++should_skip_retry=true \
    "hydra.run.dir=${OUT_ROOT}/01_sid_train" \
    > "${LOG_ROOT}/01_sid_train.log" 2>&1

CKPT=$(ls -t "${OUT_ROOT}/01_sid_train/checkpoints/"*.ckpt 2>/dev/null | head -n 1)
if [[ -z "$CKPT" || ! -f "$CKPT" ]]; then
    echo "ERROR: [G0] SID checkpoint 未生成" >&2
    exit 1
fi
echo "[G0] SID ckpt: ${CKPT}"

# ── 02 SID 推理 ──────────────────────────────────────────────────────────
echo "── [G0-Adagrad] 02 SID推理 (GPU ${GPU}) ──"

CUDA_VISIBLE_DEVICES="${GPU}" python -m src.inference \
    experiment=rqvae_vcf_inference_flat \
    data_dir="${DATA_DIR}" \
    embedding_path="${EMB_PATH}" \
    embedding_dim="${EMB_DIM}" \
    num_hierarchies="${SID_NUM_HIER}" \
    codebook_width="${CODEBOOK}" \
    "ckpt_path=${CKPT}" \
    seed="${SEED}" \
    "hydra.run.dir=${OUT_ROOT}/02_sid_inference" \
    > "${LOG_ROOT}/02_sid_inference.log" 2>&1

SID_PT="${OUT_ROOT}/02_sid_inference/pickle/merged_predictions_tensor.pt"
if [[ ! -f "$SID_PT" ]]; then
    echo "ERROR: [G0] SID tensor 未生成: ${SID_PT}" >&2
    exit 1
fi
echo "[G0] SID OK: ${SID_PT}"

# ── 碰撞分析 ──────────────────────────────────────────────────────────────
echo "── [G0-Adagrad] 碰撞分析 ──"
python scripts/analyze_sid_collisions.py --path "${SID_PT}" \
    > "${LOG_ROOT}/collision.txt" 2>&1
cat "${LOG_ROOT}/collision.txt"

# ── 03 TIGER 训练 ────────────────────────────────────────────────────────
echo "── [G0-Adagrad] 03 TIGER训练 (GPU ${GPU}, ${TIGER_MAX_STEPS} steps) ──"

CUDA_VISIBLE_DEVICES="${GPU}" python -m src.train \
    experiment=tiger_train_flat \
    data_dir="${DATA_DIR}" \
    "semantic_id_path=${SID_PT}" \
    seed="${SEED}" \
    num_hierarchies=4 \
    sequence_length=120 \
    "data_loading.train_dataloader_config.dataloader.batch_size_per_device=32" \
    "trainer.max_steps=${TIGER_MAX_STEPS}" \
    "trainer.val_check_interval=${TIGER_VAL_INTERVAL}" \
    "trainer.log_every_n_steps=100" \
    "trainer.accumulate_grad_batches=${TIGER_GRAD_ACCUM}" \
    "optim.optimizer.lr=${TIGER_LR}" \
    "optim.optimizer.weight_decay=${TIGER_WEIGHT_DECAY}" \
    "hydra.run.dir=${OUT_ROOT}/03_tiger_train" \
    > "${LOG_ROOT}/03_tiger_train.log" 2>&1

TIGER_CKPT=$(ls -t "${OUT_ROOT}/03_tiger_train/checkpoints/"*.ckpt 2>/dev/null | head -n 1)
set +e
if [[ -z "$TIGER_CKPT" ]]; then
    echo "[G0] TIGER: no checkpoint, checking log for NDCG..."
    grep -E "test/ndcg@10" "${LOG_ROOT}/03_tiger_train.log" | tail -1 || echo "WARNING: no NDCG found"
else
    echo "[G0] TIGER ckpt: ${TIGER_CKPT}"
    grep -E "test/ndcg@10" "${LOG_ROOT}/03_tiger_train.log" | tail -1
fi
set -e

echo ""
echo "══════════════ G0 (Adagrad) 完成 ══════════════"
echo "Output: ${OUT_ROOT}"
echo "Log:    ${LOG_ROOT}"
