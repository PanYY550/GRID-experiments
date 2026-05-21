#!/usr/bin/env bash
# ===========================================================================
# test_g0_stability.sh — G0 码本稳定性测试
#
# 统一使用 rqvae_vcf_online_knn_train_flat, 所有增强开关关闭。
# 测试步数: 100, 200, 400, 1000 (可配置)
# 每步: SID训练 → SID推理 → 碰撞分析
#
# 用法:
#   bash scripts/test_g0_stability.sh
#   STEPS_LIST="100 200 400 600 1000" bash scripts/test_g0_stability.sh
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

STEPS_LIST=(${STEPS_LIST:-100 200 400 1000})
GPU="${GPU:-0}"
RUN_TAG="${RUN_TAG:-g0_stab_$(date +%Y%m%d_%H%M%S)}"

# ── 可配置优化参数 (env vars 覆盖) ─────────────────────────────────
OPTIMIZER="${OPTIMIZER:-torch.optim.Adam}"
LR="${LR:-0.0003}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.00001}"
WARMUP="${WARMUP:-0}"
ENTROPY_WEIGHT="${ENTROPY_WEIGHT:-0.1}"
RESET_INTERVAL="${RESET_INTERVAL:-10}"
RESET_DECAY="${RESET_DECAY:-0.9}"
GRAD_CLIP="${GRAD_CLIP:-0.0}"
OUT_ROOT="outputs/g0_stability/${RUN_TAG}"
LOG_ROOT="logs/g0_stability/${RUN_TAG}"
mkdir -p "$OUT_ROOT" "$LOG_ROOT"

# ── 共用参数 (与 TCL_only 完全对齐) ─────────────────────────────────────
DATA_DIR=/home/pyy/GRID/src/data/amazon_data/beauty
EMB_PATH=/home/pyy/GRID/embeddings/beauty/pickle/merged_predictions_tensor.pt
EMB_DIM=768
SID_NUM_HIER=3
CODEBOOK=256
SEED=42

echo "═══════════════════════════════════════════════════════"
echo "  G0 码本稳定性测试"
echo "  Config: rqvae_vcf_online_knn_train_flat (all off)"
echo "  Steps:  ${STEPS_LIST[*]}"
echo "  GPU:    ${GPU}  Opt: ${OPTIMIZER}  LR: ${LR}  WD: ${WEIGHT_DECAY}"
echo "  Warmup: ${WARMUP}  Entropy: ${ENTROPY_WEIGHT}  ResetInt: ${RESET_INTERVAL}  ResetDecay: ${RESET_DECAY}"
echo "  Output: ${OUT_ROOT}"
echo "═══════════════════════════════════════════════════════"

SUMMARY="${LOG_ROOT}/summary.txt"
echo "steps frac_unique l0_unique l1_unique l2_unique full_collision l0_cov l1_cov l2_cov" > "$SUMMARY"

for steps in "${STEPS_LIST[@]}"; do
    echo ""
    echo "── [G0] ${steps} steps ──"

    train_dir="${OUT_ROOT}/${steps}/01_sid_train"
    infer_dir="${OUT_ROOT}/${steps}/02_sid_inference"
    mkdir -p "$train_dir" "$infer_dir"

    # ── SID 训练 ─────────────────────────────────────────────────────
    CUDA_VISIBLE_DEVICES="${GPU}" python -m src.train \
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
        seed="${SEED}" \
        trainer.max_epochs=null \
        "trainer.max_steps=${steps}" \
        "callbacks.model_checkpoint.every_n_train_steps=${steps}" \
        "optim.scheduler.warmup_steps=${WARMUP}" \
        "optim.optimizer._target_=${OPTIMIZER}" \
        "optim.optimizer.lr=${LR}" \
        "optim.optimizer.weight_decay=${WEIGHT_DECAY}" \
        "++model.codebook_entropy_weight=${ENTROPY_WEIGHT}" \
        "model.codebook_reset_interval=${RESET_INTERVAL}" \
        "model.codebook_reset_decay=${RESET_DECAY}" \
        "model.use_vcf=false" \
        "model.use_cvpm=false" \
        "model.use_tcl=false" \
        "model.use_npr=false" \
        "model.use_kpl=false" \
        "model.use_dual_tower=false" \
        "model.use_online_knn=false" \
        "model.repulsion_warmup_steps=0" \
        ++should_skip_retry=true \
        "hydra.run.dir=${train_dir}" \
        > "${LOG_ROOT}/g0_${steps}_train.log" 2>&1

    ckpt=$(ls -t "${train_dir}"/checkpoints/*.ckpt 2>/dev/null | head -n 1)
    if [[ -z "$ckpt" ]]; then
        echo "  ERROR: no checkpoint" | tee -a "$SUMMARY"
        continue
    fi

    # ── SID 推理 ─────────────────────────────────────────────────────
    CUDA_VISIBLE_DEVICES="${GPU}" python -m src.inference \
        experiment=rqvae_vcf_inference_flat \
        data_dir="${DATA_DIR}" \
        embedding_path="${EMB_PATH}" \
        embedding_dim="${EMB_DIM}" \
        num_hierarchies="${SID_NUM_HIER}" \
        codebook_width="${CODEBOOK}" \
        "ckpt_path=${ckpt}" \
        seed="${SEED}" \
        "hydra.run.dir=${infer_dir}" \
        > "${LOG_ROOT}/g0_${steps}_infer.log" 2>&1

    sid_pt="${infer_dir}/pickle/merged_predictions_tensor.pt"
    if [[ ! -f "$sid_pt" ]]; then
        echo "  ERROR: no SID tensor" | tee -a "$SUMMARY"
        continue
    fi

    # ── 碰撞分析 & 提取指标 ──────────────────────────────────────────
    python scripts/analyze_sid_collisions.py --path "${sid_pt}" \
        > "${LOG_ROOT}/g0_${steps}_collision.txt" 2>&1

    frac=$(grep 'frac_unique' "${LOG_ROOT}/g0_${steps}_collision.txt" | awk '{print $2}')
    l0=$(grep 'layer0:' "${LOG_ROOT}/g0_${steps}_collision.txt" | sed 's/.*unique_tokens=\([0-9]*\).*/\1/')
    l1=$(grep 'layer1:' "${LOG_ROOT}/g0_${steps}_collision.txt" | sed 's/.*unique_tokens=\([0-9]*\).*/\1/')
    l2=$(grep 'layer2:' "${LOG_ROOT}/g0_${steps}_collision.txt" | sed 's/.*unique_tokens=\([0-9]*\).*/\1/')
    fc=$(grep 'full_collision' "${LOG_ROOT}/g0_${steps}_collision.txt" | head -1 | awk '{print $2}')
    c0=$(grep 'layer0:' "${LOG_ROOT}/g0_${steps}_collision.txt" | sed 's/.*coverage=\([0-9.]*\).*/\1/')
    c1=$(grep 'layer1:' "${LOG_ROOT}/g0_${steps}_collision.txt" | sed 's/.*coverage=\([0-9.]*\).*/\1/')
    c2=$(grep 'layer2:' "${LOG_ROOT}/g0_${steps}_collision.txt" | sed 's/.*coverage=\([0-9.]*\).*/\1/')

    echo "${steps} ${frac} ${l0} ${l1} ${l2} ${fc} ${c0} ${c1} ${c2}" >> "$SUMMARY"
    echo "  frac_unique=${frac}  l0=${l0}  l1=${l1}  l2=${l2}  full_coll=${fc}  cov=${c0}/${c1}/${c2}"
done

echo ""
echo "═══════════════════════════════════════════════════════"
echo "  汇总"
echo "═══════════════════════════════════════════════════════"
column -t "$SUMMARY"
echo ""
echo "Logs: ${LOG_ROOT}/"
