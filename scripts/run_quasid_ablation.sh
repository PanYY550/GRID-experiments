#!/usr/bin/env bash
# ===========================================================================
# run_quasid_ablation.sh — QuaSID复现 + QuaSID+NPR+dynM 两组并行
#
# 用法:
#   bash scripts/run_quasid_ablation.sh
#   SID_MAX_STEPS=100 TIGER_MAX_STEPS=100 bash scripts/run_quasid_ablation.sh  # smoke
#
# 流水线 (每组):
#   01 SID训练 → 02 SID推理 → 碰撞分析 → 03 TIGER训练
# ===========================================================================
#
# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                    ██  QuaSID 消融实验 (2 组)  ██                           ║
# ╠══════════════════════════════════════════════════════════════════════════════╣
# ║                                                                              ║
# ║  ┌──────────────────────────────────────────────────────────────────────┐   ║
# ║  │ Exp1 (GPU 1) — QuaSID 复现 (TCL_only, QuaSID 原版参数)                │   ║
# ║  │                                                                       │   ║
# ║  │   batch=4096  warmup=1000  steps=3000  seed=42                        │   ║
# ║  │   λ_full=0.2  λ_partial=0.1     ★ QuaSID λ                           │   ║
# ║  │   m_full=0.8  m_partial=0.5     ★ QuaSID margin (静态, 无 ramp)       │   ║
# ║  │   m_start=0/0  margin_ramp=0    ★ 无渐变，直接使用最终 margin           │   ║
# ║  │   R=1                            ★ QuaSID offline 默认                 │   ║
# ║  │   use_vcf=true  use_cvpm=true  cvpm_temp=0.15  alpha=0.5              │   ║
# ║  │   use_tcl=true  λ_cl=0.1  τ=0.5  cl_ramp=0  dual_tower=true           │   ║
# ║  │   use_npr=false  use_online_knn=false                                 │   ║
# ║  │   use_dynamic_margin=false                                            │   ║
# ║  │   time_decay/asymmetric/kpl: 全关                                      │   ║
# ║  └──────────────────────────────────────────────────────────────────────┘   ║
# ║                                                                              ║
# ║  ┌──────────────────────────────────────────────────────────────────────┐   ║
# ║  │ Exp2 (GPU 2) — QuaSID+NPR+dynM (对标 NPR+TCL 3g)                      │   ║
# ║  │                                                                       │   ║
# ║  │   底座 = NPR+TCL (3g):                                                 │   ║
# ║  │     batch=4096  warmup=1000  λ=1.0/0.5  m=0.3/0.15                     │   ║
# ║  │     R=2  cvpm=true  TCL(λ=0.001,τ=0.5,ramp=1000)  dual_tower=true     │   ║
# ║  │     NPR(α=0.01)  online_knn(k=50,ema=0.99)                            │   ║
# ║  │                                                                       │   ║
# ║  │   ★ 差异:                                                              │   ║
# ║  │     use_dynamic_margin=true  (3g: false)                               │   ║
# ║  │     m0=0.8  margin_dissim_weight=0.3  ★ QuaSID m0                      │   ║
# ║  │                                                                       │   ║
# ║  │   对照基线: NPR+TCL (3g)  NDCG=0.0335  L0=100%  pcol=2.43%             │   ║
# ║  └──────────────────────────────────────────────────────────────────────┘   ║
# ║                                                                              ║
# ║  TIGER (两组相同): num_hier=4  steps=320K  batch=32  accum=16               ║
# ║                                                                              ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                        ██  控制台 (CONSOLE)  ██                            ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

# ── 1. GPU 分配 ──────────────────────────────────────────────────────────────
GPU1="${GPU1:-1}"
GPU2="${GPU2:-2}"

# ── 2. 数据路径 (共享) ───────────────────────────────────────────────────────
DATA_DIR="${DATA_DIR:-/home/pyy/GRID/src/data/amazon_data/beauty}"
EMB_PATH="${EMB_PATH:-/home/pyy/GRID/embeddings/beauty/pickle/merged_predictions_tensor.pt}"
EMB_DIM="${EMB_DIM:-768}"
EXPOSURE_COUNTS="${EXPOSURE_COUNTS:-/home/pyy/GRID/src/data/amazon_data/beauty/exposure_counts.pt}"

# ── 3. SID 通用参数 ─────────────────────────────────────────────────────────
SID_NUM_HIER="${SID_NUM_HIER:-3}"
CODEBOOK="${CODEBOOK:-256}"
SEED="${SEED:-42}"
SID_MAX_STEPS="${SID_MAX_STEPS:-3000}"
NUM_WORKERS="${NUM_WORKERS:-12}"
SCHEDULER_WARMUP="${SCHEDULER_WARMUP:-1000}"

# ── 4. TIGER 训练参数 (两组相同) ─────────────────────────────────────────────
TIGER_NUM_HIER="${TIGER_NUM_HIER:-4}"
TIGER_MAX_STEPS="${TIGER_MAX_STEPS:-320000}"
TIGER_BATCH_SIZE="${TIGER_BATCH_SIZE:-32}"
TIGER_LR="${TIGER_LR:-0.001}"
TIGER_WEIGHT_DECAY="${TIGER_WEIGHT_DECAY:-0.0001}"
TIGER_VAL_INTERVAL="${TIGER_VAL_INTERVAL:-1600}"
TIGER_GRAD_ACCUM="${TIGER_GRAD_ACCUM:-16}"
TIGER_LOG_INTERVAL="${TIGER_LOG_INTERVAL:-100}"
SEQUENCE_LENGTH="${SEQUENCE_LENGTH:-120}"

# ── 5. 实验命名 & 输出路径 ──────────────────────────────────────────────────
RUN_TAG="${RUN_TAG:-quasid_ablation_$(date +%Y%m%d_%H%M%S)}"
OUT_ROOT="${OUT_ROOT:-outputs/quasid_ablation/${RUN_TAG}}"
LOG_ROOT="${LOG_ROOT:-logs/quasid_ablation/${RUN_TAG}}"

# ── 6. 实验组定义 ────────────────────────────────────────────────────────────

# Exp1: QuaSID 复现 (TCL_only, QuaSID 原版参数)
E1_NAME="${E1_NAME:-QuaSID_TCL}"
E1_DESC="${E1_DESC:-QuaSID复现: λ=0.2/0.1 m=0.8/0.5 R=1 λ_cl=0.1 warmup=1000}"
E1_EXTRA=(
    # VCF — QuaSID 原版 λ/m/R, 静态 margin 无 ramp
    model.lambda_full=0.2
    model.lambda_partial=0.1
    model.m_full=0.8
    model.m_partial=0.5
    model.m_full_start=0.0
    model.m_partial_start=0.0
    model.margin_ramp_steps=0
    model.hamming_radius=1
    # TCL — QuaSID 原版 λ_cl=0.1, 无 ramp
    model.use_tcl=true
    model.lambda_cl=0.1
    model.quasid_cl_tau=0.5
    model.cl_ramp_steps=0
    model.use_dual_tower=true
    # 其他增强全关
    model.use_npr=false
    model.use_online_knn=false
    model.use_dynamic_margin=false
    model.use_time_decay=false
    model.use_asymmetric_repulsion=false
    model.use_kpl=false
)

# Exp2: QuaSID+NPR+dynM (对标 NPR+TCL 3g, 仅加 dynamic_margin + m0=0.8)
E2_NAME="${E2_NAME:-QuaSID_NPR_dynM}"
E2_DESC="${E2_DESC:-QuaSID+NPR+dynM: 3g底座+dynM(m0=0.8) warmup=1000}"
E2_EXTRA=(
    # 底座 = NPR+TCL 3g
    model.lambda_full=1.0
    model.lambda_partial=0.5
    model.m_full=0.3
    model.m_partial=0.15
    model.hamming_radius=2
    model.use_tcl=true
    model.lambda_cl=0.001
    model.quasid_cl_tau=0.5
    model.cl_ramp_steps=1000
    model.use_dual_tower=true
    model.use_npr=true
    model.npr_alpha_min=0.01
    model.use_online_knn=true
    model.online_knn_k=50
    model.online_knn_ema_momentum=0.99
    model.online_knn_update_interval=100
    # ★ dynamic_margin — 对齐 QuaSID m0=0.8
    model.use_dynamic_margin=true
    model.m0=0.8
    model.margin_dissim_weight=0.3
    # 其他增强全关
    model.use_time_decay=false
    model.use_asymmetric_repulsion=false
    model.use_kpl=false
)

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

# ── SID 训练公共底座 ─────────────────────────────────────────────────────────
SID_COMMON_ARGS=(
    experiment=rqvae_vcf_online_knn_train_flat
    data_dir="${DATA_DIR}"
    embedding_path="${EMB_PATH}"
    embedding_dim="${EMB_DIM}"
    num_hierarchies="${SID_NUM_HIER}"
    codebook_width="${CODEBOOK}"
    seed="${SEED}"
    trainer.max_epochs=null
    "trainer.max_steps=${SID_MAX_STEPS}"
    "data_loading.datamodule.train_dataloader_config.num_workers=${NUM_WORKERS}"
    data_loading.datamodule.train_dataloader_config.pin_memory=true
    data_loading.datamodule.train_dataloader_config.persistent_workers=true
    model.use_vcf=true
    model.use_cvpm=true
    model.cvpm_temperature=0.15
    model.alpha=0.5
    model.exposure_counts_path="${EXPOSURE_COUNTS}"
    model.repulsion_warmup_steps=1000
    "callbacks.model_checkpoint.every_n_train_steps=${SID_MAX_STEPS}"
    "optim.scheduler.warmup_steps=${SCHEDULER_WARMUP}"
    ++should_skip_retry=true
)

# ── TIGER 训练公共底座 ───────────────────────────────────────────────────────
TIGER_COMMON_ARGS=(
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
    ++should_skip_retry=true
)

# ── 辅助函数 ─────────────────────────────────────────────────────────────────
_latest_ckpt() { ls -t "${1}"/checkpoints/*.ckpt 2>/dev/null | head -n 1; }
_die() { echo "ERROR: $*" >&2; return 1; }

# ── 单组流水线 ──────────────────────────────────────────────────────────────
run_group() {
    local gpu="$1" name="$2" log_prefix="$3"
    shift 3
    local extra_args=("$@")

    local run_root="${OUT_ROOT}/${name}"
    mkdir -p "${run_root}/01_sid_train" "${run_root}/02_sid_inference" \
             "${run_root}/03_tiger_train"

    # ── 01 SID 训练 ──────────────────────────────────────────────────────────
    echo "── [${name}] 01 SID训练 (GPU ${gpu}, ${SID_MAX_STEPS} steps) ──" | tee -a "$RUN_ALL_LOG" >&2

    CUDA_VISIBLE_DEVICES="${gpu}" python -m src.train \
        "${SID_COMMON_ARGS[@]}" \
        "${extra_args[@]}" \
        "data_loading.datamodule.train_dataloader_config.batch_size_per_device=4096" \
        "hydra.run.dir=${run_root}/01_sid_train" \
        > "${LOG_ROOT}/${log_prefix}_01_sid_train.log" 2>&1

    local sid_ckpt; sid_ckpt="$(_latest_ckpt "${run_root}/01_sid_train")"
    if [[ -z "$sid_ckpt" || ! -f "$sid_ckpt" ]]; then
        _die "[${name}] SID checkpoint 未生成"; return 1
    fi
    echo "[${name}] SID ckpt: ${sid_ckpt}" | tee -a "$RUN_ALL_LOG" >&2

    # ── 02 SID 推理 ──────────────────────────────────────────────────────────
    echo "── [${name}] 02 SID推理 (GPU ${gpu}) ──" | tee -a "$RUN_ALL_LOG" >&2

    CUDA_VISIBLE_DEVICES="${gpu}" python -m src.inference \
        experiment=rqvae_vcf_inference_flat \
        data_dir="${DATA_DIR}" \
        embedding_path="${EMB_PATH}" \
        embedding_dim="${EMB_DIM}" \
        num_hierarchies="${SID_NUM_HIER}" \
        codebook_width="${CODEBOOK}" \
        "ckpt_path=${sid_ckpt}" \
        seed="${SEED}" \
        "hydra.run.dir=${run_root}/02_sid_inference" \
        > "${LOG_ROOT}/${log_prefix}_02_sid_inference.log" 2>&1

    local sid_pt="${run_root}/02_sid_inference/pickle/merged_predictions_tensor.pt"
    if [[ ! -f "$sid_pt" ]]; then
        _die "[${name}] SID tensor 未生成: ${sid_pt}"; return 1
    fi

    python -c "
import torch; t=torch.load('${sid_pt}',map_location='cpu',weights_only=True)
assert t.ndim==2 and t.shape[0]==$((SID_NUM_HIER + 1)), f'shape err: {t.shape}'
print(f'SID OK: {t.shape}')" >&2

    echo "[${name}] SID OK: ${sid_pt}" | tee -a "$RUN_ALL_LOG" >&2

    # ── 碰撞分析 ─────────────────────────────────────────────────────────────
    echo "── [${name}] 碰撞分析 ──" | tee -a "$RUN_ALL_LOG" >&2
    python scripts/analyze_sid_collisions.py --path "${sid_pt}" \
        | tee "${LOG_ROOT}/${log_prefix}_collision.txt"

    # ── 03 TIGER 训练 ────────────────────────────────────────────────────────
    echo "── [${name}] 03 TIGER训练 (GPU ${gpu}, ${TIGER_MAX_STEPS} steps) ──" | tee -a "$RUN_ALL_LOG" >&2

    CUDA_VISIBLE_DEVICES="${gpu}" python -m src.train \
        "${TIGER_COMMON_ARGS[@]}" \
        "semantic_id_path=${sid_pt}" \
        "hydra.run.dir=${run_root}/03_tiger_train" \
        > "${LOG_ROOT}/${log_prefix}_03_tiger_train.log" 2>&1

    local tiger_ckpt; tiger_ckpt="$(_latest_ckpt "${run_root}/03_tiger_train")"
    if [[ -z "$tiger_ckpt" || ! -f "$tiger_ckpt" ]]; then
        _die "[${name}] TIGER checkpoint 未生成"; return 1
    fi
    echo "[${name}] TIGER ckpt: ${tiger_ckpt}" | tee -a "$RUN_ALL_LOG" >&2

    return 0
}

# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

# ── 控制台摘要 ──────────────────────────────────────────────────────────────
{
    echo "╔══════════════════════════════════════════════════════════╗"
    echo "║     QuaSID 消融实验 — 2 组 SID+TIGER 全流程           ║"
    echo "╠══════════════════════════════════════════════════════════╣"
    printf "║  RUN_TAG: %-46s ║\n" "$RUN_TAG"
    echo "╠══════════════════════════════════════════════════════════╣"
    echo "║  [Exp1] GPU${GPU1} — ${E1_NAME}"
    echo "║         QuaSID复现: λ=0.2/0.1 m=0.8/0.5 R=1 λ_cl=0.1"
    echo "║         warmup=1000 static_margin tcl_only"
    echo "║  [Exp2] GPU${GPU2} — ${E2_NAME}"
    echo "║         3g底座+dynM(m0=0.8) warmup=1000"
    echo "║         对照: NPR+TCL(3g) NDCG=0.0335 L0=100%%"
    echo "╠══════════════════════════════════════════════════════════╣"
    printf "║  SID: batch=4096  steps=%-4s  TIGER: steps=%-6s    ║\n" \
        "$SID_MAX_STEPS" "$TIGER_MAX_STEPS"
    echo "╠══════════════════════════════════════════════════════════╣"
    printf "║  输出: %-49s ║\n" "$OUT_ROOT"
    printf "║  日志: %-49s ║\n" "$LOG_ROOT"
    echo "╚══════════════════════════════════════════════════════════╝"
} | tee "$RUN_ALL_LOG"

# ── 启动两组 (后台并行) ─────────────────────────────────────────────────────
echo "[main] 启动 ${E1_NAME} (GPU ${GPU1}) — ${E1_DESC}" | tee -a "$RUN_ALL_LOG"
(run_group "$GPU1" "$E1_NAME" "${E1_NAME}" "${E1_EXTRA[@]}") &
PID1=$!

echo "[main] 启动 ${E2_NAME} (GPU ${GPU2}) — ${E2_DESC}" | tee -a "$RUN_ALL_LOG"
(run_group "$GPU2" "$E2_NAME" "${E2_NAME}" "${E2_EXTRA[@]}") &
PID2=$!

# ── 等待完成 ─────────────────────────────────────────────────────────────────
ok1=0 ok2=0
wait "$PID1" && ok1=1 && echo "[main] ${E1_NAME} ✓ 完成" | tee -a "$RUN_ALL_LOG" \
    || echo "[main] ${E1_NAME} ✗ 失败" | tee -a "$RUN_ALL_LOG"
wait "$PID2" && ok2=1 && echo "[main] ${E2_NAME} ✓ 完成" | tee -a "$RUN_ALL_LOG" \
    || echo "[main] ${E2_NAME} ✗ 失败" | tee -a "$RUN_ALL_LOG"

# ── 汇总 ─────────────────────────────────────────────────────────────────────
echo "" | tee -a "$RUN_ALL_LOG"
echo "══════════════ 汇总 ══════════════" | tee -a "$RUN_ALL_LOG"
printf "  %-30s : %s\n" "$E1_NAME" "$([ "$ok1" -eq 1 ] && echo "OK" || echo "FAIL")" | tee -a "$RUN_ALL_LOG"
printf "  %-30s : %s\n" "$E2_NAME" "$([ "$ok2" -eq 1 ] && echo "OK" || echo "FAIL")" | tee -a "$RUN_ALL_LOG"
echo "" | tee -a "$RUN_ALL_LOG"
echo "Outputs: ${OUT_ROOT}/" | tee -a "$RUN_ALL_LOG"
echo "Logs:    ${LOG_ROOT}/" | tee -a "$RUN_ALL_LOG"
[[ "$ok1" -eq 1 && "$ok2" -eq 1 ]] && echo "All groups OK." | tee -a "$RUN_ALL_LOG" \
    || echo "Some groups FAILED — check logs." | tee -a "$RUN_ALL_LOG"

exit $(( 1 - (ok1 & ok2) ))
