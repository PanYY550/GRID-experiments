## ✅ G_QSS 实验配置确认 (QuaSID-Static Baseline) — v2 修复版

> **[v2 修复说明]** 原版 Codebook Collapse 根因分析及修复：
> - **根因**：`alpha=0.0` → 所有 pair 权重均为 1.0；`m_full=0.8` 初始时 hinge≈0.8，
>   `repulsion_loss` 在 step49 时已是 `reconstruction_loss` 的 **10倍**，导致 step99 时
>   Layer0 coverage 仅剩 **3%（codebook collapse）**。
> - **修复 1**：`lambda_full: 0.3`（原 1.0），`lambda_partial: 0.2`（原 0.5）
>   → 将排斥损失权重降为 reconstruction_loss 的可控范围。
> - **修复 2**：`repulsion_warmup_steps: 2000`（原 1000）→ 更长热身，避免训练初期崩溃。
> - **修复 3**：`m_full: 0.5`，`m_partial: 0.3`（原 0.8/0.5）→ 减小静态 margin，
>   与 cosine 距离空间匹配（D∈[0,2]，0.5 已足够区分）。
> - **修复 4**：新增 `exposure_counts_path`（VCF 路径现在也会加载，alpha=0.0 不影响结果，
>   但保持与 G1 配置一致便于对比）。

### 实验设置总览

| 配置项       | 值                     | 说明                                                           |
| ------------ | ---------------------- | -------------------------------------------------------------- |
| **实验组**   | G_QSS QuaSID-Static (v2) | 静态QuaSID基线：CVPM + HaMR(静态边界) + 无时间衰减 + 无TCCL |
| **GPU**      | CUDA_VISIBLE_DEVICES=1 | 单卡运行                                                       |
| **数据集**   | Amazon Beauty          | `/home/pyy/GRID/src/data/amazon_data/beauty`                  |
| **随机种子** | 42                     | 可复现                                                         |

---

### Step 1: SID Training (RQ-VAE + QuaSID-Static)

| 配置项                  | 旧值 (v1) | **新值 (v2)** | 修复原因                                             |
| ----------------------- | --------- | ------------- | ---------------------------------------------------- |
| experiment              | —         | `rqvae_vcf_train_flat` | 不变                                      |
| num_hierarchies         | 3         | 3             | 不变                                                 |
| codebook_width          | 256       | 256           | 不变                                                 |
| use_vcf                 | true      | ✅ true       | 不变                                                 |
| use_cvpm                | true      | ✅ true       | 不变                                                 |
| use_dynamic_margin      | false     | ❌ false      | 不变（保持静态边界语义）                              |
| **m_full**              | **0.8**   | **0.5**       | 原 0.8 对 D∈[0,2] 空间过于激进；0.5 已够区分         |
| **m_partial**           | **0.5**   | **0.3**       | 同上，适当降低                                        |
| alpha                   | 0.0       | 0.0           | 不变（无时间衰减，体现 QuaSID 静态特性）              |
| use_tcl                 | true      | ✅ true       | 不变（仅生成正样本矩阵供 CVPM 使用）                  |
| lambda_cl               | 0.0       | 0.0           | 不变（TCCL 关闭）                                    |
| **lambda_full**         | **1.0**   | **0.3**       | 原值使排斥 10× reconstruction；降低后可控             |
| **lambda_partial**      | **0.5**   | **0.2**       | 同上，比例降低                                        |
| **repulsion_warmup_steps** | **1000** | **2000**   | 更长热身防止 early collapse                           |
| hamming_radius          | 2         | 2             | 不变                                                 |
| cvpm_temperature        | 0.15      | 0.15          | 不变                                                 |
| exposure_counts_path    | _(缺失)_  | ✅ 新增       | VCF 路径现在也加载（alpha=0 时不影响权重，保持一致）   |

**运行指令（v2）**:

```bash
CUDA_VISIBLE_DEVICES=1 nohup python -m src.train \
  experiment=rqvae_vcf_train_flat \
  data_dir=/home/pyy/GRID/src/data/amazon_data/beauty \
  embedding_path=/home/pyy/GRID/embeddings/beauty/pickle/merged_predictions_tensor.pt \
  embedding_dim=768 \
  num_hierarchies=3 \
  codebook_width=256 \
  data_loading.datamodule.train_dataloader_config.batch_size_per_device=4096 \
  data_loading.datamodule.train_dataloader_config.num_workers=16 \
  trainer.max_steps=3000 \
  trainer.max_epochs=null \
  seed=42 \
  model.use_vcf=true \
  model.use_cvpm=true \
  model.cvpm_temperature=0.15 \
  model.hamming_radius=2 \
  model.use_dynamic_margin=false \
  model.m_full=0.5 \
  model.m_partial=0.3 \
  model.alpha=0.0 \
  model.lambda_full=0.3 \
  model.lambda_partial=0.2 \
  model.repulsion_warmup_steps=2000 \
  model.use_tcl=true \
  model.lambda_cl=0.0 \
  model.exposure_counts_path=/home/pyy/GRID/src/data/amazon_data/beauty/exposure_counts.pt \
  hydra.run.dir=outputs/G_QSS_v2_fixed/G_QSS_QuaSID_Static/01_sid_train \
  > outputs/G_QSS_v2_fixed/G_QSS_QuaSID_Static/01_sid_train/sid_train_g_qss_quasid_static.log 2>&1 &
```

---

### Step 2: SID Inference

| 配置项     | 值                         |
| ---------- | -------------------------- |
| experiment | `rqvae_vcf_inference_flat` |
| ckpt_path  | Step 1 生成的 checkpoint   |

**运行指令（v2）**:

```bash
CUDA_VISIBLE_DEVICES=1 nohup python -m src.inference \
  experiment=rqvae_vcf_inference_flat \
  data_dir=/home/pyy/GRID/src/data/amazon_data/beauty \
  embedding_path=/home/pyy/GRID/embeddings/beauty/pickle/merged_predictions_tensor.pt \
  embedding_dim=768 \
  num_hierarchies=3 \
  codebook_width=256 \
  ckpt_path=outputs/G_QSS_v2_fixed/G_QSS_QuaSID_Static/01_sid_train/checkpoints/checkpoint_000_003000.ckpt \
  seed=42 \
  hydra.run.dir=outputs/G_QSS_v2_fixed/G_QSS_QuaSID_Static/02_sid_inference \
  > outputs/G_QSS_v2_fixed/G_QSS_QuaSID_Static/02_sid_inference/sid_infer_g_qss_quasid_static.log 2>&1 &
```

---

### Step 3: TIGER Training

| 配置项          | 值                   |
| --------------- | -------------------- |
| experiment      | `tiger_train_flat`   |
| num_hierarchies | 4                    |
| **早停监控**    | `val/ndcg_plus_hr@5` |
| **patience**    | 10                   |
| **min_delta**   | 0.001                |
| **mode**        | max                  |

**运行指令（v2）**:

```bash
CUDA_VISIBLE_DEVICES=1 nohup python -m src.train \
  experiment=tiger_train_flat \
  data_dir=/home/pyy/GRID/src/data/amazon_data/beauty \
  num_hierarchies=4 \
  semantic_id_path=/home/pyy/GRID/outputs/G_QSS_v2_fixed/G_QSS_QuaSID_Static/02_sid_inference/pickle/merged_predictions_tensor.pt \
  seed=42 \
  hydra.run.dir=outputs/G_QSS_v2_fixed/G_QSS_QuaSID_Static/03_tiger_train \
  > outputs/G_QSS_v2_fixed/G_QSS_QuaSID_Static/03_tiger_train/tiger_train_g_qss_quasid_static.log 2>&1 &
```

---

### 输出目录结构

```
outputs/G_QSS_only_gpu1/G_QSS_QuaSID_Static/
├── 01_sid_train/
│   ├── checkpoints/
│   │   └── checkpoint_000_003000.ckpt
│   └── sid_train_g_qss_quasid_static.log
├── 02_sid_inference/
│   ├── pickle/
│   │   └── merged_predictions_tensor.pt
│   └── sid_infer_g_qss_quasid_static.log
└── 03_tiger_train/
    ├── checkpoints/
    │   └── checkpoint_epoch=000_step=00XXXX.ckpt
    └── tiger_train_g_qss_quasid_static.log
```

---

### 监控命令

```bash
# 查看SID训练日志
tail -f outputs/G_QSS_only_gpu1/G_QSS_QuaSID_Static/01_sid_train/sid_train_g_qss_quasid_static.log

# 查看TIGER训练进度
tail -f outputs/G_QSS_only_gpu1/G_QSS_QuaSID_Static/03_tiger_train/tiger_train_g_qss_quasid_static.log
```

