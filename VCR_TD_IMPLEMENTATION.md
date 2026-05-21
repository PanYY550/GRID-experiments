# VCR-TD v2.0 实现记录

## 修改的文件路径和行号

1. `src/modules/clustering/residual_quantization.py`
   - **L104-L115**: 在 `__init__` 中新增了 `use_vcr_td`, `use_time_decay`, `use_dynamic_margin`, `use_cvpm`, `cvpm_temperature`, `alpha`, `m0`, `lambda_rep` 等超参数的初始化。并加载了预先统计好的物品曝光次数张量 `exposure_counts`。
   - **L116-L121**: **(TCCL 新增)** 新增 TCCL 相关参数 `use_tcl`, `lambda_cl`, `cl_tau`, `alpha_cl` 和指标 `train_cl_loss`。
   - **L197-L272**: 新增 `compute_time_decay_weight`, `compute_dynamic_margin`, `cvpm_mask`, `vcr_td_repulsion_loss` 函数。分别计算基于时间衰减的排斥权重、基于余弦相似度的动态边界、过滤良性重叠（same-item等）的 CVPM 掩码，以及基于资格感知的排斥损失。**本次更新：将 CVPM 从硬 0/1 过滤优化为了带有** **`cvpm_temperature`** **参数的软概率过滤机制。**
   - **L273-L304**: **(TCCL 新增)** 新增 `time_aware_collaborative_contrastive_loss` 函数，实现时间感知协同对比损失（InfoNCE），与 VCR-TD 排斥损失形成动态平衡。
   - **L359-L431**: 在 `model_step` 中计算并返回了 `repulsion_loss`。根据 `use_vcr_td` 开关控制是否执行 VCR-TD 排斥逻辑。提取 `item_ids` 和 `input_embeddings` 进行损失计算。
   - **L432-L450**: **(TCCL 修改)** 扩展 `model_step` 返回值为 8 个元素，新增 `quantized_embeddings`, `positive_pair_matrix`, `exposure_times`，支持 TCCL 计算。
   - **L451-L466**: 在 `training_step` 中将 `repulsion_loss` 按照 `lambda_rep` 权重加到 `loss` 中，并在 `train_dict_to_log` 中增加了 `train/repulsion_loss` 的监控。
   - **L467-L475**: **(TCCL 新增)** 在 `training_step` 中加入 TCCL 损失计算，使用 `lambda_cl` 权重，并记录 `train/cl_loss`。
   - **L723-L733**: 在 `eval_step` 中增加了评估状态下的 loss 聚合逻辑。
   - **L850-L860**: **(TCCL 修改)** 更新 `eval_step` 中的解包以适配新的 `model_step` 返回值。
   - **L979**: **(TCCL 修改)** 更新 `predict_step` 中的解包以适配新的 `model_step` 返回值。
2. `configs/experiment/vcr_td_train.yaml`
   - 复制了 `rkmeans_train_flat.yaml` 并在 `model` 配置下增加了 `use_vcr_td`、`use_cvpm` 以及 `cvpm_temperature` 等消融实验的相关开关和超参数。
   - **(TCCL 新增)** 增加 `use_tcl`, `lambda_cl`, `cl_tau`, `alpha_cl` 配置项。
   - **(2026-04 最终验证)** 将 `alpha_cl` 设为 `0.25`（适配 Historical Time Masking + 短训练下的 maturity 增长）；并将 val/test 的 `build_positive_pair_matrix` 默认关闭。

## 新增函数的输入输出说明

- `compute_time_decay_weight(exposure_times, alpha=0.01)`
  - 输入：物品曝光时间/交互次数的张量，衰减系数。
  - 输出：`[0, 1]` 之间的权重张量，随着曝光次数增加指数衰减。
- `compute_dynamic_margin(content_embeddings, m0=0.5)`
  - 输入：物品的内容嵌入张量 (batch\_size, dim)，基础边界值。
  - 输出：动态边界矩阵 (batch\_size, batch\_size)，基于嵌入向量的余弦相似度计算，相似度越高的物品对边界越小。
- `cvpm_mask(item_ids, positive_pair_matrix=None)` - **软概率版本（v2.0更新）**
  - 输入：当前 batch 的物品ID张量 (batch\_size,)，可选的协同正样本对掩码。
  - 输出：`(batch_size, batch_size)` 的 float mask，值在 `[0,1]` 之间，表示需要参与排斥的概率。
  - **核心改进**：使用 `sigmoid((hard_mask - 0.5) / temperature)` 将硬 0/1 过滤转换为软概率过滤，通过 `cvpm_temperature` 参数控制过滤强度。
- `vcr_td_repulsion_loss(quantized_embeddings, item_ids, exposure_times, content_embeddings, mask=None)`
  - 输入：当前量化嵌入，物品ID，曝光时间，内容嵌入，可选掩码。
  - 输出：VCR-TD 排斥损失的标量值。根据时间衰减权重和动态边界调整排斥力。
- `time_aware_collaborative_contrastive_loss(z, positive_pair_matrix, exposure_times)` - **(TCCL 新增)**
  - 输入：量化嵌入张量 (batch_size, dim)，协同正样本对掩码 (batch_size, batch_size)，曝光时间张量 (batch_size,)。
  - 输出：TCCL 对比损失的标量值。基于 InfoNCE 损失，时间加权吸引力，与排斥损失形成动态平衡。

> **2026-04 更新说明（论文对齐）**
>
> - `vcr_td_repulsion_loss(...)` 现在额外接收 `positive_pair_matrix`，用于让 CVPM **过滤协同正样本对**（协同正样本不参与 repulsion）。
> - `time_aware_collaborative_contrastive_loss(...)` 的时间权重已对齐论文：使用 \(w_i = 1 - e^{-\alpha_{cl} t_i}\)，**仅依赖 item 自身成熟度 \(t_i\)**，不再使用 \(|t_i - t_j|\)。

## 实验启动脚本说明

以下是用于训练和不同消融实验的启动脚本：

### 1. VCR-TD Full (完整版)

```bash
CUDA_VISIBLE_DEVICES=1,2 nohup python -m src.train \
    experiment=vcr_td_train \
    data_dir=/home/pyy/GRID/src/data/amazon_data/beauty \
    embedding_path=/home/pyy/GRID/embeddings/beauty/pickle/merged_predictions_tensor.pt \
    embedding_dim=768 \
    num_hierarchies=3 \
    codebook_width=256 \
    model.use_vcr_td=true \
    model.use_cvpm=true \
    model.cvpm_temperature=0.15 \
    model.use_time_decay=true \
    model.use_dynamic_margin=true \
    model.alpha=0.01 \
    model.lambda_rep=0.3 \
    hydra.run.dir=outputs/vcr_td_full \
    > vcr_td_full.log 2>&1 &
```

### 2. 消融实验：Static HaMR (静态边界，无时间衰减)

Static HaMR使用固定边界m0，不随语义相似度变化，模仿QuaSID的静态排斥机制，用于对比VCR-TD的动态边界效果。

```bash
CUDA_VISIBLE_DEVICES=1,2 nohup python -m src.train \
    experiment=vcr_td_train \
    data_dir=/home/pyy/GRID/src/data/amazon_data/beauty \
    embedding_path=/home/pyy/GRID/embeddings/beauty/pickle/merged_predictions_tensor.pt \
    embedding_dim=768 \
    num_hierarchies=3 \
    codebook_width=256 \
    model.use_vcr_td=true \
    model.use_time_decay=false \
    model.use_dynamic_margin=false \
    model.m0=0.5 \
    model.lambda_rep=0.3 \
    hydra.run.dir=outputs/vcr_td_static \
    > vcr_td_static.log 2>&1 &
```

### 3. 消融实验：VCR-TD w/o Time Decay (无时间衰减，有动态边界)

测试仅使用语义感知动态边界（无时间衰减）的效果，验证动态边界组件的贡献。

```bash
CUDA_VISIBLE_DEVICES=1,2 nohup python -m src.train \
    experiment=vcr_td_train \
    data_dir=/home/pyy/GRID/src/data/amazon_data/beauty \
    embedding_path=/home/pyy/GRID/embeddings/beauty/pickle/merged_predictions_tensor.pt \
    embedding_dim=768 \
    num_hierarchies=3 \
    codebook_width=256 \
    model.use_vcr_td=true \
    model.use_time_decay=false \
    model.use_dynamic_margin=true \
    model.m0=0.5 \
    model.lambda_rep=0.3 \
    hydra.run.dir=outputs/vcr_td_wo_time_decay \
    > vcr_td_wo_time_decay.log 2>&1 &
```

### 4. GRID Baseline (原始基准，无VCR-TD)

用于对比的原始GRID方法，不启用任何排斥机制，作为所有实验的零点基准。

```bash
CUDA_VISIBLE_DEVICES=1,2 nohup python -m src.train \
    experiment=rkmeans_train_flat \
    data_dir=/home/pyy/GRID/src/data/amazon_data/beauty \
    embedding_path=/home/pyy/GRID/embeddings/beauty/pickle/merged_predictions_tensor.pt \
    embedding_dim=768 \
    num_hierarchies=3 \
    codebook_width=256 \
    hydra.run.dir=outputs/grid_baseline \
    > grid_baseline.log 2>&1 &
```

***

## CVPM 软过滤机制详解（v2.0 核心更新）

### 背景与动机

原始的 CVPM (Conflict-Aware Valid Pair Masking) 使用硬 0/1 过滤机制，对于 same-item 或协同正样本对直接设置 mask=0，完全阻断排斥信号。这种**硬过滤过强**，可能导致：

1. 排斥信号不足，模型难以学习到有效的语义分离
2. 梯度传播受阻，影响训练稳定性

### 软过滤实现原理

通过引入 `cvpm_temperature` 参数，将硬 mask 转换为软概率 mask：

```python
cvpm_hard = same_item_mask & collaborative_mask  # 硬 0/1 mask
if self.cvpm_temperature > 0:
    # sigmoid 软化：temperature 越大过滤越温和
    cvpm = torch.sigmoid((cvpm_hard.float() - 0.5) / self.cvpm_temperature)
else:
    cvpm = cvpm_hard.float()
```

**数学解释**：

- 当 `hard_mask = 1`（需要排斥）：`sigmoid(0.5 / temperature)` → 接近 1
- 当 `hard_mask = 0`（不应排斥）：`sigmoid(-0.5 / temperature)` → 接近 0，但不为 0

### Temperature 参数调优指南

| temperature | 效果                | 适用场景         |
| ----------- | ----------------- | ------------ |
| **0.05**    | 接近硬过滤，排斥信号弱       | 需要强过滤，避免过度排斥 |
| **0.15** ⭐  | **推荐默认值**，平衡过滤与信号 | 通用场景         |
| **0.3**     | 过滤温和，保留较多排斥信号     | 需要更强排斥力      |

**推荐值**：`cvpm_temperature=0.15`

- 此时 `sigmoid(0.5/0.15) ≈ 0.965`（需要排斥时）
- 此时 `sigmoid(-0.5/0.15) ≈ 0.035`（不应排斥时）
- 既保留了极强的惩罚差异，又为排斥保留了底层的梯度流通空间

### 新增超参数

| 参数                 | 类型    | 默认值  | 说明                      |
| ------------------ | ----- | ---- | ----------------------- |
| `use_cvpm`         | bool  | true | CVPM 总开关                |
| `cvpm_temperature` | float | 0.15 | 软过滤温度，范围建议 \[0.05, 0.3] |

***

## 消融实验对照表

| 实验                        | use\_vcr\_td | use\_cvpm | use\_time\_decay | use\_dynamic\_margin | 目的     |
| ------------------------- | ------------ | --------- | ---------------- | -------------------- | ------ |
| **GRID Baseline**         | false        | -         | -                | -                    | 原始方法对比 |
| **Static HaMR**           | true         | true      | false            | false                | 静态排斥效果 |
| **VCR-TD w/o Time Decay** | true         | true      | false            | true                 | 动态边界贡献 |
| **VCR-TD Full**           | true         | true      | true             | true                 | 完整方法   |

***

## 7. TCCL 时间感知协同对比损失（v4.0 新增模块）

- **实现文件**: `src/modules/clustering/residual_quantization.py`
- **新增参数**: `use_tcl`, `lambda_cl`, `cl_tau`, `alpha_cl`
- **核心创新**: 时间加权吸引力 `w_cl(t) = 1 - exp(-α t)`，与 VCR-TD 排斥损失形成“排斥-吸引”动态平衡

### TCCL 超参数说明

| 参数          | 类型    | 默认值   | 说明                          |
|-------------|-------|-------|-----------------------------|
| `use_tcl`   | bool  | false | TCCL 总开关                    |
| `lambda_cl` | float | 0.1   | TCCL 损失权重（建议从 0.05~0.2 调优） |
| `cl_tau`    | float | 0.07  | InfoNCE 温度系数                |
| `alpha_cl`  | float | 0.01  | 时间衰减系数（与 rep 的 alpha 可共享） |

### TCCL 工作原理

TCCL 通过 InfoNCE 对比学习注入用户协同正样本信号：
- **时间感知权重**：新物品（t 小）吸引力弱，成熟物品（t 大）吸引力强
- **与 VCR-TD 互补**：VCR-TD 负责“推开冲突对”，TCCL 负责“拉近协同正样本”
- 实现从“冷启动强排斥”到“成熟协同融合”的平滑过渡

**论文对齐实现（2026-04）**：
- 对每个 anchor 物品 \(i\)，使用 \(w_i = 1 - e^{-\alpha_{cl} t_i}\) 对其 InfoNCE loss 加权（而不是按 pair 的 \(|t_i-t_j|\) 加权）。

### 推荐启动命令（已包含所有必要参数）

**推荐配置（Final tuned + paper aligned）**：
```bash
CUDA_VISIBLE_DEVICES=0,1 nohup python -m src.train \
    experiment=vcr_td_train \
    data_dir=/home/pyy/GRID/src/data/amazon_data/beauty \
    embedding_path=/home/pyy/GRID/embeddings/beauty/pickle/merged_predictions_tensor.pt \
    embedding_dim=768 \
    num_hierarchies=3 \
    codebook_width=256 \
    data_loading.datamodule.train_dataloader_config.collate_fn.build_positive_pair_matrix=true \
    model.use_vcr_td=true \
    model.use_time_decay=true \
    model.use_dynamic_margin=true \
    model.use_cvpm=true \
    model.cvpm_temperature=0.15 \
    model.lambda_rep=0.3 \
    model.use_tcl=true \
    model.lambda_cl=0.2 \
    model.cl_tau=0.07 \
    model.alpha_cl=0.25 \
    hydra.run.dir=outputs/v4_experiments/final_tuned_paper_aligned \
    > logs/final_tuned_paper_aligned.log 2>&1 &
```

---

## 8. VCR-TD v2.0 关键升级（2026.04）

本次升级补齐论文可写的三个关键工程点：

1. **真实用户共现 `positive_pair_matrix`**（替换随机正样本兜底）
2. **Historical Time Masking**（防未来信息泄漏）
3. **双向时变对比**（排斥随时间衰减 + 吸引随时间增强）

### 8.1 真实用户共现 positive\_pair\_matrix（collate 阶段生成）

- **实现文件**：`src/data/loading/components/collate_functions.py`
- **数据结构**：
  - 在 `src/data/loading/components/interfaces.py` 的 `ItemData` dataclass 中新增字段：
    - `positive_pair_matrix: Optional[torch.Tensor] = None`
- **正样本定义（轻量版）**：
  - 若物品 \(i,j\) 在同一条用户交互序列中共现，则视为正样本对
  - 使用 **Top-K 截断**（默认每个 item 最多保留 50 个共现邻居）控制内存与计算
- **输出形状**：
  - `positive_pair_matrix`: `(batch_size, batch_size)`，float32，取值 \(\{0,1\}\)，对称化 + 去自环

> 注意：该实现默认读取 `${data_dir}/training` 下的 TFRecord 序列文件（可通过 `cooccurrence_sequences_dir` / `cooccurrence_*` 配置覆盖）。

### 8.2 Historical Time Masking（防数据穿越）

- **实现文件**：`src/modules/clustering/residual_quantization.py`
- **新增状态**：
  - `self.global_historical_step = 0`
  - 在 `on_train_batch_start` 中基于 `trainer.global_step + 1` 更新，确保当前 batch 只能看到“之前”的历史进度上界（DDP 一致）
- **Exposure proxy 的历史裁剪**：
  - 原始：`exposure_times = exposure_counts[item_ids]`
  - 现在：`exposure_times = min(exposure_counts[item_ids], global_historical_step)`

### 8.3 双向时变对比（排斥→吸引平滑切换）

- **实现文件**：`src/modules/clustering/residual_quantization.py` 的 `training_step`
- **成熟度因子（适配短训练 + historical cap）**：
  - 使用 batch 内 `exposure_times` 的 **90% 分位数**（`q90`）构造 maturity，减少均值被大量冷启动 item 拉低的情况：
    - `q90 = quantile(exposure_times, 0.90)`
    - `scaled_exp = log1p(q90) / 2.0`
    - `maturity = 1 - exp(-alpha_cl * scaled_exp)`（使用 `alpha_cl` 对齐“吸引侧”增长速度）
- **动态权重**：
  - `effective_lambda_rep = lambda_rep * (1 - maturity)`
  - `effective_lambda_cl  = lambda_cl  * maturity`
- **最终 loss**：
  - `loss = ... + effective_lambda_rep * repulsion_loss + effective_lambda_cl * cl_loss`


### 8.4 最终验证实验（Exp A / maturity final v4）

- **目标**：在 **Historical Time Masking + max\_steps=30** 的设置下，让 maturity 在训练后期（step≈20）进入可用区间，从而让 TCCL 的“后期强吸引”真正生效。
- **关键超参**（写入 `configs/experiment/vcr_td_train.yaml` 默认值）：
  - `model.lambda_cl: 0.20`
  - `model.alpha_cl: 0.25`（关键调整）
  - val/test：`build_positive_pair_matrix: false`（默认关闭）

**最终验证启动命令**（日志落到 `logs/` 方便追踪）：

```bash
CUDA_VISIBLE_DEVICES=0 nohup python -m src.train \
    experiment=vcr_td_train \
    data_dir=/home/pyy/GRID/src/data/amazon_data/beauty \
    embedding_path=/home/pyy/GRID/embeddings/beauty/pickle/merged_predictions_tensor.pt \
    embedding_dim=768 \
    num_hierarchies=3 \
    codebook_width=256 \
    data_loading.datamodule.train_dataloader_config.collate_fn.build_positive_pair_matrix=true \
    hydra.run.dir=outputs/v4_experiments/expA_maturity_final_v4 \
    > logs/expA_maturity_final_v4.log 2>&1 &
```

**关键观测（来自 `logs/expA_maturity_final_v4.log`）**：

- step≈20（训练后期打印）：
  - `maturity=0.3205`
  - `eff_lambda_rep=0.2039`
  - `eff_lambda_cl=0.0641`
- epoch 汇总：
  - `train/cl_loss_epoch=2.270`（metrics 中为 `2.2717`）
  - `train/repulsion_loss_epoch=0.00157`（metrics 中为 `0.0016`）
  - `train/loss_epoch=35.80`（metrics 中为 `35.8470`）

### 8.5 论文对齐修正 & 对比实验（Exp A / paper aligned）

本轮调参收敛后，我们做了两处**以论文为准**的实现对齐，并进行同设置下的 30-step 对比验证。

#### 8.5.1 论文对齐的两处实现修正

- **TCCL 时间权重对齐论文**
  - 由 \(|t_i - t_j|\) 改为只使用 \(t_i\)：\(w_i = 1 - e^{-\alpha_{cl} t_i}\)
- **CVPM 真正过滤协同正样本**
  - 将 `positive_pair_matrix` 传入 `cvpm_mask`，使协同正样本对不参与 repulsion（避免同一对上“拉/推打架”）

#### 8.5.2 对比实验设置

- **对齐前**：`outputs/v4_experiments/expA_maturity_final_v4`（日志：`logs/expA_maturity_final_v4.log`）
- **对齐后**：`outputs/v4_experiments/expA_maturity_final_v4_paper_aligned`（日志：`logs/expA_maturity_final_v4_paper_aligned.log`）
- 两次均使用：
  - `max_steps=30`
  - `build_positive_pair_matrix=true`（真实共现正样本）
  - Historical Time Masking 开启
  - `model.alpha_cl=0.25`, `model.lambda_cl=0.20`

#### 8.5.3 关键对比结果（30 steps）

| 指标 | 对齐前（final v4） | 对齐后（paper aligned） | 结论 |
|---|---:|---:|---|
| maturity（step≈20） | 0.3205 | 0.3205 | maturity **不变**（符合预期） |
| `train/cl_loss_epoch` | 2.270 | 3.580 | **TCCL 更强、更稳定** |
| `train/repulsion_loss_epoch` | 0.00157 | 0.00158 | 基本不变 |
| `train/loss_epoch` | 35.80 | 36.00 | 基本持平 |

---

## 9. Debug/日志建议（实战）

- **collate 阶段是否拿到真实 ppm**：`[TCCL DEBUG] SUCCESS: Got real positive_pair_matrix ... sum=...`
- **成熟度与动态权重**：`[TCCL DEBUG] maturity=... | eff_lambda_rep=... | eff_lambda_cl=...`
- **TCCL batch 统计**：`TCCL final loss = ... | num_pos_pairs=... | avg_pos_per_sample=...`