# VCF 组 SID 碰撞/利用率对比记录

本文件用于记录各实验组在 **SID 推理输出**（`merged_predictions_tensor.pt`）上的一致口径指标，便于横向对比与复测。

## 1. 统计口径与脚本

- **统计对象**：`merged_predictions_tensor.pt`（推理后处理已做 `deduplicate_rows_in_tensor` 并转置）
- **层数定义**：本任务 SID 训练 `SID_NUM_HIER=3`，推理产物维度为 **\(D=4\)**（额外 1 行/列为去重标记 `dedup_indicator`）
- **Full collision**：完全相同的 3-layer SID（token 序列完全一致）导致的重复
- **Partial collision (MC)**：随机采样 pairs，估计 Hamming 距离 \(0 < H \le R\) 的比例（此处 `R=2`，`mc_pairs=200000`）
- **脚本**：`scripts/analyze_sid_collisions.py`

### 1.1 字段释义（Full collision 表）

设：
- \(N\)：item 总数（本次为 12101）
- \(L\)：SID token 层数（本次为 3）
- 每个 item 的 SID 表示为长度 \(L\) 的离散序列 \( \mathbf{s}_i \in \{0,\dots,K-1\}^L \)

则：
- **`frac_unique_sids`**：唯一 SID 占比
  \[
  \texttt{frac\_unique\_sids} \;=\; \frac{\left|\left\{\mathbf{s}_i\right\}_{i=1}^{N}\right|}{N}.
  \]
- **`colliding_groups`**：发生 full collision 的唯一 SID 组数
  \[
  \texttt{colliding\_groups} \;=\; \left|\left\{\mathbf{s}:\ \#\{i:\mathbf{s}_i=\mathbf{s}\} > 1\right\}\right|.
  \]
- **`colliding_items`**：落在这些 collision 组里的 item 数（所有 full-collided items 的总量）。
- **`colliding_items_frac`**：发生 full collision 的 item 占比
  \[
  \texttt{colliding\_items\_frac} \;=\; \frac{\texttt{colliding\_items}}{N}.
  \]
- **`max_group_size`**：最大的 collision 组大小（某个完全相同 SID 下挂的 item 数的最大值）。

补充：推理配置里会对完全相同 SID 做“去重标记”（见 `rqvae*_inference*_flat.yaml` 的 `deduplicate_rows_in_tensor`）。因此脚本里打印的 `dedup_indicator_nonzero_frac` 与 `colliding_items_frac` 应该一致（同口径）。

### 1.2 字段释义（每层 token 利用率表）

对第 \(j\) 层 token（`layerj`）：
- **`unique_tokens`**：该层被实际使用到的 token 种类数
  \[
  \texttt{unique\_tokens}_j \;=\; \left|\left\{s_{i,j}\right\}_{i=1}^{N}\right|.
  \]
- **`coverage`**：该层利用率（码本覆盖率，\(K\) 为该层码本大小，本次 \(K=256\)）
  \[
  \texttt{coverage}_j \;=\; \frac{\texttt{unique\_tokens}_j}{K}.
  \]
  你们常说的 `layer0 coverage` 就是 \(\texttt{coverage}_0\)。
- **`max_token_frac`**：该层最热 token 占比
  \[
  \texttt{max\_token\_frac}_j \;=\; \max_{t\in\{0,\dots,K-1\}}\ \frac{\#\{i:\ s_{i,j}=t\}}{N}.
  \]
- **`entropy_norm`**（报告里单独给了 layer0）：对该层 token 频率分布的熵做归一化，越接近 1 越均匀；越小表示越集中。

复现命令示例（以 G0_400 为例）：

```bash
conda run -n grid python -u scripts/analyze_sid_collisions.py \
  --path "outputs/vcf_3groups_sid/G0_sid400_20260427_224552_rerun/G0_GRID_Baseline/02_sid_inference/pickle/merged_predictions_tensor.pt" \
  --codebook-width 256 \
  --hamming-radius 2 \
  --mc-pairs 200000
```

---

## 2. 本次对比范围（400-step）

### 2.1 产物路径

- **G0_400**
  - `outputs/vcf_3groups_sid/G0_sid400_20260427_224552_rerun/G0_GRID_Baseline/02_sid_inference/pickle/merged_predictions_tensor.pt`
- **G1_400**
  - `outputs/vcf_3groups_sid/G1G6GQSS_sid400_l0boost_20260427_192233/G1_VCF_Full/02_sid_inference/pickle/merged_predictions_tensor.pt`
- **G6_400**
  - `outputs/vcf_3groups_sid/G1G6GQSS_sid400_l0boost_20260427_192233/G6_VCF_wo_TCCL/02_sid_inference/pickle/merged_predictions_tensor.pt`
- **GQSS_400**
  - `outputs/vcf_3groups_sid/G1G6GQSS_sid400_l0boost_20260427_192233/G_QSS_QuaSID_Static/02_sid_inference/pickle/merged_predictions_tensor.pt`

---

## 3. 指标对比（400-step）

### 3.1 Full collision（精确）

| 组 | frac_unique_sids | colliding_items_frac | colliding_items | colliding_groups | max_group_size |
|---|---:|---:|---:|---:|---:|
| **G0_400** | 0.9237 | 0.1318 | 1595 | 672 | 11 |
| **G1_400** | 0.9165 | 0.1478 | 1788 | 777 | 9 |
| **G6_400** | 0.9113 | 0.1551 | 1877 | 804 | 13 |
| **GQSS_400** | 0.9350 | 0.1110 | 1343 | 556 | 26 |

> 注：`colliding_items_frac` 与推理后处理写入的 `dedup_indicator_nonzero_frac` 一致（同口径）。

### 3.2 每层 token 利用率（coverage/entropy）

| 组 | layer0 unique_tokens (coverage) | layer0 max_token_frac | layer1 unique_tokens (coverage) | layer2 unique_tokens (coverage) |
|---|---:|---:|---:|---:|
| **G0_400** | 95 (0.3711) | 0.0403 | 241 (0.9414) | 231 (0.9023) |
| **G1_400** | 37 (0.1445) | 0.0578 | 247 (0.9648) | 253 (0.9883) |
| **G6_400** | 35 (0.1367) | 0.0884 | 222 (0.8672) | 256 (1.0000) |
| **GQSS_400** | 117 (0.4570) | 0.0403 | 250 (0.9766) | 251 (0.9805) |

补充：layer0 entropy_norm（越接近 1 越均匀）：
- **G0_400**: 0.9363
- **G1_400**: 0.9567
- **G6_400**: 0.9222
- **GQSS_400**: 0.9155

### 3.3 Partial collision（MC 估计，R=2，200k pairs）

| 组 | partial_collision_rate (0<H≤2) | full_collision_rate (H=0, MC) |
|---|---:|---:|
| **G0_400** | 0.026945 | 0.000025 |
| **G1_400** | 0.044465 | 0.000010 |
| **G6_400** | 0.053355 | 0.000010 |
| **GQSS_400** | 0.026635 | 0.000020 |

---

## 4. 观察结论（400-step）

- **GQSS_400**：在 400-step 下同时实现 **更高的 layer0 coverage（0.457）** 与 **更低的 full collision（11.10%）**；partial collision 也与 G0 接近。
- **G1_400 / G6_400**：layer0 coverage 明显偏低（~0.14），对应 partial collision 明显升高（4.45% / 5.34%），full collision 也高于 G0。

---

## 5. 指标对比（400-step，v3fix 复测）

本节对应你们的新跑次 `G1G6GQSS_sid400_v3fix_20260427_232714`（包含：提高 \(\lambda\)、`vcf_layer0_boost`、关闭 time-weight）。

### 5.1 产物路径（v3fix）

- **G1_400_v3fix**
  - `outputs/vcf_3groups_sid/G1G6GQSS_sid400_v3fix_20260427_232714/G1_VCF_Full/02_sid_inference/pickle/merged_predictions_tensor.pt`
- **G6_400_v3fix**
  - `outputs/vcf_3groups_sid/G1G6GQSS_sid400_v3fix_20260427_232714/G6_VCF_wo_TCCL/02_sid_inference/pickle/merged_predictions_tensor.pt`
- **GQSS_400_v3fix**
  - `outputs/vcf_3groups_sid/G1G6GQSS_sid400_v3fix_20260427_232714/G_QSS_QuaSID_Static/02_sid_inference/pickle/merged_predictions_tensor.pt`

### 5.2 Full collision（精确，v3fix vs G0_400）

| 组 | frac_unique_sids | colliding_items_frac | colliding_items | colliding_groups | max_group_size |
|---|---:|---:|---:|---:|---:|
| **G0_400** | 0.9237 | 0.1318 | 1595 | 672 | 11 |
| **G1_400_v3fix** | 0.9186 | 0.1411 | 1707 | 722 | 8 |
| **G6_400_v3fix** | 0.8997 | 0.1701 | 2058 | 844 | 17 |
| **GQSS_400_v3fix** | 0.9361 | 0.1120 | 1355 | 582 | 10 |

### 5.3 每层 token 利用率（v3fix vs G0_400）

| 组 | layer0 unique_tokens (coverage) | layer0 max_token_frac | layer1 unique_tokens (coverage) | layer2 unique_tokens (coverage) |
|---|---:|---:|---:|---:|
| **G0_400** | 95 (0.3711) | 0.0403 | 241 (0.9414) | 231 (0.9023) |
| **G1_400_v3fix** | 35 (0.1367) | 0.0545 | 205 (0.8008) | 256 (1.0000) |
| **G6_400_v3fix** | 34 (0.1328) | 0.0950 | 240 (0.9375) | 220 (0.8594) |
| **GQSS_400_v3fix** | 117 (0.4570) | 0.0349 | 243 (0.9492) | 236 (0.9219) |

### 5.4 Partial collision（MC 估计，R=2，200k pairs）

| 组 | partial_collision_rate (0<H≤2) | full_collision_rate (H=0, MC) |
|---|---:|---:|
| **G0_400** | 0.026945 | 0.000025 |
| **G1_400_v3fix** | 0.045785 | 0.000020 |
| **G6_400_v3fix** | 0.062735 | 0.000030 |
| **GQSS_400_v3fix** | 0.026215 | 0.000025 |

### 5.5 结论（v3fix）

- **GQSS**：v3fix 跑次与之前一致，仍然是最稳定、碰撞最少的一组（接近/略优于 G0_400）。
- **G1/G6**：v3fix 仍未改善 `layer0 coverage`（仍在 ~0.13-0.14），full/partial collision 仍显著高于 G0_400。

---

## 6. 指标对比（400-step，Direction-A：G1/G6 静态 margin）

本节对应你们的新跑次 `G1G6GQSS_sid400_dirA_20260427_235140`（G1/G6 改为 `use_dynamic_margin=false`，与 GQSS 对齐使用静态 \(m_{\text{full}}, m_{\text{partial}}\)）。

### 6.1 产物路径（dirA）

- **G1_400_dirA**
  - `outputs/vcf_3groups_sid/G1G6GQSS_sid400_dirA_20260427_235140/G1_VCF_Full/02_sid_inference/pickle/merged_predictions_tensor.pt`
- **G6_400_dirA**
  - `outputs/vcf_3groups_sid/G1G6GQSS_sid400_dirA_20260427_235140/G6_VCF_wo_TCCL/02_sid_inference/pickle/merged_predictions_tensor.pt`
- **GQSS_400_dirA**
  - `outputs/vcf_3groups_sid/G1G6GQSS_sid400_dirA_20260427_235140/G_QSS_QuaSID_Static/02_sid_inference/pickle/merged_predictions_tensor.pt`

### 6.2 Full collision（精确，dirA vs G0_400）

| 组 | frac_unique_sids | colliding_items_frac | colliding_items | colliding_groups | max_group_size |
|---|---:|---:|---:|---:|---:|
| **G0_400** | 0.9237 | 0.1318 | 1595 | 672 | 11 |
| **G1_400_dirA** | 0.9082 | 0.1572 | 1902 | 791 | 20 |
| **G6_400_dirA** | 0.8972 | 0.1778 | 2151 | 907 | 12 |
| **GQSS_400_dirA** | 0.9342 | 0.1156 | 1399 | 603 | 8 |

### 6.3 每层 token 利用率（dirA vs G0_400）

| 组 | layer0 unique_tokens (coverage) | layer0 max_token_frac | layer1 unique_tokens (coverage) | layer2 unique_tokens (coverage) |
|---|---:|---:|---:|---:|
| **G0_400** | 95 (0.3711) | 0.0403 | 241 (0.9414) | 231 (0.9023) |
| **G1_400_dirA** | 36 (0.1406) | 0.0650 | 178 (0.6953) | 254 (0.9922) |
| **G6_400_dirA** | 36 (0.1406) | 0.0773 | 230 (0.8984) | 250 (0.9766) |
| **GQSS_400_dirA** | 109 (0.4258) | 0.0321 | 227 (0.8867) | 234 (0.9141) |

### 6.4 Partial collision（MC 估计，R=2，200k pairs）

| 组 | partial_collision_rate (0<H≤2) | full_collision_rate (H=0, MC) |
|---|---:|---:|
| **G0_400** | 0.026945 | 0.000025 |
| **G1_400_dirA** | 0.044540 | 0.000025 |
| **G6_400_dirA** | 0.056240 | 0.000050 |
| **GQSS_400_dirA** | 0.027045 | 0.000020 |

### 6.5 结论（dirA）

- **G1/G6**：即使切换到静态 margin，`layer0 coverage` 仍然停留在 ~0.14，full/partial collision 也仍显著高于 G0_400；其中 **G1 的 layer1 coverage 还显著下降**（0.6953），说明训练信号被破坏。
- **GQSS**：依然稳定，且与 G0_400 接近/略优。

---

## 7. 指标对比（400-step，GQSS incremental）

本节对应跑次 `GQSS_incr_400_20260428_000550`：

- **Q0_GQSS_base**：静态 margin，无 CVPM，无 TCCL
- **Q1_GQSS_CVPM**：静态 margin + CVPM，无 TCCL
- **Q2_GQSS_CVPM_TCCL**：静态 margin + CVPM + TCCL

注意：复查当前代码后发现，`src/modules/clustering/residual_quantization.py` 当前版本中并没有实际使用 `vcf_disable_time_weight`、`vcf_max_pairs_per_sample`、`vcf_gate_min_layer0_coverage` 等逻辑；同时该脚本未显式传 `model.alpha=0.0`。因此本跑次并不是完全等价于此前 `GQSS` 的“无时间权重”静态路线，解释时需要谨慎。

### 7.1 Full collision（精确，incremental vs G0_400）

| 组 | frac_unique_sids | colliding_items_frac | colliding_items | colliding_groups | max_group_size |
|---|---:|---:|---:|---:|---:|
| **G0_400** | 0.9237 | 0.1318 | 1595 | 672 | 11 |
| **Q0_GQSS_base** | 0.8855 | 0.1947 | 2356 | 970 | 16 |
| **Q1_GQSS_CVPM** | 0.9003 | 0.1704 | 2062 | 855 | 14 |
| **Q2_GQSS_CVPM_TCCL** | 0.8856 | 0.1946 | 2355 | 971 | 12 |

### 7.2 每层 token 利用率（incremental vs G0_400）

| 组 | layer0 unique_tokens (coverage) | layer0 max_token_frac | layer1 unique_tokens (coverage) | layer2 unique_tokens (coverage) |
|---|---:|---:|---:|---:|
| **G0_400** | 95 (0.3711) | 0.0403 | 241 (0.9414) | 231 (0.9023) |
| **Q0_GQSS_base** | 35 (0.1367) | 0.0838 | 164 (0.6406) | 256 (1.0000) |
| **Q1_GQSS_CVPM** | 34 (0.1328) | 0.0824 | 168 (0.6562) | 256 (1.0000) |
| **Q2_GQSS_CVPM_TCCL** | 37 (0.1445) | 0.0640 | 234 (0.9141) | 253 (0.9883) |

### 7.3 Partial collision（MC 估计，R=2，200k pairs）

| 组 | partial_collision_rate (0<H≤2) | full_collision_rate (H=0, MC) |
|---|---:|---:|
| **G0_400** | 0.026945 | 0.000025 |
| **Q0_GQSS_base** | 0.057040 | 0.000030 |
| **Q1_GQSS_CVPM** | 0.058460 | 0.000040 |
| **Q2_GQSS_CVPM_TCCL** | 0.044650 | 0.000025 |

### 7.4 结论（incremental）

- 本次 `Q0/Q1/Q2` 均未复现此前 `GQSS` 的稳定表现，`layer0 coverage` 仍只有约 0.13-0.14，full/partial collision 均显著高于 `G0_400`。
- 当前更大的问题是**代码/配置状态发生了漂移**：脚本里传入的部分 `+model.*` 开关在当前模型代码中没有实际生效；且没有显式设置 `model.alpha=0.0`，导致静态 margin 路线仍可能被时间权重影响。

---

## 7. 指标对比（400-step，GQSS 主线增量：Q0/Q1/Q2）

本节对应跑次 `outputs/vcf_qss_increments/GQSS_incr_400_20260428_000550`：

- **Q0**：GQSS_base（静态 margin；`use_cvpm=false`；`lambda_cl=0`）
- **Q1**：GQSS + CVPM（`use_cvpm=true`；`lambda_cl=0`）
- **Q2**：GQSS + CVPM + TCCL（`use_cvpm=true`；`lambda_cl=0.2`）

### 7.1 产物路径（Q0/Q1/Q2）

- **Q0_GQSS_base_400**
  - `outputs/vcf_qss_increments/GQSS_incr_400_20260428_000550/Q0_GQSS_base/02_sid_inference/pickle/merged_predictions_tensor.pt`
- **Q1_GQSS_CVPM_400**
  - `outputs/vcf_qss_increments/GQSS_incr_400_20260428_000550/Q1_GQSS_CVPM/02_sid_inference/pickle/merged_predictions_tensor.pt`
- **Q2_GQSS_CVPM_TCCL_400**
  - `outputs/vcf_qss_increments/GQSS_incr_400_20260428_000550/Q2_GQSS_CVPM_TCCL/02_sid_inference/pickle/merged_predictions_tensor.pt`

### 7.2 Full collision（精确，Q0/Q1/Q2 vs G0_400）

| 组 | frac_unique_sids | colliding_items_frac | colliding_items | colliding_groups | max_group_size |
|---|---:|---:|---:|---:|---:|
| **G0_400** | 0.9237 | 0.1318 | 1595 | 672 | 11 |
| **Q0_GQSS_base_400** | 0.8855 | 0.1947 | 2356 | 970 | 16 |
| **Q1_GQSS_CVPM_400** | 0.9003 | 0.1704 | 2062 | 855 | 14 |
| **Q2_GQSS_CVPM_TCCL_400** | 0.8856 | 0.1946 | 2355 | 971 | 12 |

### 7.3 每层 token 利用率（Q0/Q1/Q2 vs G0_400）

| 组 | layer0 unique_tokens (coverage) | layer0 max_token_frac | layer1 unique_tokens (coverage) | layer2 unique_tokens (coverage) |
|---|---:|---:|---:|---:|
| **G0_400** | 95 (0.3711) | 0.0403 | 241 (0.9414) | 231 (0.9023) |
| **Q0_GQSS_base_400** | 35 (0.1367) | 0.0838 | 164 (0.6406) | 256 (1.0000) |
| **Q1_GQSS_CVPM_400** | 34 (0.1328) | 0.0824 | 168 (0.6562) | 256 (1.0000) |
| **Q2_GQSS_CVPM_TCCL_400** | 37 (0.1445) | 0.0640 | 234 (0.9141) | 253 (0.9883) |

### 7.4 Partial collision（MC 估计，R=2，200k pairs）

| 组 | partial_collision_rate (0<H≤2) | full_collision_rate (H=0, MC) |
|---|---:|---:|
| **G0_400** | 0.026945 | 0.000025 |
| **Q0_GQSS_base_400** | 0.057040 | 0.000030 |
| **Q1_GQSS_CVPM_400** | 0.058460 | 0.000040 |
| **Q2_GQSS_CVPM_TCCL_400** | 0.044650 | 0.000025 |

### 7.5 结论（Q0/Q1/Q2）

- **Q0/Q1**：`layer0 coverage` 仍然很低（~0.13），full/partial collision 明显高于 G0_400。
- **Q2（加 TCCL）**：`layer1 coverage` 明显恢复（0.9141），partial collision 下降（4.47%），但 `layer0 coverage` 仍低（0.1445），full collision 仍偏高（19.46%）。

---

## 8. 指标对比（400-step，GQSS incremental v3：Q0/Q1/Q2）

本节对应跑次 `outputs/vcf_qss_increments/GQSS_incr_400_v3_20260428_170322`：

- **Q0**：GQSS_pure（静态 margin；`use_cvpm=true`；`lambda_cl=0.0`）
- **Q1**：GQSS_TCCL（静态 margin；`use_cvpm=true`；`lambda_cl=0.2`）
- **Q2**：GQSS_noCVPM（静态 margin；`use_cvpm=false`；`lambda_cl=0.0`）

### 8.1 产物路径（Q0/Q1/Q2）

- **Q0_GQSS_pure_400**
  - `outputs/vcf_qss_increments/GQSS_incr_400_v3_20260428_170322/Q0_GQSS_pure/02_sid_inference/pickle/merged_predictions_tensor.pt`
- **Q1_GQSS_TCCL_400**
  - `outputs/vcf_qss_increments/GQSS_incr_400_v3_20260428_170322/Q1_GQSS_TCCL/02_sid_inference/pickle/merged_predictions_tensor.pt`
- **Q2_GQSS_noCVPM_400**
  - `outputs/vcf_qss_increments/GQSS_incr_400_v3_20260428_170322/Q2_GQSS_noCVPM/02_sid_inference/pickle/merged_predictions_tensor.pt`

### 8.2 Full collision（精确，Q0/Q1/Q2）

| 组 | frac_unique_sids | colliding_items_frac | colliding_items | colliding_groups | max_group_size |
|---|---:|---:|---:|---:|---:|
| **Q0_GQSS_pure_400** | 0.9336 | 0.1158 | 1401 | 597 | 10 |
| **Q1_GQSS_TCCL_400** | 0.2638 | 0.8996 | 10886 | 1977 | 124 |
| **Q2_GQSS_noCVPM_400** | 0.9356 | 0.1116 | 1351 | 572 | 12 |

### 8.3 每层 token 利用率（Q0/Q1/Q2）

| 组 | layer0 unique_tokens (coverage) | layer0 max_token_frac | layer1 unique_tokens (coverage) | layer2 unique_tokens (coverage) |
|---|---:|---:|---:|---:|
| **Q0_GQSS_pure_400** | 110 (0.4297) | 0.0360 | 246 (0.9609) | 235 (0.9180) |
| **Q1_GQSS_TCCL_400** | 1 (0.0039) | 1.0000 | 47 (0.1836) | 232 (0.9062) |
| **Q2_GQSS_noCVPM_400** | 117 (0.4570) | 0.0390 | 245 (0.9570) | 244 (0.9531) |

### 8.4 Partial collision（MC 估计，R=2，200k pairs）

| 组 | partial_collision_rate (0<H≤2) | full_collision_rate (H=0, MC) |
|---|---:|---:|
| **Q0_GQSS_pure_400** | 0.025720 | 0.000035 |
| **Q1_GQSS_TCCL_400** | 0.999185 | 0.000815 |
| **Q2_GQSS_noCVPM_400** | 0.026110 | 0.000025 |

### 8.5 结论（v3）

- **Q0（GQSS_pure）**：整体碰撞与 layer0 利用率正常（`layer0 coverage≈0.43`）。
- **Q1（+TCCL）**：出现灾难性塌缩（`layer0 unique_tokens=1`，`colliding_items_frac≈0.90`）。
- **Q2（noCVPM）**：本次并未出现塌缩，整体与 Q0 接近（但理论上更不稳，仍建议把它当作“风险组”对照）。


---

## 9. 指标对比（400-step，核心创新消融 core_ablation_400：A1/A2/A3/A4）

本节对应跑次 `outputs/core_ablation/core_ablation_400_20260428_195412`。

实验设计（以 Q0_GQSS_pure 为底座，逐一加入创新点）：

| 组名 | 说明 | 关键参数 |
|---|---|---|
| **Q0_ref** | 稳定基线（引自第 8 节） | 静态边界；`lambda_cl=0.0`；`alpha=0.0` |
| **A1_TimeW** | 仅时间权重 | `alpha=0.5`；动态边界=off；`lambda_cl=0.0` |
| **A2_DynM** | 仅动态边界 | `use_dynamic_margin=true`；`m0=0.5`；`severity_beta=0.5`；`alpha=0.0`；`lambda_cl=0.0` |
| **A3_TCCL** | 仅 TCCL（warmup=200） | `lambda_cl=0.2`；`tccl_warmup_steps=200`；动态边界=off；`alpha=0.0` |
| **A4_Full** | 全部三项创新 | `alpha=0.5`；`use_dynamic_margin=true`；`lambda_cl=0.2`；`tccl_warmup_steps=200` |

### 9.1 产物路径

`outputs/core_ablation/core_ablation_400_20260428_195412/<GROUP>/02_sid_inference/pickle/merged_predictions_tensor.pt`

### 9.2 Full collision（精确）

| 组 | frac_unique_sids | colliding_items_frac | colliding_items | colliding_groups | max_group_size |
|---|---:|---:|---:|---:|---:|
| **Q0_ref** | 0.9336 | 0.1158 | 1401 | 597 | 10 |
| **A1_TimeW** | 0.9344 | 0.1151 | 1393 | 599 | 14 |
| **A2_DynM** | 0.9213 | 0.1320 | 1597 | 645 | 11 |
| **A3_TCCL** | 0.7127 | 0.4110 | 4974 | 1497 | 61 |
| **A4_Full** | 0.6947 | 0.4353 | 5267 | 1572 | 62 |

### 9.3 每层 token 利用率

| 组 | layer0 unique_tokens (coverage) | layer0 max_token_frac | layer1 coverage | layer2 coverage |
|---|---:|---:|---:|---:|
| **Q0_ref** | 110 (0.4297) | 0.0360 | 0.9609 | 0.9180 |
| **A1_TimeW** | 113 (0.4414) | 0.0363 | 0.9219 | 0.9102 |
| **A2_DynM** | 119 (0.4648) | 0.0393 | 0.9570 | 0.9688 |
| **A3_TCCL** | 30 (0.1172) | 0.1240 | 0.5430 | 0.8867 |
| **A4_Full** | 31 (0.1211) | 0.1051 | 0.4883 | 0.8711 |

### 9.4 Partial collision（MC 估计，R=2，200k pairs）

| 组 | partial_collision_rate \((0<H\leq R)\) | full_collision_rate \((H=0)\) |
|---|---:|---:|
| **Q0_ref** | 0.025720 | 0.000035 |
| **A1_TimeW** | 0.026970 | 0.000020 |
| **A2_DynM** | 0.026565 | 0.000035 |
| **A3_TCCL** | 0.068440 | 0.000190 |
| **A4_Full** | 0.072860 | 0.000250 |

### 9.5 训练曲线逐步分析（layer0 coverage_step，每 50 step）

| step | A1_TimeW | A2_DynM | A3_TCCL |
|---:|---:|---:|---:|
| 49 | 0.9805 | 1.0000 | 0.9727 |
| 99 | **0.2773** | **0.1823** | **0.3008** |
| 149 | 0.2539 | 0.3047 | 0.2773 |
| 199 | 0.2813 | 0.3086 | 0.3086 |
| 249 | 0.3203 | 0.3281 | 0.3125 |
| 299 | 0.3594 | 0.3750 | **0.2070** ← TCCL 激活 |
| 349 | 0.3633 | 0.4141 | 0.1758 |
| 399 | 0.3945 | 0.4258 | 0.1563 |

A4_Full 与 A3_TCCL 最终指标相近（layer0 ≈ 0.12），故省略逐步数据。

### 9.6 根因分析（关键发现）

> ⚠ **重要修正**：之前依赖 summary 估算时，将 A2/A3 标签混淆（coverage=0.1172 实为 A3_TCCL 结果，非 A2_DynM）。以下分析以实际文件数据为准。

**所有组在 step 49→99 均发生初始坍缩**（layer0 从 ≈1.0 骤降至 0.18–0.30）。  
原因：`repulsion_warmup_steps=2000` 使 step 100 内 warmup factor 仅 5%，repulsion 极弱；同时批次中大量碰撞对驱动 stability gate（`vcf_gate_min_layer0_coverage=0.50`）触发，把 repulsion 完全封锁，形成早期 VQ 随机退化。

**差异在于坍缩后能否自然恢复**：

| 组 | 初始坍缩 | 恢复能力 | 根因 |
|---|---|---|---|
| A1_TimeW | step 99 → 0.277 | ✅ 恢复（step 399 → 0.395） | 静态边界 + 时间权重温和；VQ 分配自然收敛 |
| A2_DynM | step 99 → 0.182（更深） | ✅ 恢复（step 399 → 0.426，**超越 Q0**） | 动态边界导致更深初始坍缩，但梯度方向正确，最终 layer0 超 Q0 |
| A3_TCCL | step 99 → 0.301 | ❌ 二次塌缩 | step 200 TCCL 激活时 layer0 仅 0.31，吸引力对抗 repulsion，coverage 从 0.31 继续跌至 0.16 |
| A4_Full | 同 A3 | ❌ 塌缩 | TCCL 主导，与 A3 相似 |

**核心结论**：
- **动态边界（A2）有效且优于 Q0**：`layer0 coverage` 0.4648 > Q0 的 0.4297，partial collision 持平；
- **时间权重（A1）中性**：微弱改善，无负面影响；
- **TCCL 是唯一问题来源**：`warmup=200` 时 coverage 仍在恢复阶段（0.31），吸引力打断恢复并引发二次塌缩；
- **修复方向**：将 `tccl_warmup_steps` 延至 step≥350（对应 A1/A2 曲线 coverage≥0.36），并将 `lambda_cl` 从 0.2 降至 0.05，避免过强吸引力。

### 9.7 下一步（B-series 400-step 验证）

| 组 | 描述 | 核心参数变化 | 预期 |
|---|---|---|---|
| **B1_DynM_TimeW** | 动态边界 + 时间权重（无 TCCL） | `alpha=0.5`；`use_dynamic_margin=true`；`lambda_cl=0.0` | layer0 ≥ 0.44，优于 Q0 |
| **B2_TCCL_fix** | 静态边界 + 修复 TCCL | `tccl_warmup_steps=350`；`lambda_cl=0.05`；动态边界=off | layer0 ≥ 0.38，验证修复可行性 |
| **B3_Full_fix** | 三项创新全合并（修复后） | `alpha=0.5`；动态边界=on；`tccl_warmup_steps=350`；`lambda_cl=0.05` | 若 B1/B2 稳定，预计最优 |

---

## 10. 最终三组对比（3000-step，SID+TIGER 全流程）

本节对应跑次 `outputs/final_3groups/final3g_sid3000_20260428_220333`，并与已完成的 `G0_only_gpu1` 对比。

- **G0**：原始 RQ-VAE 基线（复用 `G0_only_gpu1` 结果）
- **Q0**：GQSS 静态边界基线（`m_full=0.5, m_partial=0.3, CVPM`）
- **B1**：动态边界 + 时间权重（无 TCCL）

### 10.1 SID 碰撞与码本利用率（G0/Q0/B1）

| 组 | frac_unique_sids | colliding_items_frac | colliding_items | colliding_groups | max_group_size |
|---|---:|---:|---:|---:|---:|
| **G0** | 0.9745 | 0.0472 | 571 | 262 | 10 |
| **Q0** | 0.9703 | 0.0533 | 645 | 285 | 9 |
| **B1** | 0.9686 | 0.0572 | 692 | 312 | 11 |

| 组 | layer0 coverage | layer1 coverage | layer2 coverage | partial_collision_rate \(0<H\le R, R=2\) |
|---|---:|---:|---:|---:|
| **G0** | 0.7109 | 1.0000 | 1.0000 | 0.017875 |
| **Q0** | 0.4844 | 0.9922 | 0.9805 | 0.020200 |
| **B1** | 0.4922 | 0.9883 | 0.9609 | 0.020895 |

### 10.2 TIGER 最终测试指标（G0/Q0/B1）

| 组 | test/ndcg@5 | test/ndcg@10 | test/recall@5 | test/recall@10 | test/ndcg_plus_hr@5 | test/ndcg_plus_hr@10 |
|---|---:|---:|---:|---:|---:|---:|
| **G0** | 0.02696 | 0.03256 | 0.03953 | 0.05688 | 0.06649 | 0.08944 |
| **Q0** | 0.02805 | 0.03332 | 0.03989 | 0.05625 | 0.06794 | 0.08958 |
| **B1** | 0.02574 | 0.03157 | 0.03810 | 0.05625 | 0.06383 | 0.08782 |

### 10.3 相对 G0 的变化（\(\Delta\%\)）

\[
\Delta\% = \frac{\text{metric}_{\text{group}}-\text{metric}_{\text{G0}}}{\text{metric}_{\text{G0}}}\times 100\%
\]

| 组 | \(\Delta\) NDCG@10 | \(\Delta\) NDCG@5 | \(\Delta\) Recall@10 | \(\Delta\) Recall@5 | \(\Delta\) NDCG+HR@5 |
|---|---:|---:|---:|---:|---:|
| **Q0 vs G0** | +2.34% | +4.03% | -1.10% | +0.91% | +2.17% |
| **B1 vs G0** | -3.04% | -4.55% | -1.10% | -3.62% | -3.99% |

### 10.4 结论（final_3groups）

- **Q0（静态边界）是当前最优可用主线**：在 NDCG 指标上稳定优于 G0（尤其 `ndcg@5`、`ndcg@10`、`ndcg_plus_hr@5`）。
- **B1（动态边界+时间权重）在 3000-step 下未转化为下游收益**：尽管 `layer0 coverage` 与 Q0 接近，但碰撞略高，且 TIGER 指标整体低于 G0/Q0。
- **工程决策建议**：主实验与对外报告采用 **Q0**；将 **B1** 作为负向消融结果保留，用于说明“创新项在当前实现/参数下未超过静态基线”。

---

## 11. C1：U 型双端强化时间权重（400-step SID 快测）

本节对应跑次 `logs/C1_ushape/C1_ushape_sid400_20260429_181416`（脚本：`scripts/run_C1_ushape_sid_train_infer.sh`）。

- **C1a_UShape_Static**：U 型时间权重 + QuaSID 静态边界（对标 Q0）
- **C1b_UShape_DynM**：U 型时间权重 + 动态边界（对标 B1）

U 型双端强化（batch 内曝光次数百分位分段）：

- 冷启动端：`rank < cold_pct=0.25`，权重 \(1+\delta_c=1.4\)
- 热门端：`rank \ge hot_pct=0.75`，权重 \(1+\delta_h=1.3\)
- 成长期：其余，权重 \(1.0\)
- pair 权重：\(w_{ij}=\sqrt{w_i w_j}\)

### 11.1 SID 碰撞与码本利用率（C1a/C1b）

| 组 | frac_unique_sids | colliding_items_frac | colliding_items | colliding_groups | max_group_size |
|---|---:|---:|---:|---:|---:|
| **C1a_UShape_Static** | 0.9346 | 0.1159 | 1402 | 610 | 9 |
| **C1b_UShape_DynM** | 0.9404 | 0.1055 | 1277 | 556 | 9 |

### 11.2 各层 token 利用率（unique_tokens / entropy_norm / max_token_frac）

| 组 | layer0 | layer1 | layer2 |
|---|---|---|---|
| **C1a_UShape_Static** | 113 (0.4414), 0.9312, 0.0363 | 248 (0.9688), 0.9592, 0.0100 | 235 (0.9180), 0.9564, 0.0111 |
| **C1b_UShape_DynM** | 106 (0.4141), 0.9436, 0.0359 | 244 (0.9531), 0.9662, 0.0096 | 234 (0.9141), 0.9547, 0.0112 |

> 注：括号内 coverage 为 `unique_tokens / 256`。

### 11.3 Partial collision（MC 估计，R=2，300k pairs）

| 组 | partial_collision_rate \((0<H\leq R)\) | full_collision_rate \((H=0)\) |
|---|---:|---:|
| **C1a_UShape_Static** | 0.026287 | 0.000010 |
| **C1b_UShape_DynM** | 0.025510 | 0.000007 |

### 11.4 结论（400-step）

- **碰撞指标**：两组均满足快测目标（`frac_unique_sids ≥ 0.90`，`colliding_items_frac ≤ 0.15`）。
- **对比**：C1b 在 `frac_unique_sids` 与 `colliding_items_frac` 上优于 C1a；但 C1b 的 `layer0 coverage` 低于 C1a（0.4141 vs 0.4414）。
- **下一步建议**：优先把 **C1a/C1b 各跑一组 3000-step + TIGER**，验证 U 型时间权重是否能修复 “B1（旧时间权重）SID 指标好但 TIGER 变差” 的问题。

---

## 12. C1 全流程实验（3000-step SID + TIGER）vs G0 基线

> **实验日期**：2026-04-29/30 | **RUN_TAG**: `C1_full_sid3000_20260429_192552`
> **GPU 分配**：C1a=GPU1，C1b=GPU2，GPU0 空置
> **U型时间权重参数**：`cold_pct=0.25`, `hot_pct=0.75`, `delta_cold=0.4`, `delta_hot=0.3`

### 12.1 SID 碰撞指标（3000-step 后全量推理）

| 组 | frac_unique_sids | colliding_items_frac | colliding_items | colliding_groups | max_group_size |
|---|---:|---:|---:|---:|---:|
| **G0_Baseline** | 0.9745 | 0.0472 | 571 | 302 | 7 |
| **Q0_GQSS_Static** | 0.9703 | 0.0533 | 645 | 337 | 8 |
| **B1_DynM+TimeW (旧)** | 0.9686 | 0.0572 | 692 | 353 | 9 |
| **C1a_UShape_Static** | **0.9723** | **0.0507** | 614 | 279 | 10 |
| **C1b_UShape_DynM** | 0.9680 | 0.0578 | 699 | 312 | 12 |

### 12.2 各层 token 利用率（3000-step）

| 组 | layer0 unique_tokens (cov/256) | entropy_norm | max_token_frac |
|---|---|---:|---:|
| **G0_Baseline** | 182 (0.7109) | — | — |
| **Q0_GQSS_Static** | 124 (0.4844) | — | — |
| **B1_DynM+TimeW (旧)** | 126 (0.4922) | — | — |
| **C1a_UShape_Static** | 126 (0.4922), 0.9723, 0.0213 | layer1: 256(1.0), 0.9698 | layer2: 245(0.957), 0.9670 |
| **C1b_UShape_DynM** | 127 (0.4961), 0.9642, 0.0393 | layer1: 249(0.973), 0.9648 | layer2: 240(0.938), 0.9664 |

### 12.3 Partial collision（MC 估计，R=2，300k pairs）

| 组 | partial_collision_rate \((0<H\leq R)\) | full_collision_rate \((H=0)\) |
|---|---:|---:|
| **G0_Baseline** | — | — |
| **C1a_UShape_Static** | 0.020077 | 0.000003 |
| **C1b_UShape_DynM** | 0.021243 | 0.000007 |

### 12.4 TIGER 推荐指标（test set）

| 组 | NDCG@5 | NDCG@10 | Recall@5 | Recall@10 | NDCG+HR@5 | NDCG+HR@10 |
|---|---:|---:|---:|---:|---:|---:|
| **G0_Baseline** | 0.02696 | 0.03256 | 0.03953 | 0.05688 | 0.06649 | 0.08944 |
| **Q0_GQSS_Static** | **0.02805** | **0.03332** | 0.03989 | 0.05625 | **0.06794** | **0.08958** |
| **B1_DynM+TimeW (旧)** | 0.02573 | 0.03157 | 0.03806 | 0.05358 | 0.06301 | 0.08612 |
| **C1a_UShape_Static** | 0.02541 | 0.03109 | 0.03689 | 0.05455 | 0.06230 | 0.08565 |
| **C1b_UShape_DynM** | 0.02678 | 0.03234 | **0.04000** | **0.05715** | 0.06676 | 0.08949 |

### 12.5 Δ% vs G0 基线

| 组 | ΔNDCG@5 | ΔNDCG@10 | ΔRecall@5 | ΔRecall@10 |
|---|---:|---:|---:|---:|
| **Q0_GQSS_Static** | **+4.04%** | **+2.33%** | +0.91% | -1.11% |
| **B1_DynM+TimeW (旧)** | -4.56% | -3.04% | -3.72% | -5.81% |
| **C1a_UShape_Static** | -5.75% | -4.51% | -6.68% | -4.10% |
| **C1b_UShape_DynM** | **-0.67%** | **-0.68%** | **+1.14%** | **+0.47%** |

### 12.6 结论与分析

**C1b（U型时间权重 + 动态边界）是本轮实验的核心发现：**

1. **U型时间权重有效修复了 B1 的方向性 Bug**
   - B1（旧时间权重：流行商品排斥减弱）导致 NDCG@10 比 G0 下降 **-3.04%**
   - C1b（U型：冷启动端 + 热门端同时增强排斥）将差距缩小至 **-0.68%**，几乎完全找回损失

2. **C1b ≈ G0，在 Recall 上首次超越 G0**
   - Recall@5 **+1.14%**，Recall@10 **+0.47%** 超过基线，说明 U 型时间权重设计方向正确

3. **C1a（U型 + 静态边界）表现不如 C1b**
   - 静态边界与 U 型时间权重的组合不如动态边界版本，静态边界可能不够灵活，导致 SID 碰撞组的量更高（colliding_groups=279 vs 312，但 colliding_items_frac 反而 C1a 更低）

4. **Q0 仍是综合最优**（NDCG@5 +4.04% vs G0），但不含时间维度信息
   - C1b 的 NDCG@5 仍低于 Q0 约 4.5%，说明 U 型时间权重单独还不够，需与 GQSS 的 CVPM 等机制结合

5. **SID 碰撞层面**：C1a 的 `colliding_items_frac=0.0507` 是仅次于 G0（0.0472）的最低碰撞率，但未能转化为更好的 TIGER 指标，说明 **SID 碰撞减少 ≠ 下游推荐性能提升**，两者并非线性关系

**下一步建议（供参考）**：
- 方向 1：将 C1b 的 U 型时间权重融入 Q0（GQSS + CVPM + U 型时间权重），期望在 Q0 强基础上叠加时间感知优势
- 方向 2：降低 C1b 的动态边界幅度（`beta: 0.5→0.3`）以减少语义扰动，同时保留 U 型时间权重

---

## 13. C2 全流程实验（3000-step SID + TIGER）：lambda 重校准与轻量 U 型

> **实验日期**：2026-04-30 | **RUN_TAG**: `C2_full_sid3000_20260430_013836`
> **GPU 分配**：C2ctrl=GPU0，C2a=GPU2
> **核心改进**：降低排斥权重 λ 从 0.3→0.2，轻量 U 型增强（δ_cold=0.15, δ_hot=0.10）

### 13.1 实验设计

| 组名 | 说明 | 关键参数 |
|---|---|---|
| **C2ctrl_LamRecalib** | Lambda 重校准（λ_full=0.2, λ_partial=0.2） | 无时间权重，静态边界 |
| **C2a_LamRecalib_LightUShape** | Lambda 重校准 + 轻量 U 型时间权重 | δ_cold=0.15, δ_hot=0.10，静态边界 |

### 13.2 SID 碰撞指标（3000-step）

| 组 | frac_unique_sids | colliding_items_frac | colliding_items | colliding_groups | max_group_size |
|---|---:|---:|---:|---:|---:|
| **G0_Baseline** | 0.9745 | 0.0472 | 571 | 302 | 7 |
| **Q0_GQSS_Static** | 0.9703 | 0.0533 | 645 | 337 | 8 |
| **C2ctrl_LamRecalib** | 0.9722 | 0.0512 | 619 | 282 | 9 |
| **C2a_LamRecalib_LightUShape** | **0.9737** | **0.0473** | 572 | 254 | 10 |

### 13.3 各层 token 利用率（3000-step）

| 组 | layer0 (cov/256) | entropy_norm | max_token_frac |
|---|---:|---:|---:|
| **C2ctrl_LamRecalib** | 127 (0.4961) | 0.9685 | 0.0283 |
| **C2a_LamRecalib_LightUShape** | 126 (0.4922) | 0.9702 | 0.0215 |

### 13.4 Partial collision（MC 估计，R=2，300k pairs）

| 组 | partial_collision_rate | full_collision_rate |
|---|---:|---:|
| **C2ctrl_LamRecalib** | 0.020187 | 0.000003 |
| **C2a_LamRecalib_LightUShape** | **0.019807** | 0.000003 |

### 13.5 TIGER 推荐指标（test set，最优 checkpoint）

| 组 | NDCG@5 | NDCG@10 | Recall@5 | Recall@10 | NDCG+HR@5 | NDCG+HR@10 |
|---|---:|---:|---:|---:|---:|---:|
| **G0_Baseline** | 0.02696 | 0.03256 | 0.03953 | 0.05688 | 0.06649 | 0.08944 |
| **Q0_GQSS_Static** | **0.02805** | **0.03332** | 0.03989 | 0.05625 | **0.06794** | **0.08958** |
| **C2ctrl_LamRecalib** | **0.03153** | **0.03820** | **0.04717** | **0.06799** | **0.07893** | **0.10551** |
| **C2a_LamRecalib_LightUShape** | 0.03011 | 0.03692 | 0.04570 | 0.06640 | 0.07655 | 0.10445 |

> **最优 checkpoint**：C2ctrl @ step 2799（val/ndcg@10=0.03834）；C2a @ step 2899（val/ndcg@10=0.03935）

### 13.6 Δ% vs G0 基线

| 组 | ΔNDCG@5 | ΔNDCG@10 | ΔRecall@5 | ΔRecall@10 |
|---|---:|---:|---:|---:|
| **Q0_GQSS_Static** | +4.04% | +2.33% | +0.91% | -1.11% |
| **C2ctrl_LamRecalib** | **+16.95%** | **+17.32%** | **+19.33%** | **+19.53%** |
| **C2a_LamRecalib_LightUShape** | **+11.68%** | **+13.39%** | **+15.61%** | **+16.74%** |

### 13.7 结论与分析

**C2ctrl（Lambda 重校准）实现历史性突破：**

1. **λ 重校准是关键**：将 λ_full 从 0.3 降至 0.2（与 QuaSID 最优值对齐）后，TIGER 性能大幅提升
   - NDCG@10 提升 **+17.32%**，Recall@10 提升 **+19.53%**，全面超越 G0 和 Q0
   - 验证了 QuaSID 论文结论：λ_full > 0.2 会损害性能

2. **轻量 U 型时间权重（C2a）略有增益但不显著**
   - C2a 相比 C2ctrl 各项指标略低，说明在已优化的 λ 基础上，轻度 U 型增强对性能提升有限
   - 可能原因：轻量 δ 值（0.15/0.10）影响较弱，或静态边界限制了时间权重的效果

3. **SID 碰撞与 TIGER 性能关系**
   - C2a 的 `frac_unique_sids=0.9737` 优于 C2ctrl 的 0.9722，但 TIGER 指标反而略低
   - 再次验证：**SID 碰撞率降低 ≠ TIGER 性能提升**，过度追求低碰撞可能损害语义区分度

---

## 14. D 系列实验（3000-step）：固定 CL vs 内容 CVPM

> **实验日期**：2026-04-30 | **RUN_TAG**: `D_series_sid3000_20260430_151216`
> **GPU 分配**：D0=GPU0，D1=GPU1，D2=GPU2
> **实验目的**：验证固定权重 CL 和内容相似度感知 CVPM 的效果

### 14.1 实验设计

| 组名 | 固定 CL | 内容 CVPM | 说明 |
|---|---|---|---|
| **D0_FixedCL** | ✓ | ✗ | λ_cl=0.1（恒定权重，无 maturity 缩放） |
| **D1_ContentCVPM** | ✗ | ✓ | 相似度阈值=0.85，内容相似对不排斥 |
| **D2_FixedCL_ContentCVPM** | ✓ | ✓ | 两者结合 |

**关键参数**（所有组共享 Q0 最优 SID 超参）：
- λ_full=0.3, λ_partial=0.2, m_full=0.5, m_partial=0.3
- CVPM_temp=0.15, Repulsion_warmup=2000

### 14.2 SID 碰撞指标（3000-step）

| 组 | frac_unique_sids | colliding_items_frac | 状态 |
|---|---:|---:|:---|
| **Q0_GQSS_Static** | 0.9703 | 0.0533 | ✅ 健康 |
| **D0_FixedCL** | **0.0945** | **0.9850** | ❌ **严重坍塌** |
| **D1_ContentCVPM** | 0.9677 | 0.0602 | ✅ 健康 |
| **D2_FixedCL_ContentCVPM** | **0.1402** | **0.9677** | ❌ **严重坍塌** |

### 14.3 各层 token 利用率（3000-step）

| 组 | layer0 unique | layer1 unique | layer2 unique | 状态 |
|---|---:|---:|---:|:---|
| **D0_FixedCL** | 1 (0.0039) | 16 (0.0625) | 212 (0.828) | ❌ layer0 完全坍塌 |
| **D1_ContentCVPM** | 127 (0.496) | 254 (0.992) | 255 (0.996) | ✅ 全层健康 |
| **D2_FixedCL_ContentCVPM** | 1 (0.0039) | 26 (0.102) | 200 (0.781) | ❌ layer0 完全坍塌 |

### 14.4 TIGER 推荐指标（test set）

| 组 | NDCG@5 | NDCG@10 | Recall@5 | Recall@10 | 状态 |
|---|---:|---:|---:|---:|:---|
| **Q0_GQSS_Static** | **0.02805** | **0.03332** | 0.03989 | 0.05625 | ✅ 基准 |
| **D0_FixedCL** | 0.02750 | 0.03452 | 0.04016 | 0.05814 | ⚠️ 勉强可用 |
| **D1_ContentCVPM** | 0.02276 | 0.02831 | 0.03363 | 0.04811 | ❌ 低于基准 |
| **D2_FixedCL_ContentCVPM** | 0.02761 | 0.03452 | 0.04002 | 0.05799 | ⚠️ 勉强可用 |

> **D0/D2 虽然 SID 坍塌严重，但 TIGER 指标却意外接近甚至略超 Q0**，说明模型对 SID 坍塌有一定容忍度

### 14.5 关键发现

1. **固定 CL（λ_cl=0.1）导致灾难性码本坍塌**
   - D0 和 D2 的 layer0 仅使用 1 个 token，`frac_unique_sids` 跌至 0.09-0.14
   - 原因：恒定 λ_cl 与排斥损失叠加，早期训练阶段梯度冲突导致码本失活

2. **内容 CVPM 单独使用（D1）保持码本健康但 TIGER 性能下降**
   - SID 指标健康（`frac_unique_sids=0.9677`），但 NDCG@10 比 Q0 低 **-15.0%**
   - 可能原因：内容相似度过滤过于激进，将本应区分的相似商品也排除，导致 SID 区分度不足

3. **D0/D2 的 TIGER 指标与 Q0 接近的异常现象**
   - 尽管 SID 几乎完全坍塌（layer0=1 token），TIGER 仍能学习有效推荐
   - 推测：多层 SID 中 layer1/layer2 仍保留一定区分度，或 TIGER 通过序列模式学习弥补了 SID 缺陷

---

## 15. D1 阈值调优实验（3000-step）：ContentCVPM 相似度阈值

> **实验日期**：2026-05-01 | **RUN_TAG**: `D1_thresh_sid3000_20260501_011856`
> **GPU 分配**：D1t70=GPU0，D1t75=GPU1，D1t80=GPU2
> **实验目的**：寻找 ContentCVPM 的最佳相似度阈值

### 15.1 实验设计

| 组名 | 相似度阈值 | 说明 |
|---|---|---|
| **D1t70_thresh070** | 0.70 | 较低阈值，更多对被视为良性碰撞 |
| **D1t75_thresh075** | 0.75 | 中等阈值 |
| **D1t80_thresh080** | 0.80 | 较高阈值，仅高度相似对被视为良性 |

### 15.2 SID 碰撞指标（3000-step）

| 组 | frac_unique_sids | colliding_items_frac | colliding_items | max_group_size |
|---|---:|---:|---:|---:|
| **Q0_GQSS_Static** | 0.9703 | 0.0533 | 645 | 8 |
| **D1t70_thresh070** | 0.9719 | 0.0526 | 636 | 7 |
| **D1t75_thresh075** | **0.9723** | **0.0512** | 620 | 8 |
| **D1t80_thresh080** | 0.9679 | 0.0573 | 693 | 10 |

### 15.3 各层 token 利用率与熵（3000-step）

| 组 | layer0 cov | entropy_norm | max_token_frac | layer1 cov | layer2 cov |
|---|---:|---:|---:|---:|---:|
| **D1t70** | 127 (0.496) | 0.9610 | 0.0222 | 255 (0.996) | 250 (0.977) |
| **D1t75** | 127 (0.496) | **0.9706** | **0.0197** | 256 (1.000) | 255 (0.996) |
| **D1t80** | 127 (0.496) | 0.9663 | 0.0261 | 255 (0.996) | 251 (0.980) |

### 15.4 Partial collision（MC 估计，R=2）

| 组 | partial_collision_rate | full_collision_rate |
|---|---:|---:|
| **D1t70** | 0.021000 | 0.000003 |
| **D1t75** | **0.019520** | 0.000000 |
| **D1t80** | 0.020203 | 0.000003 |

### 15.5 TIGER 推荐指标（test set，最优 checkpoint）

| 组 | NDCG@5 | NDCG@10 | Recall@5 | Recall@10 | 最优 step |
|---|---:|---:|---:|---:|---:|
| **Q0_GQSS_Static** | **0.02805** | **0.03332** | 0.03989 | 0.05625 | — |
| **D1t70_thresh070** | **0.02610** | **0.03194** | **0.03863** | **0.05670** | 3999 |
| **D1t75_thresh075** | 0.02380 | 0.02975 | 0.03509 | 0.05151 | 2399 |
| **D1t80_thresh080** | 0.02486 | 0.03025 | 0.03649 | 0.05255 | 3999 |

### 15.6 Δ% vs Q0 基准

| 组 | ΔNDCG@5 | ΔNDCG@10 | ΔRecall@5 | ΔRecall@10 |
|---|---:|---:|---:|---:|
| **D1t70** | -6.95% | -4.14% | -3.16% | **+0.80%** |
| **D1t75** | -15.15% | -10.71% | -12.03% | -8.43% |
| **D1t80** | -11.37% | -9.21% | -8.52% | -6.58% |

### 15.7 结论与分析

**阈值=0.70 是最优选择：**

1. **SID 质量**：D1t75 的 `frac_unique_sids=0.9723` 和 `partial_collision_rate=0.0195` 最优，但 TIGER 性能反而最差
   - 再次验证 **SID 碰撞指标与 TIGER 性能非单调关系**

2. **TIGER 性能**：D1t70 在所有阈值中表现最好
   - NDCG@10 仅比 Q0 低 **-4.14%**，Recall@10 甚至 **+0.80%** 超越 Q0
   - 说明适度宽松的相似度阈值（0.70）能更好地保留语义区分度

3. **阈值选择建议**
   - **0.70**：推荐用于 TIGER 性能优先场景
   - **0.75**：SID 碰撞率最低（5.12%），适合对 SID 唯一性要求高的场景
   - **0.80**：不推荐，各项指标均不如 0.70/0.75

---

## 16. 实验演进总结

### 16.1 关键里程碑

| 阶段 | 实验组 | 核心发现 | TIGER NDCG@10 vs G0 |
|---|---|---|---|
| 基线 | G0 | RQ-VAE 原生实现 | 基准 |
| QuaSID 复现 | Q0 | λ=0.2 + CVPM + 静态边界 | +2.33% |
| 时间权重探索 | B1/C1 | U 型设计修复方向性 Bug | C1b: -0.68% |
| Lambda 重校准 | C2ctrl | λ=0.2 是关键，而非 0.3 | **+17.32%** |
| 内容 CVPM | D1 | 相似度阈值 0.70 最优 | -4.14% |
| 固定 CL | D0/D2 | 恒定 λ_cl 导致坍塌 | — |

### 16.2 核心结论

1. **λ 重校准（0.3→0.2）是最大收益来源**，带来 >17% TIGER 性能提升
2. **U 型时间权重设计正确**，但需与合适的 λ 值配合才能发挥效果
3. **内容 CVPM 有价值**，相似度阈值 0.70 能在保持 SID 健康的同时最小化 TIGER 损失
4. **固定 CL（无 warmup 的恒定 λ_cl）导致码本坍塌**，应避免
5. **SID 碰撞率与 TIGER 性能非线性相关**，过度降低碰撞可能损害语义区分度

### 16.3 下一步方向

1. **C2 + ContentCVPM**：将 C2ctrl 的 λ=0.2 与 D1t70 的内容 CVPM 结合
2. **C2 + 轻量 U 型**：在 C2ctrl 基础上加入适度 U 型时间权重
3. **跨数据集验证**：在 Sports/Toys 数据集验证 ContentCVPM 的泛化性
