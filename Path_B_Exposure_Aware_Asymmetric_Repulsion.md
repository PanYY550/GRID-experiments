# Path B: 曝光感知非对称排斥损失 (Exposure-Aware Asymmetric Repulsion Loss)

## 1. 动机：对称排斥的致命缺陷

### 1.1 背景：SID 碰撞与 HaMR 机制

在 GRID/TIGER 生成式推荐架构中，RQ-VAE 将海量物品的连续嵌入映射到离散 Semantic ID (SID) 空间（3 层，每层 256 个码字）。由于物品数量远超离散空间容量，不可避免发生 **SID 碰撞**——语义不同的物品被分配到相同或高度相似的 SID。

QuaSID/VCF 框架的 **HaMR (Hamming-guided Margin Repulsion)** 机制是当前解决碰撞的核心手段。它对批次内低汉明距离的碰撞对 $(i, j)$ 施加基于 Hinge Loss 的几何排斥：

$$L = \max(0, \text{margin} - \text{cosine\\_distance}(z_i, z_j))$$

### 1.2 根本缺陷：梯度的完全对称性

HaMR 的梯度满足严格的对称关系：

$$\frac{\partial L}{\partial z_i} = -\frac{\partial L}{\partial z_j}$$

碰撞双方承受数值绝对相等、方向完全相反的排斥梯度。在推荐系统的**长尾幂律分布**下，这种对称性是灾难性的。

### 1.3 梯度议价能力不对称 (Gradient Bargaining Power Asymmetry)

不同物品在梯度网络中处于完全不平等的地位：

| 物品类型 | 正样本锚定 | 梯度议价能力 | 对排斥梯度的响应 |
|----------|-----------|-------------|-----------------|
| **热门物品** (Head) | 海量用户交互正样本 | 极高 | 能轻易抵消排斥，保持位置不变 |
| **长尾物品** (Tail) | 极少或无正样本 | 极低 | 无抵抗能力，被迫承受全部位移 |

**对称排斥的结果**：热门物品纹丝不动，长尾物品被无情推向嵌入空间边缘 → **表征漂移 (Representation Drift)**。

### 1.4 实证证据

基于 Amazon Beauty 数据集的 15+ 组消融实验：

1. **λ 从 0.3 降到 0.2** → NDCG@10 **+17.32%**（减弱全局排斥 = 减少对长尾的伤害）
2. **碰撞率与性能脱钩**：多组实验中，更低碰撞率反而对应更差推荐性能
3. **Layer 0 坍缩容忍**：顶层码本坍缩至 1 个 Token，TIGER 依然强劲（浅层容错空间巨大）
4. **方向性验真**：削弱热门排斥 → NDCG **-3.04%**；增强热门排斥 → 恢复至 **-0.68%**

---

## 2. 核心创新点

### 2.1 理论创新

**首次揭示**生成式推荐中 SID 碰撞排斥的**梯度议价能力不对称性**，并指出对称排斥机制系统性地牺牲长尾物品表征质量。

### 2.2 方法创新

提出**曝光感知的非对称排斥损失**，利用曝光量比值动态分配碰撞对的梯度负担：

- **热门物品承受全量排斥梯度**（承担解决碰撞的主力）
- **长尾物品梯度被部分阻断**（受保护，避免表征漂移）

### 2.3 工程创新

- **极简实现**：仅 ~140 行 PyTorch 代码，零额外参数，零额外算力开销
- **平滑退化**：当碰撞双方曝光量趋同时自动退化为对称行为
- **安全机制**：内置 α_min 下界，防止极端长尾物品梯度完全消失导致死码

### 2.4 跨领域映射

本方案与以下领域的前沿工作形成理论映射：

- **CV 领域**：非对称损失 (ASL) 处理长尾多标签分类
- **对比学习**：Aligned Contrastive Loss (ACL) 解决长尾梯度冲突
- **度量学习**：非对称距离 (PAD) 在非均匀流形上的几何合法性

---

## 3. 数学方案

### 3.1 梯度分配系数 α

对碰撞对 $(i, j)$，设 $\text{exposure}[i] \geq \text{exposure}[j]$：

$$\alpha = \text{clamp}\left(\frac{\text{exposure}[j]}{\text{exposure}[i] + \text{exposure}[j]}, \ \alpha_{\min}, \ 0.5\right)$$

- 当两物品曝光量趋同时，$\alpha \to 0.5$，平滑退化为对称行为
- 当曝光量悬殊时，$\alpha \to \alpha_{\min}$，长尾物品获得最大保护
- $\alpha_{\min} = 0.05$（推荐值），确保极端长尾仍有微量梯度流防死码

### 3.2 前向传播：部分梯度阻断

```python
z_hot = ...   # 高曝光物品的连续嵌入（全梯度）
z_tail = ...  # 低曝光物品的连续嵌入（需保护）

# Partial gradient detachment
z_tail_mixed = α · z_tail + (1 - α) · detach(z_tail)

# 非对称余弦距离
distance = 1 - cosine_sim(normalize(z_hot), normalize(z_tail_mixed))
loss = ReLU(margin - distance)
```

### 3.3 反向传播：非对称梯度流

- **热门物品 i**：$\frac{\partial L}{\partial z_i} \propto \text{full}$ — 承担排斥主力
- **长尾物品 j**：$\frac{\partial L}{\partial z_j} = \alpha \cdot \frac{\partial L}{\partial z_{j\\_mixed}}$ — 梯度被衰减到 $\alpha$ 倍

### 3.4 动力学意义

热门物品凭借海量正样本的弹性锚定，发生微小位移即可满足排斥条件，并迅速通过正样本梯度恢复。长尾物品受到强有力的梯度保护，坚守原始语义流形，杜绝表征漂移。

---

## 4. 实现要点

### 4.1 代码结构

- **文件**：`src/modules/clustering/residual_quantization.py`
- **新增方法**：`_asymmetric_repulsion_loss()` (~140 行)
- **修改点**：`__init__` 新增 3 个参数 + `vcf_repulsion_loss` Stage 7 条件分支
- **配置文件**：`configs/experiment/rqvae_vcf_train_flat.yaml` 新增 3 个参数

### 4.2 关键设计决策

| 决策 | 选择 | 理由 |
|------|------|------|
| 操作位置 | Stage 7 (hinge loss 计算) | 已有明确碰撞对索引 (i,j)，可直接对 pair 级别操作 |
| 阻断方式 | partial detach | 改变 gradient flow topology，而非仅调节 loss magnitude |
| 距离函数 | 非对称余弦距离 | z_hot 全梯度 + z_tail_mixed 部分梯度 |

### 4.3 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `use_asymmetric_repulsion` | `false` | 是否启用非对称排斥 |
| `asymmetric_alpha_min` | `0.05` | α 安全下界，防止死码 |
| `asymmetric_temperature` | `1.0` | 温度平滑参数（保留接口） |

---

## 5. 实验设计

### 5.1 三卡对照实验

| 实验组 | GPU | 配置 | α_min | 目的 |
|--------|-----|------|-------|------|
| **B0ctrl** | GPU0 | 对称排斥（同 C2ctrl 基线） | N/A | 对照组 |
| **B0** | GPU1 | 非对称排斥 | 0.05 | 推荐安全下界 |
| **B0a** | GPU2 | 非对称排斥（极端保护） | 0.01 | 验证方向性 |

### 5.2 共享超参

| 参数 | 值 |
|------|-----|
| λ_full | 0.2 |
| λ_partial | 0.1 |
| m_full | 0.5 |
| m_partial | 0.3 |
| hamming_radius | 2 |
| repulsion_warmup_steps | 2000 |
| use_dynamic_margin | false |
| use_time_decay | false |
| use_tcl | false |
| cvpm_temperature | 0.15 |
| codebook_width | 256 |
| num_hierarchies | 3 |
| SID max_steps | 3000 |
| batch_size_per_device | 4096 |

### 5.3 评估指标

| 维度 | 指标 |
|------|------|
| **主指标** | TIGER NDCG@10, Recall@10 |
| **碰撞分析** | 全碰撞率、部分碰撞率 (H≤2)、碰撞组分布 |
| **码本健康** | 逐层唯一 Token 数、覆盖率、熵 |
| **分组公平性** (待实现) | 按曝光分位 Head/Torso/Tail 分组 NDCG |
| **表征位移** (待实现) | 训练前后 ‖z_t - z_0‖ |

### 5.4 预期判断

- **若 B0/B0a NDCG@10 > B0ctrl** → Path B 假设成立
- **若 B0a > B0 > B0ctrl** → 方向性确认，极端保护带来更大收益
- **若 B0/B0a ≤ B0ctrl** → 假设需重新审视

---

## 6. 风险评估与防御

### 6.1 "永恒碰撞"陷阱

**风险**：α → 0 使长尾物品完全停止移动，永久与热门物品共享拥挤空间。

**防御**：热门物品承受全量梯度后主动移开，相对位移同样解决碰撞。长尾不动 + 热门主动避让 = 碰撞解除。

### 6.2 码本坍缩

**风险**：梯度阻断导致部分码字长期无更新，引发死码。

**防御**：
- 内置 α_min 安全下界（默认 0.05），保证梯度涓流
- Layer 0 坍缩实验证明浅层容错空间远超预期

### 6.3 曝光特征静态滞后

**风险**：全局静态曝光量无法反映物品生命周期动态变化。

**防御**（论文/工程演进方向）：引入滑动窗口 EMA 曝光热度，替代静态全局计数，使冷启动爆款物品自动解绑梯度保护。

---

## 7. 冒烟测试结果 (SID_MAX_STEPS=400)

### 7.1 碰撞分析

| 指标 | B0ctrl (对称) | B0 (α_min=0.05) | B0a (α_min=0.01) |
|------|:--:|:--:|:--:|
| 碰撞物品占比 | 10.58% | 11.86% | 13.45% |
| 碰撞组数 | 562 | 611 | 685 |
| 最大碰撞组 | 6 | 8 | 9 |
| Partial (H≤2) | 2.67% | 2.65% | 2.69% |

### 7.2 码本利用率

| 层 | B0ctrl | B0 | B0a |
|----|:--:|:--:|:--:|
| Layer 0 | 41.8% | 42.2% | 40.2% |
| Layer 1 | 91.4% | 94.5% | 92.2% |
| Layer 2 | 89.8% | 94.9% | 92.6% |

**结论**：碰撞率随 α_min 降低呈单调递增（符合理论预期），三组码本利用健康，无坍缩。

---

## 8. 文件清单

| 文件 | 变更 | 说明 |
|------|------|------|
| `src/modules/clustering/residual_quantization.py` | 修改 +~140 行 | 核心实现 |
| `configs/experiment/rqvae_vcf_train_flat.yaml` | 修改 +3 行 | 参数声明 |
| `scripts/run_pathB_experiment.sh` | 新建 ~340 行 | 一键实验脚本 |

---

## 9. 相关文献映射

| 领域 | 工作 | 关联 |
|------|------|------|
| 生成式推荐 | QuaSID (arXiv 2603.00632) | 被改进的基线方法 |
| 生成式推荐 | AdaSID (arXiv) | 语义适应排斥，未涉及流行度 |
| 生成式推荐 | CRAB (2026) | 后处理码本重平衡，非训练期干预 |
| 生成式推荐 | RAD-DPO (2026) | Token 级梯度阻断保护共享前缀 |
| 长尾分类 | ASL (CVPR 2021) | 非对称焦点因子衰减负样本梯度 |
| 对比学习 | ACL (CVPRW 2025) | 长尾梯度冲突的数学分析 |
| 度量学习 | PAD (AAAI 2024) | 非对称距离的几何合法性 |
