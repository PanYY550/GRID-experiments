# Direction 2: Layer-wise Differential Repulsion & NPR — 实施方案

> **日期**: 2026-05-19
> **状态**: 方案设计
> **文献支撑**: Hourglass Phenomenon (arXiv:2407.21488), Hi-SAM (arXiv:2602.11799), FedMM (arXiv:2605.11433)

---

## 1. 问题陈述

### 1.1 观察

当前 NPR+TCL 系统中，3 层 RQ-VAE 的 codebook 利用率存在巨大差异：

| 层 | 利用率 | 功能 |
|:---|:---:|:---|
| L0 | 91.0% | 粗粒度语义（品类级别） |
| L1 | 68.8% | 中粒度语义 |
| L2 | **35.2%** | 细粒度语义（item 级别区分） |

所有层使用**完全相同的** VCF margin (m_full=0.8, m_partial=0.5) 和 NPR α (0.01)。

### 1.2 文献证据

**Hourglass 论文** (arXiv:2407.21488) 独立验证了这一现象：RQ-VAE 用于 Semantic ID 时，中间层 token 分布出现沙漏形——L1 宽、L2 窄、L3 宽。根因是残差量化的动力学：L0 聚类后，大部分 item 的残差集中在少数方向 → L2 可区分的信号量不足 → token 集中到少数几个。Focal loss 被证明无效——"这是结构性问题，不是 loss 权重问题"。

**Hi-SAM** 明确指出 RQ-VAE "不同层次之间缺乏解耦，shared semantics 和 modality-specific details 混杂"。

### 1.3 核心假设

不同语义层级需要**不同的排斥强度和保护力度**：

- **L0 (粗粒度)**：item 应该共享大类 token（如"护肤品""彩妆"）→ 强 NPR 保护，弱 margin
- **L1 (中粒度)**：适中的区分度 → 标准参数
- **L2 (细粒度)**：必须区分具体 item → 弱 NPR 保护，强 margin → 激活更多 L2 token

**预期效果**：L2 利用率 35.2% → 50%+，同时 L0 维持在 90% 左右，TIGER NDCG@10 提升。

---

## 2. 核心设计

### 2.1 从 SID 级碰撞 → Per-Layer 碰撞

**当前逻辑**（HaMR分类）：

```
H(i,j) = Σ_l 1[sid_i[l] ≠ sid_j[l]]          # Hamming距离
Ω_full:  H=0      → m_full=0.8, λ=0.2       # 所有层碰撞
Ω_partial: 0<H≤1  → m_partial=0.5, λ=0.1    # 2层碰撞
```

**新逻辑**（Per-Layer Mask）：

```
对每一层 l ∈ {0, 1, 2}:
  M_l[i,j] = (sid_i[l] == sid_j[l]) & CVPM[i,j]    # 该层是否碰撞
  L_l = λ_l · mean(relu(m_l - D[i,j]) · M_l[i,j])   # 该层排斥损失

L_total = L_0 + L_1 + L_2
```

**关键变化**：
- 3 层全碰撞 (H=0) 的 pair 现在收到 L0+L1+L2 三个损失项 → 排斥力自然更强
- H=1 的 pair 只从其碰撞的 2 层收到损失 → 排斥力自然更弱
- **不需要**手动区分 full vs partial——per-layer 机制自动编码了严重程度

### 2.2 Per-Layer NPR 梯度调制

**当前逻辑**：α_eff = npr_alpha_min (固定 0.01) if j ∈ kNN(i) else 1.0

**新逻辑**：

```
对每一层 l:
  α_eff_l[i,j] = npr_alpha_Ll  if (j ∈ kNN(i) AND M_l[i,j]) else 1.0

  z_i_mixed_l = α_eff_l · z_i + (1-α_eff_l) · sg[z_i]
  D_l[i,j] = 1 - cos(z_i_mixed_l, z_j)
  hinge_l = relu(m_Ll - D_l) · M_l · time_weight
```

**含义**：同一个 item pair 在不同层可以有不同的保护力度：

| 场景 | L0 α | L1 α | L2 α | 含义 |
|:---|:---:|:---:|:---:|:---|
| 碰撞 L0+L1+L2, j∈kNN(i) | 0.005 | 0.01 | 0.05 | L0 强保护, L2 弱保护 |
| 仅碰撞 L2, j∈kNN(i) | — | — | 0.05 | 仅在细粒度层保护 |
| 碰撞 L0, j∉kNN(i) | 1.0 | — | — | 不保护，全排斥 |

### 2.3 与现有 VCF pipeline 的关系

当 `use_layerwise_repulsion=false` 时，**完全走现有逻辑**，零影响。

当 `use_layerwise_repulsion=true` 时，**替代** `vcf_repulsion_loss` 中的 Stage 2-7，复用：
- Stage 1: CVPM 掩码（不变）
- Stability gate（不变）
- Pair subsampling（per-layer 执行）
- Dual-tower 架构（不变）

---

## 3. 参数设计

### 3.1 新增参数一览

```yaml
# ──── Direction 2: Layer-wise Differential Repulsion ────
use_layerwise_repulsion: false       # 总开关

# Per-layer VCF margins (当 items 在该层共享 token 时应用)
m_rep_L0: 0.67                       # L0 margin
m_rep_L1: 0.67                       # L1 margin
m_rep_L2: 0.67                       # L2 margin

# Per-layer repulsion weights
lambda_rep_L0: 0.20                  # L0 排斥权重
lambda_rep_L1: 0.10                  # L1 排斥权重
lambda_rep_L2: 0.10                  # L2 排斥权重

# Per-layer NPR alpha (梯度保护强度, 越小越保护)
npr_alpha_L0: 0.01                   # L0 NPR α
npr_alpha_L1: 0.01                   # L1 NPR α
npr_alpha_L2: 0.01                   # L2 NPR α
```

### 3.2 实验推荐值

基于"L0 粗=宽松, L2 细=严格"的直觉：

```yaml
use_layerwise_repulsion: true

m_rep_L0: 0.6        # 比默认 0.67 低 → L0 可容忍更近的距离
m_rep_L1: 0.8        # 比默认 0.67 高 → L1 标准排斥
m_rep_L2: 1.0        # 比默认 0.67 高 → L2 强排斥，激活更多 token

lambda_rep_L0: 0.15  # L0 排斥权重
lambda_rep_L1: 0.10  # L1 排斥权重
lambda_rep_L2: 0.15  # L2 排斥权重（加强细粒度区分）

npr_alpha_L0: 0.005  # L0 强保护（允许共享粗粒度语义）
npr_alpha_L1: 0.01   # L1 标准保护
npr_alpha_L2: 0.05   # L2 弱保护（允许 VCF 在细粒度层推开邻居）
```

### 3.3 参数可视化

```
Layer 0 (Coarse):    m=0.6  α=0.005  λ=0.15   ← 强保护, 弱排斥
Layer 1 (Medium):    m=0.8  α=0.01   λ=0.10   ← 标准
Layer 2 (Fine):      m=1.0  α=0.05   λ=0.15   ← 弱保护, 强排斥

        强保护                              弱保护
        ←─────── NPR α ──────────→
L0:     0.005    0.01    0.05              L2
        弱排斥                              强排斥
        ←─────── margin ──────────→
L0:     0.6      0.8      1.0              L2
```

---

## 4. 实现步骤

### 4.1 涉及文件

| 文件 | 操作 | 改动量 |
|:---|:---|:---|
| `src/modules/clustering/residual_quantization.py` | `__init__` 新增参数 + 新增 `vcf_repulsion_loss_layerwise` + 修改 `model_step` 调用 | ~100 行 |
| `src/modules/clustering/knn_preservation.py` | 新增 `build_npr_mask_per_layer` | ~20 行 |
| `configs/experiment/rqvae_vcf_online_knn_train_flat.yaml` | 新增参数块 | ~10 行 |

### 4.2 Step 1: `residual_quantization.py` — `__init__` 新增参数

在现有 NPR 参数块后面（~line 164）添加：

```python
# === Direction 2: Layer-wise Differential Repulsion & NPR ===
self.use_layerwise_repulsion = kwargs.get("use_layerwise_repulsion", False)
self.m_rep_L0 = kwargs.get("m_rep_L0", 0.67)
self.m_rep_L1 = kwargs.get("m_rep_L1", 0.67)
self.m_rep_L2 = kwargs.get("m_rep_L2", 0.67)
self.lambda_rep_L0 = kwargs.get("lambda_rep_L0", 0.20)
self.lambda_rep_L1 = kwargs.get("lambda_rep_L1", 0.10)
self.lambda_rep_L2 = kwargs.get("lambda_rep_L2", 0.10)
self.npr_alpha_L0 = kwargs.get("npr_alpha_L0", 0.01)
self.npr_alpha_L1 = kwargs.get("npr_alpha_L1", 0.01)
self.npr_alpha_L2 = kwargs.get("npr_alpha_L2", 0.01)
```

### 4.3 Step 2: `residual_quantization.py` — 新增 `vcf_repulsion_loss_layerwise`

在 `vcf_repulsion_loss` 方法之后添加新方法。核心逻辑：

```python
def vcf_repulsion_loss_layerwise(
    self,
    continuous_emb: torch.Tensor,       # (B, D)
    sid_tokens: torch.Tensor,           # (B, L)
    item_ids: torch.Tensor,             # (B,)
    exposure_times: torch.Tensor,       # (B,)
    content_embeddings: torch.Tensor,   # (B, D_in)
    positive_pair_matrix: Optional[torch.Tensor] = None,
    target_embeddings: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, dict]:
    """
    Layer-wise VCF repulsion loss.

    Key difference from vcf_repulsion_loss:
      - Instead of classifying collisions as full/partial via Hamming distance,
        computes per-layer collision masks from SID tokens directly.
      - Each layer has its own margin, λ weight, and NPR α.
    """
    B = continuous_emb.shape[0]
    device = continuous_emb.device
    if B < 2:
        return torch.tensor(0.0, device=device), {}

    # ---- Stage 1: CVPM mask (reuse existing logic) ----
    M_cvpm = self.cvpm_mask(item_ids, positive_pair_matrix,
                            content_embeddings=content_embeddings)

    # ---- Stage 2: Per-layer collision masks ----
    masks = []
    for l in range(self.n_layers):
        M_l = (sid_tokens[:, l].unsqueeze(0) == sid_tokens[:, l].unsqueeze(1))
        M_l = M_l & M_cvpm
        M_l.fill_diagonal_(False)
        masks.append(M_l)

    # ---- Stability gate (reuse existing) ----
    # ... (same as current vcf_repulsion_loss)

    # ---- Stage 3: Cosine distance (once, dual-tower or single) ----
    if target_embeddings is not None:
        norm_online = F.normalize(continuous_emb, p=2, dim=-1)
        norm_target = F.normalize(target_embeddings, p=2, dim=-1)
        D_base = 1.0 - torch.mm(norm_online, norm_target.t())
    else:
        norm_emb = F.normalize(continuous_emb, p=2, dim=-1)
        D_base = 1.0 - torch.mm(norm_emb, norm_emb.t())

    # ---- Stage 4: Per-layer repulsion ----
    margins = [self.m_rep_L0, self.m_rep_L1, self.m_rep_L2]
    lambdas = [self.lambda_rep_L0, self.lambda_rep_L1, self.lambda_rep_L2]
    alphas = [self.npr_alpha_L0, self.npr_alpha_L1, self.npr_alpha_L2]

    L_total = torch.tensor(0.0, device=device)
    stats = {}

    for l in range(self.n_layers):
        M_l = masks[l]
        n_pairs = M_l.sum().item()
        stats[f"n_L{l}"] = n_pairs

        if n_pairs == 0:
            continue

        # Find collision indices for this layer
        i_idx, j_idx = torch.where(M_l)

        # Per-layer NPR gradient modulation
        if self.use_npr and self._get_knn_buffer(device) is not None:
            knn_buf = self._get_knn_buffer(device)
            alpha_mask = build_npr_mask(
                knn_buf, item_ids, npr_alpha_min=alphas[l]
            )
            alpha_eff = alpha_mask[i_idx, j_idx]
        else:
            alpha_eff = torch.ones(n_pairs, device=device)

        # Gradient modulation: z_i_mixed = α·z_i + (1-α)·sg[z_i]
        z_i = continuous_emb[i_idx]
        z_j = (target_embeddings if target_embeddings is not None
               else continuous_emb)[j_idx]

        z_i_mixed = (alpha_eff.unsqueeze(1) * z_i
                     + (1.0 - alpha_eff.unsqueeze(1)) * z_i.detach())

        # Asymmetric cosine distance
        z_i_norm = F.normalize(z_i_mixed, p=2, dim=-1)
        z_j_norm = F.normalize(z_j, p=2, dim=-1)
        cos_sim = (z_i_norm * z_j_norm).sum(dim=-1)
        D_asym = 1.0 - cos_sim

        # Hinge loss with layer-specific margin
        hinge = F.relu(margins[l] - D_asym)
        L_l = hinge.mean()
        L_total = L_total + lambdas[l] * L_l

        stats[f"L_L{l}"] = L_l.item()
        stats[f"alpha_L{l}_mean"] = alpha_eff.mean().item()

    # Optional clipping
    if self.vcf_repulsion_clip and self.vcf_repulsion_clip > 0:
        L_total = torch.clamp(L_total, max=float(self.vcf_repulsion_clip))

    return L_total, stats
```

### 4.4 Step 3: `model_step` — 修改 VCF 调用

在 line 1381 附近，添加分支：

```python
if self.use_vcf:
    if self.use_layerwise_repulsion:
        vcf_loss, vcf_stats = self.vcf_repulsion_loss_layerwise(
            continuous_emb=encoded_embeddings,
            sid_tokens=cluster_ids,
            item_ids=item_ids,
            exposure_times=exposure_times,
            content_embeddings=input_embeddings,
            positive_pair_matrix=positive_pair_matrix,
            target_embeddings=target_embeddings,
        )
        n_full = vcf_stats.get("n_L0", 0) + vcf_stats.get("n_L1", 0) + vcf_stats.get("n_L2", 0)
        n_partial = 0  # not applicable in layerwise mode
    else:
        vcf_loss, n_full, n_partial = self.vcf_repulsion_loss(...)
```

### 4.5 Step 4: `knn_preservation.py` — 无需修改

`build_npr_mask` 已经接受 `npr_alpha_min` 参数，直接传入 per-layer 的 α 值即可。无需新增函数。

### 4.6 Step 5: 配置文件

```yaml
# configs/experiment/rqvae_vcf_online_knn_train_flat.yaml

# ──── Direction 2: Layer-wise Differential Repulsion ────
use_layerwise_repulsion: false
m_rep_L0: 0.67
m_rep_L1: 0.67
m_rep_L2: 0.67
lambda_rep_L0: 0.20
lambda_rep_L1: 0.10
lambda_rep_L2: 0.10
npr_alpha_L0: 0.01
npr_alpha_L1: 0.01
npr_alpha_L2: 0.01
```

---

## 5. 实验方案

### 5.1 Phase 1: 默认参数验证 (400 步 smoke test)

**目的**：验证 layerwise 在默认参数下与当前行为一致（正确性检查）

```bash
model.use_layerwise_repulsion=true \
model.m_rep_L0=0.67 model.m_rep_L1=0.67 model.m_rep_L2=0.67 \
model.lambda_rep_L0=0.20 model.lambda_rep_L1=0.10 model.lambda_rep_L2=0.10 \
model.npr_alpha_L0=0.01 model.npr_alpha_L1=0.01 model.npr_alpha_L2=0.01
```

检查指标：L0/L1/L2 利用率、碰撞率、vcf_loss 量级与 baseline 一致。

### 5.2 Phase 2: 实验参数 (1000 步 SID + TIGER)

```bash
model.use_layerwise_repulsion=true \
model.m_rep_L0=0.6 model.m_rep_L1=0.8 model.m_rep_L2=1.0 \
model.lambda_rep_L0=0.15 model.lambda_rep_L1=0.10 model.lambda_rep_L2=0.15 \
model.npr_alpha_L0=0.005 model.npr_alpha_L1=0.01 model.npr_alpha_L2=0.05
```

其余参数与 NPR+TCL 4g baseline 完全对齐：
```
batch=256, R=1, λ_cl=0.1, τ=0.5, Adam(lr=3e-4, wd=1e-5), warmup=0
use_npr=true, use_tcl=true, use_online_knn=true
codebook_entropy_weight=0.1, codebook_reset_interval=10
```

### 5.3 成功标准

| 指标 | Baseline (NPR+TCL) | 目标 | 说明 |
|:---|:---:|:---:|:---|
| TIGER test/ndcg@10 | 0.0355 | ≥ 0.0355 | 至少不退化 |
| L2 覆盖率 | 35.2% | ≥ 45% | 核心优化目标 |
| L0 覆盖率 | 91.0% | ≥ 85% | 允许微降但不过度 |
| max_collision | 7 | ≤ 20 | 不能爆增 |
| DSF_online | 0.175 | ≥ 0.170 | 语义保真度不显著退化 |

### 5.4 Ablation 路径

如果主实验有效，逐步消融以确定哪个组件起作用：

| Ablation | m_L0/m_L1/m_L2 | α_L0/α_L1/α_L2 | λ_L0/λ_L1/λ_L2 |
|:---|:---:|:---:|:---:|
| A: Full layerwise | 0.6/0.8/1.0 | 0.005/0.01/0.05 | 0.15/0.10/0.15 |
| B: margin only | 0.6/0.8/1.0 | 0.01/0.01/0.01 | 0.15/0.10/0.15 |
| C: NPR only | 0.67/0.67/0.67 | 0.005/0.01/0.05 | 0.15/0.10/0.15 |
| D: L2 only (仅改 L2) | 0.67/0.67/1.0 | 0.01/0.01/0.05 | 0.20/0.10/0.15 |

---

## 6. 风险分析

### 6.1 技术风险

| 风险 | 概率 | 影响 | 缓解策略 |
|:---|:---:|:---:|:---|
| L2 强排斥导致 max_collision 爆增 | 中 | 高 | smoke test 先跑 400 步，monitor max_collision；爆增则降低 m_rep_L2 |
| L0 弱 margin 导致 L0 覆盖率崩塌 | 低 | 中 | L0 margin 仅从 0.67→0.6，降幅温和；ablation 验证 |
| 残差连锁反应 (L2 变化→L0/L1 受影响) | 中 | 中 | 这是 RQ-VAE 固有特性，只能观察；encoder 有一定适应能力 |
| per-layer NPR 引入梯度噪声 | 低 | 低 | 每层仍只处理碰撞对，计算量可控 |

### 6.2 失败模式

**最可能的失败模式**：L2 利用率未显著提升

根因可能是：残差信号的方差确实不足以支持更多 L2 token 的区分。此时即使 margin 增大，encoder 也无法在 L2 残差中制造有意义的差异，只是产生噪声。这可以通过观察 L2 熵和 TIGER NDCG 来判断——如果 margin 增大但 L2 利用率和 NDCG 都不变，则方向 2 无效。

### 6.3 为什么不重蹈 KPL/RNCL

| 维度 | KPL/RNCL (失败) | Layerwise (本方案) |
|:---|:---|:---|
| 机制 | 引入新吸引力损失 | 仅调整已有 VCF 排斥力的层次分布 |
| 信号来源 | 外部 (k-NN graph) | 内部 (SID collision 本身) |
| 新增损失项 | 是 | **否** — 仍然只是 VCF 排斥 |
| 梯度冲突风险 | 高 (与 TCL 拉-推对抗) | **低** — 不改变力的方向，只改变层次分配 |
| 超参数数 | 3-4 个新参数 | 9 个 (但有清晰的层次递减结构) |

---

## 7. 与 Hourglass 论文方案的关系

| 维度 | Hourglass 论文方案 | Layerwise (本方案) |
|:---|:---|:---|
| 解决思路 | 后处理：移除 L2 集中 token / 变长 code | 训练时：通过差异化 margin 预防集中 |
| 侵入性 | 高 (改变 codebook 结构) | 低 (仅改变训练信号) |
| 与现有 pipeline 兼容 | 需要修改 SID 推理逻辑 | 完全兼容 |
| 可逆性 | 不可逆 (移除了 token) | 可逆 (改参数即可) |

Layerwise 方案可以视为 Hourglass 问题的**预防性方案**——在训练阶段就对抗 token 集中，而非事后修复。

---

## 8. 关键监控指标 (per step)

训练时新增日志输出：

```
[LAYERWISE VCF] step=500 | 
  L0: n=284 m=0.60 α_mean=0.015 L=0.034 |
  L1: n=196 m=0.80 α_mean=0.022 L=0.028 |
  L2: n=112 m=1.00 α_mean=0.068 L=0.041 |
  total_L=0.0083 | max_col=7
```

关键观察：
- `n_L2` (L2 碰撞对数量) 应随训练逐渐增加（更多 L2 token 被激活）
- `α_mean_L2` 应高于 L0（L2 层保护更弱）
- `max_col` 不应爆增（< 20）
