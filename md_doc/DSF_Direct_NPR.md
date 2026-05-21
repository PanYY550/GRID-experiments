# DSF-Direct：Neighbor-Preserving Repulsion (NPR)

**核心方案**: 模块 A (NPR) — 已验证有效。

KPL 经实验验证对 TIGER downstream 造成伤害（NPR_TCL_KPL NDCG@10=0.0321 < TCL_only=0.0341），已排除。

---

## Context

**已闭合的因果链**：α → Asymmetric Gradient → ECRD → DSF → NDCG

**关键发现**：
1. ECRD 和 DSF 仅弱相关 (r ≈ -0.25) — 漂移量 ≠ 保真度损失
2. 漂移的**方向**比漂移的**量**更重要 — 沿局部流形漂移不破坏 k-NN
3. 当前 α-detachment 是**间接**机制 — 按 popularity 调梯度, 不是按语义邻居保护需求调梯度

**原假设 vs 现实**：
- 原假设：对称排斥 → tail 被推远 → tail 漂移大 → 性能差
- 现实：不论 head/tail，**谁的 k-NN 被破坏了，谁的 downstream hit rate 就低**

**新假设**：直接保护 k-NN 结构比间接的 exposure-based α-detachment 更有效。

---

## 方案：Neighbor-Preserving Repulsion (NPR) — 替代 α-detachment

**核心思想**：α-detachment 按 popularity 保护所有 tail items（间接/粗粒度），NPR 按语义邻域关系精确保护"排斥会破坏其 k-NN 结构"的 items（直接/细粒度）。不是"tail 少推一点"，而是"会破坏语义邻居的才少推一点"。

### 梯度调制机制

NPR 的核心操作是**梯度缩放**（gradient scaling），而非完全阻断梯度。对每个碰撞对 (i, j)，在 item i 侧施加缩放因子 α：

```
z_i_mixed = α · z_i + (1 - α) · detach(z_i)
```

其中 α 由 k-NN 邻近关系决定：

- **j ∈ kNN(i)**：j 是 i 的语义邻居 → 强排斥会破坏 i 的邻域结构 → α = `npr_alpha_min`（默认 0.01），梯度降至 1%
- **j ∉ kNN(i)**：j 是 i 的语义远邻 → 安全排斥，不会破坏 k-NN → α = 1.0，全梯度

这种"混合嵌入"技巧的关键在于：`detach(z_i)` 提供前向值通路（保持余弦距离计算的数值正确），而 `α·z_i` 控制梯度回传的比例。当 α=0.01 时，i 侧只接收 1% 的排斥梯度，几乎不被推开——但其语义邻居 j 仍被完全保护。

**双塔模式**（`use_dual_tower=true`）：z_i 来自 online encoder（梯度通过 NPR 缩放），z_j 来自 target encoder（detach，永不接收梯度）。这确保排斥是单向的——只推 online 侧，target 侧始终锚定。

### build_npr_mask 算法

实现在 `src/modules/clustering/knn_preservation.py:30-70`：

1. **查表**：从预计算的 k-NN 矩阵 `knn_g0_indices[N, K]` 中取 batch 内每个 item 的 K 个邻居（全局 item ID）
2. **广播比对**：`knn_global_expanded[B, 1, K] == j_global[1, B, 1]` → `is_neighbor[B, B]` bool 矩阵
3. **赋值**：`alpha[B, B]` 初始化为 1.0，语义邻居对 (i, j) 赋值为 `npr_alpha_min`
4. **安全措施**：对角线强制为 1.0（不自保护），CVPM mask 在上游已过滤

时间复杂度 O(B²·K)，对典型 batch size (256) 和 K=50 可忽略。

### k-NN 来源：静态 G0 vs 在线 EMA

NPR 支持两种 k-NN 来源：

| 维度 | 静态 G0 k-NN | 在线 EMA k-NN |
|------|-------------|--------------|
| 数据 | 预计算文件 `knn_g0_indices.pt` | 运行时 EMA encoder outputs |
| 更新 | 固定不变（G0 推理结果） | 每步 EMA 更新，每 100 步全量重算 |
| 优势 | 无计算开销，G0 参考点稳定 | 跟踪训练中表征变化 |
| 参数 | — | `online_knn_ema_momentum=0.99`, `online_knn_update_interval=100`, `online_knn_k=50` |

**在线 k-NN 机制**（`residual_quantization.py:1418-1491`）：
- **EMA buffer** `[N, D]`：每步用 momentum=0.99 更新当前 batch 中 item 的 encoder 输出
- **首次出现**：直接赋值（无 EMA），之后开始指数滑动平均
- **周期更新**：每 `online_knn_update_interval` 步，对所有 seen items 做 all-pairs 余弦相似度 → top-K → 更新 `online_knn_indices`
- **k-NN 切换**：`_get_knn_buffer()` 优先返回在线 k-NN（若已更新过），否则回退到 G0 静态 k-NN

### 与 VCF 的共存方式

NPR 和 VCF 共用 `_asymmetric_repulsion_loss` 函数（`residual_quantization.py:801-894`）：

1. VCF 负责**检测碰撞对**（full collision / partial collision via CVPM），计算 margin 和权重
2. NPR 负责**调制碰撞对的排斥梯度**——仅替换 α 的计算逻辑：
   - `use_npr=true` 时：`alpha_eff = build_npr_mask(knn_buffer, item_ids)` → k-NN proximity 决定
   - `use_npr=false` 时：`alpha_eff = f(exposure_counts)` → popularity ratio 决定（原有逻辑）
3. 梯度缩放后的 z_i_mixed 进入 VCF 的余弦距离计算，后续 margin-based hinge loss 不变

这种设计使 NPR 成为 VCF 的**即插即用梯度调制器**，不改变 VCF 的碰撞检测、margin 机制或损失函数形式。

### 与 α-detachment 的深度对比

| 维度 | α-detachment | NPR |
|------|-------------|-----|
| 调制依据 | exposure count（协同过滤统计量） | k-NN proximity（表征拓扑） |
| 粒度 | per-item（所有碰撞对中 item 的 α 相同） | per-pair（每个碰撞对独立判断） |
| 保护范围 | 所有低曝光 item（tail） | 仅 k-NN-at-risk 的 pairs |
| 副作用 | 热门 item 之间的碰撞也被减弱的排斥保护 | 仅保护语义脆弱的 pairs |
| 理论基础 | "tail 需要保护"（经验假设） | "k-NN 结构破坏导致 downstream 性能下降"（因果链已验证） |
| 可解释性 | α=0.01 for all tail items — 为什么是 0.01？ | α=0.01 only if j∈kNN(i) — 为什么是这对？因为 j 是语义邻居 |

**关键洞察**：α-detachment 的"tail 保护"是一种统计代理（proxy）——假设低曝光 = 需要保护。但因果分析显示，真正损害 downstream 性能的不是"tail 漂移"，而是"k-NN 结构破坏"。NPR 直接针对后者，因此更精确、副作用更小。

### 实验验证

4g 实验中 NPR+TCL vs TCL_only 的直接对比（3000 步 SID + TIGER，所有其他参数相同）：

| 指标 | TCL_only | NPR+TCL | Δ |
|------|----------|---------|---|
| NDCG@10 | 0.0341 | 0.0355 | **+4.1%** |
| NDCG@5 | 0.0258 | 0.0270 | +4.7% |
| HR@10 | 0.0514 | 0.0528 | +2.7% |

NPR 在所有 TIGER 指标上一致优于不含 NPR 的 TCL_only baseline，证明 k-NN proximity-based 梯度调制比 exposure-based 更有效。

### 参数汇总

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `model.use_npr` | `false` | 启用 NPR 梯度调制 |
| `model.npr_alpha_min` | `0.01` | 语义邻居对的梯度保留比例 |
| `model.use_online_knn` | `false` | 启用在线 k-NN（否则用静态 G0 k-NN） |
| `model.online_knn_k` | `50` | k-NN 的 K 值 |
| `model.online_knn_ema_momentum` | `0.99` | EMA buffer 更新动量 |
| `model.online_knn_update_interval` | `100` | k-NN 重算间隔（步数） |
| `model.online_knn_num_items` | 由 exposure_counts 推断 | 物料池大小 |

---

## 为什么 KPL 被排除

KPL（k-NN Preservation Loss）在实验中表现负面：

| 指标 | TCL_only | NPR+TCL | NPR+TCL+KPL |
|------|----------|---------|-------------|
| NDCG@10 | 0.0341 | 0.0355 | **0.0321** |

KPL 的 InfoNCE loss 以 G0 k-NN 邻居为正样本，试图直接拉近语义邻居。但实验表明它损害了下游 TIGER 性能。可能原因：

1. **梯度冲突**：KPL 的吸引力与 VCF 的排斥力作用于相同的 item pairs，即使有 collision-aware 过滤（`collision_exclude_radius`），两个损失项的目标仍然矛盾
2. **表征坍缩**：直接拉近所有 k-NN 邻居可能导致表征空间过度收缩，削弱了 SID 的判别能力
3. **信号冗余**：VCF 排斥 + TCL 吸引力已经提供了足够的表征结构信号，额外的 KPL 吸引力过度约束了表征空间

---

## 未来探索方向

### 模块 D：Direction-Aware Repulsion — 评估

**思路**：排斥时仅沿局部流形切空间方向推开，避免正交方向（流出形面）破坏 k-NN 结构。

**理论依据**：

因果链分析的核心发现之一是"漂移方向比漂移量更重要"——沿流形漂移不破坏 k-NN，正交漂移破坏 k-NN。NPR 解决了"是否推"的问题（语义邻居不推），但未解决"往哪推"的问题。即使排斥非邻居 item，如果排斥方向正交于局部流形，仍可能将 item 推离流形面，间接破坏其 k-NN 结构。

技术上，对 item i：
1. 取其 G0 k-NN 邻居 embedding `{z_j: j ∈ kNN(i)}`
2. 计算局部切空间基：对邻居 embedding 做 PCA → 取 top-d 主成分 `V ∈ R^{D×d}` 作为切空间投影矩阵
3. 排斥梯度经投影：`g_proj = V @ (V^T @ g)` — 只保留切空间分量，丢弃法向分量

**效果评估**：理论上优雅，但实际收益可能有限。理由：

1. **NPR 已解决主要矛盾**。当前最大问题是语义邻居被直接推开（NPR 修复），而非 non-neighbor 被推向错误方向。NPR 已捕获了"方向问题"中的大部分增益。

2. **流形切空间近似质量存疑**。在 768 维空间用 K=50 个邻居估计切空间——50 个样本不足以稳定估计高维流形的局部几何。PCA 得到的"主成分"更可能反映邻居采样的噪声结构而非真实流形方向。

3. **排斥方向本就大致沿流形**。对于已经在流形上的两个 item，其差值向量 `z_i - z_j` 自然地大致位于切空间内。方向感知投影的修正量可能很小。

4. **计算成本高**。每个 item 做一次 PCA（SVD on `[K, 768]`），batch size 256 → 每步 256 次 SVD。即使使用 truncated SVD 或近似方法（如直接用邻居差值 span 代替 PCA），开销仍显著。

**结论**：方向感知排斥在理论上是 NPR 的自然延伸，但属于**精细化修正**而非**突破性改进**。边际收益大概率小于 NPR 本身带来的 +4.1% 增益。不建议作为下一优先级实验——更好的方向是调优 NPR 的 α_min、在线 k-NN 的更新频率等已有参数，或探索 NPR 与其他已有模块（如 TCL）的更深层组合。

---

## NPR 互补方案探索

以下方案与 NPR 正交——NPR 控制"推不推"（梯度缩放），这些方案分别控制"推多远""拉多近""码本多样性""训练节奏"等维度。各方案可独立验证，也可与 NPR 叠加。

### 方案 1：Warm-start G0-Adagrad + Fine-tune QuaSID+NPR（推荐优先级最高）

**动机**：G0-Adagrad 码本利用率极佳（L1: 91%, L2: 95%），说明纯 RQ-VAE + Adagrad 能学好 codebook 结构。但 G0 无碰撞处理机制，同义词可能碰撞。QuaSID+NPR 有碰撞处理但码本利用率不如 Adagrad。

**方案**：
1. 先用 G0-Adagrad 训练 SID encoder（如 1000 步，得到高利用率码本）
2. 从该 checkpoint 加载，切换为 QuaSID+NPR+TCL 微调（VCF/NPR/TCL 从此时开始生效）
3. 微调阶段：VCF 处理碰撞，NPR 保护 k-NN，TCL 注入协同信号

**预期效果**：结合 G0 的码本质量和 QuaSID 的语义优化，可能同时提升码本利用率和 TIGER NDCG。

**参考论文**：
- *Breaking the Hourglass Phenomenon of Residual Quantization* (2024) — 发现 RQ-VAE 中间层码本过度集中（"沙漏效应"），由数据稀疏驱动。两阶段训练可缓解。
- *GENPLUGIN* (2025) — 两阶段：先学习语言/ID 表征对齐，再生成式微调，缓解 exposure bias。

**实现复杂度**：低 — 只需在训练脚本中加 `ckpt_path`。

---

### 方案 2：Adaptive Per-Pair Repulsion Margin

**动机**：当前 VCF margin 是固定值（m_full=0.8, m_partial=0.5）。但碰撞对之间的语义距离不同——两个已经余弦距离 0.6 的 item 碰撞，需要的排斥比两个距离 0.2 的 item 碰撞更弱。

**方案**：
```
m_adaptive = m_base + β * cos_sim(z_i, z_j)
```
- 余弦相似度高（语义接近）→ margin 更大 → 更强排斥（需要更多分离）
- 余弦相似度低（语义已远）→ margin 更小 → 更弱排斥（避免过度推开）

**与 NPR 的关系**：
- NPR：j 是语义邻居 → α=0.01（几乎不推）
- Adaptive margin：即使 α=1.0（全梯度），margin 也根据语义距离自适应调整
- 两者正交：NPR 控制梯度（α），adaptive margin 控制目标距离

**参考论文**：
- *QuaSID* (2026) — 使用 Hamming-guided margin，但 margin 基于碰撞程度（full/partial）而非语义距离。

**实现复杂度**：低 — 修改 `_asymmetric_repulsion_loss` 中 margin 的计算。

---

### 方案 3：Reciprocal Neighbor Consistency Loss（RNCL）

**动机**：NPR 是防御性的——阻止破坏 k-NN 的排斥。但缺少进攻性的吸引力——主动让语义邻居拥有更相似的 SID。

**方案**：对互为 k-NN 邻居的 item pairs（满足 j ∈ kNN(i) 且 i ∈ kNN(j)），添加 SID 一致性损失：
```
L_rncl = -1/|R| * Σ_{(i,j)∈R} sim(SID_i, SID_j)
```
其中 R = {(i,j): i 和 j 互为 k-NN}，`sim` 可以用 Hamming 相似度或 soft token overlap。

**关键设计**：
- 仅作用于**双向** k-NN（reciprocal），比所有 k-NN 更严格、更可靠
- 鼓励 encoder 给强语义关系对分配更相似的 SID（而不仅是"不推开"）
- 可在 batch 内计算，无需额外数据

**与 NPR 的分工**：
- NPR（排斥侧）：不要推开语义邻居 — 保护已有的
- RNCL（吸引侧）：主动拉近语义邻居的 SID — 创造更好的

**潜在风险**：吸引力过强可能导致 SID 崩塌（所有 item 分到相同 SID）。需要小权重（λ_rncl ≈ 0.001）和 warmup。

**实现复杂度**：中 — 需要复用 k-NN buffer 构建 reciprocal mask，在 training_step 中加入新 loss 项。

---

### 方案 4：Codebook Diversity Regularization

**动机**：G0-Adam 码本崩塌的教训是——一旦码本崩塌（少数 code 占据几乎所有 assignment），下游 TIGER 就报废了。即使 Adagrad 不崩塌，码本向量仍可能逐渐相似化。

**方案**：对每个量化层的 codebook 向量，施加多样性正则：
```
L_div = (1/C²) * Σ_{a,b} max(0, cos_sim(e_a, e_b) - threshold)
```
- 惩罚同一层中余弦相似度过高的 codebook 向量 pair
- threshold 例如 0.3：cos_sim > 0.3 的 pair 被推开

**与 NPR 的关系**：
- NPR 作用于 item-item 排斥（侧向保护）
- L_div 作用于 codebook-codebook 排斥（结构化码本空间）

**参考论文**：
- *MACRec* (AAAI 2026) — Cross-modal quantization 改善 codebook usability。
- *CARD* (2026) — Non-uniform quantization 平衡码本利用率。
- VQ-VAE 相关工作中的 codebook diversity loss / commitment loss 变体。

**实现复杂度**：低 — 额外的 loss 项，在 `training_step` 中计算。

---

### 方案 5：Gradual VCF Repulsion Curriculum

**动机**：训练初期 encoder 输出不稳定，k-NN 关系不可靠，此时 NPR 基于不可靠的 k-NN 做保护决策可能出错。同时，早期 VCF 排斥可能干扰 encoder 的基础结构学习。

**方案**：
1. Phase 1（0 ~ warmup% 步）：仅 VQ loss（纯 RQ-VAE），建立基础码本结构
2. Phase 2（warmup% ~ 100% 步）：逐步 ramp-up VCF repulsion + TCL + NPR
   - repulsion λ 从 0 线性 ramp 到目标值
   - NPR 的 k-NN buffer 从 Phase 1 结束时的 G0 推理得到

**可行调度**：
- 3000 步训练中，前 500 步仅 VQ loss
- 500-1500 步：λ_repulsion 从 0 → 1.0（线性）
- 1500-3000 步：全 VCF + NPR + TCL

**实现复杂度**：低-中 — ramp 函数 + 条件损失项。

---

### 方案优先级矩阵

| # | 方案 | 预期收益 | 实现复杂度 | 风险 | 优先级 |
|---|------|---------|-----------|------|--------|
| 1 | Warm-start G0→QuaSID+NPR | ★★★★★ | 低 | 低 | **P0** |
| 2 | Adaptive Margin | ★★★ | 低 | 低 | **P1** |
| 3 | Reciprocal Neighbor Consistency | ★★★★ | 中 | 中（SID崩塌） | **P1** |
| 4 | Codebook Diversity Reg | ★★★ | 低 | 低 | **P2** |
| 5 | Gradual Curriculum | ★★ | 低-中 | 低 | **P2** |

**推荐实验顺序**：方案 1 最优先（最可能产生突破），方案 2/3 作为后续迭代。

---

## 代码改动范围

### 新增文件
- `src/modules/clustering/knn_preservation.py` — G0 k-NN buffer 加载、NPR mask 构建

### 修改文件
- `src/modules/clustering/residual_quantization.py`:
  - `__init__`: 新增 `use_npr`, `npr_alpha_min`, `use_online_knn`, `online_knn_*` 等参数
  - `_asymmetric_repulsion_loss`: 添加 NPR 分支（`build_npr_mask` → `alpha_eff`）
  - 在线 k-NN：`_init_online_knn_buffers`, `_update_online_knn_ema`, `_maybe_update_online_knn`, `_get_knn_buffer`
- `configs/experiment/rqvae_vcf_online_knn_train_flat.yaml`: 新增 NPR 相关参数

### 预计算
- G0 k-NN 索引文件 (`knn_g0_indices.pt`): 形状 `(N, K)`，由 G0 推理的 encoder output 计算余弦相似度 + top-K 得到

---

## 验证

### Step 1: Smoke test
- 检查 NPR 不导致 codebook 崩塌
- frac_unique > 0.5, l0cov > 0.2

### Step 2: 完整 SID 训练 + TIGER 推理
- 对比 baseline (TCL_only) 的 NDCG@10
- 核心指标：TIGER NDCG@10 提升

### Step 3: 成功标准
- TIGER NDCG@10 不低于 TCL_only baseline
- 无 codebook 崩塌
- 码本利用率（frac_unique）不低于 baseline
