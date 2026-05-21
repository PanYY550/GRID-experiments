# NPR+TCL SID 生成完整流水线：公式、原理与逻辑

> **最后更新**: 2026-05-19
> **数据来源**: `src/modules/clustering/residual_quantization.py`, `src/modules/clustering/knn_preservation.py`
> **基准配置**: QuaSID base ($B=256$, $R=1$, $\lambda_{\text{full}}=0.2$, $\lambda_{\text{partial}}=0.1$, $m_{\text{full}}=0.8$, $m_{\text{partial}}=0.5$ static, Adam $\text{lr}=3\times10^{-4}$, $\text{wd}=1\times10^{-5}$)

---

## 目录

1. [总体架构](#1-总体架构)
2. [RQ-VAE 前向传播](#2-rq-vae-前向传播)
3. [CVPM — 良性碰撞掩码](#3-cvpm--良性碰撞掩码)
4. [HaMR — 碰撞分类](#4-hamr--碰撞分类)
5. [VCF 排斥损失](#5-vcf-排斥损失)
6. [NPR — 邻居保护排斥](#6-npr--邻居保护排斥)
7. [TCL — QuaSID InfoNCE 对比学习](#7-tcl--quasid-infonce-对比学习)
8. [Online k-NN — 在线邻居追踪](#8-online-k-nn--在线邻居追踪)
9. [Dual-Tower — 双塔梯度解耦](#9-dual-tower--双塔梯度解耦)
10. [Codebook 维护](#10-codebook-维护)
11. [完整训练流程](#11-完整训练流程)
12. [SID 推理](#12-sid-推理)
13. [TIGER 下游训练](#13-tiger-下游训练)

---

## 1. 总体架构

```
输入: 预训练多模态 Embedding z ∈ R^768 (text + image, frozen from SASRec)

┌─────────────────────────────────────────────────────┐
│                  Encoder (MLP)                       │
│         768 → 256 → 128 → 64 (bottleneck)           │
│              BatchNorm + ReLU                        │
└──────────────────────┬──────────────────────────────┘
                       │ z_enc ∈ R^64
                       │
┌──────────────────────▼──────────────────────────────┐
│          Residual Quantization (3 layers)            │
│                                                      │
│  Layer 0: z → codebook[256, 64] → c0, z1 = z - c0  │
│  Layer 1: z1 → codebook[256, 64] → c1, z2 = z1 - c1 │
│  Layer 2: z2 → codebook[256, 64] → c2               │
│                                                      │
│  SID = (token0, token1, token2) ∈ {0..255}^3        │
│  Quantized: ẑ = c0 + c1 + c2                        │
└──────────────────────┬──────────────────────────────┘
                       │
         ┌─────────────┴─────────────┐
         │                           │
   Quantization Loss          Training Objectives:
   (commitment + EMA)         ├── VCF Repulsion (防御性排斥)
                              ├── TCL InfoNCE (对比吸引力)
                              ├── NPR (邻居保护调制)
                              └── Codebook Entropy (多样性正则)
```

**核心思想**：

- **RQ-VAE** 将连续 embedding 压缩为 3 层离散 SID (每层 256 codes → 16M 可能组合)
- **VCF** 排斥 SID 碰撞的 item pair，确保不同 item 获得不同 SID
- **CVPM** 屏蔽"良性碰撞"（同 item、共现正样本、语义极近 item），避免过度排斥
- **NPR** 保护 k-NN 语义邻居不被 VCF 推远，维持语义流形结构
- **TCL** 提供 InfoNCE 吸引力，将 Swing 共现 items 拉近，保证 codebook 全部活跃
- **Dual-Tower** + stop_gradient 防止 TCL 与 VCF 形成"镜像厅反馈环"

---

## 2. RQ-VAE 前向传播

### 2.1 输入归一化 + 编码

$$
\mathbf{x} = \text{BatchNorm}(\mathbf{e}_{\text{input}}), \quad \mathbf{x} \in \mathbb{R}^{B \times 768}
$$

$$
\mathbf{z} = \text{Encoder}(\mathbf{x}), \quad \mathbf{z} \in \mathbb{R}^{B \times 64}
$$

| 符号 | 含义 |
|------|------|
| $\mathbf{e}_{\text{input}}$ | 预训练多模态 embedding (SASRec, frozen) |
| $\mathbf{x}$ | BatchNorm 归一化后的输入 |
| $\mathbf{z}$ | Encoder 输出的 bottleneck 表征 (64d) |
| $B$ | batch size (=256) |

Encoder 结构: $\text{Linear}(768 \to 256) \to \text{BN} \to \text{ReLU} \to \text{Linear}(256 \to 128) \to \text{BN} \to \text{ReLU} \to \text{Linear}(128 \to 64)$

### 2.2 残差量化

对于每层 $l \in \{0, 1, 2\}$：

**Step 1 — L2 归一化残差**:

$$
\mathbf{r}_l = \frac{\mathbf{z}_l}{\|\mathbf{z}_l\|_2}, \quad \mathbf{r}_l \in \mathbb{R}^{B \times 64}
$$

| 符号 | 含义 |
|------|------|
| $\mathbf{z}_l$ | 第 $l$ 层的输入残差 ($\mathbf{z}_0$ 即 encoder 输出) |
| $\mathbf{r}_l$ | L2 归一化后的残差 |

**Step 2 — 最近邻分配**:

$$
d_l^{(k)} = \|\mathbf{r}_l - \mathbf{C}_l[k]\|_2^2, \quad k \in \{0, \dots, 255\}
$$

$$
t_l = \underset{k}{\operatorname{argmin}} \; d_l^{(k)}, \quad t_l \in \{0,\dots,255\}^B
$$

| 符号 | 含义 |
|------|------|
| $d_l^{(k)}$ | 残差与第 $k$ 个 code 的欧氏距离 |
| $\mathbf{C}_l \in \mathbb{R}^{256 \times 64}$ | 第 $l$ 层的 codebook (256 个 centroid) |
| $t_l$ | 分配的 token (最近邻 centroid 的索引) |

**Step 3 — 量化 + 残差传递**:

$$
\mathbf{c}_l = \mathbf{C}_l[t_l], \quad \mathbf{c}_l \in \mathbb{R}^{B \times 64}
$$

$$
\mathbf{z}_{l+1} = \mathbf{z}_l - \mathbf{c}_l
$$

| 符号 | 含义 |
|------|------|
| $\mathbf{c}_l$ | 量化向量 (lookup from codebook) |
| $\mathbf{z}_{l+1}$ | 第 $l+1$ 层的输入残差 (当前残差减去量化向量) |

**Step 4 — Commitment Loss** (EMA 更新模式):

$$
\mathcal{L}_{\text{commit}} = \sum_{l=0}^{2} \|\mathbf{z}_l - \operatorname{sg}[\mathbf{c}_l]\|_2^2
$$

$$
\mathcal{L}_{\text{ema}} = \sum_{l=0}^{2} \|\operatorname{sg}[\mathbf{z}_l] - \mathbf{c}_l\|_2^2
$$

| 符号 | 含义 |
|------|------|
| $\operatorname{sg}[\cdot]$ | stop_gradient 操作：前向无变化，反向梯度为 0 |
| $\mathcal{L}_{\text{commit}}$ | 惩罚 encoder 输出偏离 centroid（encoder 向 centroid 移动） |
| $\mathcal{L}_{\text{ema}}$ | 用 EMA 更新 centroid（centroid 向 encoder 输出移动，无梯度） |

**最终输出**：

- $\mathbf{T} = [t_0, t_1, t_2]$ — SID token 序列, shape $(B, 3)$
- $\hat{\mathbf{z}} = \mathbf{c}_0 + \mathbf{c}_1 + \mathbf{c}_2$ — 量化重建向量, shape $(B, 64)$
- $\mathbf{z}_0$ — pre-quantization bottleneck, 用于 TCL/DSF 计算

### 2.3 Codebook 更新 (EMA)

每次前向时，对每层独立更新：

$$
n_k \leftarrow \eta \cdot n_k + (1-\eta) \cdot |\mathcal{A}_k|
$$

$$
\mathbf{m}_k \leftarrow \eta \cdot \mathbf{m}_k + (1-\eta) \cdot \sum_{i: t_{l}^{(i)} = k} \mathbf{z}_l^{(i)}
$$

$$
\mathbf{C}_l[k] = \frac{\mathbf{m}_k}{n_k}
$$

| 符号 | 含义 |
|------|------|
| $\eta = 0.99$ | EMA decay rate |
| $\mathcal{A}_k = \{i : t_l^{(i)} = k\}$ | batch 中被分配到 centroid $k$ 的 items 集合 |
| $n_k$ | centroid $k$ 的 EMA 分配计数 |
| $\mathbf{m}_k$ | 分配给 centroid $k$ 的 $\mathbf{z}_l$ 之和的 EMA |

---

## 3. CVPM — 良性碰撞掩码

**全称**: Codebook Variance Preserving Margin (规则 1+2+3)

**目的**: 在计算 VCF 排斥损失之前，先屏蔽不应被排斥的"良性碰撞对"。

### 3.1 掩码定义

给定 batch_size=$B$，生成 CVPM mask $\mathbf{M} \in \{0,1\}^{B \times B}$：

$$
\mathbf{M}[i,j] = R_1[i,j] \;\wedge\; R_2[i,j] \;\wedge\; R_3[i,j]
$$

其中 $\wedge$ 表示逻辑与。

**当 $\mathbf{M}[i,j] = \text{False}$ 时，$(i,j)$ pair 是"良性碰撞"——从 VCF 排斥候选中移除。**

### 3.2 Rule 1 — 同 item 屏蔽

$$
R_1[i,j] = \bigl(\text{item\_id}[i] \neq \text{item\_id}[j]\bigr)
$$

同一 item 在 batch 中出现多次（如被多个用户交互过）→ 自身碰撞是良性的，不应互相排斥。

### 3.3 Rule 2 — 共现正样本屏蔽

$$
R_2[i,j] = \bigl(\mathbf{P}_{\text{swing}}[i,j] = 0\bigr)
$$

| 符号 | 含义 |
|------|------|
| $\mathbf{P}_{\text{swing}} \in \{0,1\}^{B \times B}$ | Swing 算法计算的共现正样本矩阵 |

$\mathbf{P}_{\text{swing}}[i,j] = 1$ 表示 item $i$ 和 $j$ 在用户行为序列中频繁共现（被足够多的共同用户交互过）。这些 items 共享 SID 是合理且期望的 → 不应排斥。此规则对应 QuaSID 论文 Section 5.2 的"collaborative positive masking"。

### 3.4 Rule 3 — 内容相似度屏蔽

$R_3$ 仅在 $\text{cvpm\_sim\_threshold} > 0$ 时启用（当前配置 = 0，禁用）：

$$
\cos_{ij} = \frac{\mathbf{e}_{\text{content}}^{(i)} \cdot \mathbf{e}_{\text{content}}^{(j)}}{\|\mathbf{e}_{\text{content}}^{(i)}\|_2 \cdot \|\mathbf{e}_{\text{content}}^{(j)}\|_2}
$$

$$
R_3[i,j] = \bigl(\cos_{ij} < \tau_{\text{cvpm}}\bigr)
$$

两个 items 的多模态内容嵌入高度相似 → 即使 SID 碰撞也是语义合理的 → 不排斥。

### 3.5 最终 CVPM 表达式

$$
\mathbf{M}_{\text{cvpm}}[i,j] = \underbrace{(\text{id}_i \neq \text{id}_j)}_{\text{Rule 1: 非自身}} \;\wedge\; \underbrace{(\mathbf{P}_{\text{swing}}[i,j] = 0)}_{\text{Rule 2: 非共现}} \;\wedge\; \underbrace{(\cos_{ij} < \tau_{\text{cvpm}})}_{\text{Rule 3: 内容不极似}}
$$

---

## 4. HaMR — 碰撞分类

**全称**: Hamming Radius Collision Classification

### 4.1 Hamming 距离

SID 有 $L=3$ 层，每层 256 个可能 token。对任意两个 items $i, j$：

$$
H_{ij} = \sum_{l=0}^{L-1} \mathbf{1}\bigl[t_i^{(l)} \neq t_j^{(l)}\bigr], \quad H_{ij} \in \{0, 1, 2, 3\}
$$

| 符号 | 含义 |
|------|------|
| $t_i^{(l)}$ | item $i$ 在第 $l$ 层的 SID token, $t_i^{(l)} \in \{0, \dots, 255\}$ |
| $\mathbf{1}[\cdot]$ | 指示函数 (条件为真 = 1, 否则 = 0) |
| $H_{ij}$ | 两层 SID 在逐层比较下不同的层数 |

### 4.2 碰撞类型 (R=1)

| $H_{ij}$ | 碰撞类型 | 含义 |
|:--------:|:------:|------|
| **0** | **完全碰撞** (Full Collision) | 3 层 token 完全相同 → 两个 items 获得完全相同的 SID |
| **1** | **部分碰撞** (Partial Collision, $H \leq R$) | 1 层不同, 2 层相同 → SID 部分重叠 |
| 2 | 非碰撞 | 只有 1 层相同 → 不视为碰撞 |
| 3 | 非碰撞 | 完全不同 → 不视为碰撞 |

### 4.3 碰撞集分类

将 CVPM mask 与 Hamming 距离结合：

$$
\Omega_{\text{full}} = \bigl\{(i,j) : H_{ij} = 0 \;\wedge\; \mathbf{M}_{\text{cvpm}}[i,j] = \text{True}\bigr\}
$$

$$
\Omega_{\text{partial}} = \bigl\{(i,j) : 0 < H_{ij} \leq R \;\wedge\; \mathbf{M}_{\text{cvpm}}[i,j] = \text{True}\bigr\}
$$

| 符号 | 含义 |
|------|------|
| $\Omega_{\text{full}}$ | 完全碰撞对索引集 — 需要被全力推开 |
| $\Omega_{\text{partial}}$ | 部分碰撞对索引集 — 需要被温和推开 |
| $R = 1$ | hamming_radius, 部分碰撞的 Hamming 距离上限 |

**注意**: CVPM 通过 $\mathbf{M}_{\text{cvpm}}$ 在此处介入——若某 pair 被 CVPM 判定为"良性碰撞"（$\mathbf{M}_{\text{cvpm}}[i,j]=\text{False}$），则即使 $H_{ij}=0$（同 SID）也不会进入 $\Omega_{\text{full}}$。

---

## 5. VCF 排斥损失

**全称**: VCR-TD-QuaSID Fusion Repulsion Loss

**目的**: 将 SID 碰撞的 items 在 embedding 空间推开，降低后续碰撞。

### 5.1 核心公式

$$
\mathcal{L}_{\text{rep}} = \lambda_{\text{full}} \cdot \mathcal{L}_{\text{full}} + \lambda_{\text{partial}} \cdot \mathcal{L}_{\text{partial}}
$$

其中：

$$
\mathcal{L}_{\text{full}} = \frac{1}{|\Omega_{\text{full}}|} \sum_{(i,j) \in \Omega_{\text{full}}} w_{ij} \cdot \max\bigl(0,\; m_{\text{full}} - D_{ij}\bigr)
$$

$$
\mathcal{L}_{\text{partial}} = \frac{1}{|\Omega_{\text{partial}}|} \sum_{(i,j) \in \Omega_{\text{partial}}} w_{ij} \cdot \max\bigl(0,\; m_{\text{partial}} - D_{ij}\bigr)
$$

| 符号 | 当前值 | 含义 |
|------|:------:|------|
| $\lambda_{\text{full}}$ | $0.2$ | 完全碰撞排斥损失权重 |
| $\lambda_{\text{partial}}$ | $0.1$ | 部分碰撞排斥损失权重 |
| $m_{\text{full}}$ | $0.8$ | 完全碰撞 margin — 需要推到的"安全距离" |
| $m_{\text{partial}}$ | $0.5$ | 部分碰撞 margin |
| $w_{ij}$ | $1.0$ | 时间权重 (time_decay=false 时恒为 1) |
| $D_{ij}$ | — | 余弦距离, $D_{ij} = 1 - \cos(\mathbf{z}_i, \mathbf{z}_j) \in [0, 2]$ |

### 5.2 Hinge Loss 语义

$$
\mathcal{L}_{\text{hinge}} = \max(0,\; m - D_{ij})
$$

- 当 $D_{ij} \geq m$ 时：$\mathcal{L}=0$（已经足够远，不需要再推）
- 当 $D_{ij} < m$ 时：$\mathcal{L}=m - D_{ij} > 0$（太近了，梯度方向为 $-\nabla D_{ij}$，即增大余弦距离）

**几何直观**: margin $m$ 定义了"安全距离"。碰撞的 items 在余弦空间中被推开，直到它们之间的距离达到至少 $m$。

### 5.3 为什么 $m_{\text{full}} > m_{\text{partial}}$

完全碰撞 ($H=0$) 是更严重的问题——两个 item 获得完全相同的 SID，TIGER 无法区分。部分碰撞 ($H=1$) 仅共享 2 层，仍保留了 1 层的区分能力。因此：
- $m_{\text{full}} = 0.8$：强排斥，推得远
- $m_{\text{partial}} = 0.5$：弱排斥，推一半即可

### 5.4 余弦距离计算 (Dual-Tower)

当前使用**静态 margin** ($\text{use\_dynamic\_margin} = \text{false}$)：

$$
\mathbf{z}_{\text{online}} = \text{encoder}(\mathbf{x}), \quad \mathbf{z}_{\text{target}} = \operatorname{sg}[\text{encoder}(\mathbf{x})]
$$

$$
\tilde{\mathbf{z}}_i = \frac{\mathbf{z}_{\text{online}}^{(i)}}{\|\mathbf{z}_{\text{online}}^{(i)}\|_2}, \quad \tilde{\mathbf{z}}_j = \frac{\mathbf{z}_{\text{target}}^{(j)}}{\|\mathbf{z}_{\text{target}}^{(j)}\|_2}
$$

$$
D_{ij} = 1 - \tilde{\mathbf{z}}_i^{\top} \tilde{\mathbf{z}}_j
$$

| 符号 | 含义 |
|------|------|
| $\mathbf{z}_{\text{online}}$ | online encoder 输出, 有梯度 |
| $\mathbf{z}_{\text{target}}$ | detached copy, 无梯度 (stop_gradient) |
| $\tilde{\mathbf{z}}_i, \tilde{\mathbf{z}}_j$ | L2-normalized embeddings |

**梯度流**: 只流经 $\mathbf{z}_{\text{online}}^{(i)}$ (行), $\mathbf{z}_{\text{target}}^{(j)}$ 完全冻结 → 打破"镜像厅反馈环"。

### 5.5 VCF 时序控制

#### Hard Gate (warmup)

$$
\mathcal{L}_{\text{rep}} = 0 \quad \text{if} \quad \text{step} < \text{repulsion\_warmup\_steps}
$$

当前配置 $\text{warmup}=0$（QuaSID base 的 $\lambda_{\text{cl}}=0.1$ TCL 已足够强，可同时稳定 codebook）。

#### Stability Gate

若 codebook 健康度不足，暂停排斥防止加速崩塌：

$$
\mathcal{L}_{\text{rep}} = 0 \quad \text{if} \quad \text{layer0\_coverage} < 0.50 \;\lor\; \text{frac\_unique\_ids} < 0.50
$$

#### Pair Subsampling

若碰撞对过多（$> B \times 64$），每 anchor 随机 subsample 最多 64 对，防止损失被少量 anchor 的大量碰撞主导。

---

## 6. NPR — 邻居保护排斥

**全称**: Neighbor-Preserving Repulsion

**核心创新**: 将 VCF 的梯度缩放因子从 exposure-based 改为 k-NN-based。当碰撞对 $(i,j)$ 中的 $j$ 是 $i$ 的 k-NN 语义邻居时，削弱 $i$ 侧的排斥梯度——保护 $i$ 的语义位置。

### 6.1 问题

标准 VCF 排斥：两个碰撞 items 互相推开，各承受约 50% 梯度。

如果 item $j$ 是 item $i$ 的语义最近邻（$\text{kNN}(i)$），推开 $j$ 会破坏 $i$ 的语义邻域结构 → DSF 下降 → NDCG 下降。

### 6.2 NPR 梯度缩放因子

对碰撞 pair $(i,j)$，计算 $z_i$ 侧的梯度缩放因子 $\alpha_{ij}$：

$$
\alpha_{ij} = \begin{cases}
\alpha_{\min}^{\text{npr}} & \text{if } j \in \text{kNN}_{\text{online}}(i) \\[6pt]
1.0 & \text{if } j \notin \text{kNN}_{\text{online}}(i)
\end{cases}
$$

| 符号 | 当前值 | 含义 |
|------|:------:|------|
| $\alpha_{ij}$ | $0.01$ 或 $1.0$ | item $i$ 在排斥 pair $(i,j)$ 时的梯度缩放因子 |
| $\alpha_{\min}^{\text{npr}}$ | $0.01$ | 邻居保护最小梯度缩放 (保留 1% 梯度 = 99% 被 detach) |
| $\text{kNN}_{\text{online}}(i)$ | — | item $i$ 的 online EMA k-NN 邻居集 (K=50) |

### 6.3 梯度调制机制

$$
\mathbf{z}_i^{\text{mixed}} = \alpha_{ij} \cdot \mathbf{z}_i + (1 - \alpha_{ij}) \cdot \operatorname{sg}[\mathbf{z}_i]
$$

| $\alpha_{ij}$ | $\mathbf{z}_i^{\text{mixed}}$ | 梯度比例 | 效果 |
|:---:|------|:---:|------|
| $0.01$ | $0.01 \cdot \mathbf{z}_i + 0.99 \cdot \operatorname{sg}[\mathbf{z}_i]$ | 1% | $i$ 几乎不被推开（"保护"语义位置） |
| $1.0$ | $\mathbf{z}_i$ | 100% | $i$ 正常参与排斥 |

然后用 $\mathbf{z}_i^{\text{mixed}}$ 替代 $\mathbf{z}_i$ 计算余弦距离：

$$
D_{ij}^{\text{NPR}} = 1 - \frac{\mathbf{z}_i^{\text{mixed}}}{\|\mathbf{z}_i^{\text{mixed}}\|_2} \cdot \frac{\mathbf{z}_{\text{target}}^{(j)}}{\|\mathbf{z}_{\text{target}}^{(j)}\|_2}
$$

**最终效果**: 碰撞 pair $(i,j)$ 中，若 $j$ 是 $i$ 的邻居 → $i$ 几乎不动（$\alpha = 0.01$），$j$ 承受全部推力实现分离。非邻居则正常互相推开。

### 6.4 与 Path B Exposure-Aware 的区别

| 维度 | Path B (ASYM) | NPR |
|------|:---:|:---:|
| $\alpha$ 来源 | $\frac{\min(e_i, e_j)}{e_i + e_j}$ (曝光次数) | $\mathbf{1}[j \in \text{kNN}(i)]$ (语义邻域) |
| 保护对象 | 低曝光 tail items | 所有 k-NN 语义邻居 |
| $\alpha_{\min}$ | $\alpha_{\min}^{\text{asym}} \in [0.001, 0.10]$ | $\alpha_{\min}^{\text{npr}} = 0.01$ |
| 动机 | "热门推、冷门不推" | "非邻居推、邻居保护" |

### 6.5 NPR 的关键性质

1. **纯防御性**: 不施加新损失项，只调制已有 VCF 排斥梯度
2. **无新信号**: 不引入新正/负样本对
3. **与 TCL 互补**: TCL 拉近共现 items，NPR 保护语义邻居不被 VCF 推远
4. **与 CVPM 互补**: CVPM 屏蔽共现 items 不被推（零推力），NPR 保护 k-NN 邻居不被推（几乎零推力）

---

## 7. TCL — QuaSID InfoNCE 对比学习

**全称**: Temporal Contrastive Learning (via QuaSID InfoNCE)

**目的**: 提供吸引力，将 Swing 共现 items 拉近，保证 codebook 全部活跃。

### 7.1 正样本来源

$$
\mathbf{P}_{\text{swing}}[i,j] = 1 \iff \text{Swing}(i,j) > \text{threshold}
$$

Swing 是协同过滤的共现度量：两个 items 被越多共同用户交互过 → Swing score 越高 → 它们是"行为正样本"。

### 7.2 InfoNCE 损失

**相似度矩阵**:

$$
S_{ij} = \frac{\tilde{\mathbf{z}}_i^{\top} \tilde{\mathbf{z}}_j}{\tau}, \quad \tilde{\mathbf{z}}_i = \frac{\mathbf{z}_{\text{online}}^{(i)}}{\|\mathbf{z}_{\text{online}}^{(i)}\|_2}, \quad \tilde{\mathbf{z}}_j = \frac{\mathbf{z}_{\text{target}}^{(j)}}{\|\mathbf{z}_{\text{target}}^{(j)}\|_2}
$$

| 符号 | 当前值 | 含义 |
|------|:------:|------|
| $S_{ij}$ | — | 缩放后的余弦相似度 (logit) |
| $\tau$ | $0.5$ | InfoNCE 温度参数 |
| $\mathbf{z}_{\text{target}}$ | — | detached copy of $\mathbf{z}_{\text{online}}$ |

**为什么 $\tau = 0.5$**: $\exp(\cos/\tau) \approx \exp(0.8/0.5) \approx 5$，使 softmax 梯度量级与 VCF hinge ($\approx 0.03$) 对齐，防止一种力压过另一种。

**Per-anchor InfoNCE**:

$$
\mathcal{L}_i = -\log \frac{\sum_{j \in \mathcal{P}_i} \exp(S_{ij})}{\sum_{k: k \neq i, \text{id}_k \neq \text{id}_i} \exp(S_{ik})}
$$

| 符号 | 含义 |
|------|------|
| $\mathcal{P}_i = \{j : \mathbf{P}_{\text{swing}}[i,j] = 1\}$ | anchor $i$ 的 Swing 共现正样本集 |
| 分母 | 所有有效 items (排除自身 + 同 ID 重复) 的 log-sum-exp |

**总 InfoNCE 损失**:

$$
\mathcal{L}_{\text{cl}} = \frac{1}{|\mathcal{V}|} \sum_{i \in \mathcal{V}} \mathcal{L}_i
$$

其中 $\mathcal{V} = \{i : |\mathcal{P}_i| > 0\}$ 为至少有一个正样本的有效 anchor 集。

$$
\mathcal{L}_{\text{total\_cl}} = \lambda_{\text{cl}} \cdot \mathcal{L}_{\text{cl}}, \quad \lambda_{\text{cl}} = 0.1
$$

### 7.3 防止镜像厅反馈

InfoNCE 的分母包含所有 batch items。若 $z_i$ 和 $z_k$ 双向移动 → 形成自增强循环（"镜像厅"）→ 编码器崩塌。Dual-tower 方案：

$$
\mathbf{z}_{\text{target}} = \operatorname{sg}[\mathbf{z}_{\text{online}}]
$$

梯度只流经 $\mathbf{z}_{\text{online}}$ (anchor 侧)，target 侧冻结 → 打破循环。

### 7.4 TCL 为什么驱动 Codebook 健康

**机制链**：

1. InfoNCE 将 Swing 共现 items 拉近 → 形成多个紧密的 embedding 簇
2. 不同簇覆盖 embedding 空间的不同区域
3. Codebook 的 256 个 centroid 必须分别覆盖这些簇以降低 commitment loss
4. → 所有 256 个 codes 都被分配 → L0 覆盖率高

**实验证据**: 无 TCL 时 $\text{L0} \approx 39\%$，有 TCL 时 $\text{L0} \approx 91\% - 97\%$。

---

## 8. Online k-NN — 在线邻居追踪

**目的**: 替代 G0 checkpoint 的预计算 k-NN，完全消除外部依赖。

### 8.1 EMA Buffer

为每个 item 维护其在 encoder 输出空间中的指数移动平均 (EMA) 表征：

$$
\mathbf{z}_{\text{ema}}[\text{id}] \leftarrow \gamma \cdot \mathbf{z}_{\text{ema}}[\text{id}] + (1 - \gamma) \cdot \mathbf{z}_{\text{current}}, \quad \gamma = 0.99
$$

| 符号 | 含义 |
|------|------|
| $\mathbf{z}_{\text{ema}} \in \mathbb{R}^{N \times 64}$ | EMA buffer, $N=12101$ items |
| $\gamma = 0.99$ | EMA momentum — 保留 99% 历史, 吸收 1% 新信息 |
| $\mathbf{z}_{\text{current}}$ | 当前 step 的 encoder 输出 $\mathbf{z}_{\text{online}}$ |

**$\gamma = 0.99$ 的意义**: k-NN 结构平滑演化，不会因单步参数更新剧烈变化。有效平滑窗口约 $1/(1-0.99) = 100$ 步。

### 8.2 周期性 k-NN 重建

每隔 $\Delta k = 100$ 步，基于 EMA buffer 重新计算全量 k-NN：

$$
\text{kNN}_{\text{online}}[i] = \underset{j \neq i}{\operatorname{topK}_{50}} \; \frac{\mathbf{z}_{\text{ema}}[i] \cdot \mathbf{z}_{\text{ema}}[j]}{\|\mathbf{z}_{\text{ema}}[i]\|_2 \cdot \|\mathbf{z}_{\text{ema}}[j]\|_2}
$$

### 8.3 DSF_online 的定义

$$
\text{DSF}_{\text{online}}(i; k) = \frac{1}{k} \Bigl| \text{kNN}\bigl(\mathbf{z}_{\text{current}}[i], k\bigr) \cap \text{kNN}\bigl(\mathbf{z}_{\text{ema}}[i], k\bigr) \Bigr|
$$

| 符号 | 含义 |
|------|------|
| $\text{DSF}_{\text{online}}(i)$ | item $i$ 的 online DSF, $k=50$ |
| $\mathbf{z}_{\text{current}}$ | 当前 encoder 输出 (非 EMA) |
| $\mathbf{z}_{\text{ema}}$ | EMA buffer 表征 |

$\text{DSF}_{\text{online}} \in [0, 1]$：高值 = 邻域在训练中平滑演化；低值 = 邻域被大幅扰动。（NPR+TCL 当前 $\text{DSF}_{\text{online}} \approx 0.175$）

---

## 9. Dual-Tower — 双塔梯度解耦

**动机**: 防止 VCF 排斥力和 TCL InfoNCE 吸引力的梯度形成"镜像厅反馈环"。

### 9.1 单塔的问题

单塔下 $\mathbf{z}_i$ 既是 anchor 也是 target：
- VCF: 推 $\mathbf{z}_i$ 远离 $\mathbf{z}_j$
- TCL: 拉 $\mathbf{z}_i$ 靠近 $\mathbf{z}_k$

两者同时在 $\mathbf{z}_i$ 上施加相反的力 → 梯度冲突 + 自增强循环 → 训练不稳定甚至编码器崩塌。

### 9.2 Dual-Tower 方案

$$
\mathbf{z}_{\text{online}} = \text{encoder}(\mathbf{x}), \qquad \mathbf{z}_{\text{target}} = \operatorname{sg}\bigl[\text{encoder}(\mathbf{x})\bigr]
$$

| 塔 | 定义 | 梯度 |
|:---:|------|:---:|
| Online | $\mathbf{z}_{\text{online}} = \text{encoder}(\mathbf{x})$ | ✅ 有梯度 |
| Target | $\mathbf{z}_{\text{target}} = \operatorname{sg}[\text{encoder}(\mathbf{x})]$ | ❌ 无梯度 |

**关键**: 两个塔使用**完全相同的 encoder**（非 EMA copy），表征空间完全一致。但梯度的不对称性打破了双向自增强循环。

### 9.3 各组件如何使用 Dual-Tower

$$
D_{ij}^{\text{VCF}} = 1 - \frac{\mathbf{z}_{\text{online}}^{(i)}}{\|\mathbf{z}_{\text{online}}^{(i)}\|_2} \cdot \frac{\mathbf{z}_{\text{target}}^{(j)}}{\|\mathbf{z}_{\text{target}}^{(j)}\|_2}
$$

$$
S_{ij}^{\text{TCL}} = \frac{\mathbf{z}_{\text{online}}^{(i)}}{\|\mathbf{z}_{\text{online}}^{(i)}\|_2} \cdot \frac{\mathbf{z}_{\text{target}}^{(j)}}{\|\mathbf{z}_{\text{target}}^{(j)}\|_2} \cdot \frac{1}{\tau}
$$

**梯度流**: 仅 $\mathbf{z}_{\text{online}}^{(i)}$ (行) 获梯度；$\mathbf{z}_{\text{target}}^{(j)}$ 完全冻结 → 双向反馈被切断。

---

## 10. Codebook 维护

### 10.1 Codebook Reset (每 10 步)

识别并重置"死亡"centroid（长期未被使用的 code）：

$$
\text{dead}_l = \bigl\{k : \text{EMA\_count}_l[k] < 0.5\bigr\}
$$

$$
\mathbf{C}_l[k] \leftarrow \mathbf{z}_{\text{rand}} + \epsilon, \quad \epsilon \sim \mathcal{N}(0, 0.01^2)
$$

| 符号 | 含义 |
|------|------|
| $\text{EMA\_count}_l[k]$ | centroid $k$ 的 EMA 分配计数 (decay=0.9) |
| $\mathbf{z}_{\text{rand}}$ | 从当前 batch 随机抽取的 encoder 输出 |
| $\epsilon$ | 小噪声, 防止多个死 centroid 重置到同一点 |

**目的**: 防止 VQ-VAE 经典崩塌模式——未使用的 centroid 梯度始终为零 → 永久死亡 → 有效 codebook 不断缩小。

### 10.2 Entropy Regularization

对每层 $l$ 计算 code 分配分布的归一化熵：

$$
p_l(c) = \frac{|\{i : t_l^{(i)} = c\}|}{B}, \quad c \in \{0, \dots, 255\}
$$

$$
H_{\text{norm}}^{(l)} = -\frac{\sum_{c=0}^{255} p_l(c) \cdot \log p_l(c)}{\log 256} \in [0, 1]
$$

$$
\mathcal{L}_{\text{ent}} = \frac{\lambda_{\text{ent}}}{L} \sum_{l=0}^{L-1} \bigl(1 - H_{\text{norm}}^{(l)}\bigr)
$$

| 符号 | 当前值 | 含义 |
|------|:------:|------|
| $p_l(c)$ | — | batch 中 code $c$ 被分配的比例 |
| $H_{\text{norm}}^{(l)}$ | — | 归一化熵: 1=均匀使用, 0=仅用一个 code |
| $\lambda_{\text{ent}}$ | $0.1$ | 熵正则权重 |
| $L=3$ | — | 量化层数 |

---

## 11. 完整训练流程

### 11.1 单步训练 (training_step)

```
Input: batch of B=256 items, each with:
  ├── input_embedding: pre-trained multimodal embedding (768d)
  ├── item_ids: global item indices
  ├── positive_pair_matrix: Swing co-occurrence (B × B), boolean
  └── exposure_times: per-item exposure counts

Step 1  Encoder forward
         z_enc = Encoder(Norm(input_embedding))          # (B, 64)

Step 2  Residual Quantization
         T, ẑ, L_commit = RQ-VAE(z_enc)                  # T=(B,3), ẑ=(B,64)

Step 3  Dual-Tower: z_target = sg(z_enc)                 # (B, 64), no gradient

Step 4  Online k-NN Update
         ema_z[id] = 0.99·ema_z[id] + 0.01·z_enc
         if step % 100 == 0: recompute k-NN from ema_z

Step 5  Codebook Reset
         if step % 10 == 0: reset dead centroids

Step 6  VCF Repulsion Loss
         M_cvpm = CVPM_mask(item_ids, P_swing)
         H = Hamming(T)                                  # (B, B)
         Ω_full, Ω_partial = classify(H, M_cvpm, R=1)

         D_cross = 1 - cos(z_online, z_target)           # (B, B) cross-distance

         α = NPR_alpha(knn_buffer, item_ids)             # (B, B), 0.01 or 1.0
         z_i_mixed = α·z_i + (1-α)·sg(z_i)

         L_rep = 0.2·mean(relu(0.8 - D[Ω_full]))
               + 0.1·mean(relu(0.5 - D[Ω_partial]))

Step 7  TCL InfoNCE Loss
         S = z_online^T · z_target / 0.5                  # (B, B)
         L_cl = InfoNCE(S, P_swing, item_ids)

Step 8  Entropy Regularization
         L_ent = 0.1·(1 - mean_l H_norm_l)

Step 9  Total Loss
         L_total = L_commit + L_rep + 0.1·L_cl + L_ent

Step 10 Backward + Optimizer(Adam lr=3e-4 wd=1e-5)
```

### 11.2 损失函数完整表达式

$$
\boxed{
\mathcal{L}_{\text{total}} = \underbrace{\mathcal{L}_{\text{commit}}}_{\text{RQ-VAE 量化}}
+ \underbrace{\lambda_{\text{full}} \mathcal{L}_{\text{full}} + \lambda_{\text{partial}} \mathcal{L}_{\text{partial}}}_{\text{VCF 排斥} \; (\lambda=0.2/0.1)}
+ \underbrace{\lambda_{\text{cl}} \mathcal{L}_{\text{InfoNCE}}}_{\text{TCL 吸引力} \; (\lambda=0.1)}
+ \underbrace{\frac{\lambda_{\text{ent}}}{L} \sum_{l} (1 - H_{\text{norm}}^{(l)})}_{\text{码本熵正则} \; (\lambda=0.1)}
}
$$

$$
\mathcal{L}_{\text{full}} = \frac{1}{|\Omega_{\text{full}}|} \sum_{(i,j) \in \Omega_{\text{full}}} \max\bigl(0,\; m_{\text{full}} - D_{ij}^{\text{NPR}}\bigr), \quad m_{\text{full}} = 0.8
$$

$$
\mathcal{L}_{\text{partial}} = \frac{1}{|\Omega_{\text{partial}}|} \sum_{(i,j) \in \Omega_{\text{partial}}} \max\bigl(0,\; m_{\text{partial}} - D_{ij}^{\text{NPR}}\bigr), \quad m_{\text{partial}} = 0.5
$$

$$
D_{ij}^{\text{NPR}} = 1 - \frac{\mathbf{z}_i^{\text{mixed}}}{\|\mathbf{z}_i^{\text{mixed}}\|_2} \cdot \frac{\mathbf{z}_{\text{target}}^{(j)}}{\|\mathbf{z}_{\text{target}}^{(j)}\|_2}, \quad
\mathbf{z}_i^{\text{mixed}} = \alpha_{ij} \mathbf{z}_i + (1 - \alpha_{ij}) \operatorname{sg}[\mathbf{z}_i]
$$

$$
\alpha_{ij} = \begin{cases} 0.01 & j \in \text{kNN}_{\text{online}}(i) \\ 1.0 & j \notin \text{kNN}_{\text{online}}(i) \end{cases}
$$

$$
\mathcal{L}_{\text{InfoNCE}} = \frac{1}{|\mathcal{V}|} \sum_{i \in \mathcal{V}} -\log \frac{\sum_{j \in \mathcal{P}_i} \exp(S_{ij})}{\sum_{k: k \neq i, \text{id}_k \neq \text{id}_i} \exp(S_{ik})}, \quad S_{ij} = \frac{\tilde{\mathbf{z}}_i^{\top} \tilde{\mathbf{z}}_j}{0.5}
$$

### 11.3 力的平衡

```
┌──────────────────────────────────────────────────┐
│              Item i 受到的力                      │
│                                                   │
│  ← VCF 排斥力 (推开碰撞的非邻居 items)             │
│  → TCL 吸引力 (拉近 Swing 共现 items)             │
│  ←→ Commitment (拉向分配的 centroid)              │
│  ○ NPR 调制 (将 k-NN 邻居方向的排斥力缩小 100×)    │
│  ○ CVPM 屏蔽 (对共现 items 完全不施加排斥力)       │
└──────────────────────────────────────────────────┘
```

**关键设计原则**：

- VCF 排斥力 $\approx$ TCL 吸引力（梯度量级平衡）→ codebook 稳定
- NPR 选择性削弱排斥力（仅对 k-NN 邻居）→ DSF 保真度
- CVPM 完全消除共现对的排斥 → 避免行为信号与语义信号冲突

---

## 12. SID 推理

训练完成后，全量推理（无梯度）：

```python
for each item i:
    z_enc = Encoder(Norm(embedding_i))       # (64,)
    token_0 = argmin ||z_enc - C0||          # L0 nearest neighbor
    z_1 = z_enc - C0[token_0]
    token_1 = argmin ||z_1 - C1||            # L1 nearest neighbor
    z_2 = z_1 - C1[token_1]
    token_2 = argmin ||z_2 - C2||            # L2 nearest neighbor
    SID_i = (token_0, token_1, token_2)
```

输出: `merged_predictions_tensor.pt` — shape $(N, 3)$, $N=12101$ items。

### 碰撞分析

$$
\text{碰撞对} = \{(i,j) : \text{SID}_i = \text{SID}_j\}
$$

$$
\text{唯一 SID 数} = |\{\text{distinct SIDs}\}|
$$

$$
\text{碰撞率} = \frac{N - |\text{唯一 SID 数}|}{N}
$$

$$
\text{最大碰撞组} = \max_{s} |\{i : \text{SID}_i = s\}|
$$

---

## 13. TIGER 下游训练

### 13.1 任务

SID 完成后，TIGER (Generative Retrieval) 将推荐问题转化为序列生成问题：

$$
\text{Input: } [\text{SID}_{t-119}, \dots, \text{SID}_{t-1}, \text{SID}_t] \quad \text{(user history, last 120 items)}
$$

$$
\text{Output: } \text{SID}_{t+1} = (t_0, t_1, t_2) \quad \text{(autoregressive token prediction)}
$$

### 13.2 Token-by-Token 预测

$$
P(\text{SID}_{t+1} \mid \text{history}) = P(t_0 \mid \text{history}) \cdot P(t_1 \mid \text{history}, t_0) \cdot P(t_2 \mid \text{history}, t_0, t_1)
$$

### 13.3 损失函数

$$
\mathcal{L}_{\text{TIGER}} = \sum_{l=0}^{2} \text{CE}\bigl(P_\theta(t_l \mid \text{history}, t_{<l}),\; t_l^{\text{target}}\bigr)
$$

标准 next-token cross-entropy，与 NLP 语言模型相同。

### 13.4 评估

Beam search (beam_size=10) → top-K SIDs → 映射回 item IDs：

$$
\text{NDCG@}K = \frac{1}{|\mathcal{U}|} \sum_{u \in \mathcal{U}} \frac{\sum_{k=1}^{K} \frac{2^{\text{rel}_k} - 1}{\log_2(k+1)}}{\text{IDCG}_u}
$$

$$
\text{Recall@}K = \frac{1}{|\mathcal{U}|} \sum_{u \in \mathcal{U}} \mathbf{1}\bigl[\text{GT}_u \in \text{TopK}_u\bigr]
$$

### 13.5 TIGER 参数

| 参数 | 值 | 含义 |
|------|:---:|------|
| num_hierarchies | 4 | TIGER 自身使用 4 层 (SID 有 3 层, TIGER 额外 1 层) |
| max_steps | 320000 | 总训练步数 |
| batch_size | 32 | 每设备 batch |
| grad_accum | 16 | 梯度累积 → 有效 batch = $32 \times 16 = 512$ |
| lr | 0.001 | 学习率 |
| weight_decay | 0.0001 | 权重衰减 |
| val_check_interval | 1600 | 每 1600 步验证一次 |
| sequence_length | 120 | 用户历史长度 |

### 13.6 SID → TIGER 因果链

```
SID 质量  →  TIGER 效果
  │              │
  ├─ DSF         ├─ NDCG@K
  ├─ 碰撞率      ├─ Recall@K
  ├─ 码本利用率   └─ NDCG+HR@K
  └─ 最大碰撞组
```

**已确认的因果关系**（5 组独立实验验证）:

- $\text{DSF} \uparrow \; \Longrightarrow \; \text{NDCG} \uparrow$（Path B, alpha_grid, DSF-Direct, 4g, RNCL）
- $\text{碰撞率} \not\Longrightarrow \text{NDCG}$ (alpha_grid 中碰撞率最差组 NDCG 最高)
- $\text{L0 码本覆盖率}$ 是必要条件但不充分（TCL_only L0=97.3% NDCG=0.0341 < NPR+TCL L0=91.0% NDCG=0.0355）

---

## 14. 参数速查表

### 当前最优配置 (4g NPR+TCL)

| 参数 | 值 | 类别 |
|------|:---:|------|
| $B$ (batch_size) | 256 | Data |
| SID max_steps | 1000 | Training |
| $L$ (num_hierarchies) | 3 | RQ-VAE |
| $C$ (codebook_width) | 256 | RQ-VAE |
| $d_{\text{emb}}$ | 768 | Input |
| $R$ (hamming_radius) | 1 | VCF |
| $\lambda_{\text{full}} / \lambda_{\text{partial}}$ | 0.2 / 0.1 | VCF |
| $m_{\text{full}} / m_{\text{partial}}$ | 0.8 / 0.5 (static) | VCF |
| repulsion_warmup | 0 | VCF |
| use_cvpm | true | CVPM |
| $\tau_{\text{cvpm}}$ | 0.15 | CVPM |
| $\alpha_{\min}^{\text{npr}}$ | 0.01 | NPR |
| $\lambda_{\text{cl}}$ | 0.1 | TCL |
| $\tau$ (quasid_cl_tau) | 0.5 | TCL |
| $\gamma$ (online_knn_ema) | 0.99 | Online k-NN |
| $K$ (online_knn_k) | 50 | Online k-NN |
| $\Delta k$ (knn_interval) | 100 | Online k-NN |
| $\lambda_{\text{ent}}$ | 0.1 | Regularization |
| codebook_reset_interval | 10 | Maintenance |
| optimizer | Adam | Optim |
| $\eta$ (lr) | $3\times10^{-4}$ | Optim |
| weight_decay | $1\times10^{-5}$ | Optim |
| seed | 42 | Reproducibility |

### TIGER 参数

| 参数 | 值 |
|------|:---:|
| num_hierarchies | 4 |
| max_steps | 320000 |
| $B_{\text{tiger}}$ | 32 |
| grad_accum | 16 |
| lr | 0.001 |
| weight_decay | 0.0001 |
| val_check_interval | 1600 |
| sequence_length | 120 |

---

## 15. 实验验证的因果链

```
NPR (α=0.01) ──→ 保护 k-NN 邻居不被 VCF 推远
                      │
                      ▼
              更高的 DSF_online (0.175 vs 0.168)
                      │
                      ▼
              更好的 SID 语义质量
                      │
                      ▼
              更高 TIGER NDCG@10 (+4.1% vs TCL_only)
```

**已排除的替代假说**：

- ❌ "吸引力损失可进一步提升 DSF" (KPL −5.9%, RNCL −9.3% — 与 TCL 信号冗余)
- ❌ "碰撞率越低越好" (NPR 碰撞率 9.4% > TCL_only 7.0%，但 NDCG 更高)
- ❌ "码本利用率越高越好" (TCL_only L0=97.3% 但 NDCG < NPR+TCL L0=91.0%)
