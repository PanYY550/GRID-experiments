# VCF (VCR-TD-QuaSID Fusion) 损失函数详解

本文档详细说明 GRID 项目中 RQ-VAE 训练阶段的所有损失函数，包括公式推导和代码实现对应关系。

---

## 1. 基础 RQ-VAE 损失

### 1.1 量化损失 (Commitment Loss)

RQ-VAE 使用残差量化机制，将连续嵌入 $\mathbf{z} \in \mathbb{R}^D$ 编码为 $L$ 层离散 token 序列。

对于第 $l$ 层量化：

$$
\mathbf{r}_l = \mathbf{z} - \sum_{i=0}^{l-1} \mathbf{e}_{s_i}, \quad \mathbf{r}_0 = \mathbf{z}
$$

其中 $\mathbf{e}_{s_i}$ 是第 $i$ 层选中的 codebook 向量。

**量化损失**（VQ-VAE 标准 commitment loss）：

$$
\mathcal{L}_{\text{quant}} = \sum_{l=0}^{L-1} \| \operatorname{sg}[\mathbf{r}_l] - \mathbf{e}_{s_l} \|_2^2 + \beta \| \mathbf{r}_l - \operatorname{sg}[\mathbf{e}_{s_l}] \|_2^2
$$

其中：
- $\operatorname{sg}[\cdot]$ 表示 stop-gradient 操作
- $\beta$ 是 commitment 权重（通常 $\beta = 0.25$）
- 第一项迫使 codebook 向量靠近嵌入
- 第二项迫使嵌入靠近 codebook 向量

### 1.2 重构损失 (Reconstruction Loss)

将量化后的嵌入 $\hat{\mathbf{z}} = \sum_{l=0}^{L-1} \mathbf{e}_{s_l}$ 输入解码器，重构原始嵌入：

$$
\mathcal{L}_{\text{recon}} = \| \mathbf{z} - \operatorname{Decoder}(\hat{\mathbf{z}}) \|_2^2
$$

在项目中，实际使用的是**连续嵌入上的 MSE**（简化版）：

$$
\mathcal{L}_{\text{recon}} = \frac{1}{B} \sum_{i=1}^{B} \| \mathbf{z}_i - \hat{\mathbf{z}}_i \|_2^2
$$

---

## 2. VCF 排斥损失 (VCF Repulsion Loss)

VCF 排斥损失是核心创新，融合了三项机制：
1. **HaMR** (Hamming-based Margin Regulation)：基于汉明距离的边界调节
2. **动态边界** (Dynamic Margin)：根据内容相似度和碰撞严重度自适应调整
3. **时间权重** (Time Decay)：根据曝光次数调节排斥强度

### 2.1 碰撞检测与分类

对于 batch 中的 $B$ 个物品，计算 SID 的汉明距离矩阵 $\mathbf{H} \in \mathbb{N}^{B \times B}$：

$$
H_{ij} = \sum_{l=0}^{L-1} \mathbb{1}[s_i^{(l)} \neq s_j^{(l)}]
$$

其中 $s_i^{(l)}$ 是物品 $i$ 在第 $l$ 层的 token。

**碰撞分类**（给定半径 $R$，通常 $R=2$）：

$$
\Omega_{\text{full}} = \{(i,j) : H_{ij} = 0, i \neq j\}
$$

$$
\Omega_{\text{partial}} = \{(i,j) : 0 < H_{ij} \leq R, i \neq j\}
$$

### 2.2 CVPM 掩码 (Conflict-Aware Valid Pair Masking)

在计算排斥前，先过滤掉不应该被排斥的对：

$$
M_{ij} = \begin{cases}
0 & \text{if } i = j \text{ (同物品)} \\
0 & \text{if } (i,j) \in \mathcal{P}^+ \text{ (正样本对)} \\
1 & \text{otherwise}
\end{cases}
$$

其中 $\mathcal{P}^+$ 是训练集中的正样本对（用户交互过的物品对）。

**有效碰撞对**：

$$
\tilde{\Omega}_{\text{full}} = \Omega_{\text{full}} \odot \mathbf{M}, \quad \tilde{\Omega}_{\text{partial}} = \Omega_{\text{partial}} \odot \mathbf{M}
$$

### 2.3 静态边界 (Static Margin)

**Q0 (GQSS) 基线配置**使用静态边界：

$$
m_{ij} = \begin{cases}
m_{\text{full}} & \text{if } H_{ij} = 0 \\
m_{\text{partial}} & \text{if } 0 < H_{ij} \leq R \\
0 & \text{otherwise}
\end{cases}
$$

默认参数：$m_{\text{full}} = 0.5$，$m_{\text{partial}} = 0.3$。

### 2.4 动态边界 (Dynamic Margin)

**B1/A2 配置**使用动态边界，融合内容相似度和碰撞严重度：

#### 2.4.1 内容相似度

计算内容嵌入的余弦相似度：

$$
\text{sim}_{ij} = \frac{\mathbf{c}_i^{\top} \mathbf{c}_j}{\|\mathbf{c}_i\|_2 \|\mathbf{c}_j\|_2} \in [-1, 1]
$$

语义差异度：

$$
\text{dissim}_{ij} = 1 - \text{sim}_{ij} \in [0, 2]
$$

#### 2.4.2 基础边界

$$
m_{ij}^{\text{base}} = m_0 + w_{\text{dissim}} \cdot \text{dissim}_{ij}
$$

其中：
- $m_0$：基础边界（兜底值，默认 0.5）
- $w_{\text{dissim}}$：差异权重（默认 0.5）

**关键修复**：早期版本使用 $m_{ij}^{\text{base}} = m_0 \times (1 - \text{sim}_{ij})$，导致相似物品 margin 趋近于 0，排斥失效。修复后使用加法确保 $m_{ij}^{\text{base}} \geq m_0$。

#### 2.4.3 严重度因子

根据碰撞类型和距离调整严重度：

$$
\text{severity}_{ij} = \begin{cases}
1 + \beta & \text{if } H_{ij} = 0 \text{ (完全碰撞)} \\
1 + \beta \cdot \left(1 - \frac{H_{ij}}{R}\right) & \text{if } 0 < H_{ij} \leq R \\
0 & \text{otherwise}
\end{cases}
$$

其中 $\beta$ 是严重度调节系数（默认 0.5）。

**最终动态边界**：

$$
m_{ij} = \max\left( m_{ij}^{\text{base}} \cdot \text{severity}_{ij}, m_{\min} \right)
$$

其中 $m_{\min}$ 是最小边界保护（默认 0.05）。

### 2.5 时间权重 (Time Decay)

根据曝光次数调节排斥强度，实现"冷启动强排斥、成熟物品弱排斥"。

#### 2.5.1 对数压缩

对于碰撞对 $(i,j)$，取最小曝光次数：

$$
t_{ij} = \min(t_i, t_j)
$$

使用 log1p 压缩大值：

$$
t_{ij}^{\log} = \log(1 + t_{ij})
$$

#### 2.5.2 指数衰减

基础时间权重：

$$
w_{ij}^{\text{base}} = \exp(-\alpha \cdot t_{ij}^{\log}) = (1 + t_{ij})^{-\alpha}
$$

其中 $\alpha$ 是衰减系数（默认 0.5）。

**当 $\alpha = 0.5$ 时**：

| 曝光 $t$ | $w^{\text{base}}$ | 排斥力度 |
|---------|------------------|---------|
| 0 | 1.0 | 100% |
| 10 | 0.30 | 30% |
| 100 | 0.14 | 14% |
| 1000 | 0.09 | 9% |

#### 2.5.3 冷启动增强

对于完全碰撞且处于冷启动阶段的对，增强排斥：

$$
w_{ij}^{\text{enhance}} = \min\left( 1 + \gamma \cdot \mathbb{1}[H_{ij}=0] \cdot \mathbb{1}[t_{ij} < \tau], w_{\max} \right)
$$

其中：
- $\gamma$：增强因子（默认 0.5）
- $\tau$：冷启动阈值（默认 100）
- $w_{\max}$：最大权重上限（默认 2.0）

**最终时间权重**：

$$
w_{ij} = w_{ij}^{\text{base}} \cdot w_{ij}^{\text{enhance}}
$$

### 2.6 铰链损失 (Hinge Loss)

计算连续嵌入的归一化余弦距离：

$$
D_{ij} = 1 - \frac{\mathbf{z}_i^{\top} \mathbf{z}_j}{\|\mathbf{z}_i\|_2 \|\mathbf{z}_j\|_2} \in [0, 2]
$$

**铰链损失**（推动碰撞对距离超过边界）：

$$
\mathcal{L}_{\text{hinge}}^{(ij)} = \max(0, m_{ij} - D_{ij})
$$

### 2.7 VCF 排斥损失汇总

**完全碰撞损失**：

$$
\mathcal{L}_{\text{full}} = \frac{1}{|\tilde{\Omega}_{\text{full}}|} \sum_{(i,j) \in \tilde{\Omega}_{\text{full}}} w_{ij} \cdot \max(0, m_{ij} - D_{ij})
$$

**部分碰撞损失**：

$$
\mathcal{L}_{\text{partial}} = \frac{1}{|\tilde{\Omega}_{\text{partial}}|} \sum_{(i,j) \in \tilde{\Omega}_{\text{partial}}} w_{ij} \cdot \max(0, m_{ij} - D_{ij})
$$

**总排斥损失**：

$$
\mathcal{L}_{\text{repulsion}}^{\text{VCF}} = \lambda_{\text{full}} \cdot \mathcal{L}_{\text{full}} + \lambda_{\text{partial}} \cdot \mathcal{L}_{\text{partial}}
$$

默认参数：$\lambda_{\text{full}} = 0.3$，$\lambda_{\text{partial}} = 0.2$。

---

## 3. TCCL 时间感知协同对比损失

### 3.1 成熟度计算

对于每个物品，计算其成熟度（基于曝光次数）：

$$
\text{maturity}_i = 1 - \exp(-\alpha_{\text{cl}} \cdot t_i) \in [0, 1]
$$

其中 $\alpha_{\text{cl}}$ 是成熟度衰减率（默认 0.005）。

**Batch 平均成熟度**：

$$
\bar{m} = \frac{1}{B} \sum_{i=1}^{B} \text{maturity}_i
$$

### 3.2 TCCL 损失

TCCL 使用对比学习框架，拉近正样本对，推开负样本对。

**正样本对**：用户交互过的物品对 $(i, j) \in \mathcal{P}^+$。

**负样本对**：同 batch 中未交互的物品对。

**对比损失**（InfoNCE 变体）：

$$
\mathcal{L}_{\text{TCCL}} = -\frac{1}{|\mathcal{P}^+|} \sum_{(i,j) \in \mathcal{P}^+} \log \frac{\exp(\text{sim}(\mathbf{z}_i, \mathbf{z}_j) / \tau)}{\sum_{k \in \mathcal{N}(i)} \exp(\text{sim}(\mathbf{z}_i, \mathbf{z}_k) / \tau)}
$$

其中：
- $\tau$：温度系数（默认 0.07）
- $\mathcal{N}(i)$：物品 $i$ 的负样本集合

### 3.3 时间调节

**排斥损失调节**：

$$
\lambda_{\text{rep}}^{\text{eff}} = \lambda_{\text{rep}} \cdot (1 - \bar{m})
$$

**TCCL 损失调节**：

$$
\lambda_{\text{cl}}^{\text{eff}} = \lambda_{\text{cl}} \cdot \bar{m}
$$

**含义**：
- 早期（$\bar{m} \approx 0$）：强排斥，弱对比
- 后期（$\bar{m} \approx 1$）：弱排斥，强对比

### 3.4 Warmup 机制

为防止 TCCL 过早激活导致码本坍塌，引入 warmup：

$$
\mathcal{L}_{\text{TCCL}}^{\text{gated}} = \mathbb{1}[\text{step} \geq \text{warmup\_steps}] \cdot \mathcal{L}_{\text{TCCL}}
$$

**修复版参数**：warmup_steps = 350（等 layer0 coverage 恢复后再激活）。

---

## 4. 总损失函数

### 4.1 VCF 训练总损失

$$
\mathcal{L}_{\text{total}}^{\text{VCF}} = \mathcal{L}_{\text{recon}} + \lambda_{\text{quant}} \cdot \mathcal{L}_{\text{quant}} + \mathcal{L}_{\text{repulsion}}^{\text{VCF}} + \mathcal{L}_{\text{TCCL}}^{\text{gated}}
$$

**各项系数**：

| 损失项 | 系数 | 默认/说明 |
|-------|------|----------|
| 重构损失 | 1.0 | 固定 |
| 量化损失 | $\lambda_{\text{quant}}$ | 1.0 |
| VCF排斥 | $\lambda_{\text{full}}$, $\lambda_{\text{partial}}$ | 0.3, 0.2 |
| TCCL | $\lambda_{\text{cl}}$ | 0.0 (Q0) / 0.2 (A3) |

### 4.2 不同实验配置的等价损失

#### Q0 (GQSS) - 推荐配置

$$
\mathcal{L}_{\text{Q0}} = \mathcal{L}_{\text{recon}} + \mathcal{L}_{\text{quant}} + \lambda_{\text{full}} \cdot \mathcal{L}_{\text{full}}^{\text{static}} + \lambda_{\text{partial}} \cdot \mathcal{L}_{\text{partial}}^{\text{static}}
$$

其中：
- $\mathcal{L}_{\text{full}}^{\text{static}}$：使用静态边界 $m_{\text{full}} = 0.5$
- $\mathcal{L}_{\text{partial}}^{\text{static}}$：使用静态边界 $m_{\text{partial}} = 0.3$
- 无动态边界、无时间权重、无 TCCL

#### B1 (动态边界 + 时间权重)

$$
\mathcal{L}_{\text{B1}} = \mathcal{L}_{\text{recon}} + \mathcal{L}_{\text{quant}} + \mathcal{L}_{\text{repulsion}}^{\text{dynamic}}(\alpha=0.5)
$$

其中：
- 使用动态边界（含内容相似度、严重度因子）
- 使用时间权重（$\alpha = 0.5$）
- 无 TCCL

#### A3 (TCCL)

$$
\mathcal{L}_{\text{A3}} = \mathcal{L}_{\text{recon}} + \mathcal{L}_{\text{quant}} + \mathcal{L}_{\text{repulsion}}^{\text{static}} + \mathcal{L}_{\text{TCCL}}^{\text{gated}}
$$

其中：
- 使用静态边界
- TCCL warmup = 200，$\lambda_{\text{cl}} = 0.2$

---

## 5. 稳定性机制

### 5.1 Codebook Health Gate

当码本健康度低于阈值时，临时关闭排斥损失：

$$
\mathcal{L}_{\text{repulsion}} = 0 \quad \text{if } \left( \text{coverage}_0 < \theta_{\text{cov}} \right) \lor \left( \text{frac\_unique} < \theta_{\text{unique}} \right)
$$

默认阈值：$\theta_{\text{cov}} = 0.50$，$\theta_{\text{unique}} = 0.50$。

### 5.2 Pair Subsampling

为避免 $O(B^2)$ 的碰撞对数量爆炸，对每个样本限制最大碰撞对数：

$$
|\Omega_i| \leq K \quad \text{for each item } i
$$

其中 $K$ 是每样本最大对数（默认 64）。

### 5.3 Repulsion Clipping

对排斥损失进行上限裁剪，防止梯度爆炸：

$$
\mathcal{L}_{\text{repulsion}} = \min(\mathcal{L}_{\text{repulsion}}, \text{clip\_max})
$$

---

## 6. 关键公式速查表

| 名称 | 公式 | 默认参数 |
|-----|------|---------|
| 汉明距离 | $H_{ij} = \sum_{l} \mathbb{1}[s_i^{(l)} \neq s_j^{(l)}]$ | - |
| 余弦距离 | $D_{ij} = 1 - \frac{\mathbf{z}_i^{\top} \mathbf{z}_j}{\|\mathbf{z}_i\| \|\mathbf{z}_j\|}$ | - |
| 静态边界 | $m_{ij} \in \{0, m_{\text{full}}, m_{\text{partial}}\}$ | 0.5, 0.3 |
| 动态边界基础 | $m_{ij}^{\text{base}} = m_0 + w_{\text{dissim}} \cdot \text{dissim}_{ij}$ | 0.5, 0.5 |
| 严重度 | $\text{severity}_{ij} = 1 + \beta$ (full) / $1 + \beta(1 - H/R)$ (partial) | $\beta = 0.5$ |
| 时间权重 | $w_{ij} = (1 + t_{ij})^{-\alpha} \cdot w_{ij}^{\text{enhance}}$ | $\alpha = 0.5$ |
| 铰链损失 | $\mathcal{L}_{\text{hinge}} = \max(0, m - D)$ | - |
| 成熟度 | $\text{maturity}_i = 1 - \exp(-\alpha_{\text{cl}} \cdot t_i)$ | $\alpha_{\text{cl}} = 0.005$ |

---

## 7. 实验结果与损失函数的关系

| 实验 | 动态边界 | 时间权重 | TCCL | Layer0 Coverage | TIGER NDCG@10 |
|-----|---------|---------|------|----------------|---------------|
| G0 | No | No | No | 0.71 | 0.0326 |
| Q0 | No | No | No | 0.48 | **0.0333** |
| A1 | No | Yes ($\alpha=0.5$) | No | 0.44 | - |
| A2 | Yes | No | No | **0.46** | - |
| A3 | No | No | Yes (warmup=200) | 0.12 | - |
| B1 | Yes | Yes | No | 0.49 | 0.0316 |
| B2 | No | No | Yes (warmup=350) | 0.38 | - |

**结论**：
- 动态边界在 400-step 略优，但 3000-step 未转化为 TIGER 收益
- 时间权重方向可能存在问题（热门商品排斥被削弱）
- TCCL 无论 warmup 如何都导致码本健康度下降
