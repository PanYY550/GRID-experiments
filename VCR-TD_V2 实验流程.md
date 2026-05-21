**✅ VCR-TD v2.0 完整实验流程（1.5个月冲刺版）**  
**目标**：在两张 A800 GPU 上，用 Amazon Reviews 2023 Raw 数据集完成高质量 B 类论文实验（CIKM 2026 标准），故事线按 A 类打造。  
**总时长**：6 周（每周 5–6 天实验时间）。  
**硬件约束**：两张 A800（160GB 总显存）+ 严格显存优化（BF16 + Gradient Checkpointing + LoRA + 离线特征映射）。  

---

### **实验总体流程图（6周）**

| 周次 | 阶段名称               | 主要任务                                  | 产出物                            | 预计耗时 |
| ---- | ---------------------- | ----------------------------------------- | --------------------------------- | -------- |
| 1    | 环境 & 数据准备        | GRID 搭建 + Raw 数据降采样 + 特征提取     | 内存映射特征文件 + 时间戳字典     | 5–6天    |
| 2    | 基线实现               | GRID Baseline + Static HaMR (QuaSID-like) | 两个可复现基线 + 初步指标         | 6天      |
| 3–4  | VCR-TD 核心实现与训练  | Full VCR-TD + 4组消融 + 超参搜索          | 完整实验日志 + Checkpoint         | 10–12天  |
| 5    | 评估、可视化与结果分析 | 指标计算 + t-SNE 图 + 分层曲线            | 论文级图表 + 表格数据             | 5天      |
| 6    | 论文准备与复现打包     | 实验脚本整理 + Reproducibility 文档       | GitHub 仓库 + 论文 Experiments 节 | 4–5天    |

---

### **Phase 1: 环境搭建与数据准备（Week 1）**

**1.1 服务器环境**
```bash
# 创建 conda 环境
conda create -n vcr-td python=3.11
conda activate vcr-td

# 安装核心依赖（GRID 官方 + 显存优化）
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install transformers accelerate bitsandbytes peft deepspeed
pip install -r requirements.txt   # GRID 官方 requirements

# 可选：LoRA + 8-bit 优化
pip install bitsandbytes
```

**1.2 数据准备（关键：Raw + 防泄漏）**
- 下载 Amazon Reviews 2023 Raw（Beauty 或 Toys 单一领域，控制规模适配 A800）。
- **降采样策略**（Gemini 建议）：随机选取 10%–20% 物品（约 400–800 万物品），保证长尾分布。
- **时间戳处理**：
  - 保留原始 `timestamp`（毫秒级）。
  - 构建 **Item Exposure Hash Table**（Python dict 或 Redis 轻量版）。
  - 采用 **Historical Time Masking**：训练数据加载器必须严格按时间顺序加载 mini-batch，只允许查询当前 batch 之前的历史交互次数作为 \( t \)。
- **冷启动模拟**：
  - 在测试集中随机选择 20% 长尾物品，将其历史交互全部 Mask 为 \( t=0 \)。
  - 额外划分 Extreme Cold-Start 子集（交互 ≤3 次）。

**输出目录结构**：
```
data/
├── amazon_raw_beauty.csv
├── features/          # Sentence-T5 768-dim npy/hdf5（内存映射）
├── timestamps.pkl     # Item -> 历史交互次数字典（动态更新）
├── cold_start_mask.json
```

---

### **Phase 2: 基线实现（Week 2）**

**2.1 GRID Baseline**
- 直接运行 GRID 官方 `rkmeans_train_flat.py` 或 RQ-VAE 训练脚本。
- 记录 NDCG@10、HR@10、Entropy 等指标作为零点。

**2.2 Static HaMR (QuaSID-like)**
使用我之前提供的 `haMR_loss.py` + `cvpm_mask`（已验证 <100 行）。

核心代码片段（插入 RQ-VAE 训练 loop）：
```python
# 在量化后、重建损失前插入
z_enc = encoder_output.float()          # [2B, D]
sids = quantized_indices                # [2B, L]
ids = batch_item_ids                    # underlying item ID
positive_pairs = batch_positive_mask    # contrastive positives

cvpm = cvpm_mask(ids, positive_pairs)
loss_haMR = haMR_loss(z_enc, sids, mask=cvpm,
                      m_full=0.8, m_partial=0.5, R=1,
                      lambda_full=0.2, lambda_partial=0.1)

total_loss = total_loss + 0.3 * loss_haMR   # λ_haMR = 0.3（可调）
```

**2.3 额外宏观基线（Gemini 建议）**
- SASRec（PyTorch 官方实现）
- Vanilla TIGER（GRID 自带生成器，不加任何 repulsion）

---

### **Phase 3: VCR-TD 核心实现与训练（Week 3–4）**

**3.1 VCR-TD Full 实现**
在 `haMR_loss.py` 基础上新增：
```python
def vcr_td_repulsion(z_enc, sids, ids, t_values, mask=None):
    # t_values: 当前 batch 中每个物品的历史交互次数 [2B]
    w = torch.exp(-alpha * t_values)                    # 时间衰减
    m_dynamic = m0 * (1 - cosine_sim(x_i, x_j))        # 语义感知动态边界
    loss = w * torch.relu(m_dynamic - cosine_dist)
    return (loss * mask).mean()
```

**3.2 训练命令（两张 A800）**
```bash
torchrun --nproc_per_node=2 train_vcr_td.py \
    --model GRID_RQVAE \
    --dataset amazon_beauty_raw \
    --batch_size 8192 \
    --lr 3e-4 \
    --bf16 \
    --gradient_checkpointing \
    --lora_r 16 \
    --lambda_rep 0.3 \
    --alpha 0.01 \
    --output_dir ./checkpoints/vcr_td_full
```

**3.3 4组消融实验（并行运行）**
1. GRID Baseline（官方）
2. Static HaMR（QuaSID-like）
3. VCR-TD w/o Time Decay（w(t)=1）
4. VCR-TD Full

每组保存独立 checkpoint。

---

### **Phase 4: 评估与可视化（Week 5）**

**4.1 指标计算**
- HR@10, NDCG@10
- Recall@20（Tail / Zero-shot / Extreme Cold-Start）
- Entropy（SID 多样性）
- Gini 系数（流量去中心化）

**4.2 可视化（A类必备）**
- **t-SNE 生命周期演化图**：提取 3 个典型冷启动物品在 \( t=0 \)、\( t=10 \)、\( t=100 \) 时的 pre-quantization embedding，绘制轨迹。
- **分桶性能柱状图**：冷启动 / 长尾 / 头部三个桶的 Recall 提升对比。

**4.3 结果表格模板**（直接用于论文）

| Variant               | NDCG@10  | Recall@20 (Tail) | Recall@20 (Zero-shot) | Entropy  |
| --------------------- | -------- | ---------------- | --------------------- | -------- |
| GRID Baseline         |          |                  |                       |          |
| Static HaMR           |          |                  |                       |          |
| VCR-TD w/o Time Decay |          |                  |                       |          |
| VCR-TD Full           | **Best** | **Best**         | **Best**              | **Best** |

---

### **Phase 5: 论文准备与复现打包（Week 6）**

- 所有实验脚本统一放入 `experiments/` 文件夹
- 提供 `reproduce.sh` 一键脚本
- 包含 `requirements.txt` + `seed=42` + `hardware=A800-2`
- Experiments 节必须写明：“All experiments are conducted on two A800 GPUs with BF16, Gradient Checkpointing and LoRA. We use historical time masking to strictly prevent data leakage.”

