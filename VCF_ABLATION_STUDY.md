# VCF (VCR-TD-QuaSID Fusion) 消融实验方案

## 1. 实验目标

验证VCF融合方案中各组件的有效性，量化每个模块对最终性能的贡献。

## 2. 实验组设计

### 2.1 基线组

| 组号 | 名称 | 说明 |
|------|------|------|
| **G0** | GRID Baseline | 原始RQ-VAE，无任何排斥机制 |

### 2.2 VCF核心组件消融

| 组号 | 名称 | use_vcf | CVPM | Hamming检测 | 动态边界 | 时间权重 | 说明 |
|------|------|---------|------|-------------|----------|----------|------|
| **G1** | VCF-Full | True | True | True | True | True | 完整VCF方案 |
| **G2** | VCF w/o CVPM | True | False | True | True | True | 移除协议良性对筛选 |
| **G3** | VCF w/o Hamming | True | True | False | True | True | 不检测碰撞，对所有对计算 |
| **G4** | VCF w/o Dynamic Margin | True | True | True | False | True | 使用QuaSID静态边界 m_full/m_partial |
| **G_QSS** | QuaSID-Static Baseline | True | True | True | False | False | QuaSID静态边界+无时间衰减+无TCCL |
| **G5** | VCF w/o Time Weight | True | True | True | True | False | 移除时间衰减 |
| **G6** | VCF w/o TCCL | True | True | True | True | True | use_tcl=False |

### 2.3 参数敏感性分析

| 组号 | 名称 | 参数变化 | 说明 |
|------|------|----------|------|
| **G7** | VCF-alpha-0.001 | alpha=0.001 | 极慢时间衰减 |
| **G8** | VCF-alpha-0.01 | alpha=0.01 | 默认时间衰减 |
| **G9** | VCF-alpha-0.1 | alpha=0.1 | 快速时间衰减 |
| **G10** | VCF-m0-0.3 | m0=0.3 | 较小基础边界 |
| **G11** | VCF-m0-0.5 | m0=0.5 | 默认基础边界 |
| **G12** | VCF-m0-0.7 | m0=0.7 | 较大基础边界 |
| **G13** | VCF-R-1 | hamming_radius=1 | QuaSID离线实验设置 |
| **G14** | VCF-R-2 | hamming_radius=2 | 默认部分碰撞定义 |
| **G15** | VCF-R-3 | hamming_radius=3 | 宽松部分碰撞定义 |

## 3. 评估指标

### 3.1 量化质量指标
- **Reconstruction Loss**: 内容重建误差
- **Quantization Loss**: 量化损失
- **Codebook Utilization**: 码本使用率
- **Unique ID Ratio**: 唯一SID比例

### 3.2 碰撞相关指标
- **Full Collision Rate**: 完全碰撞率 (H=0)
- **Partial Collision Rate**: 部分碰撞率 (0<H≤R)
- **Repulsion Loss**: 排斥损失值
- **Average Margin**: 平均动态边界

### 3.3 推荐性能指标（需下游任务评估）
- **Recall@K**: Top-K召回率
- **NDCG@K**: 归一化折损累计增益
- **Cold-start Recall@K**: 冷启动物品召回率

## 4. 实验配置

### 4.1 公共配置

```yaml
# 数据配置
data_dir: /path/to/data
embedding_path: /path/to/embeddings.pt
embedding_dim: 256
num_hierarchies: 4
codebook_width: 256

# 训练配置（重要：必须显式覆盖，否则默认只有30步！）
batch_size_per_device: 4096  # 参考您之前的实验设置
max_steps: 3000  # SID训练步数（必须显式设置，默认只有30步）
seed: 42
accelerator: gpu
devices: 1

# 早停配置（参考QuaSID论文：NDCG@5+HR@5连续10次未改善则停止）
# 注意：当前框架主要监控train/loss，如需监控NDCG@5+HR@5需要下游评估
# 这里使用val/loss作为替代监控指标，patience=10
callbacks:
  early_stopping:
    _target_: lightning.pytorch.callbacks.EarlyStopping
    monitor: val/loss
    patience: 10
    min_delta: 0.001
    mode: min
    verbose: true

# 优化器（参考QuaSID论文设置）
optimizer:
  _target_: torch.optim.Adam
  lr: 3e-4  # QuaSID使用3×10⁻⁴
  weight_decay: 1e-5  # QuaSID使用1×10⁻⁵
```

### 4.2 各组具体配置

#### G0: GRID Baseline
```yaml
model:
  use_vcr_td: false
  use_vcf: false
  use_tcl: false
```

#### G1: VCF-Full
```yaml
model:
  use_vcr_td: false
  use_vcf: true
  use_tcl: true
  hamming_radius: 2
  severity_beta: 0.5
  cold_start_threshold: 50
  enhancement_factor: 0.5
  lambda_full: 1.0
  lambda_partial: 0.5
  min_margin: 0.05
  max_enhance: 2.0
  min_severity_partial: 1.1
  repulsion_warmup_steps: 1000
  alpha: 0.01
  m0: 0.5
```

#### G2: VCF w/o CVPM
```yaml
model:
  use_vcf: true
  use_cvpm: false  # 关闭CVPM
```

#### G3: VCF w/o Hamming
```yaml
model:
  use_vcf: true
  # Hamming检测无法直接关闭，需修改代码或设置R极大值
  hamming_radius: 999  # 近似关闭Hamming过滤
```

#### G4: VCF w/o Dynamic Margin
```yaml
model:
  use_vcf: true
  use_dynamic_margin: false  # 关闭动态边界，退化为QuaSID静态边界
  m_full: 0.8                 # 完全碰撞静态边界
  m_partial: 0.5              # 部分碰撞静态边界
```

#### G_QSS: QuaSID-Static Baseline
```yaml
model:
  use_vcf: true
  use_dynamic_margin: false  # 静态边界
  m_full: 0.8                 # 完全碰撞边界
  m_partial: 0.5              # 部分碰撞边界
  alpha: 0.0                  # 关闭时间衰减
  use_tcl: false              # 关闭TCCL
```
**说明**: 这是证明"VCF > QuaSID"的最关键对比组。保留CVPM+HaMR的静态版本，但移除VCF的时间衰减和TCCL组件。

#### G5: VCF w/o Time Weight
```yaml
model:
  use_vcf: true
  alpha: 0.0  # 关闭时间衰减
```

#### G6: VCF w/o TCCL
```yaml
model:
  use_vcf: true
  use_tcl: false
```

## 5. 实验脚本

### 5.1 批量运行脚本

```bash
#!/bin/bash
# vcf_ablation_study.sh

# 配置（重要：SID_MAX_STEPS必须显式设置，否则默认只有30步！）
DATA_DIR="/path/to/data"
EMB_PATH="/path/to/embeddings.pt"
EMB_DIM=256
NUM_HIER=4
CODEBOOK=256
SID_MAX_STEPS=3000  # SID训练步数（必须显式设置，默认只有30步！）
GPU_LIST=(0 1 2 3)

# 实验组定义（注意：必须显式传入 trainer.max_steps，否则默认只有30步！）
declare -A EXPERIMENTS
EXPERIMENTS[G0]="model.use_vcr_td=false model.use_vcf=false model.use_cvpm=false model.use_dynamic_margin=false model.use_tcl=false model.alpha=0.0"
EXPERIMENTS[G1]="model.use_vcr_td=false model.use_vcf=true model.use_cvpm=true model.use_dynamic_margin=true model.use_tcl=true model.hamming_radius=2 model.severity_beta=0.5 model.cold_start_threshold=50 model.enhancement_factor=0.5 model.lambda_full=1.0 model.lambda_partial=0.5 model.min_margin=0.05 model.max_enhance=2.0 model.min_severity_partial=1.1 model.repulsion_warmup_steps=1000 model.alpha=0.01 model.m0=0.5 model.lambda_cl=0.2 model.cl_tau=0.07 model.alpha_cl=0.01"
EXPERIMENTS[G2]="model.use_vcf=true model.use_cvpm=false"
EXPERIMENTS[G4]="model.use_vcf=true model.use_dynamic_margin=false model.m_full=0.8 model.m_partial=0.5"
EXPERIMENTS[G_QSS]="model.use_vcf=true model.use_dynamic_margin=false model.m_full=0.8 model.m_partial=0.5 model.alpha=0.0 model.use_tcl=false"
EXPERIMENTS[G5]="model.use_vcf=true model.alpha=0.0"
EXPERIMENTS[G6]="model.use_vcf=true model.use_tcl=false"
EXPERIMENTS[G7]="model.use_vcf=true model.alpha=0.001"
EXPERIMENTS[G9]="model.use_vcf=true model.alpha=0.1"
EXPERIMENTS[G10]="model.use_vcf=true model.m0=0.3"
EXPERIMENTS[G12]="model.use_vcf=true model.m0=0.7"
EXPERIMENTS[G13]="model.use_vcf=true model.hamming_radius=1"
EXPERIMENTS[G15]="model.use_vcf=true model.hamming_radius=3"

# 运行实验
gpu_idx=0
for exp_name in G0 G1 G2 G4 G5 G6 G7 G9 G10 G12 G13 G15 G_QSS; do
    gpu=${GPU_LIST[$gpu_idx]}
    output_dir="outputs/ablation/${exp_name}"
    
    echo "[$(date)] Starting ${exp_name} on GPU ${gpu}"
    
    # 重要：必须显式传入 trainer.max_steps=3000，否则默认只有30步！
    CUDA_VISIBLE_DEVICES=$gpu python src/train.py \
        experiment=rqvae_vcf_train_flat \
        data_dir=$DATA_DIR \
        embedding_path=$EMB_PATH \
        embedding_dim=$EMB_DIM \
        num_hierarchies=$NUM_HIER \
        codebook_width=$CODEBOOK \
        trainer.max_steps=3000 \
        ${EXPERIMENTS[$exp_name]} \
        paths.output_dir=$output_dir \
        > ${output_dir}/train.log 2>&1 &
    
    gpu_idx=$(( (gpu_idx + 1) % ${#GPU_LIST[@]} ))
    sleep 5
done

echo "[$(date)] All experiments launched!"
wait
echo "[$(date)] All experiments completed!"
```

### 5.2 结果分析脚本

```python
#!/usr/bin/env python3
# analyze_vcf_ablation.py

import os
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def load_metrics(output_dir):
    """加载训练日志中的指标"""
    metrics_file = Path(output_dir) / "csv" / "metrics.csv"
    if not metrics_file.exists():
        return None
    return pd.read_csv(metrics_file)

def analyze_experiment(exp_name, output_dir):
    """分析单个实验的结果"""
    df = load_metrics(output_dir)
    if df is None:
        print(f"Warning: No metrics found for {exp_name}")
        return None
    
    # 提取最终指标
    final_row = df.iloc[-1]
    
    results = {
        'experiment': exp_name,
        'final_loss': final_row.get('train/loss', float('nan')),
        'final_quant_loss': final_row.get('train/quantization_loss', float('nan')),
        'final_repulsion_loss': final_row.get('train/repulsion_loss', float('nan')),
        'final_recon_loss': final_row.get('train/reconstruction_loss', float('nan')),
    }
    
    return results

def main():
    base_dir = Path("outputs/ablation")
    experiments = ['G0', 'G1', 'G2', 'G4', 'G_QSS', 'G5', 'G6', 'G7', 'G9', 'G10', 'G12', 'G13', 'G15']
    
    all_results = []
    for exp in experiments:
        output_dir = base_dir / exp
        result = analyze_experiment(exp, output_dir)
        if result:
            all_results.append(result)
    
    # 创建对比表格
    df = pd.DataFrame(all_results)
    df = df.sort_values('final_loss')
    
    print("\n" + "="*80)
    print("VCF Ablation Study Results")
    print("="*80)
    print(df.to_string(index=False))
    
    # 保存结果
    df.to_csv(base_dir / "ablation_summary.csv", index=False)
    print(f"\nResults saved to {base_dir / 'ablation_summary.csv'}")
    
    # 绘制对比图
    plot_comparison(df, base_dir)

def plot_comparison(df, output_dir):
    """绘制实验对比图"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Loss对比
    axes[0,0].bar(df['experiment'], df['final_loss'])
    axes[0,0].set_title('Final Training Loss')
    axes[0,0].set_ylabel('Loss')
    axes[0,0].tick_params(axis='x', rotation=45)
    
    # Quantization Loss对比
    axes[0,1].bar(df['experiment'], df['final_quant_loss'])
    axes[0,1].set_title('Final Quantization Loss')
    axes[0,1].set_ylabel('Loss')
    axes[0,1].tick_params(axis='x', rotation=45)
    
    # Repulsion Loss对比
    axes[1,0].bar(df['experiment'], df['final_repulsion_loss'])
    axes[1,0].set_title('Final Repulsion Loss')
    axes[1,0].set_ylabel('Loss')
    axes[1,0].tick_params(axis='x', rotation=45)
    
    # Reconstruction Loss对比
    axes[1,1].bar(df['experiment'], df['final_recon_loss'])
    axes[1,1].set_title('Final Reconstruction Loss')
    axes[1,1].set_ylabel('Loss')
    axes[1,1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(output_dir / "ablation_comparison.png", dpi=150)
    print(f"Plot saved to {output_dir / 'ablation_comparison.png'}")

if __name__ == "__main__":
    main()
```

## 6. 预期结果与分析

### 6.1 核心组件贡献预测

| 对比组 | 预期结果 | 理论依据 |
|--------|----------|----------|
| G1 vs G0 | G1显著优于G0 | VCF解决碰撞问题，提升SID区分度 |
| G1 vs G2 | G1优于G2 | CVPM排除协议良性对，避免误排斥 |
| G1 vs G5 | G1优于G5 | 时间权重保护冷启动物品 |
| G1 vs G6 | 取决于数据 | TCCL在协同信号强时有益 |

### 6.2 参数敏感性预测（参考QuaSID论文）

| 参数 | 预测最优值 | 影响说明 | QuaSID参考 |
|------|-----------|----------|-----------|
| alpha | 0.005~0.01 | 过大导致排斥过快消失，过小导致长期过度排斥 | - |
| m0 | 0.5~0.7 | 过小无法有效分离，过大破坏语义结构 | m_full=0.8, m_partial=0.5 |
| R | 1~2 | QuaSID离线实验使用R=1 | R=1(离线), R=2(在线) |
| lambda_full | 0.05~0.8 | QuaSID调优范围 | [0.05, 0.8] |
| lambda_partial | 0.01~0.8 | QuaSID调优范围 | [0.01, 0.8] |
| lambda_cl | 0.01~0.5 | QuaSID调优范围 | [0.01, 0.5] |

## 7. 实验执行计划

### Phase 1: 核心对比 (GPU 0-3)
- G0 (Baseline)
- G1 (VCF-Full)
- G_QSS (QuaSID-Static Baseline)
- G4 (VCF w/o Dynamic Margin)
- G2 (VCF w/o CVPM)
- G5 (VCF w/o Time)

### Phase 2: 组件消融 (GPU 0-3)
- G6 (VCF w/o TCCL)
- G7 (alpha=0.001)
- G9 (alpha=0.1)
- G10 (m0=0.3)

### Phase 3: 参数敏感性 (GPU 0-3)
- G12 (m0=0.7)
- G13 (R=1)
- G15 (R=3)

## 8. 注意事项

1. **显存管理**: 每个实验独立输出到不同目录，避免冲突
2. **随机种子**: 固定seed=42确保可复现
3. **监控指标**: 定期检查train.log确认训练正常
4. **提前停止**: 如训练发散，及时终止并调整参数
5. **日志分析**: 使用提供的脚本自动化分析结果
6. **参考QuaSID**: 批量大小256、学习率3e-4、权重衰减1e-5、早停条件NDCG@5+HR@5连续10次未改善
7. **早停实现**: 当前使用val/loss作为监控指标（patience=10），如需严格遵循QuaSID，需在下游推荐任务中计算NDCG@5+HR@5
8. **步数设置**: 必须显式传入 `trainer.max_steps=3000`，否则默认只有30步，训练完全不足

## 9. 结果解读指南

### 9.1 如何判断VCF有效？
- G1的final_loss应低于G0
- G1的repulsion_loss应显著大于0（证明排斥机制在工作）
- G1的quantization_loss不应显著高于G0（证明没有破坏量化质量）

### 9.2 如何判断组件贡献？
- G2 vs G1: CVPM贡献 = loss(G2) - loss(G1)
- G5 vs G1: Time Decay贡献 = loss(G5) - loss(G1)
- G6 vs G1: TCCL贡献 = loss(G6) - loss(G1)

### 9.3 如何判断参数最优？
- 绘制alpha/m0/R vs final_loss曲线
- 选择使loss最低的参数组合
