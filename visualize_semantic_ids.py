"""
VCR-TD Semantic ID Visualization Script
生成t-SNE和UMAP图展示码本分配动态过程
"""

import torch
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import os

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def load_semantic_ids(pickle_path, tensor_path):
    """加载语义ID数据"""
    # 加载pickle文件获取映射关系
    with open(pickle_path, 'rb') as f:
        id_mapping = pickle.load(f)
    
    # 加载tensor文件
    semantic_ids = torch.load(tensor_path)
    
    # 转置为 [num_items, num_hierarchies]
    if semantic_ids.shape[0] < semantic_ids.shape[1]:
        semantic_ids = semantic_ids.T
    
    return id_mapping, semantic_ids

def extract_hierarchical_info(semantic_ids, num_hierarchies=3):
    """
    提取层次化语义ID信息
    semantic_ids: [num_items, num_hierarchies]
    """
    num_items = semantic_ids.shape[0]
    actual_hierarchies = min(num_hierarchies, semantic_ids.shape[1])
    
    # 分离每一层的ID
    layer_ids = {}
    for layer in range(actual_hierarchies):
        layer_ids[f'layer_{layer}'] = semantic_ids[:, layer].numpy()
    
    return layer_ids, num_items

def plot_tsne_comparison(embeddings, semantic_ids_dict, save_dir='visualizations', title_prefix=''):
    """
    生成t-SNE对比图
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # 确保embeddings是numpy数组
    if isinstance(embeddings, torch.Tensor):
        embeddings = embeddings.numpy()
    
    # 采样（如果数据量太大）
    max_samples = 5000
    if len(embeddings) > max_samples:
        indices = np.random.choice(len(embeddings), max_samples, replace=False)
        embeddings_sample = embeddings[indices]
        for key in semantic_ids_dict:
            semantic_ids_dict[key] = semantic_ids_dict[key][indices]
    else:
        embeddings_sample = embeddings
    
    print(f"使用 {len(embeddings_sample)} 个样本进行可视化")
    
    # 运行t-SNE
    print("运行t-SNE降维...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(embeddings_sample)-1), max_iter=1000)
    embeddings_2d = tsne.fit_transform(embeddings_sample)
    
    # 为每个层次生成可视化
    num_layers = len(semantic_ids_dict)
    fig, axes = plt.subplots(1, num_layers, figsize=(6*num_layers, 5))
    
    if num_layers == 1:
        axes = [axes]
    
    for idx, (layer_name, layer_ids) in enumerate(semantic_ids_dict.items()):
        ax = axes[idx]
        
        # 获取唯一的ID和颜色
        unique_ids = np.unique(layer_ids)
        n_clusters = len(unique_ids)
        
        # 创建颜色映射
        colors = plt.cm.tab20(np.linspace(0, 1, min(n_clusters, 20)))
        
        # 绘制散点图
        for i, uid in enumerate(unique_ids[:20]):  # 只显示前20个聚类
            mask = layer_ids == uid
            ax.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1], 
                      c=[colors[i]], label=f'ID {uid}', alpha=0.6, s=20)
        
        ax.set_title(f'{title_prefix}{layer_name}\n({n_clusters} unique IDs)', fontsize=12)
        ax.set_xlabel('t-SNE Dimension 1')
        ax.set_ylabel('t-SNE Dimension 2')
        
        if n_clusters <= 20:
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    
    plt.tight_layout()
    filename = f'{save_dir}/tsne_hierarchical_{title_prefix.replace(" ", "_")}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"保存到: {filename}")
    plt.close()

def plot_coverage_heatmap(semantic_ids_dict, save_dir='visualizations', title_prefix=''):
    """
    生成覆盖率热力图
    """
    os.makedirs(save_dir, exist_ok=True)
    
    fig, axes = plt.subplots(1, len(semantic_ids_dict), figsize=(6*len(semantic_ids_dict), 4))
    
    if len(semantic_ids_dict) == 1:
        axes = [axes]
    
    for idx, (layer_name, layer_ids) in enumerate(semantic_ids_dict.items()):
        ax = axes[idx]
        
        # 计算每个ID的频次
        unique_ids, counts = np.unique(layer_ids, return_counts=True)
        
        # 绘制柱状图
        sorted_indices = np.argsort(counts)[::-1]
        top_n = min(30, len(unique_ids))
        
        ax.bar(range(top_n), counts[sorted_indices[:top_n]], color='steelblue', alpha=0.7)
        ax.set_title(f'{title_prefix}{layer_name} - Top {top_n} Token Coverage', fontsize=12)
        ax.set_xlabel('Token Rank')
        ax.set_ylabel('Number of Items')
        ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    filename = f'{save_dir}/token_coverage_{title_prefix.replace(" ", "_")}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"保存到: {filename}")
    plt.close()

def compare_methods_tsne(embedding_path, methods_dict, save_dir='visualizations'):
    """
    对比不同方法的t-SNE图
    methods_dict: {'method_name': (pickle_path, tensor_path), ...}
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # 加载原始嵌入
    embeddings = torch.load(embedding_path)
    if isinstance(embeddings, torch.Tensor):
        embeddings = embeddings.numpy()
    
    print(f"嵌入形状: {embeddings.shape}")
    
    # 采样
    max_samples = 3000
    if len(embeddings) > max_samples:
        indices = np.random.choice(len(embeddings), max_samples, replace=False)
        embeddings_sample = embeddings[indices]
    else:
        indices = np.arange(len(embeddings))
        embeddings_sample = embeddings
    
    # 运行t-SNE
    print("运行t-SNE降维...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(embeddings_sample)-1), max_iter=1000)
    embeddings_2d = tsne.fit_transform(embeddings_sample)
    
    # 为每个方法生成子图
    n_methods = len(methods_dict)
    fig, axes = plt.subplots(2, n_methods, figsize=(6*n_methods, 10))
    
    if n_methods == 1:
        axes = axes.reshape(-1, 1)
    
    for idx, (method_name, (pickle_path, tensor_path)) in enumerate(methods_dict.items()):
        print(f"处理 {method_name}...")
        
        # 加载语义ID
        _, semantic_ids = load_semantic_ids(pickle_path, tensor_path)
        
        # 确保索引不越界
        valid_indices = indices[indices < semantic_ids.shape[0]]
        semantic_ids_sample = semantic_ids[valid_indices]
        embeddings_2d_sample = embeddings_2d[:len(valid_indices)]
        
        # 获取层0和层1的ID
        layer_0_ids = semantic_ids_sample[:, 0].numpy()
        layer_1_ids = semantic_ids_sample[:, 1].numpy() if semantic_ids_sample.shape[1] > 1 else layer_0_ids
        
        # 绘制层0
        ax0 = axes[0, idx]
        unique_ids_0 = np.unique(layer_0_ids)
        colors_0 = plt.cm.tab20(np.linspace(0, 1, min(len(unique_ids_0), 20)))
        
        for i, uid in enumerate(unique_ids_0[:20]):
            mask = layer_0_ids == uid
            ax0.scatter(embeddings_2d_sample[mask, 0], embeddings_2d_sample[mask, 1], 
                       c=[colors_0[i]], alpha=0.6, s=20)
        
        ax0.set_title(f'{method_name}\nLayer 0 ({len(unique_ids_0)} IDs)', fontsize=11)
        ax0.set_xlabel('t-SNE Dimension 1')
        ax0.set_ylabel('t-SNE Dimension 2')
        
        # 绘制层1
        ax1 = axes[1, idx]
        unique_ids_1 = np.unique(layer_1_ids)
        colors_1 = plt.cm.tab20(np.linspace(0, 1, min(len(unique_ids_1), 20)))
        
        for i, uid in enumerate(unique_ids_1[:20]):
            mask = layer_1_ids == uid
            ax1.scatter(embeddings_2d_sample[mask, 0], embeddings_2d_sample[mask, 1], 
                       c=[colors_1[i]], alpha=0.6, s=20)
        
        ax1.set_title(f'{method_name}\nLayer 1 ({len(unique_ids_1)} IDs)', fontsize=11)
        ax1.set_xlabel('t-SNE Dimension 1')
        ax1.set_ylabel('t-SNE Dimension 2')
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/methods_comparison_tsne.png', dpi=300, bbox_inches='tight')
    print(f"保存到: {save_dir}/methods_comparison_tsne.png")
    plt.close()

def main():
    """主函数"""
    print("=" * 60)
    print("VCR-TD Semantic ID Visualization")
    print("=" * 60)
    
    # 设置路径
    base_dir = '/home/pyy/GRID'
    embedding_path = f'{base_dir}/embeddings/beauty/pickle/merged_predictions_tensor.pt'
    
    # 方法对比
    methods = {
        'Baseline': (
            f'{base_dir}/outputs/rkmeans_inference/pickle/merged_predictions.pkl',
            f'{base_dir}/outputs/rkmeans_inference/pickle/merged_predictions_tensor.pt'
        ),
        'VCR-TD Full': (
            f'{base_dir}/outputs/vcr_td_inference/pickle/merged_predictions.pkl',
            f'{base_dir}/outputs/vcr_td_inference/pickle/merged_predictions_tensor.pt'
        ),
        'VCR-TD w/o Time Decay': (
            f'{base_dir}/outputs/vcr_td_no_decay_inference/pickle/merged_predictions.pkl',
            f'{base_dir}/outputs/vcr_td_no_decay_inference/pickle/merged_predictions_tensor.pt'
        )
    }
    
    # 创建可视化目录
    save_dir = f'{base_dir}/visualizations'
    os.makedirs(save_dir, exist_ok=True)
    
    print("\n1. 生成方法对比t-SNE图...")
    try:
        compare_methods_tsne(embedding_path, methods, save_dir)
    except Exception as e:
        print(f"方法对比图生成失败: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n2. 生成各方法的层次化可视化...")
    
    # 加载嵌入
    embeddings = torch.load(embedding_path)
    if isinstance(embeddings, torch.Tensor):
        embeddings = embeddings.numpy()
    
    for method_name, (pickle_path, tensor_path) in methods.items():
        print(f"\n   处理 {method_name}...")
        try:
            _, semantic_ids = load_semantic_ids(pickle_path, tensor_path)
            layer_ids, num_items = extract_hierarchical_info(semantic_ids)
            
            print(f"   物品数量: {num_items}")
            print(f"   层次数: {len(layer_ids)}")
            print(f"   嵌入数量: {len(embeddings)}")
            
            # 确保embeddings和semantic_ids数量匹配
            min_items = min(num_items, len(embeddings))
            embeddings_subset = embeddings[:min_items]
            layer_ids_subset = {k: v[:min_items] for k, v in layer_ids.items()}
            
            plot_tsne_comparison(embeddings_subset, layer_ids_subset, save_dir, title_prefix=method_name.replace(' ', '_') + '_')
            plot_coverage_heatmap(layer_ids_subset, save_dir, title_prefix=method_name.replace(' ', '_') + '_')
            
        except Exception as e:
            print(f"   {method_name} 可视化生成失败: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("可视化完成！")
    print(f"输出目录: {save_dir}")
    print("=" * 60)

if __name__ == '__main__':
    main()
