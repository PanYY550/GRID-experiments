# 基于 Semantic ID 的生成式推荐系统前沿技术深度研究报告：从 RQ-VAE 瓶颈到端到端协同演进

## 核心引言与执行摘要

在推荐系统（Recommender Systems）向大语言模型（LLMs）和生成式范式（Generative Paradigm）演进的进程中，基于 Semantic ID（语义标识符，简称 SID）的生成式推荐架构（如 Snap Research 的 GRID 框架与 TIGER 模型）已成为桥接连续多模态语义与离散自回归生成空间的核心技术路径 1。该架构通过将物品的文本或视觉表征（Embeddings）通过残差量化变分自编码器（RQ-VAE）或 RQ-KMeans 映射为固定长度的离散 Token 序列，使得推荐任务能够完全转化为序列到序列（Seq2Seq）的自回归生成任务 2。

然而，在实际的工业级落地与项目迭代中，这种标准的两阶段管道暴露出深层次的架构缺陷。正如当前项目所经历的痛点，RQ-VAE 存在严重的“码本坍塌（Codebook Collapse）”与“沙漏效应（Hourglass Phenomenon）”，导致大量语义相近或协同信号相似的物品被强行映射到相同的 SID，引发严重的 SID 碰撞（Collision）5。更为棘手的是，简单的排斥损失（Repulsion Loss）虽然能在量化阶段推开碰撞物品，但实验表明，过度的排斥（例如排斥权重 ![img](file:///C:\Users\pyy\AppData\Local\Temp\ksohtml23176\wps1.jpg) 过大）会严重破坏物品表征在原始 LLM 隐空间中的语义保真度（Semantic Fidelity）。这种对语义流形的破坏直接导致下游的 TIGER 生成模型无法有效利用预训练语言模型的先验知识，使得简单的调参（如将 ![img](file:///C:\Users\pyy\AppData\Local\Temp\ksohtml23176\wps2.jpg) 从 0.3 降至 0.2）带来的性能回升远大于复杂的排斥机制设计。

此外，当前框架还受制于两阶段割裂（量化损失与推荐目标不一致）、静态 SID（缺乏上下文感知能力）、以及语义与协同信号错位（文本相似但用户行为不相似）等根本性障碍 3。本报告通过对 2024-2026 年间在 NeurIPS, ICLR, ICML, RecSys, KDD 等顶会及最新 arXiv 预印本的前沿文献进行详尽的深度剖析，系统性地针对上述六大痛点方向展开研究。报告不仅揭示了各项前沿技术的底层机制，还为当前项目量身定制了具备高落地可行性的具体创新点建议与优先级演进路线图，旨在打破现有框架的性能天花板。



------



## 一、 Tokenizer 范式的演进：超越 RQ-VAE 的离散量化方案

在 TIGER 或 GRID 框架中，RQ-VAE 是构建 SID 的默认选择 2。然而，RQ-VAE 依赖于显式学习的码本（Codebook），在面对呈现长尾分布（Power-law Distribution）的推荐系统物品图谱时，往往会出现大量低频物品挤占少数聚类中心，而大量码字（Codes）处于闲置状态的现象 6。这种表征空间的非均匀分布是导致 SID 碰撞的根本原因之一。

### 1.1 FSQ 与 LFQ：无查找表的隐式量化机制

为了彻底消除码本坍塌并提高计算效率，学术界开始将目光转向有限标量量化（Finite Scalar Quantization, FSQ）与无查找表量化（Lookup-Free Quantization, LFQ）。FSQ 放弃了在隐空间中寻找最近邻向量（Nearest-Neighbor Lookup）的传统 VQ 机制，转而对连续向量的每一个维度进行独立的值域约束与标量舍入 10。例如，FSQ 将输入向量投影到低维空间后，通过映射函数（如 ![img](file:///C:\Users\pyy\AppData\Local\Temp\ksohtml23176\wps3.jpg)）将其直接离散化为预设的几个离散值（Levels）11。

这种机制隐式地构建了一个正交的笛卡尔积码本（Cartesian-product Codebook），使得所有的组合在理论上都是可达的，从而彻底避免了死码（Dead Codes）问题，并在反向传播中提供了更稳定的梯度流 4。在推荐系统领域，最新的研究如 QARM V2 以及 SIDE（Semantic ID Embedding）算法已开始利用 FSQ 替代传统的 RQ 机制，证明了在无需显式码本查找的情况下，FSQ 能够以更简单的网络结构实现比 RQ-KMeans 更高的离散化保真度，并大幅降低了内存占用 13。

### 1.2 双曲空间量化：长尾分布的天然解药

除了量化机制本身的改变，量化所处的几何空间也是破局的关键。真实的推荐场景中，物品交互呈现极端的长尾分布。传统的欧几里得空间（Euclidean Space）在表示这种层级图谱时存在容量瓶颈，导致尾部物品在量化时被迫聚集，产生严重碰撞。2025 年 ICLR 的前沿工作 HypRQ-VAE（Hyperbolic Residual-Quantized Variational AutoEncoder）首次将物品索引映射到庞加莱球（Poincaré ball）的双曲空间中进行残差量化 15。由于双曲空间的体积随半径呈指数级膨胀，它能够天然地容纳推荐图谱的幂律结构。通过在双曲空间中应用指数映射（Exponential Map）与切空间内的量化，HypRQ-VAE 在保留丰富文本语义的同时，极大地提升了稀疏长尾物品的表征保真度，显著降低了尾部碰撞率 15。

### 1.3 前沿文献矩阵：Tokenizer 层面 (2024-2026)

| ***\*论文标题\****                                           | ***\*作者与年份\****  | ***\*发表会议\**** | ***\*核心贡献\****                                           | ***\*链接\****                                               | ***\*引用\**** |
| ------------------------------------------------------------ | --------------------- | ------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | -------------- |
| **HypRQ-VAE: Long-Tail-Aware Item Indexing for Generative Recommender Systems** | Wu et al., 2025       | ICLR               | 首创在双曲空间（庞加莱球）中进行残差量化，通过指数级体积膨胀天然拟合长尾物品分布，大幅提升稀疏物品推荐效果。 | [https://openreview.net/forum?id=ALJsIAmO54]                 | 15             |
| **SIDE: Semantic ID Embedding for effective learning from sequences** | Ramasamy et al., 2025 | AdKDD              | 提出无嵌入表的 SID 映射算法，引入 FSQ 机制，在工业级广告排序中显著降低内存足迹并提升归一化熵增益。 | [http://papers.adkdd.org/2025/papers/adkdd25-ramasamy-semantic.pdf] | 13             |
| **From Principles to Applications: A Comprehensive Survey of Discrete Tokenizers** | Zhan et al., 2025     | arXiv              | 系统性对比了 VQ, RQ, FSQ 与 LFQ，论证了 FSQ 通过隐式笛卡尔积避免码本崩溃的数学优势。 | [https://arxiv.org/html/2502.12448v1]                        | 11             |
| **MMQ: Multimodal Mixture-of-Quantization Tokenization for Semantic ID Generation** | Xu et al., 2025       | arXiv              | 引入多专家混合量化机制（MoQ），通过正交正则化分离模态共享与独有特征。 | [https://arxiv.org/abs/2508.15281]                           | 20             |

### 1.4 具体、可落地的创新点建议

***\*创新点：双曲有限标量量化器（Hyperbolic\**** ***\*Finite Scalar Quantization,\**** ***\*Hyp-FSQ）\****

**·** ***\*核心\**** ***\*Idea（一句话）：\**** 将有限标量量化（FSQ）的隐式笛卡尔积优势与庞加莱球双曲空间的指数级容量相结合，通过在双曲切空间内执行标量截断操作，构建一个无需显式码本且天然拟合长尾分布的离散分词器。

**·** ***\*为什么现在有\**** ***\*novelty：\**** 现有工作要么在欧式空间研究 FSQ（解决坍塌但受限于空间容量）12，要么在双曲空间研究传统 RQ-VAE（解决容量但仍存在码本查找和梯度近似问题）15。两者结合属于全新的理论交叉点。

**·** ***\*实现难度：高。\**** 难点在于需要在黎曼流形（Riemannian manifold）的切空间与流形之间精确地定义可导的 FSQ 舍入与激活函数，并保证反向传播在双曲几何框架下的数值稳定性。

**·** ***\*预期能解决的问题：\**** 从根本上消解了“过度排斥损害语义”的痛点。因为双曲空间的边缘拥有近乎无限的容量，模型无需通过人工的 Repulsion Loss 强行推开碰撞物品，长尾物品会自动散布在广阔的双曲边缘，自然降低碰撞率且完美保留语义层次结构。



------



## 二、 端到端学习：打破 Tokenizer 与 Generator 的两阶段割裂

当前项目的核心痛点之一是 Tokenizer（如 RQ-VAE）的量化损失主要关注输入特征的重构误差（Reconstruction Loss），而生成模型（如 TIGER）的优化目标是最大化序列下一步的预测概率（Next-Token Prediction Loss）7。由于 Argmax 离散化操作的不可导性，生成器的梯度无法直接反向传播回 Tokenizer 进行参数更新，导致 Tokenizer 学到的 SID 虽然在语义上聚集，但无法反映用户行为的协同过滤模式 23。

### 2.1 软标识符（Soft Identifiers）与推理对齐

传统的直通估计器（Straight-Through Estimator, STE）和 Gumbel-Softmax 虽然能提供近似梯度，但在序列极其敏感的推荐系统中，训练时的软分布（Soft Distribution）与推理时的硬采样（Hard Lookup）之间会产生巨大的偏差（Training-Inference Discrepancy），严重损害推荐精度 8。

前沿框架 UniGRec 为此提出了“可微软物品标识符（Differentiable Soft Item Identifiers）”范式。它在训练阶段彻底摒弃了硬离散化，将 SID 表达为码本向量的连续加权和，使最终的推荐损失（Recommendation Loss）能够作为统一的目标函数联合优化 Tokenizer 和 Recommender 24。为了弥合软硬转换带来的偏差，UniGRec 引入了退火推理对齐（Annealed Inference Alignment）策略。通过在训练进程中动态降低 Softmax 的温度参数（Temperature），使得生成的软权重分布逐渐逼近 One-hot 形式的硬标识符。同时，为了防止端到端优化导致的码本坍塌，它辅以码字均匀性正则化（Codeword Uniformity Regularization）以维持空间的多样性 23。

### 2.2 潜在去噪与协同表征提炼

另一种联合训练的思路源自于视觉领域的 Tokenizer 演进。例如 UNITE 框架证明，将 Tokenization 过程与潜在去噪（Latent Denoising）目标直接耦合，可以在不需要停止梯度（Stop-gradient）的情况下，让生成模型的梯度直接塑造 Tokenizer 的编码器表示 7。在推荐领域，TokenRec 虽然也是两阶段，但它在微调 LLM 骨干网络时将物品 ID 的协同特征映射回生成式掩码向量空间中，展示了表征优化的潜力 26。更进一步，Dual Collaborative Distillation 机制被提出用于端到端架构中，通过一个轻量级的协同过滤教师模型（Teacher Model），将协同先验知识同时蒸馏给 Tokenizer 和 Generator，有效弥补了纯生成模型在端到端联合训练初期协同信号严重缺失的问题 23。

### 2.3 前沿文献矩阵：端到端学习 (2024-2026)

| ***\*论文标题\****                                           | ***\*作者与年份\**** | ***\*发表会议\**** | ***\*核心贡献\****                                           | ***\*链接\****                        | ***\*引用\**** |
| ------------------------------------------------------------ | -------------------- | ------------------ | ------------------------------------------------------------ | ------------------------------------- | -------------- |
| **UniGRec: Unified Generative Recommendation with Soft Identifiers for End-to-End Optimization** | Li et al., 2026      | arXiv              | 引入可微软标识符实现端到端训练，提出退火推理对齐与码字均匀性正则化解决训练-推理不一致与码本坍塌。 | [https://arxiv.org/html/2601.17438v1] | 8              |
| **PIT: A Dynamic Personalized Item Tokenizer for End-to-End Generative Recommendation** | Wang et al., 2026    | arXiv              | 提出一种动态端到端生成框架，通过最小损失选择机制和一对多波束索引，实现分词器与生成器的协同演进。 | [https://arxiv.org/html/2602.08530v1] | 27             |
| **TokenRec: Learning to Tokenize ID for LLM-based Generative Recommendation** | Qu et al., 2025      | TKDE               | 设计 Masked Vector-Quantized 分词器将协同特征量化为离散 token，并通过生成式检索避免自回归的高延迟。 |                                       | 26             |
| **Unifying Tokenization and Latent Denoising**               | Heek et al., 2026    | arXiv              | 揭示了在联合训练中移除停止梯度操作（Stop-gradient）并结合去噪目标可以显著提升端到端 Tokenizer 质量。 | [https://arxiv.org/html/2603.22283v1] | 7              |

### 2.4 具体、可落地的创新点建议

***\*创新点：连续流形的退火残差量化（Continuous\**** ***\*Manifold Annealed Residual Quantization,\**** ***\*CM-ARQ）\****

**·** ***\*核心\**** ***\*Idea（一句话）：\**** 在两阶段之间构建一个“可学习的旁路路由（Learnable Bypass Router）”，在训练初期阶段（Epochs 1-10），TIGER 模型直接接收 RQ-VAE 编码器输出的连续嵌入（Continuous Embeddings）以获取完美的梯度反传；随着训练进行，系统通过退火函数逐步提升离散化 Token 向量在输入中的权重，直至最后阶段完全过渡到离散 SID 推荐。

**·** ***\*为什么现在有\**** ***\*novelty：\**** 现有的 STE 和 Gumbel-Softmax 是在微观层面（单个 Token）强行估算梯度，导致方差巨大。该方案在宏观特征流（Feature Flow）层面进行软硬插值（Soft-Hard Interpolation），完美避开了离散数学的局部不可导陷阱，融合了 UniGRec 的对齐思想 23 与 TIGER 的强生成能力。

**·** ***\*实现难度：中。\**** 仅需要修改输入端的 Embedding Lookup 逻辑，实现连续表征与量化表征的动态加权融合，无需对 LLM 骨干进行大规模手术。

**·** ***\*预期能解决的问题：\**** 彻底解决量化损失与推荐目标不一致的问题。Tokenizer 将不再盲目地根据文本相似度进行聚类，而是会在生成器损失的驱使下，主动将协同矩阵中具有强共现关系的物品拉入同一个 Codebook 聚类中。



------



## 三、 Context-Aware 与动态 SID：打破静态标识符的语义刚性

在标准框架中，一个物品的元数据（标题、类别、图像）一旦固定，其在 RQ-VAE 中的输出 SID 就永远是静态的（Static SIDs）2。这与真实的推荐场景严重背离：用户购买同一部手机的原因可能截然不同，有时是为了“科技尝鲜”（上下文为浏览其他电子产品），有时是为了“节日送礼”（上下文为浏览礼盒或节日用品）。静态 SID 无法捕获物品在不同序列上下文中的多面性语义。

### 3.1 基于特征集合置换的上下文感知

针对这一痛点，DeepMind 提出的 **ActionPiece** 框架是一项突破性的进展。它将序列中的每一个用户动作（Action）不再视为一个固定的 ID，而是将其解构为一个物品特征的无序集合（Unordered Feature Sets）30。在构建词表时，系统根据特征在单个集合内部以及在相邻行为之间的共现频率（Co-occurrence Frequency），将特征模式合并为新的上下文敏感 Token（类似于自然语言处理中的 BPE 算法）31。更具创新性的是，ActionPiece 引入了集合置换正则化（Set Permutation Regularization, SPR）。通过在特征集合内随机排列特征，同一个物品序列可以产生多种具有相同高层语义但 Token 序列完全不同的分词结果。这种机制使得同一个物品在不同的上下文中被编码为完全不同的 Token，不仅极大地提升了词表的利用率（从 56.9% 跃升至 95.3%），还天然实现了数据增强（Data Augmentation）和推理时的集成预测（Ensemble Predictions）31。

### 3.2 行为-语义动态对齐与演化

动态 Token 化的另一个前沿方向是基于用户意图的行为调整。在 **MMQ-v2**（也称为 ADA-SID）中，研究者提出了动态行为路由（Dynamic Behavioral Router）的概念。它通过学习为物品的不同行为 SID 分配自适应的权重，基于物品协同信号的丰富程度动态校准行为-内容对齐强度，从而在保护长尾物品免受噪声污染的同时，放大了热门物品丰富的行为上下文 33。而在实时性极强的直播推荐场景中，**OneLive** 框架构建了一个动态分词器（Dynamic Tokenizer），结合时间感知门控注意力机制（Time-Aware Gated Attention），能够持续对不断演化的实时直播内容与瞬时用户行为进行联合编码，确保 SID 随生命周期动态变化 35。

### 3.3 前沿文献矩阵：上下文与动态 SID (2024-2026)

| ***\*论文标题\****                                           | ***\*作者与年份\**** | ***\*发表会议\**** | ***\*核心贡献\****                                           | ***\*链接\****                                       | ***\*引用\**** |
| ------------------------------------------------------------ | -------------------- | ------------------ | ------------------------------------------------------------ | ---------------------------------------------------- | -------------- |
| **ActionPiece: Contextually Tokenizing Action Sequences for Generative Recommendation** | Hou et al., 2025     | ICML               | 摒弃静态 ID，将动作视为无序特征集，基于上下文共现频率进行动态分词，并引入集合置换正则化提升词表利用率。 | [https://arxiv.org/abs/2502.13581]                   | 30             |
| **MMQ-v2: Align, Denoise, and Amplify: Adaptive Behavior Mining for Semantic IDs** | Xu et al., 2025      | arXiv              | 提出自适应行为-内容对齐机制与动态行为路由器，根据协同信号丰富度为物品定制特定的动态 SID。 | [https://arxiv.org/html/2510.25622v1]                | 33             |
| **OneLive: Dynamically Unified Generative Framework for Live-Streaming Recommendation** | Wang et al., 2026    | arXiv              | 设计动态分词器与时间感知门控注意力，捕获实时变化的生命周期特征，实现直播流媒体 SID 的动态生成。 | [https://arxiv.org/html/2602.08612v1]                | 35             |
| **Context-Aware Diffusion-based Sequential Recommendation (CADSR)** | Jocelyn et al., 2024 | BigData            | 利用上下文信息在对比学习期间生成更具语义一致性的动态正样本，精准捕获偏好演化。 | [http://web.cs.wpi.edu/~kmlee/pubs/you24bigdata.pdf] | 38             |

### 3.4 具体、可落地的创新点建议

***\*创新点：前缀提示引导的上下文多态生成（Prompt-Conditioned\**** ***\*Contextual Polymorphism,\**** ***\*PCCP）\****

**·** ***\*核心\**** ***\*Idea（一句话）：\**** 不修改底层的 RQ-VAE 静态码本，而是为主推荐生成模型（TIGER）附加一个“意图提炼层”。该层以用户历史序列为输入，生成一个低维的 Context Prompt 向量；将此向量与目标物品的静态 SID 联合输入到一个轻量级的多层感知机（MLP），输出一个经过上下文调制的“虚拟动态 SID”（Virtual Dynamic SID），再以此作为最终的推荐依据。

**·** ***\*为什么现在有\**** ***\*novelty：\**** ActionPiece 需要彻底重构底层的 Tokenizer 建树逻辑 32，工程量巨大；本方案保留了原有静态 RQ-VAE 管道的高效检索优势，仅在生成模型的潜在空间中完成上下文的动态注入，实现难度低但效果显著。

**·** ***\*实现难度：低。\**** 可作为当前 GRID 框架的直接外挂组件（Plug-and-play module）。

**·** ***\*预期能解决的问题：\**** 彻底解决静态 SID 导致的语义刻板印象。同一个物品，如果用户是因为打折而浏览，生成的虚拟 SID 会偏向“价格敏感型”聚类区域；如果是为了品牌复购，虚拟 SID 会偏向“高品质”区域，极大增强推荐的上下文精准度。



------



## 四、 碰撞问题的新视角：从“盲目排斥”到“接受并利用碰撞”

您在项目中观察到的现象极其敏锐且具有代表性：“过度减少碰撞反而损害下游 TIGER 推荐性能，简单的调参（![img](file:///C:\Users\pyy\AppData\Local\Temp\ksohtml23176\wps4.jpg) 从 0.3 到 0.2）优于复杂机制”。这从数学流形的角度解释非常清晰：过度使用排斥损失（Repulsion Loss）强行推开在内容特征上本就相似的物品，会严重扭曲底层的隐空间拓扑结构。当这种扭曲的空间被量化后，生成模型将无法泛化其学到的语义规律，导致性能雪崩 4。

### 4.1 停止平等对待碰撞：资质感知与严重度缩放

工业级的解决方案不应是消除碰撞，而是“甄别碰撞”。在 Kuaishou 亿级用户的实践中，**QuaSID** (Qualification-Aware Semantic ID Learning) 框架提出了革命性的观点：并非所有的碰撞都是有害的 4。低汉明距离（Hamming distance）的重叠中，既包含了语义无关物品被强行塞入同一 Token 的“恶性碰撞”，也包含了系统通过采样策略生成的正样本（Positive pairs）或客观上完全等价的重复物品导致的“良性重叠”。

QuaSID 引入了冲突感知的有效对掩码（Conflict-Aware Valid Pair Masking, CVPM）来过滤这些良性协议引发的重叠，从而清洗出真正具有破坏性的冲突对集 4。更关键的是，它提出了汉明引导的边距排斥（Hamming-guided Margin Repulsion, HaMR）。HaMR 不再使用统一的标量权重 ![img](file:///C:\Users\pyy\AppData\Local\Temp\ksohtml23176\wps5.jpg)，而是将 SID 重叠的汉明距离转化为编码器空间中具有严重度感知能力的几何边距约束（Severity-scaled geometric constraints）。高度碰撞的无关物品受到强排斥，而部分重叠的相似物品则仅受到微调，这完美契合了您在调低 ![img](file:///C:\Users\pyy\AppData\Local\Temp\ksohtml23176\wps6.jpg) 时观察到的性能回升原理——保持了语义保真度 4。

### 4.2 接受碰撞：分层解码与语义等价类

学术界的另一个重大转向是“接受碰撞并将其视为知识”。如果多个物品共享同一个 SID，它们实际上构成了一个“语义等价类（Semantic Equivalence Class）”40。**Discrete Diffusion** 框架通过在词汇表中引入一个专用的“去重码（Dedup Token）”来打破平局。生成模型首先预测共用的 SID 序列（代表该语义等价类），然后预测这个特定的 Dedup Token 以精确定位到类内的唯一物品 42。

进一步地，在复杂的列表生成中，**HiGR**（Hierarchical Generative Slate Recommendation）提出了粗到细的层次规划（Coarse-to-fine Hierarchical Decoding）。它完全接受 SID 碰撞的存在，将生成过程解耦为两阶段：列表级的全局规划（生成包含碰撞群组的粗粒度语义意图），以及项目级的解码阶段（在碰撞群组的约束空间内挑选特定物品）。这种分层生成不仅大幅降低了序列搜索空间，还由于接受了早期 Token 的重叠，极大提高了生成效率 43。

### 4.3 前沿文献矩阵：碰撞问题新视角 (2024-2026)

| ***\*论文标题\****                                           | ***\*作者与年份\**** | ***\*发表会议\**** | ***\*核心贡献\****                                           | ***\*链接\****                                               | ***\*引用\**** |
| ------------------------------------------------------------ | -------------------- | ------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | -------------- |
| **Stop Treating Collisions Equally: Qualification-Aware Semantic ID Learning** | Hu et al., 2026      | arXiv              | 提出 QuaSID 框架，通过掩码过滤良性碰撞，利用汉明距离自适应缩放排斥力度，完美保护语义拓扑。 | [https://arxiv.org/abs/2603.00632]                           | 4              |
| **HiGR: Efficient Generative Slate Recommendation via Hierarchical Planning** | Xu et al., 2025      | arXiv              | 接受碰撞概念，采用从粗到细的分层解码策略，先生成整体语义意图（碰撞组），再解码具体物品。 | [https://arxiv.org/abs/2512.24787]                           | 43             |
| **Breaking Determinism: Fuzzy Modeling of Sequential Recommendation Using Discrete State Space Diffusion Model** | Anonymous, 2024      | NeurIPS            | 提出在词表中额外引入 Dedup Code (去重码)，在接受语义 Token 碰撞的同时保证最终解码的唯一性。 | [https://large-genrec.github.io/static/slides/www25/overall-large-genrec-tutorial-www25.pdf] | 42             |

### 4.4 具体、可落地的创新点建议

***\*创新点：基于协作哈希的碰撞组重排机制（Collaborative-Hash\**** ***\*based Collision Disambiguation,\**** ***\*CHCD）\****

**·** ***\*核心\**** ***\*Idea（一句话）：\**** 主动缩短 RQ-VAE 的层数（例如从 3 层减少到 2 层），故意拥抱大规模的语义碰撞，形成庞大的“语义等价类”；随后在生成器的解码末端，拼接一个与内容完全无关、纯基于用户协同过滤关系生成的局部 Hash ID（即 Dedup Token）来完成最终的区分。

**·** ***\*为什么现在有\**** ***\*novelty：\**** 将 DDBC 的去重机制 42 与 HiGR 的分层解码 44 平移至标准自回归模型。与其耗费大量算力在连续空间中设计复杂的 Repulsion Loss 避免碰撞，不如在离散空间中坦然接受碰撞，并用纯行为特征（Hash ID）在最后一公里破局。

**·** ***\*实现难度：低。\**** 数据预处理阶段的简单拼接，无需修改模型核心结构。

**·** ***\*预期能解决的问题：\**** 彻底绕开了“排斥损失破坏语义保真度”的死胡同。RQ-VAE 的任务被严格限定为“高层语义聚类”，而最易导致冲突的细粒度区分则交给了协同过滤哈希，实现了“语义负责召回，行为负责精准定向”的完美分工，大幅加速生成速度并提升准确率。



------



## 五、 语义-协同融合：量化前的信号注入与多模态对齐

当前生成式推荐的另一个顽疾在于“语义-协同信号错位”3。大型语言模型（LLM）擅长根据文本特征（如标题、描述）生成嵌入，导致诸如“尿布”和“啤酒”或“鼠标”和“键盘”等物品在纯文本维度可能距离较远，但它们在真实用户行为中具有极强的协同共现（Co-occurrence）概率。如果在量化前不融合这些信号，纯粹依赖文本量化生成的 SID 将完全掩盖这些潜藏的黄金推荐线索 3。

### 5.1 对比对齐与图增强（Pre-quantization Injection）

要在量化前完成融合，目前最主流且被工业界验证的技术是双向对齐与图注入。在 **DAS**（Dual-Aligned Semantic IDs）框架中，研究人员采用了一种单阶段共训练（Co-training）的方法。它引入了多视图对比对齐机制（Multi-view Contrastive Alignment），包括双向用户到物品（u2i）、物品到物品（i2i）的 InfoNCE 损失。通过这种方式，DAS 将基于交互的协同过滤（CF）去偏模块的信息强行对齐到量化器的隐空间中，使得最终量化产生的 SID 不仅具备多模态的丰富度，还深度刻画了用户行为模式 48。

更为结构化的方法则是引入图神经网络（GNN）。**SIGER**（Semantic Item Graph Enhancement）提出了将模态特定的物品图（Item-Item Semantic Graph）与基于用户行为构建的协同过滤图进行融合。为了克服图结构中常见的协同噪声，它设计了模数化个性化嵌入微调（Modulus-based Personalized Embedding Perturbation）技术。在特征进入量化器前，SIGER 运用锚点对齐（Anchor-based InfoNCE）使得语义视图和行为视图达到表示的一致性，确保了最终 SID 中协同信号的主导地位 50。

### 5.2 混合量化专家与跨模态蒸馏

**PRORec** (Progressive Collaborative and Semantic Knowledge Fusion) 通过跨模态知识对齐任务（CKA），在第一阶段就利用 AdaLN 将语义知识投射到协同嵌入中，增强表示能力；在第二阶段通过模态内知识蒸馏（IKD）进行深度融合融合 3。另一方面，**MMQ**（Multimodal Mixture-of-Quantization）摒弃了强行融合为单一向量的做法，采用多专家架构（Multi-expert architecture），并行运行模态特异性专家（捕捉独特性）与模态共享专家（捕捉协同增益），通过正交正则化保证两者不冗余，并通过行为感知的微调（Behavior-aware fine-tuning）自适应下游推荐目标 20。

### 5.3 前沿文献矩阵：语义-协同融合 (2024-2026)

| ***\*论文标题\****                                           | ***\*作者与年份\**** | ***\*发表会议\**** | ***\*核心贡献\****                                           | ***\*链接\****                        | ***\*引用\**** |
| ------------------------------------------------------------ | -------------------- | ------------------ | ------------------------------------------------------------ | ------------------------------------- | -------------- |
| **DAS: Dual-Aligned Semantic IDs Empowered Industrial Recommender System** | Ye et al., 2025      | CIKM               | 提出多视图对比对齐（u2i, i2i, u2u），在量化阶段之前实现协同信号与语义表征的单阶段联合优化。 | [https://arxiv.org/abs/2508.10584]    | 49             |
| **Semantic Item Graph Enhancement for Multimodal Recommendation (SIGER)** | Hu et al., 2025      | arXiv              | 通过将协同信号注入物品语义图，并利用基于锚点的 InfoNCE 对齐与嵌入微调来降噪，实现统一建模。 | [https://arxiv.org/html/2508.06154v1] | 50             |
| **Progressive Collaborative and Semantic Knowledge Fusion for Generative Recommendation** | Liu et al., 2025     | arXiv              | 提出 PRORec，通过 AdaLN 进行跨模态对齐，并在量化后应用知识蒸馏来统一不同维度的表征。 | [https://arxiv.org/html/2502.06269v1] | 3              |
| **MMQ: Multimodal Mixture-of-Quantization Tokenization for Semantic ID Generation** | Xu et al., 2025      | arXiv              | 引入多专家架构和正交正则化，分离模态共享与独有信息，并在微调阶段进行行为对齐。 | [https://arxiv.org/abs/2508.15281]    | 20             |

### 5.4 具体、可落地的创新点建议

***\*创新点：预量化图卷积对齐（Pre-Quantization\**** ***\*Graph Convolutional Alignment,\**** ***\*PQ-GCA）\****

**·** ***\*核心\**** ***\*Idea（一句话）：\**** 在将物品送入 RQ-VAE 量化前，先将基于 LLM 生成的纯内容 Embeddings 作为初始节点特征，输入到一个轻量级的双层 LightGCN 网络中（边由全局用户-物品共现矩阵决定），使得协同相似的物品在连续隐空间中相互拉近后，再进行联合离散化量化。

**·** ***\*为什么现在有\**** ***\*novelty：\**** 与极其复杂的 SIGER 图嵌入融合 51 或 DAS 的双向对齐 49 相比，PQ-GCA 剥离了复杂的对比学习目标，将图神经网络退化为一个纯粹的“特征平滑滤波器（Feature Smoother）”，以最小的代价完成了行为先验特征的强行注入。

**·** ***\*实现难度：中。\**** 只需在预处理流水线中加入一个基于协同过滤图的聚合步骤，计算开销可控。

**·** ***\*预期能解决的问题：\**** 直接消除语义相似度与协同信号的背离。经过 LightGCN 聚合后，“鼠标”的 Embedding 会不可避免地向“键盘”靠近，当它们进入 RQ-VAE 时，将极有可能在前几层共享同一个 Semantic ID，从而为下游生成模型提供极其强烈的共买暗示，大幅提高基于关联规则的推荐转化率。



------



## 六、 冷启动与时序演化：零样本注入与动态分布适应

生成式推荐系统存在独特的“冷启动坍塌（Cold-Start Collapse）”问题 55。当新的物品上线时，即使我们可以立即通过 LLM 和量化器计算出它的新 SID，但在重度依赖序列训练的自回归模型（TIGER）的记忆中，这种全新的 Token 组合是从未出现过的“域外分布（Out-of-Distribution）”。模型会产生幻觉，倾向于无视新物品而继续生成历史训练数据中常见的旧物品 SID 55。

### 6.1 模型编辑技术（Model Editing）在推荐系统中的跨界应用

为打破必须等待天级或周级全量重新训练才能推荐冷启动物品的魔咒，前沿框架 **GenRecEdit** 首次将大语言模型中的知识编辑（Knowledge Editing）技术（如 ROME 和 MEMIT）引入到生成式推荐中 55。它的核心思想是将冷启动物品的引入视为“为 LLM 注入新事实”。通过“定位-然后-编辑（Locate-then-Edit）”流水线，它首先精准定位生成模型中最负责存储物品表征的特定前馈神经网络（FFN）层，然后利用线性映射原理，通过秩一更新（Rank-One Update）或闭式解，直接微调 FFN 的隐状态权重 55。这一突破性机制使得系统能够在一瞬间（耗时仅为重新训练的 9.5%）将新物品的 SID 关联到现有物品的语义邻域中，赋予模型“零样本推荐（Zero-Shot Recommendation）”冷启动物品的能力 55。

### 6.2 基于不对称对齐的 CF 信号迁移

由于冷启动物品本质上缺乏交互日志，直接训练极易陷入过拟合。**SMILE** (SeMantic Ids Enhanced CoLd Item Representation) 提出了一种极具工程价值的两步转移对齐架构。为了从热门物品（Head items）向冷启动物品（Cold items）转移协同信息，SMILE 利用 RQ-OPQ 联合编码机制：第一步，利用 RQ 编码的层级结构，将高热度物品的广义协同信号作为模板，强制冷启动物品对齐共享；第二步，利用 OPQ（优化乘积量化）编码来维持冷启动物品侧重于细粒度内容的差异化特征 57。这种设计承认了内容与协同之间的不对称性，在电商搜索场景中取得了点击率和订单量的显著双升。此外，对于高度时序演化的场景，**MMGRid** 构建了一个上下文网格（Contextual Grid），通过权重合并与子空间合并算法动态组合不同时期保存的 GR 检查点（Checkpoints），无需重新训练即可推断最新的意图演化 59。

### 6.3 前沿文献矩阵：冷启动与时序 (2024-2026)

| ***\*论文标题\****                                           | ***\*作者与年份\**** | ***\*发表会议\**** | ***\*核心贡献\****                                           | ***\*链接\****                        | ***\*引用\**** |
| ------------------------------------------------------------ | -------------------- | ------------------ | ------------------------------------------------------------ | ------------------------------------- | -------------- |
| **Bringing Model Editing to Generative Recommendation in Cold-Start Scenarios (GenRecEdit)** | Shi et al., 2026     | arXiv              | 将 LLM 的模型编辑技术（ROME/MEMIT）首次引入生成式推荐，通过直接修改 FFN 权重实现冷启动物品的零样本关联推荐。 | [https://arxiv.org/abs/2603.14259]    | 55             |
| **SMILE: SeMantic Ids Enhanced CoLd Item Representation for Click-through Rate Prediction** | Zhao et al., 2025    | arXiv              | 采用 RQ-OPQ 两步对齐编码，将热门物品的共享协同信号精准迁移至冷启动物品的表征中。 | [https://arxiv.org/html/2510.12604v1] | 57             |
| **MMGRid: Contextual Grid for Generative Recommendation**    | Anonymous, 2026      | arXiv              | 提出一种统一的实验框架，通过对生成模型的不同 Checkpoints 进行子空间融合，有效适应物品的时序演化。 | [https://arxiv.org/html/2601.15930v1] | 59             |

### 6.4 具体、可落地的创新点建议

***\*创新点：实时知识注入的局部秩一更新（Local\**** ***\*Rank-One Update for Real-Time Injection,\**** ***\*LRUI）\****

**·** ***\*核心\**** ***\*Idea（一句话）：\**** 借鉴 GenRecEdit 的思想，当新物品上架并生成全新的 SID 时，无需等待夜间离线重训，而是利用缓存系统中与该新物品内容最相似的 ![img](file:///C:\Users\pyy\AppData\Local\Temp\ksohtml23176\wps7.jpg) 个热门物品的历史交互隐状态，通过闭式解（Closed-form solution）计算出一个极其微小的增量矩阵，在推理阶段动态附加到 Transformer 最后一层 FFN 的输出上。

**·** ***\*为什么现在有\**** ***\*novelty：\**** GenRecEdit 虽然证明了模型编辑在推荐系统的有效性，但其底层计算协方差矩阵求逆极为耗时 55。本方案采用运行时的局部增量附加，彻底避开了持久化模型权重的修改，属于即插即用的工程级近似创新。

**·** ***\*实现难度：高。\**** 数学理论复杂，需解决微小权重干扰导致的全局序列模式衰退（Catastrophic Forgetting）。

**·** ***\*预期能解决的问题：\**** 打破生成式推荐系统面临的最致命短板——无法推荐实时产生的新鲜内容。该机制能欺骗生成模型，使其认为新物品的 SID 是一个“旧有熟悉但被稍微改造”的 Token 组合，从而自然地将其放置于生成序列的候选队列中，彻底盘活冷启动流量。



------



## 七、 优先级排序与实施路线图

综合考量创新层级、技术实现门槛、实验验证周期以及发表顶会论文的潜力，建议按照以下优先级策略推进当前基于 GRID/TIGER 架构的优化与演进：

***\*优先级\**** ***\*1：预量化图卷积对齐（PQ-GCA）-\**** ***\*解决“语义-协同错位”\****

**·** ***\*考量依据：\**** 创新性（中），实现难度（低），验证周期（极短），论文可行性（高，适配 CIKM / WWW）。

**·** ***\*推荐理由：\**** 投入产出比极高。只需要在现有流水线的数据预处理端增加一层简单的协同过滤图谱聚合（如 LightGCN 传播步骤），即可在完全不改动复杂 TIGER 生成器代码的前提下，强行纠正 RQ-VAE 的偏误聚类。线下离线召回指标通常会获得立竿见影的巨大提升。

***\*优先级\**** ***\*2：基于协作哈希的碰撞组重排机制（CHCD）-\**** ***\*解决“碰撞问题”\****

**·** ***\*考量依据：\**** 创新性（高），实现难度（低），验证周期（短），论文可行性（高，适配 SIGIR / KDD）。

**·** ***\*推荐理由：\**** 直击目前项目的痛点。它从理论高度解释了为什么您的“调低 ![img](file:///C:\Users\pyy\AppData\Local\Temp\ksohtml23176\wps8.jpg) 参数反而更好”的观察是完全正确的。通过主动削减 RQ-VAE 深度制造碰撞，并辅以协同 Hash 码进行最后消歧，不仅完美规避了强制推开对语义的破坏，还大幅降低了自回归生成的时间延迟。

***\*优先级\**** ***\*3：连续流形的退火残差量化（CM-ARQ）-\**** ***\*解决“两阶段割裂”\****

**·** ***\*考量依据：\**** 创新性（极高），实现难度（中），验证周期（中），论文可行性（极高，适配 NeurIPS / ICLR）。

**·** ***\*推荐理由：\**** 端到端学习是生成式推荐的“圣杯”。通过退火技术平滑跨越量化的非连续性鸿沟，能够真正使得 Tokenizer 的聚类被下游推荐任务的损失所驱动。它从根本上重塑了训练流水线，具备极高的学术价值。

***\*优先级\**** ***\*4：前缀提示引导的上下文多态生成（PCCP）-\**** ***\*解决“静态\**** ***\*SID”\****

**·** ***\*考量依据：\**** 创新性（高），实现难度（中高），验证周期（长），论文可行性（高，适配 ICML / RecSys）。

**·** ***\*推荐理由：\**** 打破一个物品一个 ID 的静态藩篱是必然趋势，但动态重组词表（如 ActionPiece）工程量浩大。通过外挂提示层（Prompt layer）对隐空间向量进行上下文调控，是在现有框架下的最优折中解。

***\*优先级\**** ***\*5：实时知识注入的局部秩一更新（LRUI）-\**** ***\*解决“冷启动”\****

**·** ***\*考量依据：\**** 创新性（极高），实现难度（极高），验证周期（极长），论文可行性（极高）。

**·** ***\*推荐理由：\**** 模型编辑在 RecSys 是一片绝对的蓝海，学术红利极高。但由于其对底层张量操作的极高要求以及灾难性遗忘的风险，建议作为长期的前沿预研课题进行学术攻坚，不建议作为短期提升业务指标的快反方案。

***\*优先级\**** ***\*6：双曲有限标量量化器（Hyp-FSQ）-\**** ***\*解决“Tokenizer\**** ***\*缺陷”\****

**·** ***\*考量依据：\**** 创新性（极高），实现难度（最高），验证周期（长），论文可行性（高，重理论的 NeurIPS/ICLR）。

**·** ***\*推荐理由：\**** 尽管 FSQ 能够完美解决码本坍塌，且双曲空间对长尾分布极具亲和力，但这种深水区的数学理论创新需要重写底层的黎曼几何算子，工程沉没成本极大。建议除非现有架构遭遇无法逾越的容量瓶颈，否则延后实施。

#### **引用的著作**

\1. Generative Recommendation with Semantic IDs: A Practitioner’s Handbook - arXiv, 访问时间为 五月 1, 2026， https://arxiv.org/html/2507.22224v1

\2. Generative Recommendation with Semantic IDs: A Practitioner’s Handbook - arXiv, 访问时间为 五月 1, 2026， https://arxiv.org/abs/2507.22224

\3. Progressive Collaborative and Semantic Knowledge Fusion for Generative Recommendation - arXiv, 访问时间为 五月 1, 2026， https://arxiv.org/html/2502.06269v1

\4. Stop Treating Collisions Equally: Qualification-Aware Semantic ID Learning for Recommendation at Industrial Scale - arXiv, 访问时间为 五月 1, 2026， https://arxiv.org/html/2603.00632v1

\5. Beyond Static Collision Handling: Adaptive Semantic ID Learning for Multimodal Recommendation at Industrial Scale - arXiv, 访问时间为 五月 1, 2026， https://arxiv.org/html/2604.23522v1

\6. A Survey of Generative Recommendation from a Tri-Decoupled Perspective: Tokenization, Architecture, and Optimization - Preprints.org, 访问时间为 五月 1, 2026， https://www.preprints.org/manuscript/202512.0203

\7. End-to-End Training for Unified Tokenization and Latent Denoising - arXiv, 访问时间为 五月 1, 2026， https://arxiv.org/html/2603.22283v1

\8. Official implementation of “UniGRec: Unified Generative Recommendation with Soft Identifiers for End-to-End Optimization” - GitHub, 访问时间为 五月 1, 2026， https://github.com/Jialei-03/UniGRec

\9. Generative Recommender with End-to-End Learnable Item Tokenization - arXiv, 访问时间为 五月 1, 2026， https://arxiv.org/pdf/2409.05546

\10. A Survey of Item Identifiers in Generative Recommendation - TechRxiv, 访问时间为 五月 1, 2026， https://www.techrxiv.org/doi/pdf/10.36227/techrxiv.176945895.52184668

\11. arXiv:2502.12448v1 [cs.IR] 18 Feb 2025, 访问时间为 五月 1, 2026， https://arxiv.org/pdf/2502.12448

\12. vector-quantize-pytorch - PyPI, 访问时间为 五月 1, 2026， https://pypi.org/project/vector-quantize-pytorch/

\13. SIDE : Semantic ID Embedding for, 访问时间为 五月 1, 2026， http://papers.adkdd.org/2025/paper-presentations/slides-adkdd25-ramasamy-semantic.pdf

\14. SIDE: Semantic ID Embedding for effective learning from sequences, 访问时间为 五月 1, 2026， http://papers.adkdd.org/2025/papers/adkdd25-ramasamy-semantic.pdf

\15. HypRQ-VAE: Long-Tail-Aware Item Indexing for Generative Recommender Systems, 访问时间为 五月 1, 2026， https://openreview.net/forum?id=ALJsIAmO54

\16. HYPRQ-VAE: LONG-TAIL-AWARE ITEM INDEXING - OpenReview, 访问时间为 五月 1, 2026， https://openreview.net/pdf/2e5108920fd339e84d6fc959d3f2c82a470782d2.pdf

\17. Dawei Zhou’s Homepage - Publications, 访问时间为 五月 1, 2026， https://sites.google.com/view/dawei-zhou/publications

\18. SIDE: Semantic ID Embedding for effective learning from sequences - arXiv, 访问时间为 五月 1, 2026， https://arxiv.org/html/2506.16698v1

\19. From Principles to Applications: A Comprehensive Survey of Discrete Tokenizers in Generation, Comprehension, Recommendation, and Information Retrieval - arXiv, 访问时间为 五月 1, 2026， https://arxiv.org/html/2502.12448v1

\20. [2508.15281] MMQ: Multimodal Mixture-of-Quantization Tokenization for Semantic ID Generation and User Behavioral Adaptation - arXiv, 访问时间为 五月 1, 2026， https://arxiv.org/abs/2508.15281

\21. MMQ: Multimodal Mixture-of-Quantization Tokenization for Semantic ID Generation and User Behavioral Adaptation - arXiv, 访问时间为 五月 1, 2026， https://arxiv.org/html/2508.15281v1

\22. Generative Recommender with End-to-End Learnable Item Tokenization - arXiv, 访问时间为 五月 1, 2026， https://arxiv.org/html/2409.05546v3

\23. UniGRec: Unified Generative Recommendation with Soft Identifiers for End-to-End Optimization - arXiv, 访问时间为 五月 1, 2026， https://arxiv.org/html/2601.17438v1

\24. UniGRec: Unified Generative Recommendation with Soft Identifiers for End-to-End Optimization | Request PDF - ResearchGate, 访问时间为 五月 1, 2026， https://www.researchgate.net/publication/400084442_UniGRec_Unified_Generative_Recommendation_with_Soft_Identifiers_for_End-to-End_Optimization

\25. UniGRec: Unified Generative Recommendation with Soft Identifiers for End-to-End Optimization - arXiv, 访问时间为 五月 1, 2026， https://arxiv.org/pdf/2601.17438

\26. [Literature Review] TokenRec: Learning to Tokenize ID for LLM-based Generative Recommendation - Moonlight | AI Colleague for Research Papers, 访问时间为 五月 1, 2026， https://www.themoonlight.io/en/review/tokenrec-learning-to-tokenize-id-for-llm-based-generative-recommendation

\27. PIT: A Dynamic Personalized Item Tokenizer for End-to-End Generative Recommendation, 访问时间为 五月 1, 2026， https://arxiv.org/html/2602.08530v1

\28. PIT: A Dynamic Personalized Item Tokenizer for End-to-End Generative Recommendation | Request PDF - ResearchGate, 访问时间为 五月 1, 2026， https://www.researchgate.net/publication/400603866_PIT_A_Dynamic_Personalized_Item_Tokenizer_for_End-to-End_Generative_Recommendation

\29. TokenRec: Learning to Tokenize ID for LLM-based Generative Recommendations, 访问时间为 五月 1, 2026， https://ira.lib.polyu.edu.hk/bitstream/10397/115697/1/Qu_TokenRec_Learning_Tokenize.pdf

\30. ActionPiece: Contextually Tokenizing Action Sequences for Generative Recommendation - GitHub, 访问时间为 五月 1, 2026， https://raw.githubusercontent.com/mlresearch/v267/main/assets/hou25f/hou25f.pdf

\31. ActionPiece: Contextually Tokenizing Action Sequences for Generative Recommendation | OpenReview, 访问时间为 五月 1, 2026， https://openreview.net/forum?id=h2oNQOzbc5

\32. ICML Poster ActionPiece: Contextually Tokenizing Action Sequences for Generative Recommendation, 访问时间为 五月 1, 2026， https://icml.cc/virtual/2025/poster/44439

\33. Semantic ID Learning - Emergent Mind, 访问时间为 五月 1, 2026， https://www.emergentmind.com/topics/semantic-id-learning

\34. MMQ-v2: Align, Denoise, and Amplify: Adaptive Behavior Mining for Semantic IDs Learning in Recommendation - arXiv, 访问时间为 五月 1, 2026， https://arxiv.org/html/2510.25622v1

\35. OneLive: Dynamically Unified Generative Framework for Live-Streaming Recommendation, 访问时间为 五月 1, 2026， https://www.researchgate.net/publication/400603685_OneLive_Dynamically_Unified_Generative_Framework_for_Live-Streaming_Recommendation

\36. OneLive: Dynamically Unified Generative Framework for Live-Streaming Recommendation, 访问时间为 五月 1, 2026， https://arxiv.org/html/2602.08612v1

\37. [2502.13581] ActionPiece: Contextually Tokenizing Action Sequences for Generative Recommendation - arXiv, 访问时间为 五月 1, 2026， https://arxiv.org/abs/2502.13581

\38. Context-Aware Diffusion-based Sequential Recommendation - Worcester Polytechnic Institute, 访问时间为 五月 1, 2026， http://web.cs.wpi.edu/~kmlee/pubs/you24bigdata.pdf

\39. Stop Treating Collisions Equally: Qualification-Aware Semantic ID Learning for Recommendation at Industrial Scale | Request PDF - ResearchGate, 访问时间为 五月 1, 2026， https://www.researchgate.net/publication/401469266_Stop_Treating_Collisions_Equally_Qualification-Aware_Semantic_ID_Learning_for_Recommendation_at_Industrial_Scale

\40. A Wolf in Sheep’s Clothing: Bypassing Commercial LLM Guardrails via Harmless Prompt Weaving and Adaptive Tree Search - ResearchGate, 访问时间为 五月 1, 2026， [https://www.researchgate.net/publication/398226037_A_Wolf_in_Sheep’s_Clothing_Bypassing_Commercial_LLM_Guardrails_via_Harmless_Prompt_Weaving_and_Adaptive_Tree_Search](https://www.researchgate.net/publication/398226037_A_Wolf_in_Sheep's_Clothing_Bypassing_Commercial_LLM_Guardrails_via_Harmless_Prompt_Weaving_and_Adaptive_Tree_Search)

\41. Inference in Computational Semantics ICoS-5 Workshop Proceedings - ACL Anthology, 访问时间为 五月 1, 2026， https://aclanthology.org/W06-39.pdf

\42. Discrete Diffusion for Bundle Construction - OpenReview, 访问时间为 五月 1, 2026， https://openreview.net/forum?id=dKyhgfe50H

\43. (PDF) HiGR: Efficient Generative Slate Recommendation via Hierarchical Planning and Multi-Objective Preference Alignment - ResearchGate, 访问时间为 五月 1, 2026， https://www.researchgate.net/publication/399276650_HiGR_Efficient_Generative_Slate_Recommendation_via_Hierarchical_Planning_and_Multi-Objective_Preference_Alignment

\44. HiGR: Efficient Generative Slate Recommendation via Hierarchical Planning and Multi-Objective Preference Alignment - arXiv, 访问时间为 五月 1, 2026， https://arxiv.org/pdf/2512.24787

\45. Stop Treating Collisions Equally: Qualification-Aware Semantic ID Learning for Recommendation at Industrial Scale - arXiv, 访问时间为 五月 1, 2026， https://arxiv.org/pdf/2603.00632

\46. Generative Recommendation Models: Progress and Directions - GitHub Pages, 访问时间为 五月 1, 2026， https://large-genrec.github.io/static/slides/www25/overall-large-genrec-tutorial-www25.pdf

\47. CEMG: Collaborative-Enhanced Multimodal Generative Recommendation | Request PDF, 访问时间为 五月 1, 2026， https://www.researchgate.net/publication/400872455_CEMG_Collaborative-Enhanced_Multimodal_Generative_Recommendation

\48. DAS: Dual-Aligned Semantic IDs Empowered Industrial Recommender System, 访问时间为 五月 1, 2026， https://www.researchgate.net/publication/394488149_DAS_Dual-Aligned_Semantic_IDs_Empowered_Industrial_Recommender_System

\49. DAS: Dual-Aligned Semantic IDs Empowered Industrial Recommender System - arXiv, 访问时间为 五月 1, 2026， https://arxiv.org/html/2508.10584v1

\50. Semantic Item Graph Enhancement for Multimodal Recommendation - arXiv, 访问时间为 五月 1, 2026， https://arxiv.org/pdf/2508.06154

\51. Semantic Item Graph Enhancement for Multimodal Recommendation - arXiv, 访问时间为 五月 1, 2026， https://arxiv.org/html/2508.06154v1

\52. Progressive Collaborative and Semantic Knowledge Fusion for Generative Recommendation | Request PDF - ResearchGate, 访问时间为 五月 1, 2026， https://www.researchgate.net/publication/388884135_Progressive_Collaborative_and_Semantic_Knowledge_Fusion_for_Generative_Recommendation

\53. Collaborative-Aware Multimodal Semantic IDs - Emergent Mind, 访问时间为 五月 1, 2026， https://www.emergentmind.com/topics/collaborative-aware-multimodal-semantic-ids

\54. DAS: Dual-Aligned Semantic IDs Empowered Industrial … - dblp, 访问时间为 五月 1, 2026， https://dblp.org/rec/conf/cikm/YeSSWWJ25.html

\55. Bringing Model Editing to Generative Recommendation in Cold-Start Scenarios - arXiv, 访问时间为 五月 1, 2026， https://arxiv.org/html/2603.14259v1

\56. Bringing Model Editing to Generative Recommendation in Cold-Start Scenarios | Request PDF - ResearchGate, 访问时间为 五月 1, 2026， https://www.researchgate.net/publication/402479787_Bringing_Model_Editing_to_Generative_Recommendation_in_Cold-Start_Scenarios

\57. SMILE: SeMantic Ids Enhanced CoLd Item Representation for Click-through Rate Prediction in E-commerce SEarch - arXiv, 访问时间为 五月 1, 2026， https://arxiv.org/html/2510.12604v1

\58. SMILE: SeMantic Ids Enhanced CoLd Item Representation for Click-through Rate Prediction in E-commerce SEarch - ResearchGate, 访问时间为 五月 1, 2026， https://www.researchgate.net/publication/396499863_SMILE_SeMantic_Ids_Enhanced_CoLd_Item_Representation_for_Click-through_Rate_Prediction_in_E-commerce_SEarch

\59. MMGRid: Navigating Temporal-aware and Cross-domain Generative Recommendation via Model Merging - arXiv, 访问时间为 五月 1, 2026， https://arxiv.org/html/2601.15930v1

\60. Better Generalization with Semantic IDs: A Case Study in Ranking for Recommendations, 访问时间为 五月 1, 2026， https://www.semanticscholar.org/paper/Better-Generalization-with-Semantic-IDs%3A-A-Case-in-Singh-Vu/049b6e2c053d289c6e9f15cd9562836cdeece2c2