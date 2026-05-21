import copy
import logging
from typing import Any, Dict, Optional, Tuple

import torch
from lightning import LightningModule
from lightning.pytorch.trainer.states import TrainerFn
from torch import nn
from torch.distributions import Categorical
from torchmetrics import MeanMetric

from src.data.loading.components.interfaces import ItemData
from src.models.components.interfaces import OneKeyPerPredictionOutput
from src.models.modules.clustering.base_clustering_module import BaseClusteringModule
from src.modules.clustering.knn_preservation import (
    build_npr_mask,
    build_reciprocal_mask,
    compute_kpl_loss,
    compute_rncl_loss,
    load_knn_g0,
)
import torch.nn.functional as F

class ResidualQuantization(LightningModule):
    def __init__(
        self,
        n_layers: Optional[int] = None,
        normalization_layer: nn.Module = nn.Identity(),
        encoder: nn.Module = nn.Identity(),
        decoder: nn.Module = nn.Identity(),
        quantization_layer: Optional[BaseClusteringModule] = None,
        quantization_layer_list: Optional[nn.ModuleList] = None,
        init_buffer_size: int = 1000,
        training_loop_function: callable = None,
        quantization_loss_weight: float = 1.0,
        reconstruction_loss_function: Optional[nn.Module] = None,
        reconstruction_loss_weight: float = 0.0,
        normalize_residuals: bool = True,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
        train_layer_wise: bool = False,
        track_residuals: bool = False,
        verbose: bool = False,
        **kwargs,
    ) -> None:
        """
        Initialize the Residual Quantization module.

        Args:
            n_layers: The number of quantization layers.
            quantization_layer: The quantization layer to use.
            quantization_layer_list: The list of quantization layers to use.
            init_buffer_size: The size of the buffer for initializing the centroids.
            training_loop_function: The custom training loop function to use.
            normalize_residuals: Whether to normalize the residuals before quantization.
            optimizer: The optimizer to use.
            scheduler: The learning rate scheduler to use.
            quantization_loss_weight: The weight of the quantization loss.
            reconstruction_loss_function: The loss function to use for reconstruction.
            reconstruction_loss_weight: The weight of the reconstruction loss.
            normalize_inputs: Whether to normalize the input embeddings.
            batch_norm_inputs: Whether to apply batch normalization to the input embeddings.
            train_layer_wise: Whether to train the layers one at a time. If true, each layer
                will be trained for the same, plus or minus one, number of steps.
            track_residuals: Whether to track residuals at each layer.
            verbose: Whether to log progress during training.
        """
        super().__init__()
        self.save_hyperparameters(
            logger=False,
            ignore=[
                "optimizer",
                "scheduler",
                "training_loop_function",
                "normalization_layer",
                "encoder",
                "decoder",
                "quantization_layer_list",
                "quantization_layer",
            ],
        )

        self.optimizer = optimizer
        self.scheduler = scheduler

        self.verbose = verbose
        self.log_if_true("Verbose mode enabled", self.verbose)
        # We always track residuals if verbose mode is enabled
        self.track_residuals = track_residuals or self.verbose

        self.normalization_layer = normalization_layer
        self.encoder = encoder
        self.decoder = decoder
        self.quantization_layer_list = self._instantiate_quantization_layer_list(
            quantization_layer,
            quantization_layer_list,
            n_layers,
        )
        self.n_layers = len(self.quantization_layer_list)

        self.training_loop_function = training_loop_function
        if self.training_loop_function is not None:
            self.log_if_true("Using custom training loop function", self.verbose)
            self.automatic_optimization = False
        self.train_layer_wise = train_layer_wise
        self.normalize_residuals = normalize_residuals

        self.quantization_loss_weight = quantization_loss_weight
        self.reconstruction_loss_function = reconstruction_loss_function
        self.reconstruction_loss_weight = reconstruction_loss_weight

        # VCR-TD properties
        self.use_vcr_td = kwargs.get("use_vcr_td", False)
        self.use_time_decay = kwargs.get("use_time_decay", False)
        self.use_dynamic_margin = kwargs.get("use_dynamic_margin", False)
        self.use_cvpm = kwargs.get("use_cvpm", True) # 新增 CVPM 开关，默认开启
        self.cvpm_temperature = kwargs.get("cvpm_temperature", 0.15)  # 默认0.15，可调（0.05~0.3区间）
        self.alpha = kwargs.get("alpha", 0.01)
        self.m0 = kwargs.get("m0", 0.5)
        self.lambda_rep = kwargs.get("lambda_rep", 0.3)
        self.exposure_counts_path = kwargs.get("exposure_counts_path", None)

        # === VCF (VCR-TD-QuaSID Fusion) properties ===
        self.use_vcf = kwargs.get("use_vcf", False)  # VCF融合方案总开关
        
        # 互斥检查：use_vcr_td 和 use_vcf 不能同时开启
        if self.use_vcr_td and self.use_vcf:
            raise ValueError(
                "use_vcr_td and use_vcf cannot both be True. "
                "VCF (use_vcf=True) is the upgraded version of VCR-TD and replaces it. "
                "Please set use_vcr_td=False when using use_vcf=True."
            )
        self.hamming_radius = kwargs.get("hamming_radius", 2)  # 部分碰撞半径R
        self.severity_beta = kwargs.get("severity_beta", 0.5)  # 严重程度调节系数
        self.cold_start_threshold = kwargs.get("cold_start_threshold", 50)  # 冷启动阈值τ
        self.enhancement_factor = kwargs.get("enhancement_factor", 0.5)  # 增强因子γ
        self.lambda_full = kwargs.get("lambda_full", 1.0)  # 完全碰撞损失权重
        self.lambda_partial = kwargs.get("lambda_partial", 0.5)  # 部分碰撞损失权重
        self.min_margin = kwargs.get("min_margin", 0.05)  # 最小边界保护
        self.max_enhance = kwargs.get("max_enhance", 2.0)  # 增强因子上限
        self.min_severity_partial = kwargs.get("min_severity_partial", 1.1)  # 部分碰撞最小severity
        self.repulsion_warmup_steps = kwargs.get("repulsion_warmup_steps", 1000)  # Phase-1 hard-gate steps
        # Phase-2 soft ramp: linear increase from 0→1 over soft_ramp_steps after warmup.
        # Prevents gradient shock from discontinuous 0→full repulsion at warmup boundary.
        # 0 = disabled (single hard gate, original behavior).
        self.repulsion_soft_ramp_steps = kwargs.get("repulsion_soft_ramp_steps", 0)
        # margin_dissim_weight: 语义差异加成权重（>0 时对不同语义的碰撞增加额外 margin）
        # 设 0.0 时退化为纯固定 margin（安全默认）；推荐训练时使用 0.3
        self.margin_dissim_weight = kwargs.get("margin_dissim_weight", 0.0)

        # === Path B: Exposure-Aware Asymmetric Repulsion ===
        self.use_asymmetric_repulsion = kwargs.get("use_asymmetric_repulsion", False)
        self.asymmetric_alpha_min = kwargs.get("asymmetric_alpha_min", 0.05)
        self.asymmetric_temperature = kwargs.get("asymmetric_temperature", 1.0)
        # ASYM reduces total repulsion gradient to ~(1+α)/2 of symmetric.
        # lambda_scale compensates: >1.0 restores total gradient magnitude while
        # preserving the hot/tail gradient asymmetry. Recommend 1.5–2.0.
        self.asymmetric_lambda_scale = kwargs.get("asymmetric_lambda_scale", 1.0)

        # === Module A: Neighbor-Preserving Repulsion (NPR) ===
        # Modulates repulsion gradient by G0 k-NN proximity instead of exposure.
        # If j ∈ kNN_G0(i): protect (α=npr_alpha_min), else full repulsion (α=1.0).
        self.use_npr = kwargs.get("use_npr", False)
        self.npr_alpha_min = kwargs.get("npr_alpha_min", 0.01)

        # === Direction 2: Layer-wise Differential Repulsion & NPR ===
        # Per-layer VCF margins and NPR alpha.
        # When enabled, replaces the SID-level full/partial collision classification
        # with per-layer collision masks. Each layer gets its own margin, λ, and α.
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
        # Hamming filter: only pairs with H ≤ this are repelled in layerwise mode.
        # 0 = H=0 only (strictest), 1 = H≤1, etc. Default 0.
        self.layerwise_hamming_radius = kwargs.get("layerwise_hamming_radius", 0)

        # === Module B: k-NN Preservation Loss (KPL) ===
        # InfoNCE loss with G0 k-NN neighbors as positives.
        # Complementary to QuaSID InfoNCE (co-occurrence positives).
        self.use_kpl = kwargs.get("use_kpl", False)
        self.lambda_kpl = kwargs.get("lambda_kpl", 0.001)
        self.kpl_tau = kwargs.get("kpl_tau", 0.5)
        self.kpl_ramp_steps = kwargs.get("kpl_ramp_steps", 1000)
        self.kpl_collision_exclude_radius = kwargs.get("kpl_collision_exclude_radius", 0)
        self.knn_g0_path = kwargs.get("knn_g0_path", None)

        # === Module C: Reciprocal Neighbor Consistency Loss (RNCL) ===
        # Gentle cosine-distance loss on reciprocal k-NN pairs only.
        # Unlike KPL: reciprocal only (not all k-NN), cosine distance (not InfoNCE),
        # 10x smaller λ.  Purely attractive — no softmax denominator, no pushing away.
        self.use_rncl = kwargs.get("use_rncl", False)
        self.lambda_rncl = kwargs.get("lambda_rncl", 0.0001)
        self.rncl_ramp_steps = kwargs.get("rncl_ramp_steps", 1000)

        # === Online k-NN: eliminate G0 dependency ===
        # Maintain EMA of encoder bottleneck outputs (N, 64) and periodically
        # recompute k-NN to use as neighbor reference for NPR and KPL.
        # This lets the model protect its OWN current neighborhood structure
        # instead of relying on a precomputed G0 checkpoint.
        self.use_online_knn = kwargs.get("use_online_knn", False)
        self.online_knn_ema_momentum = kwargs.get("online_knn_ema_momentum", 0.99)
        self.online_knn_update_interval = kwargs.get("online_knn_update_interval", 100)
        self.online_knn_k = kwargs.get("online_knn_k", 50)
        self._online_knn_ema = None          # (num_items, bottleneck_dim) float
        self._online_knn_indices = None      # legacy, replaced by register_buffer below
        self._online_knn_items_seen = None   # (num_items,) bool — which items have EMA entries
        self._online_knn_num_items = None

        # === QuaSID-style Dual-Tower (same-encoder + stop_gradient) ===
        # Uses encoded_embeddings.detach() as target instead of an EMA copy.
        # This keeps both sides in the same embedding space (no EMA lag artifacts)
        # while breaking the mirror-hall feedback loop via unidirectional gradients.
        # Set to False to fall back to single-tower behavior (not recommended).
        self.use_dual_tower = kwargs.get("use_dual_tower", True)
        # Legacy: _target_encoder kept as None for backward checkpoint compatibility.
        self._target_encoder = None
        self.target_encoder_ema_tau = 0.0  # deprecated, kept for old configs

        # === VCF stability knobs (critical; prevents codebook collapse) ===
        # Limit collision pairs to avoid O(B^2) domination.
        self.vcf_max_pairs_per_sample = kwargs.get("vcf_max_pairs_per_sample", 64)
        # Gate repulsion when codebook health is poor.
        self.vcf_gate_min_layer0_coverage = kwargs.get("vcf_gate_min_layer0_coverage", 0.50)
        self.vcf_gate_min_frac_unique_ids = kwargs.get("vcf_gate_min_frac_unique_ids", 0.50)
        # Optional hard cap for repulsion loss to prevent spikes.
        self.vcf_repulsion_clip = kwargs.get("vcf_repulsion_clip", 0.0)

        # QuaSID 静态边界（消融 G4: use_dynamic_margin=False 时回退使用）
        self.m_full = kwargs.get("m_full", 0.8)       # 完全碰撞静态边界，与 QuaSID 默认值一致
        self.m_partial = kwargs.get("m_partial", 0.5)  # 部分碰撞静态边界，与 QuaSID 默认值一致

        # === Adaptive margin ramp ===
        # Margin linearly increases from m_*_start to m_* over margin_ramp_steps.
        # Ramp begins at repulsion_warmup_steps. Before ramp: use start values.
        # Set margin_ramp_steps=0 to disable (use static margins).
        self.m_full_start = kwargs.get("m_full_start", 0.3)
        self.m_partial_start = kwargs.get("m_partial_start", 0.15)
        self.margin_ramp_steps = kwargs.get("margin_ramp_steps", 0)

        # === Codebook entropy regularization ===
        # Lightweight diversity bonus; 0.0 = disabled.
        self.codebook_entropy_weight = kwargs.get("codebook_entropy_weight", 0.0)

        # === Codebook reset (prevents encoder/index collapse) ===
        # Standard VQ-VAE fix: periodically replace dead centroids with random
        # encoder outputs from the current batch, preventing the runaway feedback
        # where unused centroids get zero gradient and become permanently stale.
        self.codebook_reset_interval = kwargs.get("codebook_reset_interval", 10)
        # Minimum EMA assignment count below which a centroid is considered "dead"
        self.codebook_reset_decay = kwargs.get("codebook_reset_decay", 0.9)

        # === VCR-TD v2.0: U型双端强化时间权重参数 ===
        # 基于 batch 内曝光次数百分位将商品分为三段生命周期:
        #   冷启动 (rank < cold_pct)         → w = 1 + delta_cold  (需建立独立SID路径)
        #   成长期 (cold_pct ≤ rank < hot_pct) → w = 1.0            (基准，不干扰自然聚类)
        #   热门期 (rank ≥ hot_pct)           → w = 1 + delta_hot   (TIGER序列中最关键)
        # pair权重: w_ij = sqrt(w_i * w_j)，几何均值保守插值
        self.cold_pct = kwargs.get("cold_pct", 0.25)      # 冷启动上界百分位
        self.hot_pct = kwargs.get("hot_pct", 0.75)        # 热门下界百分位
        self.delta_cold = kwargs.get("delta_cold", 0.4)   # 冷启动排斥增益
        self.delta_hot = kwargs.get("delta_hot", 0.3)     # 热门排斥增益

        # VCF 默认启用动态边界；消融 G4 可通过显式传 use_dynamic_margin=false 关闭
        # （注意：这会覆盖上方 VCR-TD 路径中 use_dynamic_margin 的默认值 False）
        if self.use_vcf:
            self.use_dynamic_margin = kwargs.get("use_dynamic_margin", True)

        # === QuaSID InfoNCE 对比学习（替代旧 TCCL）===
        # 复用 use_tcl 标志位作为 QuaSID CL 开关；旧 TCCL 方法保留但废弃。
        self.use_tcl = kwargs.get("use_tcl", False)
        self.lambda_cl = kwargs.get("lambda_cl", 0.001)  # v3: 0.001 — gradient-aligned with VCF hinge
        self.cl_tau = kwargs.get("cl_tau", 0.07)
        self.alpha_cl = kwargs.get("alpha_cl", 0.25)     # 保留（旧 TCCL 用），QuaSID CL 不使用
        self.tccl_warmup_steps = kwargs.get("tccl_warmup_steps", 0)
        self.quasid_cl_tau = kwargs.get("quasid_cl_tau", 0.5)  # v3: 0.5 — exp(cos_sim/τ) ≈ 5, ~VCF hinge scale
        self.cl_ramp_steps = kwargs.get("cl_ramp_steps", 1000)  # v3: λ ramp 1000→2000 steps
        self.train_cl_loss = MeanMetric()                # 用于日志记录
        self.train_kpl_loss = MeanMetric()               # Module B: KPL loss

        # === 内容相似度感知 CVPM 扩展（方向 4）===
        # 若 cvpm_sim_threshold > 0，CVPM 额外将内容余弦相似度 ≥ threshold 的 pair
        # 视为"语义相似良性碰撞"并屏蔽出排斥集（对齐 QuaSID Section 6 未来工作）。
        # 0.0 = 禁用（默认，保持 Q0 基准行为）
        self.cvpm_sim_threshold = kwargs.get("cvpm_sim_threshold", 0.0)

        # Step-level diagnostics for TCCL in-batch positives.
        # - no_pos_steps: how often a training step has zero in-batch positives
        # - pos_sample_frac: fraction of samples that have >=1 in-batch positive
        # - avg_pos_per_sample: average number of positive pairs per sample
        self._tccl_steps_seen = 0
        self._tccl_steps_no_pos = 0
        self._tccl_steps_pos_sample_frac_sum = 0.0
        self._tccl_steps_avg_pos_per_sample_sum = 0.0

        # === Historical Time Masking state ===
        # Interpreted as "how much history has been observed so far" (monotonic).
        # Used to cap exposure counts to avoid leaking future information.
        self.global_historical_step = 0

        # exposure_counts 供 VCR-TD 和 VCF 两条路径共用；不注册为 buffer 以避免 ckpt 兼容问题
        self._exposure_counts = None
        if (self.use_vcr_td or self.use_vcf) and self.exposure_counts_path:
            self._exposure_counts = torch.load(self.exposure_counts_path)

        # G0 k-NN indices for NPR (Module A) and KPL (Module B).
        # When use_online_knn=True, the online buffer replaces the precomputed G0 file.
        self._knn_g0_buffer = None
        need_knn = (self.use_npr or self.use_kpl)
        if need_knn and self.knn_g0_path is not None and not self.use_online_knn:
            self._knn_g0_buffer = load_knn_g0(
                self.knn_g0_path,
                num_items=len(self._exposure_counts) if self._exposure_counts is not None else 0,
                device=torch.device("cpu"),
            )

        if need_knn and self.use_online_knn:
            self._online_knn_num_items = (
                len(self._exposure_counts) if self._exposure_counts is not None else 0
            )

        # Per-layer EMA centroid assignment counters for codebook reset.
        # Shape: (n_clusters,) per layer. Filled in on_train_start / on the first step.
        self._codebook_assignment_ema: list = []

        self.train_loss = MeanMetric()
        self.train_quantization_loss = MeanMetric()
        self.train_reconstruction_loss = MeanMetric()
        self.train_repulsion_loss = MeanMetric()
        if self.verbose:
            # Note that if normalize_residuals is True, the residuals norm metrics below are uninformative
            self.train_first_residuals_norm_ratio = MeanMetric()
            self.train_last_residuals_norm_ratio = MeanMetric()
            self.first_centroids_norm = MeanMetric()
            self.last_centroids_norm = MeanMetric()
            self.train_frac_unique_ids = MeanMetric()
            self.train_mse = MeanMetric()
            for layer_idx in range(self.n_layers):
                # We use MeanMetric to track the fraction of unique ids and the
                # entropy of the cluster ids for each layer
                # Note that we don't need to move these metrics to the device here,
                # because they will be moved to the device in the training_step method
                setattr(
                    self,
                    f"train_layer_coverages_{layer_idx}",
                    MeanMetric(),
                )
                setattr(
                    self,
                    f"train_layer_id_entropy_{layer_idx}",
                    MeanMetric(),
                )

        self.val_loss = MeanMetric()
        self.val_first_residuals_norm_ratio = MeanMetric()
        self.val_last_residuals_norm_ratio = MeanMetric()
        self.val_mse = MeanMetric()
        self.val_frac_unique_ids = MeanMetric()

        self.test_loss = MeanMetric()
        self.test_first_residuals_norm_ratio = MeanMetric()
        self.test_last_residuals_norm_ratio = MeanMetric()
        self.test_mse = MeanMetric()
        self.test_frac_unique_ids = MeanMetric()

        # We set the initialization buffer sizes for each layer to the same value
        self.init_buffer_size = init_buffer_size
        for layer in self.quantization_layer_list:
            layer.init_buffer_size = init_buffer_size

    def _instantiate_quantization_layer_list(
        self,
        quantization_layer: Optional[BaseClusteringModule] = None,
        quantization_layer_list: Optional[nn.ModuleList] = None,
        n_layers: Optional[int] = None,
    ) -> None:
        """
        Instantiate the quantization layers. If quantization_layer_list is provided,
        it is used directly. Otherwise, a list of quantization layers is created using
        the provided quantization_layer and n_layers.

        Args:
            quantization_layer: The quantization layer to use, if
                quantization_layer_list is not provided.
            quantization_layer_list: The list of quantization layers to use.
            n_layers: The number of quantization layers to create, if
                quantization_layer_list is not provided.

        Returns:
            An nn.ModuleList of quantization layers.
        """
        if quantization_layer_list is not None:
            return quantization_layer_list
        else:
            if n_layers is None:
                raise ValueError(
                    "Since a quantization layer list was not provided, n_layers must be provided."
                )
            if quantization_layer is None:
                raise ValueError(
                    "Either quantization_layer or quantization_layer_list must be provided."
                )
            return nn.ModuleList(
                modules=[copy.deepcopy(quantization_layer) for _ in range(n_layers)]
            )

    def compute_time_decay_weight(self, exposure_times: torch.Tensor, alpha: float = 0.01) -> torch.Tensor:
        """
        计算时间衰减权重 w(t) = e^(-αt)
        Args:
            exposure_times: 物品曝光时间/交互次数
            alpha: 衰减系数
        Returns:
            weight: [0, 1]之间的权重值
        """
        return torch.exp(-alpha * exposure_times)

    def on_train_batch_start(self, batch: Any, batch_idx: int) -> None:
        """Use global_step for finer-grained, DDP-consistent historical progress."""
        if getattr(self, "trainer", None) is not None:
            self.global_historical_step = max(1, int(self.trainer.global_step) + 1)

    def on_train_batch_end(self, outputs, batch: Any, batch_idx: int) -> None:
        """Called after each training batch; no-op since QuaSID-style dual-tower uses detach."""
        pass

    def compute_dynamic_margin(self, content_embeddings: torch.Tensor, m0: float = 0.5) -> torch.Tensor:
        """
        计算语义感知动态边界矩阵 m(i,j) = m_0 * (1 - cos(x_i, x_j))
        Args:
            content_embeddings: 多模态内容嵌入 (batch_size, dim)
            m0: 基础边界值
        Returns:
            margin_matrix: 动态边界矩阵 (batch_size, batch_size)
        """
        # (batch_size, batch_size)
        norm_embeddings = F.normalize(content_embeddings, p=2, dim=-1)
        cosine_sim = torch.mm(norm_embeddings, norm_embeddings.t())
        return m0 * (1 - cosine_sim)

    def cvpm_mask(
        self,
        item_ids: torch.Tensor,
        positive_pair_matrix: Optional[torch.Tensor] = None,
        content_embeddings: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Conflict-Aware Valid Pair Masking (CVPM) - VCF版本
        硬掩码，排除同物品、正样本对，以及（可选）内容语义高度相似对。

        规则 1：同物品 pair（item_id 相同）→ 良性，屏蔽
        规则 2：协同正样本 pair（collaborative positive）→ 良性，屏蔽
        规则 3（可选）：内容嵌入余弦相似度 ≥ cvpm_sim_threshold → 语义相似良性碰撞，屏蔽
                        （需 cvpm_sim_threshold > 0 且传入 content_embeddings）

        Args:
            item_ids: 当前 batch 中每个样本的物品 ID (batch_size,)
            positive_pair_matrix: 协同正样本对 mask (batch_size, batch_size), 1=正样本
            content_embeddings: 多模态内容嵌入 (batch_size, D)，用于规则 3
        Returns:
            cvpm: (batch_size, batch_size) bool mask，True 表示参与排斥
        """
        item_ids = item_ids.to(dtype=torch.long, device=self.device)
        batch_size = item_ids.shape[0]

        # 规则 1: same-item mask
        same_item_mask = (item_ids.unsqueeze(0) != item_ids.unsqueeze(1))

        # 规则 2: collaborative positives mask
        if positive_pair_matrix is not None:
            positive_pair_matrix = positive_pair_matrix.to(device=self.device)
            collaborative_mask = ~positive_pair_matrix.bool()
        else:
            collaborative_mask = torch.ones(batch_size, batch_size, dtype=torch.bool, device=self.device)

        # 规则 3: 内容相似度感知掩码（cvpm_sim_threshold > 0 时启用）
        # cosine_sim(i,j) >= threshold → 视为语义相似良性碰撞 → 从排斥集中屏蔽
        if self.cvpm_sim_threshold > 0.0 and content_embeddings is not None:
            norm_emb = F.normalize(content_embeddings.to(device=self.device), p=2, dim=-1)
            cosine_sim = torch.mm(norm_emb, norm_emb.T)          # (B, B)
            content_dissimilar_mask = cosine_sim < self.cvpm_sim_threshold
        else:
            content_dissimilar_mask = torch.ones(batch_size, batch_size, dtype=torch.bool, device=self.device)

        cvpm = same_item_mask & collaborative_mask & content_dissimilar_mask
        return cvpm

    def compute_hamming_distance(self, sid_tokens: torch.Tensor) -> torch.Tensor:
        """
        计算Hamming距离矩阵
        Args:
            sid_tokens: SID token序列 (batch_size, n_layers)
        Returns:
            H: Hamming距离矩阵 (batch_size, batch_size)
        """
        sid_i = sid_tokens.unsqueeze(1)  # (B, 1, L)
        sid_j = sid_tokens.unsqueeze(0)  # (1, B, L)
        H = (sid_i != sid_j).sum(dim=-1).float()  # (B, B)
        return H

    def classify_collisions(self, H: torch.Tensor, M: torch.Tensor, R: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        分类碰撞类型：完全碰撞 vs 部分碰撞
        Args:
            H: Hamming距离矩阵 (B, B)
            M: CVPM掩码 (B, B) bool
            R: 部分碰撞半径
        Returns:
            Omega_full: 完全碰撞的索引 (tuple of tensors)
            Omega_partial: 部分碰撞的索引 (tuple of tensors)
        """
        # 完全碰撞: H=0 且 M=1
        is_full = (H == 0) & M
        Omega_full = torch.where(is_full)
        
        # 部分碰撞: 0 < H <= R 且 M=1
        is_partial = (H > 0) & (H <= R) & M
        Omega_partial = torch.where(is_partial)
        
        return Omega_full, Omega_partial

    def compute_dynamic_margin_vcf(self, content_embeddings: torch.Tensor, H: torch.Tensor,
                                    m0: float, R: int, beta: float) -> torch.Tensor:
        """
        计算动态边界 - VCF版本（含severity和最小边界保护）

        核心修复（原公式 Bug 说明）:
          原公式: margin_base = m0 * (1 - cos_sim)
          问题：RQ-VAE 将语义相似 item 聚到同一 SID，导致碰撞 pair 的 cos_sim 极高（0.8~0.95），
                margin_base → 0，排斥损失完全失效（实测 repulsion_loss ≈ 3.9e-9）。

          新公式: margin_base = m0 + margin_dissim_weight * (1 - cos_sim).clamp(0)
          - m0 提供固定兜底（始终 ≥ m0，对相似碰撞同样有效）
          - margin_dissim_weight * (1 - cos_sim) 给"语义差异越大的碰撞"额外加成
            （语义相近 pair 加成 ≈ 0，语义迥异 pair 加成最大达 margin_dissim_weight）

        Args:
            content_embeddings: 多模态内容嵌入 (B, D)
            H: Hamming距离矩阵 (B, B)
            m0: 基础边界值（兜底）
            R: 部分碰撞半径
            beta: 严重程度调节系数
        Returns:
            margin: 动态边界矩阵 (B, B)；非碰撞对为 0
        """
        B = content_embeddings.size(0)
        device = content_embeddings.device

        # ── Step 1: 语义相似度（仅用于加成，不再压缩 base margin） ──────────────
        norm_embeddings = F.normalize(content_embeddings, p=2, dim=-1)
        cos_sim = torch.mm(norm_embeddings, norm_embeddings.t()).clamp(-1.0, 1.0)
        # dissim ∈ [0, 1]（语义越不同值越大）
        dissim = (1.0 - cos_sim).clamp(min=0.0, max=1.0)

        # ── Step 2: 基础边界 = 固定兜底 + 差异加成 ────────────────────────────
        # margin_dissim_weight=0 时等价于纯固定 margin（消融 G4 退化路径）
        margin_base = m0 + self.margin_dissim_weight * dissim  # ≥ m0 始终成立

        # ── Step 3: 严重程度因子（仅对碰撞对赋非零值） ────────────────────────
        severity = torch.zeros(B, B, device=device)
        is_full = (H == 0)
        severity[is_full] = 1.0 + beta                        # 完全碰撞：最严重

        is_partial = (H > 0) & (H <= R)
        if is_partial.any():
            severity[is_partial] = torch.clamp(
                1.0 + beta * (1.0 - H[is_partial] / R),
                min=self.min_severity_partial
            )

        # ── Step 4: 合并 margin；非碰撞对保持 0 ──────────────────────────────
        collision_mask = is_full | is_partial
        margin = torch.zeros(B, B, device=device)
        margin[collision_mask] = (margin_base * severity)[collision_mask]

        # ── Step 5: 最小边界保护（只作用于碰撞对） ────────────────────────────
        margin[collision_mask] = torch.clamp(margin[collision_mask], min=self.min_margin)

        return margin

    def compute_time_weight_vcf(self, exposure_times: torch.Tensor, H: torch.Tensor) -> torch.Tensor:
        """
        U型双端强化时间权重 (VCR-TD v2.0)

        设计逻辑:
          旧版单调递减权重存在根本缺陷：热门商品（高曝光）在TIGER训练序列中出现最频繁，
          其SID碰撞对推荐性能危害最大，单调递减策略反而削弱了对热门商品的排斥，
          导致B1实验中TIGER NDCG@10低于无时间权重的Q0基准。

          U型双端强化策略 (修正版方案四):
            - 冷启动期 (rank < cold_pct)           : w = 1 + delta_cold
              → 商品处于生命周期初期，SID路径尚未稳定，需强隔离防止被头部吸收
            - 成长期   (cold_pct ≤ rank < hot_pct) : w = 1.0
              → 基准权重，不干扰RQ-VAE自然语义聚类
            - 热门期   (rank ≥ hot_pct)             : w = 1 + delta_hot
              → 高频出现于用户序列，SID质量对TIGER性能影响最大，优先保护

          pair权重: w_ij = sqrt(w_i * w_j)，几何均值保守插值

        百分位在batch内动态计算，对不同曝光分布的数据集天然鲁棒。

        Args:
            exposure_times: 曝光计数 (B,)
            H: Hamming距离矩阵 (B, B)
        Returns:
            w_final: 时间权重矩阵 (B, B)，非碰撞对权重为0
        """
        B = exposure_times.size(0)
        device = exposure_times.device

        is_collision = (H <= self.hamming_radius)

        # Batch-level percentile ranking: rank_i ∈ [0, 1]
        sorted_idx = torch.argsort(exposure_times.float())
        ranks = torch.zeros(B, device=device)
        ranks[sorted_idx] = torch.arange(B, device=device, dtype=torch.float32) / max(B - 1, 1)

        # Per-item lifecycle stage weight
        w_stage = torch.ones(B, device=device)
        w_stage[ranks < self.cold_pct] += self.delta_cold
        w_stage[ranks >= self.hot_pct] += self.delta_hot

        # Pair weight: geometric mean of item-level weights
        w_pair = torch.sqrt(w_stage.unsqueeze(1) * w_stage.unsqueeze(0))  # (B, B)

        # Apply only to collision pairs; non-collision pairs → 0
        w_final = w_pair * is_collision.float()

        return w_final

    def vcf_repulsion_loss(
        self,
        continuous_emb: torch.Tensor,
        sid_tokens: torch.Tensor,
        item_ids: torch.Tensor,
        exposure_times: torch.Tensor,
        content_embeddings: torch.Tensor,
        positive_pair_matrix: Optional[torch.Tensor] = None,
        target_embeddings: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, int, int]:
        """
        VCF融合排斥损失 (HaMR + Dynamic Margin + Time Decay)

        Args:
            continuous_emb: 连续嵌入 (B, D) from online encoder
            sid_tokens: SID token序列 (B, L)
            item_ids: 物品ID (B,)
            exposure_times: 曝光时间 (B,)
            content_embeddings: 多模态内容嵌入 (B, D_in)
            positive_pair_matrix: 正样本对标记 (B, B)
            target_embeddings: Detached target embeddings (B, D) from same encoder. When provided,
                uses cross-distance online vs target (dual-tower) to break
                the mirror-hall feedback loop.

        Returns:
            L_repulsion: 排斥损失标量
            n_full: 完全碰撞对数量
            n_partial: 部分碰撞对数量
        """
        B = continuous_emb.shape[0]
        if B < 2:
            return torch.tensor(0.0, device=continuous_emb.device), 0, 0
        
        device = continuous_emb.device
        
        # Stage 1: CVPM掩码（传入 content_embeddings 以启用规则 3：内容相似度感知屏蔽）
        M = self.cvpm_mask(item_ids, positive_pair_matrix, content_embeddings=content_embeddings)
        
        # Stage 2: Hamming距离计算
        H = self.compute_hamming_distance(sid_tokens)
        
        # Stage 3: 碰撞分类
        Omega_full, Omega_partial = self.classify_collisions(H, M, self.hamming_radius)
        n_full = len(Omega_full[0])
        n_partial = len(Omega_partial[0])
        
        # 如果没有碰撞对，返回0
        if n_full == 0 and n_partial == 0:
            return torch.tensor(0.0, device=device), 0, 0

        # ---- Stability gate (prevent codebook collapse) ----
        # If layer0 coverage or SID uniqueness is too low, temporarily disable repulsion.
        try:
            n_clusters = int(getattr(self.quantization_layer_list[0], "n_clusters", 0))
            if n_clusters > 0:
                layer0_cov = float(sid_tokens[:, 0].unique().numel()) / float(n_clusters)
            else:
                layer0_cov = 1.0
            frac_unique_rows = float(sid_tokens.unique(dim=0).size(0)) / float(B)
        except Exception:
            layer0_cov = 1.0
            frac_unique_rows = 1.0

        if (layer0_cov < self.vcf_gate_min_layer0_coverage) or (
            frac_unique_rows < self.vcf_gate_min_frac_unique_ids
        ):
            if self.global_step % 50 == 0:
                print(f"[STABILITY GATE] step={self.global_step} | "
                      f"layer0_cov={layer0_cov:.4f} (thresh={self.vcf_gate_min_layer0_coverage}) | "
                      f"frac_unique={frac_unique_rows:.4f} (thresh={self.vcf_gate_min_frac_unique_ids}) | "
                      f"BLOCKING repulsion", flush=True)
            return torch.tensor(0.0, device=device), n_full, n_partial

        # ---- Pair subsampling (avoid O(B^2) domination) ----
        K = int(self.vcf_max_pairs_per_sample)
        if K > 0 and (n_full + n_partial) > B * K:
            def _subsample_pairs(omega: Tuple[torch.Tensor, torch.Tensor], k: int) -> Tuple[torch.Tensor, torch.Tensor]:
                ii, jj = omega
                if ii.numel() == 0:
                    return omega
                new_i = []
                new_j = []
                for i in range(B):
                    mask = (ii == i)
                    if not mask.any():
                        continue
                    cand_j = jj[mask]
                    if cand_j.numel() <= k:
                        new_i.append(ii[mask])
                        new_j.append(cand_j)
                    else:
                        perm = torch.randperm(cand_j.numel(), device=device)[:k]
                        sel_j = cand_j[perm]
                        sel_i = torch.full((sel_j.numel(),), i, device=device, dtype=ii.dtype)
                        new_i.append(sel_i)
                        new_j.append(sel_j)
                if len(new_i) == 0:
                    return omega
                return (torch.cat(new_i, dim=0), torch.cat(new_j, dim=0))

            Omega_full = _subsample_pairs(Omega_full, K)
            Omega_partial = _subsample_pairs(Omega_partial, K)
            n_full = int(Omega_full[0].numel())
            n_partial = int(Omega_partial[0].numel())
        
        # Stage 4: 动态/静态边界计算
        # use_dynamic_margin=True (VCF默认)：内容相似度 × 碰撞严重度的融合动态边界
        # use_dynamic_margin=False (消融 G4)：退化为 QuaSID 静态边界 m_full / m_partial
        if self.use_dynamic_margin:
            margin = self.compute_dynamic_margin_vcf(
                content_embeddings, H, self.m0, self.hamming_radius, self.severity_beta
            )
        else:
            # Adaptive margin: ramp from m_*_start to m_* over margin_ramp_steps.
            # Ramp starts after warmup, giving codebook time to stabilize before
            # imposing strong repulsion. Set margin_ramp_steps=0 for static margins.
            if self.margin_ramp_steps > 0 and self.training:
                ramp_start = int(self.repulsion_warmup_steps)
                ramp_progress = (int(self.global_step) - ramp_start) / max(int(self.margin_ramp_steps), 1)
                ramp_progress = max(0.0, min(1.0, ramp_progress))
                eff_m_full = self.m_full_start + ramp_progress * (self.m_full - self.m_full_start)
                eff_m_partial = self.m_partial_start + ramp_progress * (self.m_partial - self.m_partial_start)
            else:
                eff_m_full = self.m_full
                eff_m_partial = self.m_partial
                ramp_progress = -1.0  # sentinel: static margin

            margin = torch.zeros(B, B, device=device)
            margin[H == 0] = eff_m_full
            mask_partial = (H > 0) & (H <= self.hamming_radius)
            margin[mask_partial] = eff_m_partial

            if self.global_step % 50 == 0 and self.training:
                ramp_str = f"ramp={ramp_progress:.3f}" if ramp_progress >= 0 else "ramp=off"
                print(f"[ADAPTIVE MARGIN] step={int(self.global_step)} | "
                      f"eff_m_full={eff_m_full:.3f} | eff_m_partial={eff_m_partial:.3f} | "
                      f"{ramp_str}")
        
        # Stage 5: 时间权重计算 (U型双端强化；use_time_decay=False 时退化为均等权重)
        if self.use_time_decay and exposure_times is not None:
            w_final = self.compute_time_weight_vcf(exposure_times, H)
        else:
            # 无时间权重：碰撞对均赋权重 1.0（Q0/GQSS基准行为）
            w_final = (H <= self.hamming_radius).float()
        
        # Stage 6: 余弦距离计算
        if target_embeddings is not None:
            # Dual-tower: cross-distance online vs target.
            # D[i,j] = 1 - cos(z_online_i, z_target_j)
            # Gradient flows only through z_online_i (row); z_target_j is anchored.
            norm_online = F.normalize(continuous_emb, p=2, dim=-1)
            norm_target = F.normalize(target_embeddings, p=2, dim=-1)
            D = 1.0 - torch.mm(norm_online, norm_target.t())  # (B, B) asymmetric
        else:
            # Single-tower: self-distance (backward compatible)
            norm_emb = F.normalize(continuous_emb, p=2, dim=-1)
            D = 1.0 - torch.mm(norm_emb, norm_emb.t())  # (B, B)

        # Stage 7: 排斥损失计算（对称 / 曝光感知非对称 / NPR）
        if (self.use_asymmetric_repulsion or self.use_npr) and exposure_times is not None:
            L_repulsion, n_full, n_partial = self._asymmetric_repulsion_loss(
                continuous_emb=continuous_emb,
                margin=margin,
                w_final=w_final,
                Omega_full=Omega_full,
                Omega_partial=Omega_partial,
                exposure_times=exposure_times,
                n_full=n_full,
                n_partial=n_partial,
                target_embeddings=target_embeddings,
                item_ids=item_ids,
            )
        else:
            # 原始对称 hinge loss（保持向后兼容）
            hinge_loss = F.relu(margin - D)

            L_full = torch.tensor(0.0, device=device)
            if n_full > 0:
                L_full = (hinge_loss[Omega_full] * w_final[Omega_full]).mean()

            L_partial = torch.tensor(0.0, device=device)
            if n_partial > 0:
                L_partial = (hinge_loss[Omega_partial] * w_final[Omega_partial]).mean()

            L_repulsion = self.lambda_full * L_full + self.lambda_partial * L_partial

        # Optional clipping for stability
        if self.vcf_repulsion_clip and self.vcf_repulsion_clip > 0:
            L_repulsion = torch.clamp(L_repulsion, max=float(self.vcf_repulsion_clip))

        return L_repulsion, n_full, n_partial

    def _asymmetric_repulsion_loss(
        self,
        continuous_emb: torch.Tensor,
        margin: torch.Tensor,
        w_final: torch.Tensor,
        Omega_full: Tuple[torch.Tensor, torch.Tensor],
        Omega_partial: Tuple[torch.Tensor, torch.Tensor],
        exposure_times: torch.Tensor,
        n_full: int,
        n_partial: int,
        target_embeddings: Optional[torch.Tensor] = None,
        item_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, int, int]:
        """
        Stage 7 (Asymmetric / NPR): 非对称排斥损失。

        Two modes:
          - Exposure-aware (default, use_asymmetric_repulsion=True, use_npr=False):
            Gradient scaling by popularity ratio: α = min(exp_i, exp_j) / (exp_i + exp_j).
          - Neighbor-Preserving (use_npr=True):
            Gradient scaling by G0 k-NN proximity: α = npr_alpha_min if j ∈ kNN_G0(i).

        Single-tower: z_i, z_j from same encoder. z_protected = α·z_i + (1-α)·detach(z_i).
        Dual-tower: z_i from online, z_j from target (frozen).
        Target side is always anchored (no gradient from detached z_j).
        """
        device = continuous_emb.device

        i_full, j_full = Omega_full
        i_partial, j_partial = Omega_partial

        all_i = torch.cat([i_full, i_partial], dim=0)  # (M,)
        all_j = torch.cat([j_full, j_partial], dim=0)  # (M,)
        M_total = all_i.numel()

        if M_total == 0:
            return torch.tensor(0.0, device=device), n_full, n_partial

        # ── Step A: 获取每对 embedding ─────────────────────────────────
        z_i = continuous_emb[all_i]            # (M, D) online side
        if target_embeddings is not None:
            z_j = target_embeddings[all_j]      # (M, D) target side (frozen, no grad)
        else:
            z_j = continuous_emb[all_j]         # single-tower fallback

        # ── Step B: 计算 alpha_eff (梯度缩放因子) ───────────────────────
        if self.use_npr and item_ids is not None and self._get_knn_buffer(device) is not None:
            # Module A: Neighbor-Preserving Repulsion.
            # alpha_eff[i] = npr_alpha_min if j ∈ kNN_G0(i) else 1.0.
            knn_buffer = self._get_knn_buffer(device)
            alpha_mask = build_npr_mask(knn_buffer, item_ids, self.npr_alpha_min)
            # Map (B,B) mask → per-pair alpha_eff (M,)
            alpha_eff = alpha_mask[all_i, all_j]
        else:
            # Original: exposure-aware asymmetric repulsion.
            exp_i = exposure_times[all_i].float()  # (M,)
            exp_j = exposure_times[all_j].float()  # (M,)
            i_is_tail = exp_i < exp_j              # (M,) bool — i is tail relative to j

            # α_raw = min(exp_i, exp_j) / (exp_i + exp_j) → clamped to [α_min, 0.5]
            exp_min = torch.min(exp_i, exp_j)
            exp_sum = exp_i + exp_j
            alpha_raw = exp_min / exp_sum.clamp(min=1e-8)
            if self.asymmetric_temperature != 1.0:
                alpha_raw = 0.5 * (2.0 * alpha_raw).pow(self.asymmetric_temperature)
            alpha = torch.clamp(alpha_raw, min=self.asymmetric_alpha_min, max=0.5)

            # Effective gradient scale: 1.0 when i=hot, α when i=tail
            alpha_eff = torch.where(i_is_tail, alpha, torch.tensor(1.0, device=device))

        # ── Step C: 梯度缩放应用于 online 侧 ────────────────────────────
        # z_i_mixed = α_eff · z_i + (1-α_eff) · detach(z_i)
        z_i_mixed = (
            alpha_eff.unsqueeze(1) * z_i
            + (1.0 - alpha_eff.unsqueeze(1)) * z_i.detach()
        )

        # ── Step D: 非对称余弦距离 ────────────────────────────────────
        z_i_norm = F.normalize(z_i_mixed, p=2, dim=-1)
        z_j_norm = F.normalize(z_j, p=2, dim=-1)
        cos_sim = (z_i_norm * z_j_norm).sum(dim=-1)  # (M,)
        D_asym = 1.0 - cos_sim

        # ── Step E: 索引对应 margin 与时间权重 ────────────────────────
        if n_full > 0:
            margin_full = margin[i_full, j_full]
            w_full = w_final[i_full, j_full]
        else:
            margin_full = torch.zeros(0, device=device)
            w_full = torch.zeros(0, device=device)

        if n_partial > 0:
            margin_partial = margin[i_partial, j_partial]
            w_partial = w_final[i_partial, j_partial]
        else:
            margin_partial = torch.zeros(0, device=device)
            w_partial = torch.zeros(0, device=device)

        margin_all = torch.cat([margin_full, margin_partial], dim=0)
        w_all = torch.cat([w_full, w_partial], dim=0)

        # ── Step F: 铰链损失 ──────────────────────────────────────────
        hinge_loss = F.relu(margin_all - D_asym)

        # ── Step G: 分离 full/partial 以应用各自 λ ────────────────────
        L_full = torch.tensor(0.0, device=device)
        if n_full > 0:
            L_full = (hinge_loss[:n_full] * w_all[:n_full]).mean()

        L_partial = torch.tensor(0.0, device=device)
        if n_partial > 0:
            L_partial = (hinge_loss[n_full:] * w_all[n_full:]).mean()

        L_repulsion = self.lambda_full * L_full + self.lambda_partial * L_partial

        if self.asymmetric_lambda_scale != 1.0:
            L_repulsion = L_repulsion * self.asymmetric_lambda_scale

        if self.global_step % 50 == 0:
            dual_str = "dual" if target_embeddings is not None else "single"
            mode_str = "NPR" if self.use_npr else "ASYM"
            alpha_mean = alpha_eff.mean().item()
            alpha_min = alpha_eff.min().item()
            alpha_max = alpha_eff.max().item()
            print(
                f"[{mode_str} REPULSION] step={int(self.global_step)} | "
                f"tower={dual_str} | M={M_total} | "
                f"alpha_mean={alpha_mean:.4f} "
                f"alpha_min={alpha_min:.4f} alpha_max={alpha_max:.4f} | "
                f"D_asym_mean={D_asym.mean().item():.4f} | "
                f"L_full={L_full.item():.6f} L_partial={L_partial.item():.6f}"
            )

        return L_repulsion, n_full, n_partial

    def vcf_repulsion_loss_layerwise(
        self,
        continuous_emb: torch.Tensor,
        sid_tokens: torch.Tensor,
        item_ids: torch.Tensor,
        exposure_times: torch.Tensor,
        content_embeddings: torch.Tensor,
        positive_pair_matrix: Optional[torch.Tensor] = None,
        target_embeddings: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Direction 2: Layer-wise VCF repulsion loss.

        Only repels H=0 pairs (full collision: all 3 SID layers identical).
        For these pairs, each layer contributes its own margin/λ/α, so a
        3-layer collision naturally receives 3× the repulsion signal.

        H≥1 pairs are NOT repelled — the model already distinguishes them
        at one or more layers. Repelling them would be semantically arbitrary.
        """
        B = continuous_emb.shape[0]
        device = continuous_emb.device
        stats: dict = {}

        if B < 2:
            return torch.tensor(0.0, device=device), stats

        # ── Stage 1: CVPM mask (reuse existing) ────────────────────────
        M_cvpm = self.cvpm_mask(
            item_ids, positive_pair_matrix, content_embeddings=content_embeddings
        )

        # ── Stage 2: Per-layer collision masks ─────────────────────────
        # Each layer gets its own collision mask: M_l[i,j] = (sid_i[l]==sid_j[l]) & M_cvpm
        per_layer_masks: list[torch.Tensor] = []
        for l in range(self.n_layers):
            M_l = (sid_tokens[:, l].unsqueeze(0) == sid_tokens[:, l].unsqueeze(1))
            M_l = M_l & M_cvpm
            M_l.fill_diagonal_(False)
            per_layer_masks.append(M_l)

        # ── Hamming filter (only pairs with H ≤ layerwise_hamming_radius) ─
        R = int(self.layerwise_hamming_radius)
        if R < self.n_layers:
            H = torch.zeros(B, B, dtype=torch.long, device=device)
            for l in range(self.n_layers):
                H = H + (sid_tokens[:, l].unsqueeze(0) != sid_tokens[:, l].unsqueeze(1)).long()
            M_h_filter = (H <= R)
        else:
            # R >= n_layers → no Hamming filter
            M_h_filter = torch.ones(B, B, dtype=torch.bool, device=device)

        # ── Stability gate ────────────────────────────────────────────
        try:
            n_clusters = int(getattr(self.quantization_layer_list[0], "n_clusters", 0))
            if n_clusters > 0:
                layer0_cov = float(sid_tokens[:, 0].unique().numel()) / float(n_clusters)
            else:
                layer0_cov = 1.0
            frac_unique_rows = float(sid_tokens.unique(dim=0).size(0)) / float(B)
        except Exception:
            layer0_cov = 1.0
            frac_unique_rows = 1.0

        if (layer0_cov < self.vcf_gate_min_layer0_coverage) or (
            frac_unique_rows < self.vcf_gate_min_frac_unique_ids
        ):
            if self.global_step % 50 == 0:
                print(
                    f"[LAYERWISE STABILITY GATE] step={self.global_step} | "
                    f"layer0_cov={layer0_cov:.4f} (thresh={self.vcf_gate_min_layer0_coverage}) | "
                    f"frac_unique={frac_unique_rows:.4f} (thresh={self.vcf_gate_min_frac_unique_ids}) | "
                    f"BLOCKING repulsion"
                )
            return torch.tensor(0.0, device=device), stats

        # ── Stage 3: Cosine distance (once, dual-tower or single) ─────
        if target_embeddings is not None:
            norm_online = F.normalize(continuous_emb, p=2, dim=-1)
            norm_target = F.normalize(target_embeddings, p=2, dim=-1)
            D_base = 1.0 - torch.mm(norm_online, norm_target.t())
        else:
            norm_emb = F.normalize(continuous_emb, p=2, dim=-1)
            D_base = 1.0 - torch.mm(norm_emb, norm_emb.t())

        # ── Stage 4: Per-layer repulsion ──────────────────────────────
        # Each layer uses its own collision mask, margin, λ, and NPR α.
        # The Hamming filter (M_h_filter) removes pairs with H > R.
        # For a given pair: only layers where that pair actually collides
        # contribute repulsion, naturally encoding collision severity.
        margins = [self.m_rep_L0, self.m_rep_L1, self.m_rep_L2]
        lambdas = [self.lambda_rep_L0, self.lambda_rep_L1, self.lambda_rep_L2]
        alphas = [self.npr_alpha_L0, self.npr_alpha_L1, self.npr_alpha_L2]

        L_total = torch.tensor(0.0, device=device)
        total_pairs = 0
        w = torch.ones(B, B, device=device)

        for l in range(self.n_layers):
            # Intersect per-layer mask with Hamming filter
            M_l = per_layer_masks[l] & M_h_filter
            i_idx, j_idx = torch.where(M_l)
            n_pairs = i_idx.numel()
            stats[f"n_L{l}"] = int(n_pairs)

            if n_pairs == 0:
                continue

            # ── Pair subsampling per layer ─────────────────────────────
            K = int(self.vcf_max_pairs_per_sample)
            if K > 0 and n_pairs > B * K:
                keep = []
                for b in range(B):
                    mask_b = (i_idx == b)
                    if not mask_b.any():
                        continue
                    cand = torch.where(mask_b)[0]
                    if cand.numel() <= K:
                        keep.append(cand)
                    else:
                        perm = torch.randperm(cand.numel(), device=device)[:K]
                        keep.append(cand[perm])
                if len(keep) == 0:
                    continue
                keep = torch.cat(keep, dim=0)
                i_idx = i_idx[keep]
                j_idx = j_idx[keep]
                n_pairs = i_idx.numel()
                stats[f"n_L{l}"] = int(n_pairs)

            # ── Per-layer NPR gradient modulation ──────────────────────
            if self.use_npr and self._get_knn_buffer(device) is not None:
                knn_buf = self._get_knn_buffer(device)
                alpha_mask = build_npr_mask(knn_buf, item_ids, npr_alpha_min=alphas[l])
                alpha_eff = alpha_mask[i_idx, j_idx]
            else:
                alpha_eff = torch.ones(n_pairs, device=device)

            # ── Gradient modulation: z_i_mixed = α·z_i + (1-α)·sg[z_i] ─
            z_i = continuous_emb[i_idx]
            if target_embeddings is not None:
                z_j = target_embeddings[j_idx]
            else:
                z_j = continuous_emb[j_idx]

            z_i_mixed = (
                alpha_eff.unsqueeze(1) * z_i
                + (1.0 - alpha_eff.unsqueeze(1)) * z_i.detach()
            )

            # ── Asymmetric cosine distance ─────────────────────────────
            z_i_norm = F.normalize(z_i_mixed, p=2, dim=-1)
            z_j_norm = F.normalize(z_j, p=2, dim=-1)
            cos_sim = (z_i_norm * z_j_norm).sum(dim=-1)
            D_asym = 1.0 - cos_sim

            # ── Hinge loss with layer-specific margin ──────────────────
            w_pairs = w[i_idx, j_idx]
            hinge = F.relu(margins[l] - D_asym)
            L_l = (hinge * w_pairs).sum() / max(n_pairs, 1)
            L_total = L_total + lambdas[l] * L_l
            total_pairs += n_pairs

            stats[f"L_L{l}"] = float(L_l.item())
            stats[f"alpha_L{l}_mean"] = float(alpha_eff.mean().item())

        # ── Optional clipping ──────────────────────────────────────────
        if self.vcf_repulsion_clip and self.vcf_repulsion_clip > 0:
            L_total = torch.clamp(L_total, max=float(self.vcf_repulsion_clip))

        # ── Periodic logging ───────────────────────────────────────────
        if self.global_step % 50 == 0:
            parts = [f"R={R} total_pairs={total_pairs}"]
            for l in range(self.n_layers):
                n = stats.get(f"n_L{l}", 0)
                alpha = stats.get(f"alpha_L{l}_mean", 1.0)
                L_val = stats.get(f"L_L{l}", 0.0)
                parts.append(f"L{l}: n={n} α={alpha:.3f} L={L_val:.4f}")
            print(
                f"[LAYERWISE VCF] step={int(self.global_step)} | "
                + " | ".join(parts)
                + f" | total_L={L_total.item():.6f}"
            )

        return L_total, stats

    def vcr_td_repulsion_loss(
        self,
        quantized_embeddings: torch.Tensor,
        item_ids: torch.Tensor,
        exposure_times: torch.Tensor,
        content_embeddings: torch.Tensor,
        positive_pair_matrix: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        VCR-TD核心排斥损失
        L_rep = (1/|C|) * Σ w(t_ij) * max(0, m(i,j) - d(z_i, z_j))
        """
        batch_size = quantized_embeddings.shape[0]
        # 防御性代码：处理 batch_size < 2 的边缘情况
        if batch_size < 2:
            return torch.tensor(0.0, device=quantized_embeddings.device)

        # 确保输入数据都在同一设备上
        device = quantized_embeddings.device
        item_ids = item_ids.to(device)
        exposure_times = exposure_times.to(device)
        content_embeddings = content_embeddings.to(device)

        # 1. 计算 CVPM Mask
        if self.use_cvpm:
            # cvpm_mask 处理了 same-item 逻辑 (天然包含了排除自身对角线)
            cvpm = self.cvpm_mask(item_ids, positive_pair_matrix=positive_pair_matrix)
            valid_mask = cvpm.float()
        else:
            # 降级方案：只排除对角线 (自身排斥)
            valid_mask = 1.0 - torch.eye(batch_size, device=device)
            
        if mask is not None:
            valid_mask = valid_mask * mask.to(device)

        # 2. 计算时间衰减权重矩阵
        # 曝光时间计算组合：使用最大值 w(max(t_i, t_j))
        if self.use_time_decay:
            exp_times_i = exposure_times.unsqueeze(1).expand(batch_size, batch_size)
            exp_times_j = exposure_times.unsqueeze(0).expand(batch_size, batch_size)
            # t_ij can be defined as max(t_i, t_j) to reflect the matureness of the collision
            t_ij = torch.max(exp_times_i, exp_times_j)
            w_t = self.compute_time_decay_weight(t_ij, self.alpha)
        else:
            w_t = torch.ones(batch_size, batch_size, device=device)

        # 3. 计算动态边界矩阵
        if self.use_dynamic_margin:
            margin = self.compute_dynamic_margin(content_embeddings, self.m0)
        else:
            margin = torch.full((batch_size, batch_size), self.m0, device=device)

        # 4. 计算量化嵌入距离矩阵
        distance = torch.cdist(quantized_embeddings, quantized_embeddings, p=2)

        # 5. 计算排斥损失并应用 mask 与权重
        repulsion = torch.relu(margin - distance)
        repulsion = repulsion * valid_mask * w_t

        # 使用有效冲突对的数量进行归一化，使用 clamp(min=1) 防止除零错误
        valid_count = valid_mask.sum().clamp(min=1.0)
        
        return repulsion.sum() / valid_count

    def quasid_infonce_loss(
        self,
        z: torch.Tensor,
        positive_pair_matrix: torch.Tensor,
        item_ids: torch.Tensor,
        target_embeddings: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """QuaSID InfoNCE contrastive loss — balancing force for HaMR repulsion.

        QuaSID uses cross-product S = e_t @ e_p.T / τ between triggers and targets
        (different items from co-occurrence pairs). When target_embeddings (detached)
        is provided, we use S = z_online @ z_target.T / τ, so only the anchor side
        gets gradient. This breaks the self-reinforcing double-pull cycle that would
        otherwise collapse the encoder representation space.

        Formula (QuaSID Eq 8-9, adapted to multi-positive in-batch):
          S_{i,j} = z_online_i^T z_target_j / τ
          L_i = -log( Σ_{j∈P_i} exp(S_{i,j}) / Σ_{j: M_{i,j}=1} exp(S_{i,j}) )
          L_cl = λ_cl · mean(L_i over anchors with |P_i| > 0)

        where P_i = co-occurring positives (from positive_pair_matrix),
              M_{i,j}=0 when item_ids[i]==item_ids[j] (same-ID false-negative mask).
        """
        B = z.shape[0]
        if B < 2:
            return torch.tensor(0.0, device=z.device, dtype=torch.float32)

        z_online_norm = F.normalize(z, dim=1)
        if target_embeddings is not None:
            z_target_norm = F.normalize(target_embeddings, dim=1)
            S = torch.mm(z_online_norm, z_target_norm.t()) / self.quasid_cl_tau
        else:
            S = torch.mm(z_online_norm, z_online_norm.t()) / self.quasid_cl_tau

        # QuaSID same-ID false-negative mask: exclude self and duplicate item IDs
        same_id = (item_ids.unsqueeze(0) == item_ids.unsqueeze(1))
        M = (~same_id).float()  # (B, B)

        exp_S = torch.exp(S) * M

        loss = torch.tensor(0.0, device=z.device, dtype=torch.float32)
        num_valid = 0
        for i in range(B):
            pos_mask = positive_pair_matrix[i] > 0.5
            if not pos_mask.any():
                continue
            pos_sum = exp_S[i, pos_mask].sum()
            neg_sum = exp_S[i].sum()  # M already masks self + same-ID
            denom = pos_sum + neg_sum + 1e-8
            loss += -torch.log(pos_sum / denom)
            num_valid += 1

        if num_valid == 0:
            return torch.tensor(0.0, device=z.device, dtype=torch.float32)
        return loss / float(num_valid)

    def time_aware_collaborative_contrastive_loss(self, z: torch.Tensor,
                                                  positive_pair_matrix: Optional[torch.Tensor] = None,
                                                  exposure_times: torch.Tensor = None,
                                                  target_embeddings: Optional[torch.Tensor] = None) -> torch.Tensor:
        """[DEPRECATED] Time-Aware Collaborative Contrastive Loss (TCCL)

        Replaced by quasid_infonce_loss(). Kept for backward compatibility
        with old checkpoints that may reference this method symbolically.
        """
        if not self.use_tcl:
            return torch.tensor(0.0, device=z.device, dtype=torch.float32)

        if positive_pair_matrix is None:
            return torch.tensor(0.0, device=z.device, dtype=torch.float32)

        B = z.shape[0]
        if self.global_step % 50 == 0:
            print(f"[TCCL DEBUG] TCCL activated | batch_size={B} | positive_pair_matrix shape={positive_pair_matrix.shape} | sum={positive_pair_matrix.sum().item()}")

        z_norm = F.normalize(z, dim=1)
        if target_embeddings is not None:
            z_target_norm = F.normalize(target_embeddings, dim=1)
            sim = torch.mm(z_norm, z_target_norm.T)
        else:
            sim = torch.mm(z_norm, z_norm.T)

        # 时间感知吸引力权重（论文定义）：w_cl(t_i) = 1 - exp(-alpha_cl * t_i)
        # 只依赖 item 自身成熟度，不使用 |t_i - t_j|
        if exposure_times is not None:
            t_i = exposure_times.to(device=sim.device, dtype=torch.float32).clamp(min=0.0)
            w_i = (1.0 - torch.exp(-self.alpha_cl * t_i)).clamp(0.0, 1.0)  # (B,)
        else:
            w_i = torch.ones((B,), device=sim.device, dtype=torch.float32)

        # InfoNCE loss
        loss = 0.0
        num_pos_pairs = 0
        num_pos_samples = 0
        step_has_any_pos = False
        for i in range(B):
            pos_mask = positive_pair_matrix[i] > 0.5
            pos_count = pos_mask.sum().item()
            num_pos_pairs += pos_count

            if pos_count == 0:
                continue
            step_has_any_pos = True
            num_pos_samples += 1

            pos = sim[i, pos_mask]
            # 修复：排除自身，避免自对比
            self_mask = torch.arange(B, device=sim.device) == i
            neg_mask = (~pos_mask) & (~self_mask)
            neg = sim[i, neg_mask]

            if len(neg) == 0:
                if self.global_step % 50 == 0:
                    print(f"[TCCL DEBUG] Batch {i} has no negative samples")
                continue

            numerator = torch.sum(torch.exp(pos / self.cl_tau))
            denominator = numerator + torch.sum(torch.exp(neg / self.cl_tau))
            sample_loss = -torch.log(numerator / (denominator + 1e-8)) * w_i[i]
            loss += sample_loss

        final_loss = loss / B if B > 0 else torch.tensor(0.0, device=z.device)
        if self.global_step % 50 == 0:
            print(f"[TCCL DEBUG] TCCL final loss = {final_loss.item():.6f} | num_pos_pairs={num_pos_pairs} | avg_pos_per_sample={num_pos_pairs/B if B>0 else 0:.2f}")

        # --- Step-level coverage + density accounting (during fitting only) ---
        if getattr(self, "trainer", None) is not None and self.trainer.state.fn == TrainerFn.FITTING:
            self._tccl_steps_seen += 1
            if not step_has_any_pos:
                self._tccl_steps_no_pos += 1

            pos_sample_frac = float(num_pos_samples) / float(B) if B > 0 else 0.0
            avg_pos_per_sample = float(num_pos_pairs) / float(B) if B > 0 else 0.0
            self._tccl_steps_pos_sample_frac_sum += pos_sample_frac
            self._tccl_steps_avg_pos_per_sample_sum += avg_pos_per_sample

            if self.global_step % 50 == 0:
                ratio = self._tccl_steps_no_pos / max(1, self._tccl_steps_seen)
                mean_pos_sample_frac = self._tccl_steps_pos_sample_frac_sum / max(1, self._tccl_steps_seen)
                mean_avg_pos_per_sample = self._tccl_steps_avg_pos_per_sample_sum / max(1, self._tccl_steps_seen)
                print(
                    f"[TCCL COVERAGE] step={int(self.global_step)} "
                    f"no_pos_steps={self._tccl_steps_no_pos}/{self._tccl_steps_seen} "
                    f"ratio={ratio:.4f} "
                    f"pos_sample_frac={pos_sample_frac:.4f} (mean={mean_pos_sample_frac:.4f}) "
                    f"avg_pos_per_sample={avg_pos_per_sample:.2f} (mean={mean_avg_pos_per_sample:.2f})"
                )

        return final_loss

    def forward(
        self, embeddings: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through the quantization layers.

        Args:
            embeddings: The input embeddings to quantize.
                Shape (batch_size, n_features)
        Returns:
            cluster_ids: The cluster ids assigned to the input items.
            all_residuals: The residuals at each layer, unless self.track_residuals is
                False, in which case this is None.
            quantized_embeddings: The full quantized embeddings after passing through
                all quantization layers. These are the sum of the quantized embeddings
                from each layer.
                  Shape (batch_size, n_features).
            quantization_loss: The total quantization loss value, summed across layers.
        """
        cluster_ids = []
        current_residuals = embeddings
        all_residuals = [] if self.track_residuals else None
        quantized_embeddings = torch.zeros_like(embeddings)
        quantization_loss = torch.tensor(0.0).to(self.device)

        for idx, layer in enumerate(self.quantization_layer_list):
            if self.normalize_residuals:
                current_residuals = nn.functional.normalize(
                    current_residuals, dim=-1
                )  # normalize along the feature dimension

            # Determine whether to train the current layer
            train_layer = False
            if self.trainer.state.fn == TrainerFn.FITTING:
                # If we are training layer-wise, we only train the current layer.
                if self.train_layer_wise:
                    train_layer = idx == self.current_layer
                # If we are training all layers simultaneously, there are multiple cases
                else:
                    # If the current layer is already initialized, but not all layers
                    # are initialized, we do not train the current layer because the
                    # initialization of subsequent layers could require a special
                    # optimization step that should not be applied to
                    # already-initialized layers.
                    if (
                        self.quantization_layer_list[idx].is_initialized
                        and not self.quantization_layer_list[-1].is_initialized
                    ):
                        train_layer = False
                    # Otherwise, we always train the first layer, and train subsequent
                    # layers as long as the previous layer produced valid quantized
                    # embeddings, meaning it has been initialized or is currently in
                    # its initialization step.
                    elif idx == 0:
                        train_layer = True
                    elif (
                        self.quantization_layer_list[idx - 1].is_initialized
                        or self.quantization_layer_list[idx - 1].is_initial_step
                    ):
                        train_layer = True

            if train_layer:
                # We call model step inside forward because we need to get the
                # quantization layer's loss, which is computed in the model step
                layer_ids, layer_embeddings, layer_loss = layer.model_step(
                    current_residuals
                )
                quantization_loss += layer_loss
            else:
                layer_ids, layer_embeddings = layer.predict_step(current_residuals)

            cluster_ids.append(layer_ids)  # batch_size
            quantized_embeddings = quantized_embeddings + layer_embeddings
            current_residuals = current_residuals - layer_embeddings
            if self.track_residuals:
                all_residuals.append(current_residuals)

        cluster_ids = torch.stack(cluster_ids, dim=-1)  # batch_size x n_layers
        all_residuals = (
            torch.stack(all_residuals, dim=-1) if self.track_residuals else None
        )

        return cluster_ids, all_residuals, quantized_embeddings, quantization_loss

    def model_step(
        self, model_input: ItemData
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        Optional[torch.Tensor],
        torch.Tensor,
        torch.Tensor,
        Optional[torch.Tensor],
    ]:
        """
        Perform a forward pass and compute the loss for a single batch.

        Args:
            model_input: ItemData consisting of the batch of input features.

        Returns:
            cluster_ids: The cluster ids assigned to the input items.
                    Shape (batch_size, n_layers)
            all_residuals: The residuals at each layer, unless self.track_residuals is
                False, in which case this is None.
                    Shape (batch_size, n_features, n_layers)
            quantized_embeddings: The quantized embeddings for TCCL.
            quantization_loss: The cumulative loss from the quantization layers.
            reconstruction_loss: The reconstruction loss.
            repulsion_loss: The VCR-TD repulsion loss.
            positive_pair_matrix: The positive pair matrix for TCCL (optional).
            exposure_times: The exposure times for TCCL and VCR-TD.
            encoded_embeddings: Encoder output before quantization (for TCCL stability).
            target_embeddings: Detached target embeddings (None if single-tower).
        """
        input_embeddings = model_input.transformed_features["input_embedding"].to(
            self.device
        )
        normalized_input_embeddings = self.normalization_layer(input_embeddings)
        encoded_embeddings = self.encoder(normalized_input_embeddings)

        # Initialize online k-NN buffers on first forward pass
        if (
            self.use_online_knn
            and self._online_knn_ema is None
            and self.trainer.state.fn != TrainerFn.PREDICTING
        ):
            self._init_online_knn_buffers(
                encoded_embeddings.device, encoded_embeddings.shape[-1]
            )

        # QuaSID-style dual-tower: stop_gradient on the same encoder output.
        # Both sides share the same embedding space (no EMA lag), while
        # cross-distance D = 1 - cos(z_online_i, z_target_j) only gives
        # gradient through the online side, breaking the mirror-hall loop.
        target_embeddings: Optional[torch.Tensor] = None
        if self.use_dual_tower:
            target_embeddings = encoded_embeddings.detach()

        (
            cluster_ids,
            all_residuals,
            quantized_embeddings,
            quantization_loss,
        ) = self.forward(encoded_embeddings)

        if (
            self.trainer.state.fn != TrainerFn.PREDICTING
            and self.reconstruction_loss_function is not None
            and self.quantization_layer_list[-1].is_initialized
        ):
            # Compute the reconstruction loss
            reconstructed_embeddings = self.decoder(quantized_embeddings)
            reconstruction_loss = self.reconstruction_loss_function(
                reconstructed_embeddings, normalized_input_embeddings
            )
        else:
            reconstruction_loss = torch.tensor(0.0).to(self.device)

        repulsion_loss = torch.tensor(0.0).to(self.device)
        positive_pair_matrix = None
        exposure_times = torch.zeros(cluster_ids.shape[0], device=self.device, dtype=torch.float32)
        
        condition = (self.use_vcr_td or self.use_tcl or self.use_vcf) and self.trainer.state.fn != TrainerFn.PREDICTING

        if condition:
            # Gather item_ids and compute exposure times
            item_ids = model_input.item_ids
            # 兼容 list, tuple 或 tensor 的情况，并转为 torch.long 以确保后续比较和索引安全
            if not isinstance(item_ids, torch.Tensor):
                item_ids = torch.tensor(item_ids, dtype=torch.long, device=self.device)
            else:
                item_ids = item_ids.to(dtype=torch.long, device=self.device)

            if self._exposure_counts is not None:
                # clamp item_ids to avoid out of bounds
                # 确保exposure_counts在正确的设备上
                exposure_counts = self._exposure_counts.to(self.device)
                valid_ids = torch.clamp(item_ids, 0, exposure_counts.size(0) - 1)
                # === Historical Time Masking（防未来信息泄漏）===
                # Cap the global exposure proxy by "history seen so far".
                # This makes t only reflect information available up to current training progress.
                exposure_times_full = exposure_counts[valid_ids].to(dtype=torch.float32)
                hist_cap = torch.tensor(
                    float(self.global_historical_step),
                    device=self.device,
                    dtype=torch.float32,
                ).clamp(min=1.0)
                exposure_times = torch.minimum(exposure_times_full, hist_cap)
            else:
                exposure_times = torch.zeros_like(item_ids, dtype=torch.float32, device=self.device)
            
            # === Get positive_pair_matrix for QuaSID InfoNCE / CVPM ===
            if self.use_tcl:
                positive_pair_matrix = None
                if hasattr(model_input, 'positive_pair_matrix') and model_input.positive_pair_matrix is not None:
                    positive_pair_matrix = model_input.positive_pair_matrix
                elif hasattr(model_input, 'collab_positive_mask') and model_input.collab_positive_mask is not None:
                    positive_pair_matrix = model_input.collab_positive_mask
                else:
                    positive_pair_matrix = None

                if positive_pair_matrix is not None:
                    positive_pair_matrix = positive_pair_matrix.to(self.device)
            
            if self.use_vcr_td:
                repulsion_loss = self.vcr_td_repulsion_loss(
                    quantized_embeddings=quantized_embeddings,
                    item_ids=item_ids,
                    exposure_times=exposure_times,
                    content_embeddings=input_embeddings,
                    positive_pair_matrix=positive_pair_matrix,
                    mask=None
                )
            
            # === VCF (VCR-TD-QuaSID Fusion) 排斥损失 ===
            if self.use_vcf:
                if self.use_layerwise_repulsion:
                    # Direction 2: per-layer collision masks with layer-specific
                    # margins, λ weights, and NPR α values.
                    vcf_loss, vcf_stats = self.vcf_repulsion_loss_layerwise(
                        continuous_emb=encoded_embeddings,
                        sid_tokens=cluster_ids,
                        item_ids=item_ids,
                        exposure_times=exposure_times,
                        content_embeddings=input_embeddings,
                        positive_pair_matrix=positive_pair_matrix,
                        target_embeddings=target_embeddings,
                    )
                    n_full = sum(vcf_stats.get(f"n_L{l}", 0) for l in range(self.n_layers))
                    n_partial = 0  # H≥1 not repelled in layerwise mode
                else:
                    vcf_loss, n_full, n_partial = self.vcf_repulsion_loss(
                        continuous_emb=encoded_embeddings,
                        sid_tokens=cluster_ids,
                        item_ids=item_ids,
                        exposure_times=exposure_times,
                        content_embeddings=input_embeddings,
                        positive_pair_matrix=positive_pair_matrix,
                        target_embeddings=target_embeddings,
                    )

                # Two-phase warmup (prevents gradient shock at warmup boundary):
                #   Phase 1 (step < warmup):   Hard gate — vcf_loss = 0.
                #     Zero gradient lets VQ codebook stabilize without interference.
                #   Phase 2 (warmup ≤ step < warmup+soft_ramp): Linear ramp 0→1.
                #     Smoothly introduces repulsion, avoiding the discontinuous jump
                #     that causes catastrophic VQ bifurcation (centroid death).
                #   After ramp:                 Full repulsion (original behavior).
                if self.training:
                    step = int(self.global_step)
                    warmup = int(self.repulsion_warmup_steps)
                    if step < warmup:
                        vcf_loss = torch.tensor(0.0, device=vcf_loss.device)
                    elif self.repulsion_soft_ramp_steps > 0 and step < warmup + int(self.repulsion_soft_ramp_steps):
                        ramp_progress = (step - warmup) / max(int(self.repulsion_soft_ramp_steps), 1)
                        vcf_loss = vcf_loss * ramp_progress

                repulsion_loss = vcf_loss

                # 日志记录（降低频率）
                if self.global_step % 50 == 0:
                    if self.use_layerwise_repulsion:
                        print(f"[VCF LAYERWISE] step={self.global_step} | total_pairs={n_full} | vcf_loss={vcf_loss.item():.6f}")
                    else:
                        print(f"[VCF DEBUG] step={self.global_step} | n_full={n_full} | n_partial={n_partial} | vcf_loss={vcf_loss.item():.6f}")

        return (
            cluster_ids,
            all_residuals,
            quantized_embeddings,
            quantization_loss,
            reconstruction_loss,
            repulsion_loss,
            positive_pair_matrix,
            exposure_times,
            encoded_embeddings,
            target_embeddings,
        )

    # ─── Online k-NN helpers ─────────────────────────────────────────────────
    def _init_online_knn_buffers(self, device: torch.device, encoder_dim: int) -> None:
        """Allocate EMA and tracking buffers for online k-NN."""
        if self._online_knn_num_items is None or self._online_knn_num_items <= 0:
            raise ValueError(
                "online_knn_num_items must be set. Provide exposure_counts_path "
                "or set use_online_knn=False."
            )
        N = self._online_knn_num_items
        self._online_knn_ema = torch.zeros(N, encoder_dim, device="cpu")
        self._online_knn_items_seen = torch.zeros(N, dtype=torch.bool)
        self.register_buffer(
            "online_knn_indices",
            torch.zeros(N, self.online_knn_k, dtype=torch.long),
            persistent=True,
        )

    @torch.no_grad()
    def _update_online_knn_ema(
        self, encoded: torch.Tensor, item_ids: torch.Tensor
    ) -> None:
        """Update EMA buffer with encoder outputs from the current batch."""
        enc_cpu = encoded.detach().cpu()
        ids_cpu = item_ids.long().cpu()
        momentum = self.online_knn_ema_momentum

        for i_local, i_global in enumerate(ids_cpu):
            ig = int(i_global)
            if ig >= len(self._online_knn_items_seen):
                continue
            if not self._online_knn_items_seen[ig]:
                self._online_knn_ema[ig] = enc_cpu[i_local]
                self._online_knn_items_seen[ig] = True
            else:
                self._online_knn_ema[ig] = (
                    momentum * self._online_knn_ema[ig] + (1.0 - momentum) * enc_cpu[i_local]
                )

    @torch.no_grad()
    def _maybe_update_online_knn(self) -> None:
        """Recompute k-NN from EMA buffer every online_knn_update_interval steps."""
        if self._online_knn_ema is None:
            return
        step = int(self.global_step)
        if step % self.online_knn_update_interval != 0:
            return
        # Guard against multiple calls at the same step (Lightning can call
        # training_step several times before incrementing global_step).
        if step == getattr(self, "_online_knn_last_update_step", None):
            return
        if not self._online_knn_items_seen.any():
            return
        self._online_knn_last_update_step = step

        import torch.nn.functional as F
        ema = self._online_knn_ema.float()
        # Only use items that have been seen at least once
        ema_norm = F.normalize(ema, dim=-1)
        sim = torch.mm(ema_norm, ema_norm.t())
        sim.fill_diagonal_(-float("inf"))
        _, knn = torch.topk(sim, self.online_knn_k, dim=-1)
        self.online_knn_indices.copy_(knn.long())

        seen_frac = self._online_knn_items_seen.float().mean().item()
        print(
            f"[Online k-NN] step={step} | updated k-NN | items_seen={seen_frac*100:.1f}%"
        )

    def _get_knn_buffer(self, device: torch.device) -> torch.Tensor | None:
        """Return the active k-NN buffer (online if enabled, else G0 precomputed)."""
        if self.use_online_knn and hasattr(self, "online_knn_indices"):
            buf = self.online_knn_indices
            if buf is not None and buf.abs().sum() > 0:  # has been updated at least once
                return buf.to(device)
        return self._knn_g0_buffer

    def training_step(self, batch: Tuple[ItemData]) -> torch.Tensor:
        """
        Perform a single training step on a batch of data.

        Args:
            batch: A batch of data of ItemData type wrapped in a Tuple.

        Returns:
            loss: The loss value.
        """
        # Lightning wraps the batch in a tuple for training, we get the batch from
        # position 0. This behavior only happens for training_step.
        model_input: ItemData = batch[0]
        (
            cluster_ids,
            all_residuals,
            quantized_embeddings,
            quantization_loss,
            reconstruction_loss,
            repulsion_loss,
            positive_pair_matrix,
            exposure_times,
            encoded_embeddings,
            target_embeddings,
        ) = self.model_step(model_input)

        # === Online k-NN: update EMA and periodically recompute k-NN graph ===
        if self.use_online_knn and self._online_knn_ema is not None:
            raw_ids = model_input.item_ids
            if not isinstance(raw_ids, torch.Tensor):
                raw_ids = torch.tensor(raw_ids, dtype=torch.long, device=self.device)
            else:
                raw_ids = raw_ids.to(dtype=torch.long, device=self.device)
            self._update_online_knn_ema(encoded_embeddings, raw_ids)
            self._maybe_update_online_knn()

        # === Codebook reset: replace dead centroids with random encoder outputs ===
        # Standard VQ-VAE fix (van den Oord 2017, Razavi 2019):
        # Centroids that receive zero assignments across many batches become
        # permanently stale. Resetting them to live encoder outputs brings them
        # back into competition and prevents the runaway feedback where the
        # encoder collapses to an ever-shrinking subset of the codebook.
        if self.training and self.codebook_reset_interval > 0:
            step = int(self.global_step)
            if step % self.codebook_reset_interval == 0 and step > 0:
                with torch.no_grad():
                    for layer_idx in range(self.n_layers):
                        centroids = self.quantization_layer_list[layer_idx].get_centroids()
                        n_clusters = centroids.shape[0]
                        layer_ids = cluster_ids[:, layer_idx]
                        batch_counts = torch.bincount(layer_ids, minlength=n_clusters).float()

                        # Init or update EMA
                        if layer_idx >= len(self._codebook_assignment_ema):
                            self._codebook_assignment_ema.append(batch_counts)
                        else:
                            ema = self._codebook_assignment_ema[layer_idx]
                            ema = (self.codebook_reset_decay * ema +
                                   (1.0 - self.codebook_reset_decay) * batch_counts)
                            self._codebook_assignment_ema[layer_idx] = ema

                        # Find dead centroids: zero EMA assignment count
                        dead_mask = self._codebook_assignment_ema[layer_idx] < 0.5
                        n_dead = dead_mask.sum().item()
                        if n_dead > 0 and n_dead < n_clusters:
                            # Pick random encoder outputs as replacement seeds
                            idxs = torch.randint(0, encoded_embeddings.shape[0],
                                                 (n_dead,), device=centroids.device)
                            # Add small noise so they don't all collapse to the same point
                            noise = torch.randn(n_dead, centroids.shape[1],
                                               device=centroids.device) * 0.01
                            replacement = encoded_embeddings.detach()[idxs] + noise
                            centroids.data[dead_mask] = replacement
                            # Reset EMA for revived centroids
                            self._codebook_assignment_ema[layer_idx][dead_mask] = 1.0

        # === 双向时变对比（排斥→吸引平滑切换）===
        # NOTE: effective_lambda_rep is only used for VCR-TD path (use_vcf=false).
        # VCF path handles its own weighting via lambda_full/lambda_partial internally.
        effective_lambda_rep = self.lambda_rep  # simplified; maturity gating removed

        # VCF 内部已经使用 lambda_full/partial 进行加权，不需要再用 effective_lambda_rep 缩放
        # VCR-TD 需要使用 effective_lambda_rep 缩放
        if self.use_vcf:
            repulsion_weight = 1.0  # VCF 内部已处理权重
        else:
            repulsion_weight = effective_lambda_rep

        loss = (
            self.quantization_loss_weight * quantization_loss
            + self.reconstruction_loss_weight * reconstruction_loss
            + repulsion_weight * repulsion_loss
        )
        
        # === QuaSID InfoNCE 对比损失（平衡力）===
        # v3: CL activates at the SAME step as VCF (repulsion_warmup_steps), with a
        # λ ramp that starts at 10% and linearly increases to 100% over cl_ramp_steps.
        # This gives VCF time to establish repulsion before CL's attraction grows strong.
        # τ=0.5 aligns gradient scales: exp(cos/τ) ≈ 5, on par with VCF hinge ≈ 0.03.
        if self.use_tcl and float(self.lambda_cl) > 0.0 and positive_pair_matrix is not None:
            step = int(self.global_step)
            warmup = int(self.repulsion_warmup_steps)

            if step >= warmup:
                # λ ramp: 0.0001 → lambda_cl over cl_ramp_steps (default 1000 steps)
                ramp_progress = min(1.0, (step - warmup) / max(int(self.cl_ramp_steps), 1))
                eff_lambda = self.lambda_cl * (0.1 + 0.9 * ramp_progress)

                # Resolve item_ids for same-ID false-negative masking
                raw_ids = model_input.item_ids
                if not isinstance(raw_ids, torch.Tensor):
                    raw_ids = torch.tensor(raw_ids, dtype=torch.long, device=self.device)
                else:
                    raw_ids = raw_ids.to(dtype=torch.long, device=self.device)

                l_cl = self.quasid_infonce_loss(
                    encoded_embeddings,
                    positive_pair_matrix=positive_pair_matrix,
                    item_ids=raw_ids,
                    target_embeddings=target_embeddings,
                )
                loss = loss + eff_lambda * l_cl
                self.train_cl_loss.update(l_cl)

                if step % 50 == 0:
                    print(f"[QuaSID CL] step={step} | l_cl={l_cl.item():.6f} | "
                          f"eff_lambda={eff_lambda:.6f} | total_loss={loss.item():.6f}")

        # === Module B: k-NN Preservation Loss (KPL) ===
        # InfoNCE with G0 k-NN neighbors as positives, complementary to QuaSID CL
        # (co-occurrence positives). Same warmup as VCF, separate λ ramp.
        if self.use_kpl and self._get_knn_buffer(self.device) is not None:
            step = int(self.global_step)
            warmup = int(self.repulsion_warmup_steps)

            if step >= warmup:
                ramp_progress = min(1.0, (step - warmup) / max(int(self.kpl_ramp_steps), 1))
                eff_lambda_kpl = self.lambda_kpl * (0.1 + 0.9 * ramp_progress)

                raw_ids = model_input.item_ids
                if not isinstance(raw_ids, torch.Tensor):
                    raw_ids = torch.tensor(raw_ids, dtype=torch.long, device=self.device)
                else:
                    raw_ids = raw_ids.to(dtype=torch.long, device=self.device)

                knn_buffer = self._get_knn_buffer(self.device)
                l_kpl = compute_kpl_loss(
                    z_online=encoded_embeddings,
                    z_target=target_embeddings if target_embeddings is not None else encoded_embeddings,
                    knn_g0_indices=knn_buffer,
                    item_ids=raw_ids,
                    tau=self.kpl_tau,
                    cluster_ids=cluster_ids,
                    collision_exclude_radius=self.kpl_collision_exclude_radius,
                )
                loss = loss + eff_lambda_kpl * l_kpl
                self.train_kpl_loss.update(l_kpl)

                if step % 50 == 0:
                    print(f"[KPL] step={step} | l_kpl={l_kpl.item():.6f} | "
                          f"eff_lambda={eff_lambda_kpl:.6f} | total_loss={loss.item():.6f}")

        # === Module C: Reciprocal Neighbor Consistency Loss (RNCL) ===
        # Gentle cosine-distance loss on reciprocal k-NN pairs — pure attraction,
        # no softmax denominator (unlike KPL's InfoNCE).  Same warmup as VCF/KPL.
        if self.use_rncl and self._get_knn_buffer(self.device) is not None:
            step = int(self.global_step)
            warmup = int(self.repulsion_warmup_steps)

            if step >= warmup:
                ramp_progress = min(1.0, (step - warmup) / max(int(self.rncl_ramp_steps), 1))
                eff_lambda_rncl = self.lambda_rncl * (0.1 + 0.9 * ramp_progress)

                raw_ids = model_input.item_ids
                if not isinstance(raw_ids, torch.Tensor):
                    raw_ids = torch.tensor(raw_ids, dtype=torch.long, device=self.device)
                else:
                    raw_ids = raw_ids.to(dtype=torch.long, device=self.device)

                knn_buf = self._get_knn_buffer(self.device)
                rncl_mask = build_reciprocal_mask(knn_buf, raw_ids)
                l_rncl = compute_rncl_loss(
                    z_online=encoded_embeddings,
                    z_target=target_embeddings if target_embeddings is not None else encoded_embeddings,
                    reciprocal_mask=rncl_mask,
                )
                loss = loss + eff_lambda_rncl * l_rncl
                self.log("train/rncl_loss", l_rncl, on_step=True, on_epoch=False, prog_bar=False)
                self.log("train/rncl_pairs", rncl_mask.sum().float(), on_step=True, on_epoch=False)

                if step % 50 == 0:
                    n_pairs = rncl_mask.sum().item()
                    print(f"[RNCL] step={step} | l_rncl={l_rncl.item():.6f} | "
                          f"pairs={n_pairs} | eff_lambda={eff_lambda_rncl:.6f}")

        # Codebook entropy regularization: normalized diversity bonus to prevent collapse.
        # entropy_norm ∈ [0, 1]: 1 = perfectly uniform codebook usage, 0 = single-code.
        if self.codebook_entropy_weight > 0.0:
            entropy_sum = torch.tensor(0.0, device=self.device)
            for layer_idx in range(self.n_layers):
                layer_ids = cluster_ids[:, layer_idx]
                n_clusters = int(self.quantization_layer_list[layer_idx].n_clusters)
                counts = torch.bincount(layer_ids, minlength=n_clusters).float()
                probs = counts / counts.sum().clamp(min=1)
                layer_entropy = -(probs * (probs + 1e-12).log()).sum()
                max_entropy = torch.tensor(float(n_clusters)).log()
                entropy_sum += layer_entropy / max_entropy.clamp(min=1e-8)
            entropy_bonus = entropy_sum / float(self.n_layers)
            loss = loss - self.codebook_entropy_weight * entropy_bonus

        self.train_loss(loss)
        self.train_quantization_loss(quantization_loss)
        self.train_reconstruction_loss(reconstruction_loss)
        self.train_repulsion_loss(repulsion_loss)
        train_dict_to_log = {
            "train/loss": self.train_loss,
            "train/quantization_loss": self.train_quantization_loss,
            "train/reconstruction_loss": self.train_reconstruction_loss,
            "train/repulsion_loss": self.train_repulsion_loss,
        }
        if self.use_tcl:
            train_dict_to_log["train/cl_loss"] = self.train_cl_loss
        if self.use_kpl:
            train_dict_to_log["train/kpl_loss"] = self.train_kpl_loss

        with torch.no_grad():
            if self.verbose and self.global_step % self.trainer.log_every_n_steps == 0:
                # Compute verbose ID statistics
                (
                    train_first_residuals_norm_ratio,
                    train_last_residuals_norm_ratio,
                    first_centroids_norm,
                    last_centroids_norm,
                    train_frac_unique_ids,
                    train_mse,
                    train_layer_coverages,
                    train_layer_id_entropies,
                ) = self._compute_output_stats(
                    cluster_ids=cluster_ids,
                    all_residuals=all_residuals,
                    input_embeddings=model_input.transformed_features[
                        "input_embedding"
                    ],
                )
                # Update the metrics
                self.train_first_residuals_norm_ratio(train_first_residuals_norm_ratio)
                self.train_last_residuals_norm_ratio(train_last_residuals_norm_ratio)
                self.first_centroids_norm(first_centroids_norm)
                self.last_centroids_norm(last_centroids_norm)
                self.train_frac_unique_ids(train_frac_unique_ids)
                self.train_mse(train_mse)
                for layer_idx in range(self.n_layers):
                    layer_frac_unique_metric = getattr(
                        self, f"train_layer_coverages_{layer_idx}"
                    )
                    layer_id_entropy_metric = getattr(
                        self, f"train_layer_id_entropy_{layer_idx}"
                    )
                    layer_frac_unique_metric(train_layer_coverages[layer_idx])
                    layer_id_entropy_metric(train_layer_id_entropies[layer_idx])

                train_dict_to_log.update(
                    {
                        "train/last_residuals_norm_ratio": self.train_last_residuals_norm_ratio,
                        "train/first_residuals_norm_ratio": self.train_first_residuals_norm_ratio,
                        "train/first_centroids_norm": self.first_centroids_norm,
                        "train/last_centroids_norm": self.last_centroids_norm,
                        "train/frac_unique_ids": self.train_frac_unique_ids,
                        "train/mse": self.train_mse,
                    }
                )
                train_dict_to_log.update(
                    {
                        f"train/layer_{layer_idx}/frac_layer_coverages": getattr(
                            self,
                            f"train_layer_coverages_{layer_idx}",
                        )
                        for layer_idx in range(self.n_layers)
                    }
                )
                train_dict_to_log.update(
                    {
                        f"train/layer_{layer_idx}/id_entropy": getattr(
                            self,
                            f"train_layer_id_entropy_{layer_idx}",
                        )
                        for layer_idx in range(self.n_layers)
                    }
                )

        self.log_dict(
            train_dict_to_log,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )

        # If a training loop function is passed, we call it with the module and the loss
        # Otherwise we use the automatic optimization provided by Lightning
        if self.training_loop_function is not None:
            if self.train_layer_wise:
                layer_to_check = self.current_layer
            else:
                layer_to_check = -1
            is_initialized = self.quantization_layer_list[layer_to_check].is_initialized

            self.training_loop_function(
                self,
                loss=loss,
                world_size=self.trainer.world_size,
                is_initialized=is_initialized,
            )

        if (
            self.train_layer_wise
            and self.global_step % self.steps_per_layer == 0
            and (
                self.quantization_layer_list[self.current_layer].is_initialized
                or self.current_layer < 0
            )
            and self.current_layer < self.n_layers - 1
        ):
            self.log_if_true(
                f"Finished training layer {self.current_layer} of {self.n_layers}",
                self.verbose,
            )
            self.current_layer += 1

        return loss

    def on_train_start(self):
        """Lightning hook that is called when training begins."""
        if hasattr(self, "train_loss"):
            self.train_loss.reset()

        # Reset TCCL diagnostics.
        self._tccl_steps_seen = 0
        self._tccl_steps_no_pos = 0
        self._tccl_steps_pos_sample_frac_sum = 0.0
        self._tccl_steps_avg_pos_per_sample_sum = 0.0

        # Reset codebook assignment EMA counters.
        self._codebook_assignment_ema = []

        self.current_layer = 0
        for layer in self.quantization_layer_list:
            layer.on_train_start()

        if self.train_layer_wise:
            total_steps = self.trainer.max_steps
            if self.reconstruction_loss_function is None:
                eff_n_layers = self.n_layers
            else:
                eff_n_layers = self.n_layers + 1
                self.current_layer = -1
            self.steps_per_layer = total_steps // eff_n_layers
            self.log_if_true(
                f"Training layers one-at-a-time, each for {self.steps_per_layer} steps."
                " Ensure that early stopping callbacks are disabled.",
                self.verbose,
            )
        else:
            self.log_if_true("Training all layers simultaneously", self.verbose)

        if self.verbose:
            self.train_first_residuals_norm_ratio.reset()
            self.train_last_residuals_norm_ratio.reset()
            self.train_frac_unique_ids.reset()
            self.first_centroids_norm.reset()
            self.last_centroids_norm.reset()
            self.train_mse.reset()
            for layer_idx in range(self.n_layers):
                layer_frac_unique_metric = getattr(
                    self, f"train_layer_coverages_{layer_idx}"
                )
                layer_id_entropy_metric = getattr(
                    self, f"train_layer_id_entropy_{layer_idx}"
                )
                layer_frac_unique_metric.reset()
                layer_id_entropy_metric.reset()

    def on_train_end(self) -> None:
        """Lightning hook that is called when training ends."""
        if self._tccl_steps_seen > 0:
            ratio = self._tccl_steps_no_pos / max(1, self._tccl_steps_seen)
            mean_pos_sample_frac = self._tccl_steps_pos_sample_frac_sum / max(1, self._tccl_steps_seen)
            mean_avg_pos_per_sample = self._tccl_steps_avg_pos_per_sample_sum / max(1, self._tccl_steps_seen)
            print(
                f"[TCCL COVERAGE][FINAL] no_pos_steps={self._tccl_steps_no_pos}/{self._tccl_steps_seen} "
                f"ratio={ratio:.4f} "
                f"mean_pos_sample_frac={mean_pos_sample_frac:.4f} "
                f"mean_avg_pos_per_sample={mean_avg_pos_per_sample:.2f}"
            )

    def _compute_output_stats(
        self,
        cluster_ids: torch.Tensor,
        all_residuals: torch.Tensor,
        input_embeddings: torch.Tensor,
    ) -> Tuple[MeanMetric, MeanMetric, MeanMetric, MeanMetric, MeanMetric]:
        """
        Compute output statistics for the model.

        Args:
            cluster_ids: The cluster ids assigned to the input items.
                    Shape (batch_size, n_layers)
            all_residuals: The residuals at each layer, unless self.track_residuals is
                False, in which case this is None.
                    Shape (batch_size, n_features, n_layers)
            input_embeddings: The cluster embeddings. These are returned for
                debugging purposes.
                    Shape (batch_size, n_features)

        Returns:
            A tuple containing:
                - first_residuals_norm_ratio: The ratio of the norm of the first
                    residuals to the norm of the input embeddings. Note that if
                    self.normalize_residuals is True, this metric is uninformative.
                - last_residuals_norm_ratio: The ratio of the norm of the last
                    residuals to the norm of the input embeddings. Note that if
                    self.normalize_residuals is True, this metric is uninformative.
                - first_centroids_norm: The norm of the centroids of the first
                    quantization layer.
                - last_centroids_norm: The norm of the centroids of the last
                    quantization layer.
                - frac_unique_ids: # distinct item ID sequences / batch_size.
                - mse: The mean squared error of the last residuals (target is 0).
                - layer_coverages: A list containing, for each quantization layer,
                    # distinct IDs / # clusters in that layer.
                - layer_id_entropies: A list containing, for each quantization layer,
                    the batch entropy of the cluster ids in that layer.
        """
        input_embedding_norm = torch.linalg.matrix_norm(input_embeddings)
        first_residuals_norm_ratio = (
            torch.linalg.matrix_norm(all_residuals[:, :, 0]) / input_embedding_norm
        )
        last_residuals_norm = torch.linalg.matrix_norm(all_residuals[:, :, -1])
        last_residuals_norm_ratio = last_residuals_norm / input_embedding_norm
        mse = last_residuals_norm**2 / all_residuals[:, :, -1].numel()

        first_centroids_norm = torch.linalg.matrix_norm(
            self.quantization_layer_list[0].get_centroids()
        )
        last_centroids_norm = torch.linalg.matrix_norm(
            self.quantization_layer_list[-1].get_centroids()
        )

        frac_unique_ids = (
            torch.unique(cluster_ids, dim=0).shape[0] / cluster_ids.shape[0]
        )

        layer_coverages = []
        layer_id_entropies = []
        for layer_idx in range(self.n_layers):
            _, cluster_counts = torch.unique(
                cluster_ids[:, layer_idx], return_counts=True
            )
            cluster_counts = (cluster_counts / cluster_ids.shape[0]).to(self.device)
            entropy = Categorical(probs=cluster_counts).entropy().to(self.device)
            layer_coverages.append(
                cluster_counts.shape[0]
                / self.quantization_layer_list[layer_idx].n_clusters
            )
            layer_id_entropies.append(entropy)

        return (
            first_residuals_norm_ratio,
            last_residuals_norm_ratio,
            first_centroids_norm,
            last_centroids_norm,
            frac_unique_ids,
            mse,
            layer_coverages,
            layer_id_entropies,
        )

    def eval_step(
        self,
        batch: ItemData,
        loss_to_aggregate: MeanMetric,
        first_residuals_norm_ratio_metric: MeanMetric,
        last_residuals_norm_ratio_metric: MeanMetric,
        frac_unique_ids_metric: MeanMetric,
        mse_metric: MeanMetric,
    ):
        """
        Perform a single evaluation step on a batch of data.

        Args:
            batch: A batch of data of ItemData type.
            loss_to_aggregate: The metric for the loss.
            first_residuals_norm_ratio_metric: The metric for the first residuals norm ratio.
            last_residuals_norm_ratio_metric: The metric for the last residuals norm ratio.
            frac_unique_ids_metric: The metric for the fraction of unique ids.
            mse_metric: The metric for the mean squared error.
        """
        (
            cluster_ids,
            all_residuals,
            _,
            quantization_loss,
            reconstruction_loss,
            repulsion_loss,
            _,
            _,
            _,
            _,
        ) = self.model_step(batch)
        # VCF 的排斥损失内部已含 lambda_full/lambda_partial 加权，不能再乘 lambda_rep
        # VCR-TD 的排斥损失是未加权的裸值，需要乘 lambda_rep
        repulsion_weight = 1.0 if self.use_vcf else self.lambda_rep
        loss = (
            self.quantization_loss_weight * quantization_loss
            + self.reconstruction_loss_weight * reconstruction_loss
            + repulsion_weight * repulsion_loss
        )
        loss_to_aggregate(loss)

        (
            first_residuals_norm_ratio,
            last_residuals_norm_ratio,
            _,
            _,
            frac_unique_ids,
            mse,
            _,
            _,
        ) = self._compute_output_stats(
            cluster_ids=cluster_ids,
            all_residuals=all_residuals,
            input_embeddings=batch.transformed_features["input_embedding"],
        )
        last_residuals_norm_ratio_metric(last_residuals_norm_ratio)
        first_residuals_norm_ratio_metric(first_residuals_norm_ratio)
        frac_unique_ids_metric(frac_unique_ids)
        mse_metric(mse)

    def validation_step(self, batch: ItemData, batch_idx: int):
        """
        Perform a single validation step on a batch of data.

        Args:
            batch: A batch of data of ItemData type.
            batch_idx: The index of the batch.
        """
        self.eval_step(
            batch,
            self.val_loss,
            self.val_first_residuals_norm_ratio,
            self.val_last_residuals_norm_ratio,
            self.val_frac_unique_ids,
            self.val_mse,
        )

        val_dict_to_log = {
            "val/loss": self.val_loss,
            "val/first_residuals_norm_ratio": self.val_first_residuals_norm_ratio,
            "val/last_residuals_norm_ratio": self.val_last_residuals_norm_ratio,
            "val/frac_unique_ids": self.val_frac_unique_ids,
            "val/mse": self.val_mse,
        }
        self.log_dict(
            val_dict_to_log,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )

    def on_validation_start(self):
        """Lightning hook that is called when validation begins."""
        self.val_loss.reset()
        self.val_first_residuals_norm_ratio.reset()
        self.val_last_residuals_norm_ratio.reset()
        self.val_frac_unique_ids.reset()
        self.val_mse.reset()

    def test_step(self, batch: ItemData, batch_idx: int) -> None:
        """
        Perform a single test step on a batch of data.

        Args:
            batch: A batch of data of ItemData type.
            batch_idx: The index of the batch.
        """
        self.eval_step(
            batch,
            self.test_loss,
            self.test_first_residuals_norm_ratio,
            self.test_last_residuals_norm_ratio,
            self.test_frac_unique_ids,
            self.test_mse,
        )

        test_dict_to_log = {
            "test/loss": self.test_loss,
            "test/first_residuals_norm_ratio": self.test_first_residuals_norm_ratio,
            "test/last_residuals_norm_ratio": self.test_last_residuals_norm_ratio,
            "test/frac_unique_ids": self.test_frac_unique_ids,
            "test/mse": self.test_mse,
        }
        self.log_dict(
            test_dict_to_log,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )

    def on_test_start(self):
        """Lightning hook that is called when testing begins."""
        self.test_loss.reset()
        self.test_first_residuals_norm_ratio.reset()
        self.test_last_residuals_norm_ratio.reset()
        self.test_frac_unique_ids.reset()
        self.test_mse.reset()

    def predict_step(self, batch: ItemData) -> OneKeyPerPredictionOutput:
        """
        Perform a single prediction step on a batch of data.

        Save the cluster ids assigned to the input items and the corresponding item ids
        in a OneKeyPerPredictionOutput object.

        Args:
            batch: A batch of data of ItemData type.

        Returns:
            model_output: A OneKeyPerPredictionOutput object containing the item
                ids as keys and the cluster ids as predictions.
        """
        cluster_ids, _, _, _, _, _, _, _, _, _ = self.model_step(batch)

        item_ids = [
            item_id.item() if isinstance(item_id, torch.Tensor) else item_id
            for item_id in batch.item_ids
        ]

        model_output = OneKeyPerPredictionOutput(
            keys=item_ids,
            predictions=cluster_ids,
            key_name="item_id",
            prediction_name="cluster_ids",
        )
        return model_output

    def configure_optimizers(self) -> Dict[str, Any]:
        """
        Configure the optimizer and learning rate scheduler.

        Returns:
            A dictionary containing the optimizer and learning rate scheduler.
        """
        if self.optimizer is not None:
            optimizer = self.optimizer(params=self.trainer.model.parameters())
            if self.scheduler is not None:
                scheduler = self.scheduler(optimizer=optimizer)
                return {
                    "optimizer": optimizer,
                    "lr_scheduler": {
                        "scheduler": scheduler,
                        "monitor": "val/loss",
                        "interval": "step",
                        "frequency": 1,
                    },
                }
            return {"optimizer": optimizer}

    def on_load_checkpoint(self, checkpoint):
        """
        Lightning hook that is called to load the model state from a checkpoint.

        Args:
            checkpoint: The checkpoint to load the model state from.
        """
        # Strip legacy _target_encoder keys from old EMA-based checkpoints.
        # Since we now use QuaSID-style same-encoder detach, the EMA copy is no
        # longer needed and its keys would cause "unexpected key" errors.
        state_dict = checkpoint.get("state_dict", {})
        target_keys = [k for k in state_dict if k.startswith("_target_encoder.")]
        for k in target_keys:
            del state_dict[k]
        if target_keys:
            print(f"[on_load_checkpoint] Stripped {len(target_keys)} legacy _target_encoder keys.")

        # Strip online_knn_indices when loading into a model without the buffer
        # (e.g. during inference where _init_online_knn_buffers is gated out).
        if "online_knn_indices" in state_dict and not hasattr(self, "online_knn_indices"):
            del state_dict["online_knn_indices"]

        self.current_layer = checkpoint["current_layer"]
        for idx, layer in enumerate(self.quantization_layer_list):
            layer.is_initialized = checkpoint["layers_initialized"][idx]
        return super().on_load_checkpoint(checkpoint)

    def on_save_checkpoint(self, checkpoint):
        """
        Lightning hook that is called to save the model state to a checkpoint.

        Args:
            checkpoint: The checkpoint to save the model state to.
        """
        checkpoint["current_layer"] = self.current_layer
        checkpoint["layers_initialized"] = [
            layer.is_initialized for layer in self.quantization_layer_list
        ]
        # We do not save the input embedding cache as this can be very large
        return super().on_save_checkpoint(checkpoint)

    def log_if_true(self, message: str, condition: bool) -> None:
        """Log a message if condition is True."""
        if condition:
            logging.info(f"Device {self.device}: {message}")
