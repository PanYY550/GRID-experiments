#!/usr/bin/env python3
"""
ECRD (Embedding Cosine-drift Representation Drift) and encoder-output utilities.

Shared utilities for DSF analysis scripts:
  - load_checkpoint:  Load an RQ-VAE model from a training checkpoint.
  - compute_encoder_outputs:  Run embeddings through the encoder to get
    the pre-quantization latent representations used for k-NN / DSF.
"""

import torch
import torch.nn as nn

EMBEDDING_PATH = "/home/pyy/GRID/embeddings/beauty/pickle/merged_predictions_tensor.pt"
EXPOSURE_PATH = "/home/pyy/GRID/src/data/amazon_data/beauty/exposure_counts.pt"


def _build_encoder_from_state_dict(state_dict: dict) -> nn.Module:
    """Infer encoder architecture from state_dict keys and build it."""
    prefix = "encoder.model."
    layers = {}
    for k, v in state_dict.items():
        if k.startswith(prefix):
            idx = int(k[len(prefix) :].split(".")[0])
            if idx not in layers:
                layers[idx] = {}
            param_name = ".".join(k[len(prefix) :].split(".")[1:])
            layers[idx][param_name] = (v.shape, v.dtype)

    # Sort indices and infer architecture
    sorted_idx = sorted(layers.keys())
    # Pattern: Linear, ReLU, Linear, Dropout, ReLU, Linear, Dropout, ReLU, Linear, Dropout
    # We know the architecture from the config: MLP with hidden=[768,256,128], out=64
    dims = []
    for idx in sorted_idx:
        if "weight" in layers[idx]:
            shape = layers[idx]["weight"][0]  # torch.Size
            out_dim, in_dim = shape[0], shape[1]
            dims.append((in_dim, out_dim))

    from src.models.components.network_blocks.mlp import MLP

    hiddens = [d[0] for d in dims[1:]]  # input_dims of subsequent layers
    out_dim = dims[-1][1]
    encoder = MLP(
        input_dim=dims[0][1],
        output_dim=out_dim,
        hidden_dim_list=hiddens,
        activation=nn.ReLU,
        bias=True,
        dropout=0.0,
    )
    # Load just the encoder state
    encoder_state = {}
    for k, v in state_dict.items():
        if k.startswith("encoder."):
            encoder_state[k[len("encoder.") :]] = v
    encoder.load_state_dict(encoder_state)
    return encoder


def _build_normalization_from_state_dict(state_dict: dict) -> nn.Module:
    """Infer normalization layer from state_dict keys and build it."""
    # Check if it's BatchNorm1d
    has_running = any("running_mean" in k for k in state_dict if "normalization" in k)
    if has_running:
        # Find the dimension
        for k, v in state_dict.items():
            if "normalization" in k and hasattr(v, "shape") and len(v.shape) == 1:
                dim = v.shape[0]
                break
        norm = nn.BatchNorm1d(dim)
    else:
        norm = nn.Identity()

    norm_state = {}
    for k, v in state_dict.items():
        if k.startswith("normalization_layer."):
            # Strip prefix; also handles Sequential wrapper ('0.' prefix)
            sub = k[len("normalization_layer."):]
            if sub.startswith("0."):
                sub = sub[2:]
            norm_state[sub] = v
    if norm_state:
        norm.load_state_dict(norm_state)
    return norm


class _EncoderWrapper(nn.Module):
    """Minimal wrapper holding normalization + encoder for compute_encoder_outputs."""

    def __init__(self, normalization_layer: nn.Module, encoder: nn.Module):
        super().__init__()
        self.normalization_layer = normalization_layer
        self.encoder = encoder


def load_checkpoint(path: str, device: str = "cpu") -> _EncoderWrapper:
    """Load encoder + normalization from a training checkpoint."""
    # Pre-import to avoid circular imports during torch.load unpickling
    import src.utils.decorators  # noqa: F401
    import src.data.loading.components.iterators  # noqa: F401
    import src.data.loading.components.interfaces  # noqa: F401

    ckpt = torch.load(path, map_location=device, weights_only=False)
    state_dict = ckpt.get("state_dict", ckpt)

    # Strip legacy _target_encoder prefix
    cleaned = {}
    for k, v in state_dict.items():
        if k.startswith("_target_encoder."):
            cleaned[k[len("_target_encoder.") :]] = v
        else:
            cleaned[k] = v

    encoder = _build_encoder_from_state_dict(cleaned)
    norm = _build_normalization_from_state_dict(cleaned)

    wrapper = _EncoderWrapper(norm, encoder)
    wrapper.to(device)
    wrapper.eval()
    return wrapper


@torch.no_grad()
def compute_encoder_outputs(
    embeddings: torch.Tensor,
    model: _EncoderWrapper,
    batch_size: int = 4096,
    device: str = "cpu",
) -> torch.Tensor:
    """Compute pre-quantization encoder outputs for all embeddings."""
    if embeddings.device.type != device:
        embeddings = embeddings.to(device)
    model.to(device)
    outputs = []
    for i in range(0, len(embeddings), batch_size):
        batch = embeddings[i : i + batch_size]
        normed = model.normalization_layer(batch)
        enc = model.encoder(normed)
        outputs.append(enc.cpu())
    return torch.cat(outputs, dim=0)
