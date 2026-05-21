from typing import Any, Dict, List

import torch
import torch.nn.functional as F
import torchmetrics
from torchmetrics.metric import Metric
from torchmetrics.utilities.distributed import gather_all_tensors

## Custom Metrics


class CustomMeanReductionMetric(torchmetrics.Metric):
    """
    Custom metric class that uses mean reduction and supports distributed training.
    """

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        self.metric_values = 0
        self.total_values = 0

    def compute(self) -> torch.Tensor:
        # Aggregates the metric accross workers and returns the final value
        metric_values_tensor = torch.tensor(self.metric_values).to(self.device)
        total_values_tensor = torch.tensor(self.total_values).to(self.device)
        # Compute final metric
        if self.total_values == 0:
            return torch.tensor(0.0, device=self.device)
        # Checks if using more than one GPU
        # If so, gather all metric values and total values from all GPUs. Else, return the current
        # worker's metric value
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            # Gather all metric values and total values from all GPUs

            metric_values_tensor_list = [
                t.unsqueeze(0) if t.dim() == 0 else t
                for t in gather_all_tensors(metric_values_tensor)
            ]
            metric_values_tensor = torch.cat(metric_values_tensor_list).sum()

            total_values_tensor_list = [
                t.unsqueeze(0) if t.dim() == 0 else t
                for t in gather_all_tensors(total_values_tensor)
            ]

            total_values_tensor = torch.cat(total_values_tensor_list).sum()

        return metric_values_tensor / total_values_tensor

    def reset(self) -> None:
        self.metric_values = 0
        self.total_values = 0

    def update(self) -> None:
        raise NotImplementedError


class CustomRetrievalMetric(CustomMeanReductionMetric):
    """
    Custom retrieval metric class to calculate ranking metrics.
    """

    def __init__(
        self,
        top_k: int,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.top_k = top_k

    def update(
        self, preds: torch.Tensor, target: torch.Tensor, indexes: torch.Tensor, **kwargs
    ) -> None:

        batch_size = int(len(indexes) / (indexes == 0).sum().item())
        preds = preds.reshape(batch_size, -1)
        target = target.reshape(batch_size, -1).int()

        metric = self._metric(preds, target)
        self.metric_values += metric.sum().item()
        self.total_values += batch_size

    def _metric(self, preds: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class NDCG(CustomRetrievalMetric):
    """
    Metric to calculate Normalized Discounted Cumulative Gain@K (NDCG@K).
    """

    def _metric(self, preds: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        topk_indices = torch.topk(preds, self.top_k)[1]
        topk_true = target.gather(1, topk_indices)

        # Compute DCG
        dcg = torch.sum(
            topk_true
            / torch.log2(
                torch.arange(2, self.top_k + 2, device=target.device).unsqueeze(0)
            ),
            dim=1,
        )

        # Compute IDCG
        ideal_indices = torch.topk(target, self.top_k)[1]
        ideal_dcg = torch.sum(
            target.gather(1, ideal_indices)
            / torch.log2(
                torch.arange(2, self.top_k + 2, device=target.device).unsqueeze(0)
            ),
            dim=1,
        )

        # Handle cases where IDCG is zero
        ndcg = dcg / torch.where(ideal_dcg == 0, torch.ones_like(ideal_dcg), ideal_dcg)
        return ndcg


class Recall(CustomRetrievalMetric):
    """
    Metric to calculate Recall@K.
    """

    def _metric(self, preds: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        topk_indices = torch.topk(preds, self.top_k)[1]
        topk_true = target.gather(1, topk_indices)

        true_positives = topk_true.sum(dim=1)
        total_relevant = target.sum(dim=1)

        recall = true_positives / total_relevant.minimum(
            torch.tensor(self.top_k, device=self.device)
        ).clamp(
            min=1
        )  # Use clamp to avoid zero
        return recall


class NDCGPlusHR(CustomRetrievalMetric):
    """
    Metric to calculate NDCG@K + HR@K (composite metric as in QuaSID paper).
    This combines Normalized Discounted Cumulative Gain and Hit Rate (Recall).
    """

    def _metric(self, preds: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        topk_indices = torch.topk(preds, self.top_k)[1]
        topk_true = target.gather(1, topk_indices)

        # Compute NDCG component
        dcg = torch.sum(
            topk_true
            / torch.log2(
                torch.arange(2, self.top_k + 2, device=target.device).unsqueeze(0)
            ),
            dim=1,
        )
        ideal_indices = torch.topk(target, self.top_k)[1]
        ideal_dcg = torch.sum(
            target.gather(1, ideal_indices)
            / torch.log2(
                torch.arange(2, self.top_k + 2, device=target.device).unsqueeze(0)
            ),
            dim=1,
        )
        ndcg = dcg / torch.where(ideal_dcg == 0, torch.ones_like(ideal_dcg), ideal_dcg)

        # Compute HR (Recall) component
        true_positives = topk_true.sum(dim=1)
        total_relevant = target.sum(dim=1)
        hr = true_positives / total_relevant.minimum(
            torch.tensor(self.top_k, device=self.device)
        ).clamp(min=1)

        # Return composite metric: NDCG + HR
        return ndcg + hr


## Evaluators

class Evaluator:
    def __init__(self, metrics: Dict[str, Metric], *args, **kwargs):
        self.metrics = metrics

    def __call__(self, *args, **kwargs):
        raise NotImplementedError

    def reset(self):
        for metric in self.metrics.values():
            metric.reset()

    def to(self, device: torch.device):
        for metric in self.metrics.values():
            metric.to(device=device)


class RetrievalEvaluator(Evaluator):
    """
    Wrapper for retrieval evaluation metrics.
    It takes model outputs and automatically calculates the retrieval metrics.
    """

    def __init__(
        self,
        metrics: Dict[str, CustomRetrievalMetric],
        top_k_list: List[int],
        should_sample_negatives_from_vocab: bool = True,
        num_negatives: int = 500,
        placeholder_token_buffer: int = 100,
    ):
        self.metrics = {
            f"{metric_name}@{top_k}": metric_object(
                top_k=top_k, sync_on_compute=False, compute_with_cache=False
            )
            for metric_name, metric_object in metrics.items()
            for top_k in top_k_list
        }
        self.should_sample_negatives_from_vocab = should_sample_negatives_from_vocab
        self.num_negatives = num_negatives
        self.placeholder_token_buffer = placeholder_token_buffer

    def __call__(
        self,
        query_embeddings: torch.Tensor,
        key_embeddings: torch.Tensor,
        labels: torch.Tensor,
    ):
        num_of_samples = query_embeddings.shape[0]
        num_of_candidates = key_embeddings.shape[0]

        if self.should_sample_negatives_from_vocab:
            inbatch_negatives = self.sample_negative_ids_from_vocab(
                num_of_samples=num_of_samples,
                num_of_candidates=num_of_candidates,
                num_negatives=self.num_negatives,
            )
            # we +1 here because we need to include the positive sample
            num_of_candidates = self.num_negatives + 1
            pos_embeddings = key_embeddings[labels]
            key_embeddings = key_embeddings[inbatch_negatives]
            # key_embeddings shape: (bsz, num_negatives+1, emb_dim)
            key_embeddings = torch.cat(
                [pos_embeddings.unsqueeze(1), key_embeddings], dim=1
            )
            # the positive index will always be 0 because the pos embedding will always be the first one.
            labels = torch.zeros(num_of_samples).long()

        # following examples from https://lightning.ai/docs/torchmetrics/stable/retrieval/precision.html
        # indexes refers to the mask of the labels
        indexes = torch.arange(0, query_embeddings.shape[0])
        expanded_indexes = (
            indexes.unsqueeze(-1).expand(num_of_samples, num_of_candidates).reshape(-1)
        )

        if self.should_sample_negatives_from_vocab:
            preds = (
                torch.mul(
                    query_embeddings.unsqueeze(1).expand_as(key_embeddings),
                    key_embeddings,
                )
                .sum(-1)
                .reshape(-1)
            )
        else:
            preds = torch.mm(query_embeddings, key_embeddings.t()).reshape(-1)

        target = torch.zeros(num_of_samples, num_of_candidates).bool()
        target[torch.arange(num_of_samples), labels] = True
        target = target.reshape(-1)

        for _, metric_object in self.metrics.items():
            metric_object.update(
                preds,
                target.to(preds.device),
                indexes=expanded_indexes.to(preds.device),
            )

    # this method samples random negative samples from the whole vocab
    def sample_negative_ids_from_vocab(
        self,
        num_of_samples: int,
        num_of_candidates: int,
        num_negatives: int,
    ) -> torch.Tensor:
        # num_of_samples: batch size
        # num_of_candidates: number of total vocabs
        # num_negatives: number of negative samples

        # we do randint to accelerate the negative sampling
        # this could have collision with positive pairs but the chance is very low

        # TODO (Clark): in the future we might need to have non-collision negative sampling
        # when K in top-k is very small (e.g., hits@1) and num_negatives is very large
        negative_candidates = torch.randint(
            self.placeholder_token_buffer,
            num_of_candidates,
            (num_of_samples, num_negatives),
        )

        return negative_candidates


class SIDRetrievalEvaluator(Evaluator):
    """
    Wrapper for retrieval evaluation metrics for semantic IDs.
    It takes model outputs in semantic IDs and automatically calculates the retrieval metrics.
    """

    def __init__(
        self,
        metrics: Dict[str, CustomRetrievalMetric],
        top_k_list: List[int],
    ):
        self.metrics = {
            f"{metric_name}@{top_k}": metric_object(
                top_k=top_k, sync_on_compute=False, compute_with_cache=False
            )
            for metric_name, metric_object in metrics.items()
            for top_k in top_k_list
        }

    def __call__(
        self,
        marginal_probs: torch.Tensor,
        generated_ids: torch.Tensor,
        labels: torch.Tensor,
        **kwargs,
    ):
        batch_size, num_candidates, num_hierarchies = generated_ids.shape
        labels = labels.reshape(batch_size, 1, num_hierarchies)
        preds = marginal_probs.reshape(-1)

        # check if the generated IDs contain the labels
        # if so, we get the coordinates of the matched IDs
        matched_id_coord = torch.all((generated_ids == labels), dim=2).nonzero()

        # we initialize the ground truth as all false
        target = torch.zeros(batch_size, num_candidates).bool()

        # we set the matched IDs to true if they are in the generated IDs
        target[matched_id_coord[:, 0], matched_id_coord[:, 1]] = True
        target = target.reshape(-1)
        expanded_indexes = (
            torch.arange(batch_size)
            .unsqueeze(-1)
            .expand(batch_size, num_candidates)
            .reshape(-1)
        )

        for _, metric_object in self.metrics.items():
            metric_object.update(
                preds,
                target.to(preds.device),
                indexes=expanded_indexes.to(preds.device),
            )


class SelectionRate(CustomMeanReductionMetric):
    """Tracks selected/total ratio for filtered evaluation."""

    def update(self, selected: int, total: int) -> None:
        self.metric_values += int(selected)
        self.total_values += int(total)


class TailFilteredSIDRetrievalEvaluator(Evaluator):
    """
    SID retrieval evaluator that only updates metrics on a subset of samples whose
    label semantic IDs appear in an allow-list (e.g., tail items mapped to SIDs).

    Notes:
    - The allow-list is a tensor of shape (M, num_hierarchies) saved via torch.save.
    - Metrics are logged under names prefixed with `allowed_label_name`.
    """

    def __init__(
        self,
        metrics: Dict[str, CustomRetrievalMetric],
        top_k_list: List[int],
        allowed_labels_path: str,
        allowed_label_name: str = "tail",
    ):
        # Prefix metric keys so they don't collide with the default evaluator
        self.metrics = {
            f"{allowed_label_name}_{metric_name}@{top_k}": metric_object(
                top_k=top_k, sync_on_compute=False, compute_with_cache=False
            )
            for metric_name, metric_object in metrics.items()
            for top_k in top_k_list
        }
        self.metrics[f"{allowed_label_name}_selection_rate"] = SelectionRate(
            sync_on_compute=False, compute_with_cache=False
        )

        self.allowed_label_name = allowed_label_name
        self.allowed_labels = torch.load(allowed_labels_path, map_location="cpu")
        if not isinstance(self.allowed_labels, torch.Tensor):
            raise TypeError(
                f"allowed_labels must be a torch.Tensor. got {type(self.allowed_labels)}"
            )
        if self.allowed_labels.ndim != 2:
            raise ValueError(
                f"allowed_labels must be 2D (M x H). got shape={tuple(self.allowed_labels.shape)}"
            )

    def __call__(
        self,
        marginal_probs: torch.Tensor,
        generated_ids: torch.Tensor,
        labels: torch.Tensor,
        **kwargs,
    ):
        batch_size, num_candidates, num_hierarchies = generated_ids.shape
        labels_2d = labels.reshape(batch_size, num_hierarchies).to("cpu")

        if self.allowed_labels.shape[1] != num_hierarchies:
            raise ValueError(
                f"allowed_labels H mismatch: allowed={self.allowed_labels.shape[1]} model={num_hierarchies}"
            )

        # mask samples where the label SID matches any allow-listed SID
        # (bs, 1, H) == (1, M, H) -> (bs, M, H) -> (bs, M) -> (bs,)
        mask = (labels_2d.unsqueeze(1) == self.allowed_labels.unsqueeze(0)).all(
            dim=2
        ).any(dim=1)

        selected = int(mask.sum().item())
        total = int(mask.numel())
        self.metrics[f"{self.allowed_label_name}_selection_rate"].update(
            selected=selected, total=total
        )
        if selected == 0:
            return

        # Filter tensors on-device for metric computation
        mask_dev = mask.to(marginal_probs.device)
        marginal_probs_f = marginal_probs.reshape(batch_size, num_candidates)[mask_dev]
        generated_ids_f = generated_ids[mask_dev]
        labels_f = labels.reshape(batch_size, num_hierarchies)[mask_dev]

        preds = marginal_probs_f.reshape(-1)
        labels_f = labels_f.reshape(selected, 1, num_hierarchies)

        matched_id_coord = torch.all((generated_ids_f == labels_f), dim=2).nonzero()
        target = torch.zeros(selected, num_candidates).bool()
        target[matched_id_coord[:, 0], matched_id_coord[:, 1]] = True
        target = target.reshape(-1)

        expanded_indexes = (
            torch.arange(selected, device=preds.device)
            .unsqueeze(-1)
            .expand(selected, num_candidates)
            .reshape(-1)
        )

        for metric_name, metric_object in self.metrics.items():
            if metric_name.endswith("_selection_rate"):
                continue
            metric_object.update(
                preds,
                target.to(preds.device),
                indexes=expanded_indexes,
            )
