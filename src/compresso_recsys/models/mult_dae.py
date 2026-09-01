"""Multinomial denoising autoencoder for implicit collaborative filtering."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Hashable, Sequence

import numpy as np
import torch
from scipy.sparse import csr_matrix
from torch import nn
from torch.nn import functional as F

from compresso import SRPTensor
from compresso_recsys.models._autoencoder_batching import (
    dense_training_batch,
    prepare_dense_training_data,
)
from compresso_recsys.models._ranking import validate_candidate_topk
from compresso_recsys.models._validation import canonical_csr
from compresso_recsys.models.base import BaseCollaborativeRecommender
from compresso_recsys.persistence import ModelCheckpointReader, ModelCheckpointWriter

__all__ = ["MultDAE", "MultDAEConfig", "MultDAETrainer"]


def _progress(iterable, *, enabled: bool, desc: str):
    if not enabled:
        return iterable
    try:
        from tqdm.auto import tqdm
    except Exception:  # pragma: no cover - optional display helper
        return iterable
    return tqdm(iterable, desc=desc)


@dataclass
class MultDAEConfig:
    """Configuration for :class:`MultDAETrainer`.

    ``latent_dim`` is the deterministic bottleneck width. ``dropout`` corrupts
    normalized interaction vectors during training only, as in Mult-DAE.
    ``l2_reg`` is the coefficient on the squared L2 norm of the encoder and
    decoder weight matrices; biases are not regularized. The default matches
    the original implementation's ``0.01 / 500`` setting.
    ``preload_training_data=True`` caches the dense interaction matrix on the
    training device by default. On CUDA it falls back to CSR minibatch streaming
    when the matrix would use more than half of currently free device memory.
    """

    latent_dim: int = 200
    dropout: float = 0.5
    epochs: int = 20
    batch_size: int = 256
    lr: float = 1e-3
    l2_reg: float = 0.01 / 500
    preload_training_data: bool = True
    device: str | torch.device = "cpu"
    show_progress: bool = True
    seed: int = 0

    def __post_init__(self) -> None:
        for name in ("latent_dim", "epochs", "batch_size"):
            value = getattr(self, name)
            if isinstance(value, (bool, np.bool_)) or not isinstance(
                value, (int, np.integer)
            ):
                raise TypeError(f"{name} must be an integer")
            if value < 1:
                raise ValueError(f"{name} must be >= 1, got {value}")
        if not np.isfinite(self.dropout) or not 0.0 <= self.dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {self.dropout}")
        if not np.isfinite(self.lr) or self.lr <= 0.0:
            raise ValueError(f"lr must be finite and > 0, got {self.lr}")
        if not np.isfinite(self.l2_reg) or self.l2_reg < 0.0:
            raise ValueError(
                "l2_reg must be finite and >= 0, got "
                f"{self.l2_reg}"
            )
        if not isinstance(self.preload_training_data, (bool, np.bool_)):
            raise TypeError("preload_training_data must be a bool")
        if isinstance(self.seed, (bool, np.bool_)) or not isinstance(
            self.seed, (int, np.integer)
        ):
            raise TypeError("seed must be an integer")
        torch.device(self.device)


class MultDAE(nn.Module):
    """The deterministic ``n_items -> latent -> n_items`` Mult-DAE network."""

    def __init__(self, n_items: int, latent_dim: int, dropout: float) -> None:
        super().__init__()
        if n_items < 1:
            raise ValueError("n_items must be >= 1")
        if latent_dim < 1:
            raise ValueError("latent_dim must be >= 1")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        self.n_items = int(n_items)
        self.input_dropout = nn.Dropout(float(dropout))
        self.encoder = nn.Linear(self.n_items, int(latent_dim))
        self.decoder = nn.Linear(int(latent_dim), self.n_items)

    def forward(self, interactions: torch.Tensor) -> torch.Tensor:
        """Return one unnormalized multinomial score per catalog item."""
        if interactions.ndim != 2 or interactions.shape[1] != self.n_items:
            raise ValueError(
                "interactions must have shape (rows, "
                f"{self.n_items}), got {tuple(interactions.shape)}"
            )
        normalized = F.normalize(interactions, p=2, dim=1)
        hidden = torch.tanh(self.encoder(self.input_dropout(normalized)))
        return self.decoder(hidden)


class MultDAETrainer(BaseCollaborativeRecommender):
    """Train and serve Mult-DAE on complete implicit-feedback user rows."""

    checkpoint_type = "mult_dae_trainer"

    def __init__(self, config: MultDAEConfig | None = None) -> None:
        self.cfg = config if config is not None else MultDAEConfig()
        self.device = torch.device(self.cfg.device)
        self.model: MultDAE | None = None
        self.optimizer: torch.optim.Optimizer | None = None
        self.history: list[dict[str, float]] = []
        self._n_items: int | None = None
        self.training_data_preloaded_: bool | None = None

    @property
    def is_fitted(self) -> bool:
        return self.model is not None

    @property
    def n_items(self) -> int | None:
        return self._n_items

    def fit(
        self,
        interactions: csr_matrix,
        *,
        item_ids: Sequence[Hashable] | np.ndarray | None = None,
    ) -> MultDAETrainer:
        """Fit Mult-DAE using multinomial reconstruction likelihood."""
        interactions = canonical_csr(interactions, name="interactions")
        if interactions.shape[0] < 1 or interactions.shape[1] < 1:
            raise ValueError(
                "interactions must contain at least one user and one item"
            )
        if np.any(interactions.data < 0):
            raise ValueError("interactions must contain nonnegative values")
        active_rows = np.flatnonzero(np.diff(interactions.indptr) > 0)
        if active_rows.size == 0:
            raise ValueError("interactions must contain at least one nonempty user")

        torch.manual_seed(int(self.cfg.seed))
        rng = np.random.default_rng(int(self.cfg.seed))
        self._n_items = int(interactions.shape[1])
        self._set_item_ids(item_ids, n_items=self._n_items)
        self.model = self._build_model()
        self._build_checkpoint_optimizer()
        assert self.optimizer is not None
        self.history = []
        training_data = prepare_dense_training_data(
            interactions,
            device=self.device,
            preload=self.cfg.preload_training_data,
        )
        self.training_data_preloaded_ = training_data is not None

        epochs = _progress(
            range(1, int(self.cfg.epochs) + 1),
            enabled=self.cfg.show_progress,
            desc="MultDAE fit",
        )
        for epoch in epochs:
            self.model.train()
            order = rng.permutation(active_rows)
            loss_sum = torch.zeros((), device=self.device)
            users = 0
            for start in range(0, order.size, int(self.cfg.batch_size)):
                selected = order[start : start + int(self.cfg.batch_size)]
                target = dense_training_batch(
                    interactions,
                    selected,
                    device=self.device,
                    preloaded=training_data,
                )
                logits = self.model(target)
                loss = -(target * F.log_softmax(logits, dim=1)).sum(dim=1).mean()
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                self.optimizer.step()
                batch_users = int(selected.size)
                loss_sum += loss.detach() * batch_users
                users += batch_users
            mean_loss = float((loss_sum / users).item())
            self.history.append({"epoch": float(epoch), "loss": mean_loss})
            if hasattr(epochs, "set_postfix"):
                epochs.set_postfix({"loss": f"{mean_loss:.4f}"})
        return self

    def _build_model(self) -> MultDAE:
        if self._n_items is None:
            raise RuntimeError("MultDAE catalog size is unavailable")
        return MultDAE(
            self._n_items,
            int(self.cfg.latent_dim),
            float(self.cfg.dropout),
        ).to(self.device)

    def predict_on_batch(
        self,
        source: csr_matrix,
        *,
        k: int,
        exclude_seen: bool = True,
        candidate_ids: Sequence[Hashable] | np.ndarray | None = None,
    ) -> SRPTensor:
        source = self._prepare_source(source)
        candidate_rows = self._candidate_rows(candidate_ids)
        validate_candidate_topk(
            source,
            candidate_rows,
            k=k,
            exclude_seen=exclude_seen,
        )
        assert self.model is not None and self._n_items is not None
        if source.shape[0] == 0:
            return SRPTensor(
                cols=torch.empty((0, k), dtype=torch.long, device=self.device),
                vals=torch.empty((0, k), dtype=torch.float32, device=self.device),
                shape=source.shape,
            )

        dense = source.toarray().astype(np.float32, copy=False)
        inputs = torch.from_numpy(dense).to(self.device)
        candidates = torch.from_numpy(candidate_rows).long().to(self.device)
        self.model.eval()
        with torch.no_grad():
            logits = self.model(inputs)
            selected_logits = logits[:, candidates]
            if exclude_seen and source.indices.size:
                candidate_to_local = np.full(source.shape[1], -1, dtype=np.int64)
                candidate_to_local[candidate_rows] = np.arange(candidate_rows.size)
                seen_counts = np.diff(source.indptr)
                seen_rows = np.repeat(
                    np.arange(source.shape[0], dtype=np.int64),
                    seen_counts,
                )
                seen_local = candidate_to_local[source.indices]
                included = seen_local >= 0
                selected_logits[
                    torch.as_tensor(
                        seen_rows[included], dtype=torch.long, device=self.device
                    ),
                    torch.as_tensor(
                        seen_local[included], dtype=torch.long, device=self.device
                    ),
                ] = -torch.inf
            values, local_columns = torch.topk(selected_logits, int(k), dim=1)
            columns = candidates[local_columns]
        return SRPTensor(cols=columns, vals=values, shape=source.shape)

    def _checkpoint_config(self) -> dict[str, Any]:
        config = asdict(self.cfg)
        config["device"] = str(self.device)
        return config

    @classmethod
    def _from_checkpoint_config(
        cls,
        config: dict,
        reader: ModelCheckpointReader,
        *,
        device: torch.device,
    ) -> MultDAETrainer:
        state = reader.read_json("state/trainer.json")
        n_items = state.get("n_items")
        if isinstance(n_items, bool) or not isinstance(n_items, int) or n_items < 1:
            raise ValueError("MultDAE n_items must be a positive integer")
        config = dict(config)
        config["device"] = str(device)
        trainer = cls(MultDAEConfig(**config))
        trainer._n_items = n_items
        trainer.model = trainer._build_model()
        return trainer

    def _checkpoint_module(self) -> nn.Module | None:
        return self.model

    def _save_checkpoint_state(self, writer: ModelCheckpointWriter) -> None:
        assert self._n_items is not None
        writer.write_json(
            "state/trainer.json",
            {"n_items": self._n_items, "history": self.history},
        )

    def _load_checkpoint_state(self, reader: ModelCheckpointReader) -> None:
        state = reader.read_json("state/trainer.json")
        history = state.get("history")
        if not isinstance(history, list):
            raise ValueError("MultDAE training history must be a list")
        self.history = list(history)

    def _build_checkpoint_optimizer(self) -> None:
        if self.model is None:
            raise RuntimeError("MultDAE model must be built before its optimizer")
        self.optimizer = torch.optim.Adam(
            [
                {
                    "params": [
                        self.model.encoder.weight,
                        self.model.decoder.weight,
                    ],
                    "weight_decay": 2.0 * float(self.cfg.l2_reg),
                },
                {
                    "params": [
                        self.model.encoder.bias,
                        self.model.decoder.bias,
                    ],
                    "weight_decay": 0.0,
                },
            ],
            lr=float(self.cfg.lr),
        )
