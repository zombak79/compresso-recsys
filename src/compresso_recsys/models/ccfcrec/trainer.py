"""The training procedure, and the fitted model's prediction path.

CCFCRec is implemented from the paper (Zhou, Zhang and Yang, *Contrastive
Collaborative Filtering for Cold-Start Item Recommendation*, WWW 2023,
arXiv:2302.02151v2); no third-party code was consulted. Equation numbers refer
to that version.
"""

from __future__ import annotations

import time
import warnings
from typing import Any, Hashable, Sequence

import numpy as np
import pandas as pd
import torch
from scipy.sparse import csr_matrix, isspmatrix_csr

from compresso import SRPTensor
from compresso_recsys._reporting import (
    _INHERIT,
    _Inherit,
    TrainingProgress,
    _format_duration,
)
from compresso_recsys.persistence import ModelCheckpointReader, ModelCheckpointWriter
from compresso_recsys.training import EarlyStopping, Validation, _make_monitor
from compresso_recsys.models.core.identifiers import canonical_item_ids
from compresso_recsys.models.core.validation import canonical_csr
from compresso_recsys.models.core.catalog import (
    CandidateCatalog,
    CandidateSelection,
)
from compresso_recsys.models.core.cold_start import BaseColdStartRecommender
from compresso_recsys.models.core.features import (
    ItemFeatures,
    canonical_feature_space_id,
    canonical_item_features,
    canonical_metadata,
    take_features,
)
from compresso_recsys.models.ccfcrec.config import CCFCRecConfig
from compresso_recsys.models.ccfcrec.model import CCFCRec
from compresso_recsys.models.clcrec.sampling import NegativeSampler


def _dense_rows(
    features: csr_matrix | np.ndarray,
    rows: np.ndarray,
    *,
    device: torch.device,
) -> torch.Tensor:
    selected = features[rows]
    array = selected.toarray() if isspmatrix_csr(selected) else np.asarray(selected)
    return torch.from_numpy(np.array(array, dtype=np.float32, order="C", copy=True)).to(
        device
    )


def _sparse_tensor(matrix: csr_matrix, device: torch.device) -> torch.Tensor:
    """A binary CSR matrix as a torch sparse COO tensor, for ``matrix @ columns``."""
    matrix = matrix.tocsr(copy=True)
    matrix.sum_duplicates()  # also sorts indices: the COO form is then coalesced
    coo = matrix.tocoo()
    indices = torch.from_numpy(
        np.vstack((coo.row.astype(np.int64), coo.col.astype(np.int64)))
    )
    values = torch.ones(coo.nnz, dtype=torch.float32)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Sparse invariant checks are implicitly disabled.*",
            category=UserWarning,
        )
        return torch.sparse_coo_tensor(
            indices,
            values,
            coo.shape,
            is_coalesced=True,
            check_invariants=False,
        ).to(device)


class _CooccurrenceSampler:
    """Positive and negative contrastive items of Eq. 4.

    An item's positives ``N+_v`` are the other items some user interacted
    with together with it; its negatives are every other item. Positives are
    drawn uniformly from ``N+_v``. An item without any co-occurring item (its
    users interacted with nothing else) uses itself as the positive, which is
    counted in :attr:`self_positives`.
    """

    def __init__(self, training: csr_matrix, *, max_attempts: int) -> None:
        columns = training.tocsc().astype(np.float32)
        cooccurrence = (columns.T @ columns).tocsr()
        cooccurrence.data[:] = 1.0
        cooccurrence.setdiag(1.0)
        cooccurrence.sort_indices()
        self.negatives = NegativeSampler(cooccurrence, max_attempts=max_attempts)
        others = cooccurrence.copy()
        others.setdiag(0.0)
        others.eliminate_zeros()
        others.sort_indices()
        self._others = others
        self.self_positives = 0

    def positives(
        self,
        items: np.ndarray,
        n_positive: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        starts = self._others.indptr[items]
        counts = self._others.indptr[items + 1] - starts
        draws = rng.random((items.size, n_positive))
        offsets = np.floor(draws * np.maximum(counts, 1)[:, None]).astype(np.int64)
        picked = self._others.indices[
            np.minimum(starts[:, None] + offsets, max(self._others.nnz - 1, 0))
        ]
        lonely = counts == 0
        if lonely.any():
            picked[lonely] = items[lonely, None]
            self.self_positives += int(lonely.sum()) * n_positive
        return picked.astype(np.int64, copy=False)


class CCFCRecTrainer(BaseColdStartRecommender):
    """Fit CCFCRec and rank warm and cold candidates for any user history.

    The objective is Eq. 9: ``L_q + L_z + λ·L_c`` plus weight decay.

    - ``L_q`` and ``L_z`` (Eq. 7-8) are BPR losses: for each observed
      ``(item v, user u+)`` a user ``u-`` who did not interact with ``v`` is
      sampled, and ``v`` must score ``u+`` above ``u-``. ``L_q`` scores with the
      content-based embedding ``q_v`` and ``L_z`` with the co-occurrence
      embedding ``z_v``; both share the user embedding ``s_u``.
    - ``L_c`` (Eq. 4) contrasts ``q_v`` with the ``z`` of ``n_positive_items``
      items that co-occur with ``v`` against ``n_negative_items`` that do not.

    A user embedding is computed from the user's history (Eq. 3), never looked
    up by user ID, so any history can be scored and no user IDs are needed.

    Deviation from the paper: the content vector ``c_v`` is the supplied item
    feature vector. The paper concatenates per-attribute embeddings and learns
    lookup tables for one-hot and multi-hot attributes; those lookups are not
    implemented. Several dense modalities concatenated by
    :class:`~compresso_recsys.models.MMConcatWrapper` play the role of the
    concatenation.
    """

    _fit_name = "CCFCRec"
    checkpoint_type = "ccfcrec_trainer"

    def __init__(
        self,
        config: CCFCRecConfig | None = None,
        logger: Any | None = None,
    ) -> None:
        super().__init__()
        self.cfg = config if config is not None else CCFCRecConfig()
        self.logger = logger
        self.device = torch.device(self.cfg.device)
        self.module: CCFCRec | None = None
        self.optimizer: torch.optim.Optimizer | None = None
        self.history: list[dict[str, float]] = []
        self.train_item_indices_: np.ndarray | None = None
        self.n_items_: int | None = None
        self.n_features_: int | None = None
        self._source_to_train: np.ndarray | None = None
        self._candidate_cache: tuple[int, torch.Tensor] | None = None
        self._is_fitted = False

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted

    # -- Persistence ---------------------------------------------------------

    def _on_catalog_published(self, catalog: CandidateCatalog) -> None:
        self._candidate_cache = None

    @classmethod
    def _from_checkpoint_config(
        cls,
        config: dict,
        reader: ModelCheckpointReader,
        *,
        device: torch.device,
    ) -> CCFCRecTrainer:
        config = dict(config)
        config["device"] = str(device)
        trainer = cls(CCFCRecConfig(**config))
        state = reader.read_json("state/trainer.json")
        trainer.module = trainer._build_module(
            n_users=int(state["n_users"]),
            n_train_items=int(state["n_train_items"]),
            n_features=int(state["n_features"]),
        ).to(device)
        return trainer

    def _checkpoint_module(self) -> CCFCRec | None:
        return self.module

    def _move_checkpoint_state(self, device: torch.device) -> None:
        self._candidate_cache = None

    def _save_checkpoint_state(self, writer: ModelCheckpointWriter) -> None:
        assert self.train_item_indices_ is not None
        assert self.module is not None
        writer.write_numpy("state/train_item_indices.npy", self.train_item_indices_)
        writer.write_json(
            "state/trainer.json",
            {
                "n_items": self.n_items_,
                "n_features": self.n_features_,
                "n_train_items": int(self.train_item_indices_.size),
                "n_users": int(self.module.item_columns.shape[0]),
                "history": self.history,
            },
        )
        self.candidates._save_checkpoint(writer)

    def _load_checkpoint_state(self, reader: ModelCheckpointReader) -> None:
        state = reader.read_json("state/trainer.json")
        self.train_item_indices_ = np.asarray(
            reader.read_numpy("state/train_item_indices.npy"),
            dtype=np.int64,
        )
        self.n_items_ = int(state["n_items"])
        self.n_features_ = int(state["n_features"])
        if self.train_item_indices_.size != int(state["n_train_items"]):
            raise ValueError("CCFCRec training-item count is inconsistent")
        history = state.get("history")
        if not isinstance(history, list):
            raise ValueError("CCFCRec training history must be a list")
        self.history = list(history)
        self._source_to_train = self._train_row_map(
            self.n_items_,
            self.train_item_indices_,
        )
        self._candidate_cache = None
        self.candidates._load_checkpoint(reader)
        source_ids = self.candidates.source_item_ids
        assert source_ids is not None
        if source_ids.size != self.n_items_:
            raise ValueError("CCFCRec source catalog does not match n_items")

    def _build_checkpoint_optimizer(self) -> None:
        self._reset_optimizer()

    def _finish_checkpoint_load(self) -> None:
        self._is_fitted = True
        if self.module is not None:
            self.module.eval()

    # -- Fitting -------------------------------------------------------------

    def _build_module(
        self,
        *,
        n_users: int,
        n_train_items: int,
        n_features: int,
    ) -> CCFCRec:
        return CCFCRec(
            n_users=n_users,
            n_items=n_train_items,
            n_features=n_features,
            embedding_dim=self.cfg.embedding_dim,
            hidden_dim=self.cfg.hidden_dim,
        )

    def _reset_optimizer(self) -> None:
        if self.module is None:
            raise RuntimeError("trainer must be built before creating optimizer")
        self.optimizer = torch.optim.Adam(
            self.module.parameters(),
            lr=float(self.cfg.lr),
            weight_decay=float(self.cfg.weight_decay),
        )

    @staticmethod
    def _train_row_map(n_items: int, train_indices: np.ndarray) -> np.ndarray:
        mapping = np.full(n_items, -1, dtype=np.int64)
        mapping[train_indices] = np.arange(train_indices.size, dtype=np.int64)
        return mapping

    def fit(
        self,
        interactions: csr_matrix,
        item_features: ItemFeatures,
        *,
        train_item_indices: np.ndarray | Sequence[int] | None = None,
        item_ids: Sequence[Hashable] | np.ndarray | None = None,
        metadata: pd.DataFrame | None = None,
        feature_space_id: str | None = None,
        validation: Validation | None = None,
        early_stopping: EarlyStopping | None = None,
        logger: Any | None = _INHERIT,
        show_progress: bool | None | _Inherit = _INHERIT,
    ) -> CCFCRecTrainer:
        """Fit the encoders on the training items.

        ``item_features`` has one row per interaction column and every row
        enters the initial candidate catalog, so items without interactions
        (the cold ones) are rankable straight after fitting. Only the
        ``train_item_indices`` columns (default: every column with an
        interaction) are trained on and get a co-occurrence embedding.

        ``validation`` records ``val_*`` metrics in :attr:`history` after each
        due epoch, and ``early_stopping`` stops on one of them and restores the
        best weights.
        """
        reporter = self._reporter(logger, show_progress)
        interactions = canonical_csr(interactions, name="interactions")
        if interactions.shape[0] < 2 or interactions.shape[1] < 1:
            raise ValueError(
                "interactions must contain at least two users and one item"
            )
        if interactions.data.size and not np.all(interactions.data == 1):
            raise ValueError(
                "interactions must contain binary implicit values equal to 1"
            )
        dtype = np.dtype("float32")
        features = canonical_item_features(item_features, dtype=dtype)
        n_users, n_items = (int(size) for size in interactions.shape)
        if features.shape[0] != n_items:
            raise ValueError(
                f"item_features has {features.shape[0]} rows, but interactions "
                f"has {n_items} items"
            )
        resolved_ids = canonical_item_ids(
            np.arange(n_items, dtype=np.int64) if item_ids is None else item_ids,
            expected_rows=n_items,
        )
        resolved_metadata = canonical_metadata(metadata, item_ids=resolved_ids)
        resolved_space = canonical_feature_space_id(feature_space_id)
        if train_item_indices is None:
            train_indices = np.flatnonzero(np.diff(interactions.tocsc().indptr) > 0)
        else:
            train_indices = np.unique(np.asarray(train_item_indices, dtype=np.int64))
            if train_indices.size and (
                train_indices[0] < 0 or train_indices[-1] >= n_items
            ):
                raise ValueError("train_item_indices must be valid item columns")
        if train_indices.size < 2:
            raise ValueError("CCFCRec needs at least two training items")
        training = interactions[:, train_indices].tocsr()
        if training.nnz < 1:
            raise ValueError("training item columns must contain an interaction")
        outside = interactions.nnz - training.nnz
        if outside:
            raise ValueError(
                f"interactions has {outside} entries outside train_item_indices"
            )

        torch.manual_seed(int(self.cfg.seed))
        self.module = self._build_module(
            n_users=n_users,
            n_train_items=int(train_indices.size),
            n_features=int(features.shape[1]),
        ).to(self.device)
        self._reset_optimizer()
        self.train_item_indices_ = train_indices.copy()
        self._source_to_train = self._train_row_map(n_items, train_indices)
        self.n_items_ = n_items
        self.n_features_ = int(features.shape[1])
        self.history = []
        self.candidates.install(
            source_item_ids=resolved_ids,
            source_popularity=np.zeros(n_items, dtype=dtype),
            n_input_features=int(features.shape[1]),
            candidate_features=features,
            metadata=resolved_metadata,
            feature_space_id=resolved_space,
            dtype=dtype,
            include_popularity=False,
        )

        training_features = take_features(features, train_indices)
        feature_tensor = (
            None
            if isspmatrix_csr(training_features)
            else torch.from_numpy(
                np.array(training_features, dtype=np.float32, order="C", copy=True)
            ).to(self.device)
        )
        item_users = training.T.tocsr()
        user_sampler = NegativeSampler(
            item_users,
            max_attempts=self.cfg.max_negative_attempts,
        )
        item_sampler = _CooccurrenceSampler(
            training,
            max_attempts=self.cfg.max_negative_attempts,
        )
        all_item_users = _sparse_tensor(item_users, self.device)
        pair_items = np.repeat(
            np.arange(train_indices.size, dtype=np.int64),
            np.diff(item_users.indptr),
        )
        pair_users = item_users.indices.astype(np.int64, copy=True)
        rng = np.random.default_rng(self.cfg.seed + 1)
        batch_size = self.cfg.batch_size
        n_batches = -(-pair_items.size // batch_size)

        monitor = _make_monitor(self, validation, early_stopping)
        with TrainingProgress(
            reporter,
            label=self._fit_name,
            epochs=self.cfg.epochs,
            batches=n_batches,
            monitor=monitor,
        ) as progress:
            progress.start(
                f"{n_users} users | {train_indices.size} training items | "
                f"{n_items} catalog items | {pair_items.size} interactions | "
                f"{n_batches} batches of {batch_size} | "
                f"{self.cfg.epochs} epochs | device {self.device}"
            )
            for epoch in progress.epochs():
                order = rng.permutation(pair_items.size)
                sums = {
                    "loss": 0.0,
                    "bpr_content": 0.0,
                    "bpr_cooc": 0.0,
                    "contrastive": 0.0,
                }
                batches = 0

                def running() -> dict[str, float]:
                    """Mean of each running sum, built only when logging."""
                    return {key: value / batches for key, value in sums.items()}

                user_collisions = user_sampler.residual_collisions
                item_collisions = item_sampler.negatives.residual_collisions
                for start in range(0, order.size, batch_size):
                    rows = order[start : start + batch_size]
                    stats = self._train_step(
                        pair_items[rows],
                        pair_users[rows],
                        training=training,
                        item_users=item_users,
                        user_sampler=user_sampler,
                        item_sampler=item_sampler,
                        rng=rng,
                        training_features=training_features,
                        feature_tensor=feature_tensor,
                    )
                    for key, value in stats.items():
                        sums[key] += value
                    batches += 1
                    progress.batch(batches, metrics=running)
                record = {key: value / max(1, batches) for key, value in sums.items()}
                record["epoch"] = float(epoch)
                record["residual_negative_users"] = float(
                    user_sampler.residual_collisions - user_collisions
                )
                record["residual_negative_items"] = float(
                    item_sampler.negatives.residual_collisions - item_collisions
                )
                self.history.append(record)
                # Prediction reads the co-occurrence embeddings and the
                # candidate cache, both of which the epoch made stale.
                self._refresh_cooccurrence(all_item_users)
                self._candidate_cache = None
                progress.epoch_done(record)
            # Early stopping may have restored earlier weights, buffer included.
            self._candidate_cache = None
            self.module.eval()
            self._is_fitted = True
        return self

    @torch.no_grad()
    def _refresh_cooccurrence(self, item_users: torch.Tensor) -> None:
        assert self.module is not None
        self.module.cooccurrence.copy_(
            torch.sparse.mm(item_users, self.module.item_columns)
        )

    def _content(
        self,
        rows: np.ndarray,
        *,
        training_features: csr_matrix | np.ndarray,
        feature_tensor: torch.Tensor | None,
    ) -> torch.Tensor:
        """Content embeddings ``q`` of the training items at ``rows``."""
        assert self.module is not None
        if feature_tensor is not None:
            inputs = feature_tensor[torch.from_numpy(rows).to(self.device)]
        else:
            inputs = _dense_rows(training_features, rows, device=self.device)
        return self.module.content(inputs)

    def _histories(self, matrix: csr_matrix, rows: np.ndarray, columns: torch.Tensor):
        """``Σ`` of ``columns`` over each row's interactions (Eq. 2-3), deduplicated."""
        unique, inverse = np.unique(rows, return_inverse=True)
        summed = torch.sparse.mm(_sparse_tensor(matrix[unique], self.device), columns)
        index = torch.from_numpy(inverse.reshape(rows.shape)).to(self.device)
        return summed[index]

    def _train_step(
        self,
        items: np.ndarray,
        users: np.ndarray,
        *,
        training: csr_matrix,
        item_users: csr_matrix,
        user_sampler: NegativeSampler,
        item_sampler: _CooccurrenceSampler,
        rng: np.random.Generator,
        training_features: csr_matrix | np.ndarray,
        feature_tensor: torch.Tensor | None,
    ) -> dict[str, float]:
        assert self.module is not None and self.optimizer is not None
        cfg = self.cfg
        module = self.module
        negative_users = user_sampler.sample(items, 1, rng)[:, 0]
        positive_items = item_sampler.positives(items, cfg.n_positive_items, rng)
        negative_items = item_sampler.negatives.sample(items, cfg.n_negative_items, rng)

        module.train()
        self.optimizer.zero_grad(set_to_none=True)
        q = self._content(
            items,
            training_features=training_features,
            feature_tensor=feature_tensor,
        )
        s_positive = self._histories(training, users, module.user_columns)
        s_negative = self._histories(training, negative_users, module.user_columns)
        cooc_rows = np.concatenate(
            (items[:, None], positive_items, negative_items), axis=1
        )
        z = self._histories(item_users, cooc_rows, module.item_columns)
        z_item = z[:, 0]
        z_positive = z[:, 1 : 1 + cfg.n_positive_items]
        z_negative = z[:, 1 + cfg.n_positive_items :]

        bpr_content = CCFCRec.bpr_loss(q, s_positive, s_negative)
        bpr_cooc = CCFCRec.bpr_loss(z_item, s_positive, s_negative)
        contrastive = CCFCRec.contrastive_loss(
            q, z_positive, z_negative, cfg.temperature
        )
        loss = bpr_content + bpr_cooc + cfg.contrastive_weight * contrastive
        loss.backward()
        self.optimizer.step()
        return {
            "loss": float(loss.detach().cpu()),
            "bpr_content": float(bpr_content.detach().cpu()),
            "bpr_cooc": float(bpr_cooc.detach().cpu()),
            "contrastive": float(contrastive.detach().cpu()),
        }

    # -- Prediction ----------------------------------------------------------

    def _prepare_source(self, source: csr_matrix) -> csr_matrix:
        if not self.is_fitted or self.n_items_ is None:
            raise RuntimeError("CCFCRecTrainer must be fitted before prediction")
        source = canonical_csr(source, name="source")
        if source.shape[1] != self.n_items_:
            raise ValueError(
                f"source has {source.shape[1]} items, but CCFCRec was fitted "
                f"with {self.n_items_} items"
            )
        if source.data.size and not np.all(source.data == 1):
            raise ValueError("source must contain binary implicit values equal to 1")
        return source

    @torch.no_grad()
    def _encode_features(self, features: csr_matrix | np.ndarray) -> torch.Tensor:
        assert self.module is not None
        chunk = self.cfg.encoder_batch_size
        parts = [
            self.module.content(
                _dense_rows(
                    features,
                    np.arange(start, min(start + chunk, features.shape[0])),
                    device=self.device,
                )
            )
            for start in range(0, features.shape[0], chunk)
        ]
        if not parts:
            return torch.empty(
                (0, self.cfg.embedding_dim), dtype=torch.float32, device=self.device
            )
        return torch.cat(parts)

    @torch.no_grad()
    def _candidate_matrix(self, selection: CandidateSelection) -> torch.Tensor:
        """Item vectors for the selected candidates: content, or co-occurrence when hybrid."""
        full = selection.rows.size == selection.catalog.n_items
        if full and self._candidate_cache is not None:
            version, cached = self._candidate_cache
            if version == selection.catalog.version:
                return cached
        assert self.module is not None and self._source_to_train is not None
        matrix = self._encode_features(selection.features)
        if self.cfg.inference == "hybrid":
            source_to_candidate = selection.source_to_candidate
            registered = np.flatnonzero(source_to_candidate >= 0)
            candidate_to_source = np.full(selection.catalog.n_items, -1, dtype=np.int64)
            candidate_to_source[source_to_candidate[registered]] = registered
            sources = candidate_to_source[selection.rows]
            train_rows = np.where(
                sources >= 0,
                self._source_to_train[np.maximum(sources, 0)],
                -1,
            )
            warm = np.flatnonzero(train_rows >= 0)
            if warm.size:
                matrix[torch.from_numpy(warm).to(self.device)] = (
                    self.module.cooccurrence[
                        torch.from_numpy(train_rows[warm]).to(self.device)
                    ]
                )
        if full:
            self._candidate_cache = (selection.catalog.version, matrix)
        return matrix

    @torch.no_grad()
    def user_embeddings(self, source: csr_matrix) -> torch.Tensor:
        """``s_u`` (Eq. 3) of each source history; items without a column are ignored."""
        assert self.module is not None and self.train_item_indices_ is not None
        compact = source[:, self.train_item_indices_].tocsr()
        if compact.shape[0] == 0:
            return torch.empty(
                (0, self.cfg.embedding_dim), dtype=torch.float32, device=self.device
            )
        return torch.sparse.mm(
            _sparse_tensor(compact, self.device), self.module.user_columns
        )

    @torch.no_grad()
    def _predict_prepared_batch(
        self,
        source: csr_matrix,
        *,
        k: int,
        exclude_seen: bool,
        selection: CandidateSelection,
        candidates: torch.Tensor,
    ) -> SRPTensor:
        if not 1 <= int(k) <= selection.rows.size:
            raise ValueError(f"k must be in [1, {selection.rows.size}], got {k}")
        if source.shape[0] == 0:
            return SRPTensor(
                cols=torch.empty((0, k), dtype=torch.long, device=self.device),
                vals=torch.empty((0, k), dtype=torch.float32, device=self.device),
                shape=(0, selection.catalog.n_items),
            )
        scores = self.user_embeddings(source) @ candidates.T

        seen_counts = np.diff(source.indptr)
        seen_users = np.repeat(np.arange(source.shape[0], dtype=np.int64), seen_counts)
        seen_catalog = selection.source_to_candidate[source.indices]
        registered = seen_catalog >= 0
        seen_local = np.full(seen_catalog.shape, -1, dtype=np.int64)
        seen_local[registered] = selection.candidate_to_local[seen_catalog[registered]]
        selected = seen_local >= 0
        if exclude_seen:
            available = selection.rows.size - np.bincount(
                seen_users[selected],
                minlength=source.shape[0],
            )
            if available.size and np.any(available < k):
                row = int(np.flatnonzero(available < k)[0])
                raise ValueError(
                    f"source row {row} has only {available[row]} unseen items "
                    f"among the selected candidates, fewer than k={k}"
                )
            if bool(selected.any()):
                rows = torch.from_numpy(seen_users[selected]).to(self.device)
                columns = torch.from_numpy(seen_local[selected]).to(self.device)
                scores[rows, columns] = -torch.inf
        local = SRPTensor.from_dense(scores, k=int(k), score_mode="raw")
        global_rows = torch.from_numpy(selection.rows).long().to(self.device)
        return SRPTensor(
            cols=global_rows[local.cols],
            vals=local.vals,
            shape=(source.shape[0], selection.catalog.n_items),
        )

    def predict_on_batch(
        self,
        source: csr_matrix,
        *,
        k: int,
        exclude_seen: bool = True,
        candidate_ids: Sequence[Hashable] | np.ndarray | None = None,
    ) -> SRPTensor:
        """Predict one batch against the current or restricted catalog."""
        source = self._prepare_source(source)
        selection = self.candidates.resolve_selection(candidate_ids)
        return self._predict_prepared_batch(
            source,
            k=k,
            exclude_seen=exclude_seen,
            selection=selection,
            candidates=self._candidate_matrix(selection),
        )

    def predict(
        self,
        source: csr_matrix,
        *,
        k: int = 100,
        batch_size: int = 1024,
        exclude_seen: bool = True,
        candidate_ids: Sequence[Hashable] | np.ndarray | None = None,
        logger: Any | None = _INHERIT,
        show_progress: bool | None | _Inherit = _INHERIT,
    ) -> SRPTensor:
        """Predict all source rows while reusing one candidate matrix."""
        reporter = self._reporter(
            logger,
            show_progress,
            default_show_progress=False,
        )
        source = self._prepare_source(source)
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        selection = self.candidates.resolve_selection(candidate_ids)
        candidates = self._candidate_matrix(selection)
        columns: list[torch.Tensor] = []
        values: list[torch.Tensor] = []
        starts = range(0, source.shape[0], batch_size)
        steps = len(starts)
        started = time.monotonic()
        reporter.log(
            f"predict@{k} started: {source.shape[0]} rows | "
            f"{steps} batches of {batch_size} | device {self.device}"
        )
        for step, start in enumerate(
            reporter.wrap(starts, total=steps, desc=f"CCFCRec predict@{k}"),
            start=1,
        ):
            result = self._predict_prepared_batch(
                source[start : start + batch_size],
                k=k,
                exclude_seen=exclude_seen,
                selection=selection,
                candidates=candidates,
            )
            columns.append(result.cols)
            values.append(result.vals)
            log_steps = reporter.log_every_n_steps
            if log_steps and step % log_steps == 0:
                reporter.step(
                    f"predict@{k} step {step}/{steps}",
                    step,
                    steps,
                    started,
                )
        if not columns:
            prediction = self._predict_prepared_batch(
                source,
                k=k,
                exclude_seen=exclude_seen,
                selection=selection,
                candidates=candidates,
            )
        else:
            prediction = SRPTensor(
                cols=torch.vstack(columns),
                vals=torch.vstack(values),
                shape=(source.shape[0], selection.catalog.n_items),
            )
        reporter.log(
            f"predict@{k} finished: "
            f"{_format_duration(time.monotonic() - started)} total | "
            f"{source.shape[0]} rows"
        )
        return prediction
