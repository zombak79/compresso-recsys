"""The training procedure, and the fitted model's prediction path.

CLCRec is implemented from the paper (Wei et al., *Contrastive Learning for
Cold-Start Recommendation*, ACM MM 2021, arXiv:2107.05315v3); no third-party
code was consulted. Equation numbers refer to that version.
"""

from __future__ import annotations

import time
from typing import Any, ClassVar, Hashable, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.sparse import csr_matrix, isspmatrix_csr

from compresso import SRPTensor
from compresso_recsys._reporting import (
    _INHERIT,
    _Inherit,
    TrainingProgress,
    _format_duration,
)
from compresso_recsys.persistence import ModelCheckpointReader, ModelCheckpointWriter
from compresso_recsys.models.core.identifiers import (
    ItemVocabulary,
    canonical_item_ids,
)
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
from compresso_recsys.models.clcrec.config import CLCRecConfig
from compresso_recsys.models.clcrec.model import CLCRec
from compresso_recsys.models.clcrec.sampling import (
    NegativeSampler,
    sample_other_items,
)


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


class CLCRecTrainer(BaseColdStartRecommender):
    """Fit CLCRec and rank warm and cold candidates for known training users.

    The objective is Eq. 13 with the MF encoder: ``λ·L_RE + (1−λ)·L_UI +
    η‖Θ‖²``. ``L_UI`` contrasts each observed pair against ``n_negatives``
    items the user has not interacted with, scoring ``z_u·v/τ``; with
    probability ``hybrid_prob`` each item vector ``v`` is the item's content
    representation ``f`` instead of its ID embedding (hybrid contrastive
    training, Fig. 4). ``L_RE`` contrasts an item's ID embedding with its own
    content representation against other items' ones, scoring cosine/τ. The
    anchor of ``L_RE`` is the observed item of the same pair, as in Fig. 1.

    Implementation choices where the paper is silent, all documented in
    ``CR-Notes/audit/clcrec-design.md`` §5: the hybrid mask is drawn per item
    slot; ``η‖Θ‖²`` is applied to the embedding rows a batch uses plus the
    encoder weights spread over an epoch; R-E negatives reuse the U-I ones
    unless ``re_n_negatives`` is set.

    Scores depend on which training user a row is, so the model declares
    ``accepts_user_ids``: ``predict_on_batch(..., user_ids=)`` selects learned
    user embeddings, and ``unknown_users="fold_in"`` estimates an embedding for
    any other history (an extension, not part of the paper).
    """

    _fit_name = "CLCRec"
    checkpoint_type = "clcrec_trainer"
    accepts_user_ids: ClassVar[bool] = True

    def __init__(
        self,
        config: CLCRecConfig | None = None,
        logger: Any | None = None,
    ) -> None:
        super().__init__()
        self.cfg = config if config is not None else CLCRecConfig()
        self.logger = logger
        self.device = torch.device(self.cfg.device)
        self.module: CLCRec | None = None
        self.optimizer: torch.optim.Optimizer | None = None
        self.history: list[dict[str, float]] = []
        self.train_item_indices_: np.ndarray | None = None
        self.n_items_: int | None = None
        self.n_features_: int | None = None
        self._source_to_train: np.ndarray | None = None
        self._candidate_cache: tuple[int, torch.Tensor] | None = None
        self._user_vocabulary: ItemVocabulary | None = None
        self._is_fitted = False

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted

    # -- User identity ------------------------------------------------------
    # STUB until the user-identity plumbing lands on release/0.4.0
    # (CR-Notes/audit/user-identity-channel.md §4); these move to
    # BaseIdentifiedRecommender and are deleted here, not merged.

    def _set_user_ids(
        self,
        user_ids: Sequence[Hashable] | np.ndarray | None,
        n_rows: int,
    ) -> None:
        if user_ids is None:
            self._user_vocabulary = ItemVocabulary.positional(n_rows)
            return
        vocabulary = ItemVocabulary.from_ids(user_ids, name="user_ids")
        if vocabulary.n_items != n_rows:
            raise ValueError(
                f"user_ids has {vocabulary.n_items} entries, but interactions "
                f"has {n_rows} rows"
            )
        self._user_vocabulary = vocabulary

    @property
    def user_ids(self) -> np.ndarray | None:
        vocabulary = self._user_vocabulary
        return None if vocabulary is None else vocabulary.item_ids

    def _user_rows(
        self,
        user_ids: Sequence[Hashable] | np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        assert self._user_vocabulary is not None
        lookup = self._user_vocabulary.id_to_row
        values = np.asarray(user_ids, dtype=object).tolist()
        rows = np.fromiter(
            (lookup.get(value, -1) for value in values),
            dtype=np.int64,
            count=len(values),
        )
        return rows, rows >= 0

    def _resolve_user_rows(
        self,
        user_ids: Sequence[Hashable] | np.ndarray | None,
        n_rows: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        if user_ids is None:
            rows = np.full(n_rows, -1, dtype=np.int64)
            known = np.zeros(n_rows, dtype=bool)
        else:
            if len(user_ids) != n_rows:
                raise ValueError(
                    f"user_ids has {len(user_ids)} entries for {n_rows} source rows"
                )
            rows, known = self._user_rows(user_ids)
        policy = getattr(self.cfg, "unknown_users", "error")
        if policy == "error" and not known.all():
            unknown = (
                ["<no user_ids>"]
                if user_ids is None
                else [
                    value
                    for value, flag in zip(
                        np.asarray(user_ids, dtype=object).tolist(), known
                    )
                    if not flag
                ]
            )
            raise KeyError(
                f"unknown user IDs {unknown[:5]!r} under unknown_users='error'"
            )
        return rows, known

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
    ) -> CLCRecTrainer:
        config = dict(config)
        config["device"] = str(device)
        trainer = cls(CLCRecConfig(**config))
        state = reader.read_json("state/trainer.json")
        trainer.module = trainer._build_module(
            n_users=int(state["n_users"]),
            n_train_items=int(state["n_train_items"]),
            n_features=int(state["n_features"]),
        ).to(device)
        return trainer

    def _checkpoint_module(self) -> CLCRec | None:
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
                "n_users": int(self.module.user_emb.num_embeddings),
                "history": self.history,
            },
        )
        # STUB: the identity plumbing writes this from the base class.
        assert self._user_vocabulary is not None
        writer.write_item_ids("identity/user_ids.json", self._user_vocabulary.item_ids)
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
            raise ValueError("CLCRec training-item count is inconsistent")
        history = state.get("history")
        if not isinstance(history, list):
            raise ValueError("CLCRec training history must be a list")
        self.history = list(history)
        self._user_vocabulary = ItemVocabulary.from_ids(
            reader.read_item_ids("identity/user_ids.json"),
            name="user_ids",
        )
        self._source_to_train = self._train_row_map(
            self.n_items_,
            self.train_item_indices_,
        )
        self._candidate_cache = None
        self.candidates._load_checkpoint(reader)
        source_ids = self.candidates.source_item_ids
        assert source_ids is not None
        if source_ids.size != self.n_items_:
            raise ValueError("CLCRec source catalog does not match n_items")

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
    ) -> CLCRec:
        return CLCRec(
            n_users=n_users,
            n_items=n_train_items,
            n_features=n_features,
            embedding_dim=self.cfg.embedding_dim,
            hidden_dim=self.cfg.hidden_dim,
            activation=self.cfg.activation,
        )

    def _reset_optimizer(self) -> None:
        if self.module is None:
            raise RuntimeError("trainer must be built before creating optimizer")
        self.optimizer = torch.optim.Adam(
            self.module.parameters(), lr=float(self.cfg.lr)
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
        user_ids: Sequence[Hashable] | np.ndarray | None = None,
        metadata: pd.DataFrame | None = None,
        feature_space_id: str | None = None,
        validation: Any | None = None,
        early_stopping: Any | None = None,
        logger: Any | None = _INHERIT,
        show_progress: bool | None | _Inherit = _INHERIT,
    ) -> CLCRecTrainer:
        """Fit ID embeddings and the content encoder on the training items.

        ``item_features`` has one row per interaction column and every row
        enters the initial candidate catalog, so items without interactions
        (the cold ones) are rankable straight after fitting. Only the
        ``train_item_indices`` columns (default: every column with an
        interaction) get ID embeddings and are used in training.
        """
        if validation is not None or early_stopping is not None:
            raise NotImplementedError(
                "validation and early stopping arrive with the shared training "
                "hook (compresso_recsys.training)"
            )
        reporter = self._reporter(logger, show_progress)
        interactions = canonical_csr(interactions, name="interactions")
        if interactions.shape[0] < 1 or interactions.shape[1] < 1:
            raise ValueError("interactions must contain at least one user and one item")
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
            raise ValueError("CLCRec needs at least two training items")
        training = interactions[:, train_indices]
        if training.nnz < 1:
            raise ValueError("training item columns must contain an interaction")
        outside = interactions.nnz - training.nnz
        if outside:
            raise ValueError(
                f"interactions has {outside} entries outside train_item_indices"
            )
        self._set_user_ids(user_ids, n_users)

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
        sampler = NegativeSampler(
            training,
            max_attempts=self.cfg.max_negative_attempts,
        )
        pair_users = np.repeat(
            np.arange(n_users, dtype=np.int64),
            np.diff(training.indptr),
        )
        pair_items = training.indices.astype(np.int64, copy=True)
        rng = np.random.default_rng(self.cfg.seed + 1)
        generator = torch.Generator(device=self.device)
        generator.manual_seed(int(self.cfg.seed) + 2)
        batch_size = self.cfg.batch_size
        n_batches = -(-pair_items.size // batch_size)
        cold_users = int(np.sum(np.diff(training.indptr) == 0))

        with TrainingProgress(
            reporter,
            label=self._fit_name,
            epochs=self.cfg.epochs,
            batches=n_batches,
        ) as progress:
            progress.start(
                f"{n_users} users | {train_indices.size} training items | "
                f"{n_items} catalog items | {pair_items.size} interactions | "
                f"{n_batches} batches of {batch_size} | "
                f"{self.cfg.epochs} epochs | device {self.device}"
            )
            for epoch in progress.epochs():
                order = rng.permutation(pair_items.size)
                sums = {"loss": 0.0, "ui": 0.0, "re": 0.0, "l2": 0.0}
                batches = 0

                def running() -> dict[str, float]:
                    """Mean of each running sum, built only when logging."""
                    return {key: value / batches for key, value in sums.items()}

                collisions_before = sampler.residual_collisions
                for start in range(0, order.size, batch_size):
                    rows = order[start : start + batch_size]
                    stats = self._train_step(
                        pair_users[rows],
                        pair_items[rows],
                        sampler=sampler,
                        rng=rng,
                        generator=generator,
                        training_features=training_features,
                        feature_tensor=feature_tensor,
                        n_batches=n_batches,
                    )
                    for key, value in stats.items():
                        sums[key] += value
                    batches += 1
                    progress.batch(batches, metrics=running)
                record = {key: value / max(1, batches) for key, value in sums.items()}
                record["epoch"] = float(epoch)
                record["residual_negative_collisions"] = float(
                    sampler.residual_collisions - collisions_before
                )
                record["users_without_interactions"] = float(cold_users)
                self.history.append(record)
                progress.epoch_done(record)
            self.module.eval()
            self._is_fitted = True
        return self

    def _content(
        self,
        rows: np.ndarray,
        *,
        training_features: csr_matrix | np.ndarray,
        feature_tensor: torch.Tensor | None,
    ) -> torch.Tensor:
        """Encode the training items at ``rows`` (deduplicated) and gather back."""
        assert self.module is not None
        unique, inverse = np.unique(rows, return_inverse=True)
        if feature_tensor is not None:
            inputs = feature_tensor[torch.from_numpy(unique).to(self.device)]
        else:
            inputs = _dense_rows(training_features, unique, device=self.device)
        encoded = self.module.encoder(inputs)
        index = torch.from_numpy(inverse.reshape(rows.shape)).to(self.device)
        return encoded[index]

    def _train_step(
        self,
        users: np.ndarray,
        positives: np.ndarray,
        *,
        sampler: NegativeSampler,
        rng: np.random.Generator,
        generator: torch.Generator,
        training_features: csr_matrix | np.ndarray,
        feature_tensor: torch.Tensor | None,
        n_batches: int,
    ) -> dict[str, float]:
        assert self.module is not None and self.optimizer is not None
        cfg = self.cfg
        module = self.module
        negatives = sampler.sample(users, cfg.n_negatives, rng)
        items = np.concatenate((positives[:, None], negatives), axis=1)
        re_negatives = (
            None
            if cfg.re_n_negatives is None
            else sample_other_items(
                positives,
                cfg.re_n_negatives,
                sampler.n_items,
                rng,
            )
        )
        content_rows = (
            items
            if re_negatives is None
            else np.concatenate((items, re_negatives), axis=1)
        )

        module.train()
        self.optimizer.zero_grad(set_to_none=True)
        user_tensor = torch.from_numpy(users).to(self.device)
        item_tensor = torch.from_numpy(items).to(self.device)
        z_users = module.user_emb(user_tensor)
        z_items = module.item_emb(item_tensor)
        content = self._content(
            content_rows,
            training_features=training_features,
            feature_tensor=feature_tensor,
        )
        item_content = content[:, : items.shape[1]]

        # U-I term with hybrid contrastive training (Eq. 13, Fig. 4).
        swap = (
            torch.rand(items.shape, generator=generator, device=self.device)
            < cfg.hybrid_prob
        )
        item_side = torch.where(swap.unsqueeze(-1), item_content, z_items)
        targets = torch.zeros(users.size, dtype=torch.long, device=self.device)
        ui = F.cross_entropy(
            module.ui_logits(z_users, item_side, cfg.temperature),
            targets,
        )

        # R-E term: the pair's item against content representations (Eq. 12-13).
        re_content = (
            item_content
            if re_negatives is None
            else torch.cat(
                (item_content[:, :1], content[:, items.shape[1] :]),
                dim=1,
            )
        )
        re = F.cross_entropy(
            module.re_logits(z_items[:, 0], re_content, cfg.temperature),
            targets,
        )

        l2 = (z_users.square().sum() + z_items.square().sum()) / users.size
        l2 = (
            l2
            + sum(parameter.square().sum() for parameter in module.encoder.parameters())
            / n_batches
        )
        loss = cfg.re_weight * re + (1.0 - cfg.re_weight) * ui + cfg.l2_weight * l2
        loss.backward()
        self.optimizer.step()
        return {
            "loss": float(loss.detach().cpu()),
            "ui": float(ui.detach().cpu()),
            "re": float(re.detach().cpu()),
            "l2": float(l2.detach().cpu()),
        }

    # -- Prediction ----------------------------------------------------------

    def _prepare_source(self, source: csr_matrix) -> csr_matrix:
        if not self.is_fitted or self.n_items_ is None:
            raise RuntimeError("CLCRecTrainer must be fitted before prediction")
        source = canonical_csr(source, name="source")
        if source.shape[1] != self.n_items_:
            raise ValueError(
                f"source has {source.shape[1]} items, but CLCRec was fitted "
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
            self.module.encoder(
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
        """Item vectors for the selected candidates: ID rows or content."""
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
                matrix[torch.from_numpy(warm).to(self.device)] = self.module.item_emb(
                    torch.from_numpy(train_rows[warm]).to(self.device)
                )
        if full:
            self._candidate_cache = (selection.catalog.version, matrix)
        return matrix

    def _fold_in(self, source: csr_matrix) -> torch.Tensor:
        """Estimate user embeddings for histories of users unseen in training.

        This is an extension, not part of the paper's protocol: with every item
        table and the encoder frozen, the user vectors start at zero and take
        ``fold_in_steps`` Adam steps on the U-I term (ID item vectors, no
        hybrid swap) plus its L2 penalty, with negatives drawn from a generator
        seeded by ``seed``. Columns without an ID embedding are ignored, so an
        empty or all-cold history keeps the zero vector.
        """
        assert self.module is not None and self.train_item_indices_ is not None
        cfg = self.cfg
        compact = source[:, self.train_item_indices_].tocsr()
        users = torch.zeros(
            (source.shape[0], cfg.embedding_dim),
            dtype=torch.float32,
            device=self.device,
        )
        if compact.nnz == 0 or cfg.fold_in_steps == 0:
            return users
        pair_users = np.repeat(
            np.arange(compact.shape[0], dtype=np.int64),
            np.diff(compact.indptr),
        )
        pair_items = compact.indices.astype(np.int64, copy=True)
        sampler = NegativeSampler(compact, max_attempts=cfg.max_negative_attempts)
        rng = np.random.default_rng(cfg.seed + 3)
        item_weight = self.module.item_emb.weight.detach()
        positives = item_weight[torch.from_numpy(pair_items).to(self.device)]
        user_index = torch.from_numpy(pair_users).to(self.device)
        targets = torch.zeros(pair_items.size, dtype=torch.long, device=self.device)
        users.requires_grad_(True)
        optimizer = torch.optim.Adam([users], lr=float(cfg.fold_in_lr))
        with torch.enable_grad():
            for _ in range(cfg.fold_in_steps):
                negatives = sampler.sample(pair_users, cfg.n_negatives, rng)
                items = torch.cat(
                    (
                        positives.unsqueeze(1),
                        item_weight[torch.from_numpy(negatives).to(self.device)],
                    ),
                    dim=1,
                )
                optimizer.zero_grad(set_to_none=True)
                loss = F.cross_entropy(
                    CLCRec.ui_logits(users[user_index], items, cfg.temperature),
                    targets,
                ) + cfg.l2_weight * users.square().sum() / max(1, source.shape[0])
                loss.backward()
                optimizer.step()
        return users.detach()

    @torch.no_grad()
    def _user_matrix(
        self,
        source: csr_matrix,
        user_ids: Sequence[Hashable] | np.ndarray | None,
    ) -> torch.Tensor:
        assert self.module is not None
        rows, known = self._resolve_user_rows(user_ids, source.shape[0])
        users = torch.empty(
            (source.shape[0], self.cfg.embedding_dim),
            dtype=torch.float32,
            device=self.device,
        )
        if known.any():
            positions = torch.from_numpy(np.flatnonzero(known)).to(self.device)
            users[positions] = self.module.user_emb(
                torch.from_numpy(rows[known]).to(self.device)
            )
        if not known.all():
            positions = np.flatnonzero(~known)
            users[torch.from_numpy(positions).to(self.device)] = self._fold_in(
                source[positions]
            )
        return users

    @torch.no_grad()
    def _predict_prepared_batch(
        self,
        source: csr_matrix,
        *,
        k: int,
        exclude_seen: bool,
        selection: CandidateSelection,
        candidates: torch.Tensor,
        user_ids: Sequence[Hashable] | np.ndarray | None,
    ) -> SRPTensor:
        if not 1 <= int(k) <= selection.rows.size:
            raise ValueError(f"k must be in [1, {selection.rows.size}], got {k}")
        if source.shape[0] == 0:
            return SRPTensor(
                cols=torch.empty((0, k), dtype=torch.long, device=self.device),
                vals=torch.empty((0, k), dtype=torch.float32, device=self.device),
                shape=(0, selection.catalog.n_items),
            )
        users = self._user_matrix(source, user_ids)
        scores = users @ candidates.T

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
        user_ids: Sequence[Hashable] | np.ndarray | None = None,
    ) -> SRPTensor:
        """Predict one batch; ``user_ids`` names the training user of each row."""
        source = self._prepare_source(source)
        selection = self.candidates.resolve_selection(candidate_ids)
        return self._predict_prepared_batch(
            source,
            k=k,
            exclude_seen=exclude_seen,
            selection=selection,
            candidates=self._candidate_matrix(selection),
            user_ids=user_ids,
        )

    def _predict_identified(
        self,
        source: csr_matrix,
        *,
        k: int,
        exclude_seen: bool,
        candidate_ids: np.ndarray,
        user_ids: Sequence[Hashable] | np.ndarray | None = None,
    ) -> SRPTensor:
        if not isinstance(source, csr_matrix):
            raise TypeError("cold-start recommendations require a CSR source")
        return self.predict(
            source,
            k=k,
            exclude_seen=exclude_seen,
            candidate_ids=candidate_ids,
            user_ids=user_ids,
        )

    def _predict_identified_with_reporting(
        self,
        source: csr_matrix,
        *,
        k: int,
        exclude_seen: bool,
        candidate_ids: np.ndarray,
        reporter: Any,
        user_ids: Sequence[Hashable] | np.ndarray | None = None,
    ) -> SRPTensor:
        if not isinstance(source, csr_matrix):
            raise TypeError("cold-start recommendations require a CSR source")
        return self.predict(
            source,
            k=k,
            exclude_seen=exclude_seen,
            candidate_ids=candidate_ids,
            user_ids=user_ids,
            logger=reporter,
            show_progress=_INHERIT,
        )

    def predict(
        self,
        source: csr_matrix,
        *,
        k: int = 100,
        batch_size: int = 1024,
        exclude_seen: bool = True,
        candidate_ids: Sequence[Hashable] | np.ndarray | None = None,
        user_ids: Sequence[Hashable] | np.ndarray | None = None,
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
        if user_ids is not None and len(user_ids) != source.shape[0]:
            raise ValueError(
                f"user_ids has {len(user_ids)} entries for {source.shape[0]} rows"
            )
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
            reporter.wrap(starts, total=steps, desc=f"CLCRec predict@{k}"),
            start=1,
        ):
            end = min(start + batch_size, source.shape[0])
            result = self._predict_prepared_batch(
                source[start:end],
                k=k,
                exclude_seen=exclude_seen,
                selection=selection,
                candidates=candidates,
                user_ids=None if user_ids is None else user_ids[start:end],
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
                user_ids=user_ids,
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
