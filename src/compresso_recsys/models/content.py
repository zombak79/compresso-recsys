from __future__ import annotations

from dataclasses import dataclass
from typing import Hashable, Literal, Sequence

import numpy as np
import torch
from scipy.sparse import csr_matrix, issparse

from compresso import SRPTensor
from compresso_recsys.models.cold_start import BaseColdStartRecommender

__all__ = ["ContentRecommender", "ContentRecommenderConfig"]

ContentDataType = Literal["float32", "float64"]

_TORCH_DTYPES: dict[str, torch.dtype] = {
    "float32": torch.float32,
    "float64": torch.float64,
}


def _l2_normalize(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Row-wise L2 normalization, matching ``F.normalize(x, dim=-1)``."""
    return x / x.norm(dim=1, keepdim=True).clamp_min(eps)


@dataclass(frozen=True)
class ContentRecommenderConfig:
    """Configuration for :class:`ContentRecommender`.

    Parameters
    ----------
    normalize:
        L2-normalize item feature vectors before scoring, making the
        profile/candidate product a cosine similarity instead of a raw dot
        product. Leaving this off lets high-norm items dominate the ranking.
    elsa_forward:
        Subtract the user's own interaction vector from the scores and apply
        ReLU, reproducing the ELSA-forward scoring used by
        ``evaluate_item_embeddings_with_holdout``. This has no effect on the
        ranking when predicting with ``exclude_seen=True``, since it only
        touches entries that seen-item masking then sets to ``-inf``.
    device:
        Torch device for every matrix product, for example ``"cuda"`` or
        ``"mps"``. Only the final score matrix is copied back to the host.
    dtype:
        Floating-point precision used for the stored features and all products.
    """

    normalize: bool = True
    elsa_forward: bool = True
    device: str = "cpu"
    dtype: ContentDataType = "float32"

    def __post_init__(self) -> None:
        if self.dtype not in _TORCH_DTYPES:
            raise ValueError(
                f"dtype must be one of {sorted(_TORCH_DTYPES)}, got {self.dtype!r}"
            )


class ContentRecommender(BaseColdStartRecommender):
    """Cold-start baseline scoring items by content-feature similarity.

    The model learns nothing. A user profile is the sum of the feature vectors
    of the items they interacted with, and candidates are ranked by their
    similarity to that profile. Because items are scored from features alone,
    unseen items are recommendable as soon as they are registered on the
    catalog.

    >>> model = ContentRecommender(ContentRecommenderConfig(device="cuda"))
    >>> model.fit(item_features, item_ids=item_ids)
    >>> top = model.predict(source, k=20)

    With the default configuration this reproduces the scoring in
    :func:`compresso_recsys.retrieval.evaluate_item_embeddings_with_holdout`
    exactly, so the same item embeddings yield the same metrics through either
    path. That function is an ELSA-forward recommender fused with an evaluator
    rather than a neutral evaluator, which is why ``normalize`` and
    ``elsa_forward`` exist at all.
    """

    def __init__(self, config: ContentRecommenderConfig | None = None) -> None:
        super().__init__()
        self.cfg = config if config is not None else ContentRecommenderConfig()
        self.device = torch.device(self.cfg.device)
        self.source_features_: torch.Tensor | None = None
        self._torch_dtype = _TORCH_DTYPES[self.cfg.dtype]
        self._candidate_cache: tuple[object, torch.Tensor] | None = None

    @property
    def is_fitted(self) -> bool:
        """Whether the model is ready for prediction."""
        return self.source_features_ is not None

    def to(self, device: str | torch.device) -> "ContentRecommender":
        """Move stored features to ``device`` and return ``self``."""
        self.device = torch.device(device)
        if self.source_features_ is not None:
            self.source_features_ = self.source_features_.to(self.device)
        self._candidate_cache = None
        return self

    def fit(
        self,
        item_features: csr_matrix | np.ndarray,
        *,
        item_ids: Sequence[Hashable] | np.ndarray | None = None,
    ) -> "ContentRecommender":
        """Publish ``item_features`` as both the source and candidate space.

        Unlike :class:`TEASER`, this model takes no interaction matrix. It
        holds no parameters and scores directly in feature space, so there is
        nothing to learn from user histories.
        """
        features = np.array(
            item_features.todense() if issparse(item_features) else item_features,
            dtype=self.cfg.dtype,
            copy=True,
        )
        if features.ndim != 2:
            raise ValueError(
                f"item_features must be 2D, got shape {tuple(features.shape)}"
            )
        if features.shape[0] < 1:
            raise ValueError("item_features must contain at least one item")
        if not np.isfinite(features).all():
            raise ValueError("item_features must contain only finite values")

        ids = (
            np.arange(features.shape[0])
            if item_ids is None
            else np.asarray(item_ids, dtype=object)
        )
        if ids.shape[0] != features.shape[0]:
            raise ValueError(
                f"item_features has {features.shape[0]} rows but got "
                f"{ids.shape[0]} item_ids"
            )

        source = torch.as_tensor(features, dtype=self._torch_dtype, device=self.device)
        self.source_features_ = (
            _l2_normalize(source) if self.cfg.normalize else source
        ).contiguous()
        self._candidate_cache = None

        self._install_feature_catalog(
            source_item_ids=ids,
            source_popularity=np.zeros(ids.shape[0], dtype=self.cfg.dtype),
            n_input_features=int(features.shape[1]),
            # Stored unnormalized so update_candidates() can accept raw
            # features; normalization happens per batch in _candidate_matrix.
            candidate_features=features,
            metadata=None,
            feature_space_id=None,
            dtype=np.dtype(self.cfg.dtype),
            include_popularity=False,
        )
        return self

    def _candidate_matrix(
        self, features: csr_matrix | np.ndarray
    ) -> torch.Tensor:
        """Candidate features on device, normalized, cached by identity.

        The cache holds a reference to ``features`` so the identity test stays
        valid. It hits for whole-catalog selections, where the frozen catalog
        array is reused, and misses for ``candidate_ids=`` selections, which
        build a fresh slice per call.
        """
        if self._candidate_cache is not None and self._candidate_cache[0] is features:
            return self._candidate_cache[1]
        # Copy: catalog features are frozen read-only and torch warns when
        # wrapping a non-writable array.
        dense = np.array(
            features.todense() if issparse(features) else features,
            dtype=self.cfg.dtype,
            copy=True,
        )
        matrix = torch.as_tensor(dense, dtype=self._torch_dtype, device=self.device)
        if self.cfg.normalize:
            matrix = _l2_normalize(matrix)
        matrix = matrix.contiguous()
        self._candidate_cache = (features, matrix)
        return matrix

    def _profiles(self, source: csr_matrix) -> torch.Tensor:
        """User profiles as a sparse-times-dense product on ``self.device``."""
        assert self.source_features_ is not None
        coo = source.tocoo()
        indices = torch.as_tensor(
            np.vstack([coo.row, coo.col]), dtype=torch.long, device=self.device
        )
        values = torch.as_tensor(
            coo.data, dtype=self._torch_dtype, device=self.device
        )
        # Built from a valid scipy COO and coalesced below, so the invariant
        # check is redundant; opting out explicitly silences torch's warning.
        sparse = torch.sparse_coo_tensor(
            indices,
            values,
            size=source.shape,
            device=self.device,
            check_invariants=False,
        ).coalesce()
        return torch.sparse.mm(sparse, self.source_features_)

    def predict_on_batch(
        self,
        source: csr_matrix,
        *,
        k: int,
        exclude_seen: bool = True,
        candidate_ids: Sequence[Hashable] | np.ndarray | None = None,
    ) -> SRPTensor:
        """Predict ranked top-``k`` items for one source batch."""
        source = self._prepare_source(source)
        selection = self._resolve_candidate_selection(candidate_ids)
        n_candidates = int(selection.rows.size)
        if not 1 <= int(k) <= n_candidates:
            raise ValueError(f"k must be in [1, {n_candidates}], got {k}")

        # Two hops: source-vocabulary column -> catalog row -> selection-local
        # column. Indexing scores with source.indices directly is only correct
        # when the selection covers the whole catalog.
        seen_counts = np.diff(source.indptr)
        seen_users = np.repeat(
            np.arange(source.shape[0], dtype=np.int64), seen_counts
        )
        seen_catalog = selection.source_to_candidate[source.indices]
        registered = seen_catalog >= 0
        seen_local = np.full(seen_catalog.shape, -1, dtype=np.int64)
        seen_local[registered] = selection.candidate_to_local[
            seen_catalog[registered]
        ]
        selected_seen = seen_local >= 0

        if exclude_seen:
            available = n_candidates - np.bincount(
                seen_users[selected_seen], minlength=source.shape[0]
            )
            if available.size and np.any(available < k):
                row = int(np.flatnonzero(available < k)[0])
                raise ValueError(
                    f"source row {row} has only {available[row]} unseen items "
                    f"among the selected candidates, fewer than k={k}"
                )

        if source.shape[0] == 0:
            return SRPTensor(
                cols=torch.empty((0, k), dtype=torch.long),
                vals=torch.empty((0, k), dtype=self._torch_dtype),
                shape=(0, self.candidates.n_items),
            )

        scores = self._profiles(source) @ self._candidate_matrix(
            selection.features
        ).T

        rows = cols = None
        if bool(selected_seen.any()):
            rows = torch.as_tensor(
                seen_users[selected_seen], dtype=torch.long, device=self.device
            )
            cols = torch.as_tensor(
                seen_local[selected_seen], dtype=torch.long, device=self.device
            )

        if self.cfg.elsa_forward:
            if rows is not None:
                # The reference subtracts a binary interaction vector, so this
                # is exactly 1.0 regardless of the values carried by `source`.
                scores[rows, cols] -= 1.0
            scores = torch.relu(scores)

        if exclude_seen and rows is not None:
            scores[rows, cols] = -torch.inf

        # from_dense needs a host tensor; this copy must not be non-blocking.
        local = SRPTensor.from_dense(scores.cpu(), k=int(k), score_mode="raw")
        catalog_rows = torch.from_numpy(np.ascontiguousarray(selection.rows))
        return SRPTensor(
            cols=catalog_rows[local.cols],
            vals=local.vals,
            shape=(source.shape[0], self.candidates.n_items),
        )
