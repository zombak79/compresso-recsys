"""Serving a warm collaborative model through the cold-start contract."""

from __future__ import annotations

from threading import RLock
from typing import (
    Any,
    Callable,
    Hashable,
    Literal,
    Mapping,
    Protocol,
    Sequence,
    runtime_checkable,
)

import numpy as np
import torch
from scipy.sparse import csr_matrix, hstack, issparse, isspmatrix_csr, vstack

from compresso import SRPTensor
from compresso_recsys._reporting import _INHERIT, _Inherit, _Reporter, _format_duration
from compresso_recsys.models.core.validation import canonical_csr
from compresso_recsys.models.base import (
    BaseIdentifiedRecommender,
    BasePersistableRecommender,
    Recommender,
    SequentialRecommender,
    _accepts_reporting_keywords,
)
from compresso_recsys.models.core.identifiers import (
    ItemVocabulary,
    canonical_item_ids,
)
from compresso_recsys.sequences import ItemSequences

__all__ = [
    "BaseColdStartRecommender",
    "CandidateCatalog",
    "ColdStartRecommender",
    "ItemVocabulary",
    "MutableCandidateCatalog",
    "WarmCatalogAdapter",
]

ItemFeatures = csr_matrix | SRPTensor | np.ndarray | torch.Tensor
CandidateConflict = Literal["error", "replace", "ignore"]

_NOT_INSTALLED = (
    "no candidate catalog is installed: the model has not been fitted, or "
    "install() was never called on the catalog"
)




class WarmCatalogAdapter(BaseIdentifiedRecommender):
    """Expose a fixed-catalog recommender in a larger identified catalog.

    The wrapped model continues to consume and rank only its training items.
    :meth:`align_source` expresses a source over the expanded catalog in the
    fitted item space, while :meth:`predict_on_batch` remaps the resulting ranked
    columns back into that catalog. Cold candidates remain valid target items but
    can never be emitted by the wrapped model.

    Stage catalogs must follow the checkpoint invariant: the training IDs are an
    exact ordered prefix and cold items are appended. A ``csr_matrix`` is projected
    to that fitted prefix. An :class:`~compresso_recsys.ItemSequences` is passed
    through whole, because its warm indices already mean the same thing and the
    wrapped model's tokenizer turns appended cold indices into ``unk`` without
    deleting positions. Rows survive either way, so alignment with the targets is
    preserved.

    This is mandatory whenever the model's item space is narrower than the
    evaluation catalog, which the ``temporal`` split mode guarantees by
    construction. It is also worth reaching for under ``leave_last_out``, where
    the catalogs do match but items whose every occurrence falls in a held-out
    tail are still absent from training -- and the model families do not treat
    such columns alike. A softmax next-item objective pushes every non-target
    logit down on every step, and a never-trained item is never a target, so it
    is buried: on MovieLens-1M such items land at the 95th rank percentile for
    :class:`~compresso_recsys.models.SimpleRNNTrainer` against the 60th for
    :class:`~compresso_recsys.models.ELSATrainer`, which leaves them near their
    initialization. Neither number is about recommendation quality, so a
    comparison spanning both families is sounder with the cold items made
    unreachable for each. Whether it matters is a question about the data rather
    than the protocol: count the evaluation rows whose target is absent from
    training before deciding.

    Parameters
    ----------
    model:
        Fitted recommender whose prediction columns follow ``train_item_ids``.
    train_item_ids:
        Item IDs in the exact column order used to fit ``model``.
    catalog_item_ids:
        Expanded source and target catalog. ``train_item_ids`` must be its exact
        ordered prefix; additional cold items are appended after it.
    """

    def __init__(
        self,
        model: Recommender | SequentialRecommender,
        train_item_ids: Sequence[Hashable] | np.ndarray,
        catalog_item_ids: Sequence[Hashable] | np.ndarray,
    ) -> None:
        if not isinstance(model, Recommender):
            raise TypeError("model must implement predict_on_batch(source, *, k)")
        train_vocabulary = ItemVocabulary.from_ids(
            train_item_ids,
            name="train_item_ids",
        )
        catalog_vocabulary = ItemVocabulary.from_ids(
            catalog_item_ids,
            name="catalog_item_ids",
        )
        missing = [
            item_id
            for item_id in train_vocabulary.item_ids.tolist()
            if item_id not in catalog_vocabulary.id_to_row
        ]
        if missing:
            raise ValueError(
                "catalog_item_ids is missing training item ID: "
                f"{missing[0]!r}"
            )
        if not np.array_equal(
            train_vocabulary.item_ids,
            catalog_vocabulary.item_ids[: train_vocabulary.n_items],
        ):
            raise ValueError(
                "train_item_ids must be an exact ordered prefix of "
                "catalog_item_ids; checkpoint stage catalogs may only grow by "
                "appending cold items"
            )

        train_to_catalog = np.fromiter(
            (
                catalog_vocabulary.id_to_row[item_id]
                for item_id in train_vocabulary.item_ids.tolist()
            ),
            dtype=np.int64,
            count=train_vocabulary.n_items,
        )
        train_to_catalog.setflags(write=False)

        self.model = model
        self._train_vocabulary = train_vocabulary
        self._catalog_vocabulary = catalog_vocabulary
        self.train_item_ids = train_vocabulary.item_ids
        self.catalog_item_ids = catalog_vocabulary.item_ids
        self.train_to_catalog = train_to_catalog
        self.catalog_size = catalog_vocabulary.n_items
        self._identity_alignment = np.array_equal(
            self.train_item_ids,
            self.catalog_item_ids,
        )
        self._mapping_lock = RLock()
        self._mapping_by_device: dict[torch.device, torch.Tensor] = {}

        if isinstance(model, BaseIdentifiedRecommender) and not np.array_equal(
            model.source_item_ids,
            self.train_item_ids,
        ):
            raise ValueError(
                "model source_item_ids must match train_item_ids in row order"
            )

    @property
    def source_item_ids(self) -> np.ndarray:
        """Stable IDs accepted from the expanded stage catalog."""
        return self.catalog_item_ids

    @property
    def candidate_item_ids(self) -> np.ndarray:
        """Stable IDs in the expanded output catalog."""
        return self.catalog_item_ids

    def _recommend_vocabularies(
        self,
    ) -> tuple[ItemVocabulary, ItemVocabulary]:
        return self._catalog_vocabulary, self._catalog_vocabulary

    def _prediction_reporter(self, logger: Any, show_progress: Any) -> _Reporter:
        if isinstance(self.model, BaseIdentifiedRecommender):
            return self.model._prediction_reporter(logger, show_progress)
        return super()._prediction_reporter(logger, show_progress)

    def _scoreable_candidate_rows(
        self,
        vocabulary: ItemVocabulary,
    ) -> np.ndarray:
        del vocabulary
        return self.train_to_catalog

    def _effective_exclude_seen(self, exclude_seen: bool) -> bool:
        if isinstance(self.model, BaseIdentifiedRecommender):
            return self.model._effective_exclude_seen(exclude_seen)
        return super()._effective_exclude_seen(exclude_seen)

    def _recommendation_source(
        self,
        rows: list[np.ndarray],
        *,
        vocabulary: ItemVocabulary,
    ) -> csr_matrix | ItemSequences:
        if not isinstance(self.model, BaseIdentifiedRecommender):
            raise TypeError(
                "recommend() requires the wrapped model to inherit an "
                "identified recommender base"
            )
        return self.model._recommendation_source(rows, vocabulary=vocabulary)

    def _predict_identified(
        self,
        source: csr_matrix | ItemSequences,
        *,
        k: int,
        exclude_seen: bool,
        candidate_ids: np.ndarray,
    ) -> SRPTensor:
        if not isinstance(self.model, BaseIdentifiedRecommender):
            raise TypeError(
                "recommend() requires the wrapped model to inherit an "
                "identified recommender base"
            )
        if isinstance(source, csr_matrix):
            source = self.align_source(source)
        predictions = self.model._predict_identified(
            source,
            k=k,
            exclude_seen=exclude_seen,
            candidate_ids=candidate_ids,
        )
        n_rows = source.shape[0] if isinstance(source, csr_matrix) else source.n_rows
        return self._remap_predictions(predictions, n_rows=n_rows)

    def _predict_identified_with_reporting(
        self,
        source: csr_matrix | ItemSequences,
        *,
        k: int,
        exclude_seen: bool,
        candidate_ids: np.ndarray,
        reporter: _Reporter,
    ) -> SRPTensor:
        if not isinstance(self.model, BaseIdentifiedRecommender):
            raise TypeError(
                "recommend() requires the wrapped model to inherit an "
                "identified recommender base"
            )
        if isinstance(source, csr_matrix):
            source = self.align_source(source)
        predictions = self.model._predict_identified_with_reporting(
            source,
            k=k,
            exclude_seen=exclude_seen,
            candidate_ids=candidate_ids,
            reporter=reporter,
        )
        n_rows = source.shape[0] if isinstance(source, csr_matrix) else source.n_rows
        return self._remap_predictions(predictions, n_rows=n_rows)

    def align_source(self, source: csr_matrix) -> csr_matrix:
        """Select the fitted training-item columns from an expanded-catalog matrix.

        Matrices only. A history needs no alignment: a sequential model's
        tokenizer maps an out-of-catalog index to its own ``unk`` token, keeping
        the position, and *projecting* one instead would delete interior events
        and thereby assert transitions that never happened. Pass sequences
        straight to :meth:`predict_on_batch`.
        """
        if isinstance(source, ItemSequences):
            raise TypeError(
                "sequences need no alignment: a model's tokenizer turns an "
                "out-of-catalog index into its own 'unk' token, in place. "
                "Dropping those items instead would join their neighbours as if "
                "they had been consecutive. Pass the sequences to "
                "predict_on_batch() directly"
            )
        source = canonical_csr(source, name="source")
        if source.shape[1] != self.catalog_size:
            raise ValueError(
                f"source has {source.shape[1]} items, but catalog_item_ids has "
                f"{self.catalog_size} entries"
            )
        if self._identity_alignment:
            return source
        return source[:, self.train_to_catalog].tocsr()

    def _mapping_on(self, device: torch.device) -> torch.Tensor:
        with self._mapping_lock:
            mapping = self._mapping_by_device.get(device)
            if mapping is None:
                mapping = torch.tensor(
                    self.train_to_catalog,
                    dtype=torch.long,
                    device=device,
                )
                self._mapping_by_device[device] = mapping
            return mapping

    def _remap_predictions(
        self,
        predictions: SRPTensor,
        *,
        n_rows: int,
    ) -> SRPTensor:
        """Express wrapped-model prediction columns in the expanded catalog."""
        if not isinstance(predictions, SRPTensor):
            raise TypeError("model prediction must be an SRPTensor")
        if predictions.rows != n_rows:
            raise ValueError("model prediction rows must match the source rows")
        if predictions.cols_total != len(self.train_item_ids):
            raise ValueError(
                "model prediction items must match train_item_ids: expected "
                f"{len(self.train_item_ids)}, got {predictions.cols_total}"
            )

        mapping = self._mapping_on(predictions.cols.device)
        return SRPTensor(
            cols=mapping[predictions.cols],
            vals=predictions.vals,
            shape=(predictions.rows, self.catalog_size),
            validate=False,
        )

    def predict_on_batch(
        self,
        source: csr_matrix | ItemSequences,
        *,
        k: int,
        exclude_seen: bool = True,
        candidate_ids: Sequence[Hashable] | np.ndarray | None = None,
    ) -> SRPTensor:
        """Predict warm items and express their columns in the full catalog."""
        if isinstance(source, ItemSequences):
            # Unaligned by design: the model's tokenizer decides what an
            # out-of-catalog index becomes, so a history arrives in catalog
            # space and only the prediction columns need widening.
            n_rows, n_items = source.n_rows, source.n_items
            if n_items != self.catalog_size:
                raise ValueError(
                    f"source spans {n_items} items, but catalog_item_ids has "
                    f"{self.catalog_size} entries"
                )
        else:
            source = canonical_csr(source, name="source")
            n_rows, n_items = source.shape
            if n_items != len(self.train_item_ids):
                raise ValueError(
                    f"source has {n_items} items, but train_item_ids has "
                    f"{len(self.train_item_ids)} entries; call align_source() first"
                )
        if candidate_ids is not None:
            self._train_vocabulary.rows_for(
                candidate_ids,
                name="candidate_ids",
            )
        kwargs = (
            {}
            if candidate_ids is None
            else {"candidate_ids": candidate_ids}
        )
        predictions = self.model.predict_on_batch(
            source,
            k=k,
            exclude_seen=exclude_seen,
            **kwargs,
        )
        return self._remap_predictions(predictions, n_rows=n_rows)
