"""Identifier handling shared by every recommender that names its items."""

from __future__ import annotations


from abc import ABC, abstractmethod
import inspect
from pathlib import Path
import re
from typing import (
    Any,
    Hashable,
    Literal,
    Sequence,
)

import numpy as np
from scipy.sparse import csr_matrix
from torch import nn

from compresso import SRPTensor
from compresso_recsys._reporting import (
    _INHERIT,
    _Inherit,
    _Reporter,
    _resolve_reporter,
)
from compresso_recsys.persistence import (
    ModelCheckpointReader,
    ModelCheckpointWriter,
)
from compresso_recsys.sequences import ItemSequences
from compresso_recsys.models.core.identifiers import ItemVocabulary, Recommendations


_MODEL_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
_MODELS_DIR = "models"


def _embedded_model_path(root: Path, name: str) -> Path:
    if not isinstance(name, str) or _MODEL_NAME.fullmatch(name) is None:
        raise ValueError(
            "model name must start with an ASCII letter or digit and contain "
            "only letters, digits, '.', '_', or '-'"
        )
    if name.lower().endswith(".zip"):
        raise ValueError("model name must omit the .zip extension")
    return root / _MODELS_DIR / f"{name}.zip"


def _unwrapped_module(module: nn.Module) -> nn.Module:
    """Return the eager module underneath a compiled Torch wrapper."""
    original = getattr(module, "_orig_mod", None)
    return module if not isinstance(original, nn.Module) else original


def _accepts_reporting_keywords(method: Any) -> bool:
    """Whether a prediction override accepts the new reporting keywords."""
    try:
        parameters = inspect.signature(method).parameters.values()
    except (TypeError, ValueError):
        return False
    if any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in parameters
    ):
        return True
    names = {parameter.name for parameter in parameters}
    return {"logger", "show_progress"} <= names


class BaseIdentifiedRecommender(ABC):
    """Shared production-facing recommendation workflow.

    Histories and candidate filters enter as stable IDs. Source-specific bases
    turn the mapped rows into a CSR matrix or :class:`ItemSequences`; concrete
    models only need to apply the selected candidates before their top-k.
    """

    def _fixed_vocabulary(self) -> ItemVocabulary:
        vocabulary = getattr(self, "_item_vocabulary", None)
        if isinstance(vocabulary, ItemVocabulary):
            return vocabulary
        n_items = getattr(self, "n_items", None)
        if n_items is None:
            raise RuntimeError(
                f"{type(self).__name__} has no fitted item vocabulary"
            )
        vocabulary = ItemVocabulary.positional(int(n_items))
        self._item_vocabulary = vocabulary
        return vocabulary

    def _prepare_item_vocabulary(
        self,
        item_ids: Sequence[Hashable] | np.ndarray | None,
        *,
        n_items: int,
    ) -> ItemVocabulary:
        """Validate a fitted catalog without publishing it on the model."""
        vocabulary = (
            ItemVocabulary.positional(n_items)
            if item_ids is None
            else ItemVocabulary.from_ids(item_ids)
        )
        if vocabulary.n_items != int(n_items):
            raise ValueError(
                f"item_ids has {vocabulary.n_items} entries, but the fitted "
                f"catalog has {n_items} items"
            )
        return vocabulary

    def _set_item_ids(
        self,
        item_ids: Sequence[Hashable] | np.ndarray | None,
        *,
        n_items: int,
    ) -> None:
        """Validate and publish the fitted catalog on the model."""
        vocabulary = self._prepare_item_vocabulary(item_ids, n_items=n_items)
        self._publish_item_vocabulary(vocabulary)

    def _publish_item_vocabulary(self, vocabulary: ItemVocabulary) -> None:
        """Publish a vocabulary previously prepared for a successful fit."""
        self._item_vocabulary = vocabulary

    @property
    def source_item_ids(self) -> np.ndarray:
        """Stable IDs accepted in recommendation histories."""
        return self._fixed_vocabulary().item_ids

    @property
    def candidate_item_ids(self) -> np.ndarray:
        """Stable IDs that can be returned by :meth:`recommend`."""
        return self._fixed_vocabulary().item_ids

    def _recommend_vocabularies(
        self,
    ) -> tuple[ItemVocabulary, ItemVocabulary]:
        vocabulary = self._fixed_vocabulary()
        return vocabulary, vocabulary

    def _restore_source_item_ids(self, item_ids: np.ndarray) -> None:
        self._item_vocabulary = ItemVocabulary.from_ids(
            item_ids,
            name="source_item_ids",
        )

    def _save_checkpoint_common_state(
        self,
        writer: ModelCheckpointWriter,
    ) -> None:
        source, _ = self._recommend_vocabularies()
        writer.write_item_ids("identity/source_item_ids.json", source.item_ids)

    def _load_checkpoint_common_state(
        self,
        reader: ModelCheckpointReader,
    ) -> None:
        self._restore_source_item_ids(
            reader.read_item_ids("identity/source_item_ids.json")
        )

    @abstractmethod
    def _recommendation_source(
        self,
        rows: list[np.ndarray],
        *,
        vocabulary: ItemVocabulary,
    ) -> csr_matrix | ItemSequences:
        """Build the low-level batched source for mapped history rows."""

    @abstractmethod
    def _predict_identified(
        self,
        source: csr_matrix | ItemSequences,
        *,
        k: int,
        exclude_seen: bool,
        candidate_ids: np.ndarray,
    ) -> SRPTensor:
        """Predict after candidate IDs have been resolved and filtered."""

    def _predict_identified_with_reporting(
        self,
        source: csr_matrix | ItemSequences,
        *,
        k: int,
        exclude_seen: bool,
        candidate_ids: np.ndarray,
        reporter: _Reporter,
    ) -> SRPTensor:
        """Reporting-aware prediction hook with a legacy-compatible fallback."""
        del reporter
        return self._predict_identified(
            source,
            k=k,
            exclude_seen=exclude_seen,
            candidate_ids=candidate_ids,
        )

    def _candidate_rows(
        self,
        candidate_ids: Sequence[Hashable] | np.ndarray | None,
    ) -> np.ndarray:
        _, vocabulary = self._recommend_vocabularies()
        if candidate_ids is None:
            return np.arange(vocabulary.n_items, dtype=np.int64)
        return np.sort(
            np.unique(vocabulary.rows_for(candidate_ids, name="candidate_ids"))
        )

    def _scoreable_candidate_rows(
        self,
        vocabulary: ItemVocabulary,
    ) -> np.ndarray:
        """Candidate rows eligible before request-specific filters."""
        return np.arange(vocabulary.n_items, dtype=np.int64)

    def _effective_exclude_seen(self, exclude_seen: bool) -> bool:
        """Resolve the masking policy before capacity and prediction agree on it."""
        return exclude_seen

    def _prediction_reporter(
        self,
        logger: Any,
        show_progress: Any,
    ) -> _Reporter:
        """Resolve reporting for a base-provided batched prediction call."""
        config = getattr(self, "cfg", None)
        return _resolve_reporter(
            default_logger=getattr(self, "logger", None),
            logger=logger,
            # Base prediction has always defaulted to a quiet serving path,
            # independently of a trainer's fit-time progress setting.
            default_show_progress=False,
            show_progress=show_progress,
            prefix=getattr(config, "log_prefix", type(self).__name__),
            log_every_n_steps=getattr(config, "log_every_n_steps", 0),
        )

    def _reporter(
        self,
        logger: Any,
        show_progress: Any,
        *,
        default_show_progress: bool | None = None,
    ) -> _Reporter:
        """Resolve reporting for a fit call.

        Unlike :meth:`_prediction_reporter`, this honours the trainer's
        configured ``show_progress``: fitting is the loud path by default.
        ``default_show_progress`` overrides that for a trainer whose fit runs in
        more than one phase and wants one of them kept quiet.
        """
        return _resolve_reporter(
            default_logger=self.logger,
            logger=logger,
            default_show_progress=(
                self.cfg.show_progress
                if default_show_progress is None
                else default_show_progress
            ),
            show_progress=show_progress,
            prefix=self.cfg.log_prefix,
            log_every_n_steps=self.cfg.log_every_n_steps,
        )

    def recommend(
        self,
        histories: Sequence[Sequence[Hashable]],
        *,
        k: int = 100,
        exclude_seen: bool = False,
        allowlist: Sequence[Hashable] | np.ndarray | None = None,
        blocklist: Sequence[Hashable] | np.ndarray | None = None,
        on_insufficient: Literal["truncate", "raise"] = "truncate",
        logger: Any | None = _INHERIT,
        show_progress: bool | None | _Inherit = _INHERIT,
    ) -> Recommendations:
        """Recommend up to ``k`` item IDs for each item-ID history.

        ``logger`` and ``show_progress`` override prediction reporting for this
        call. Passing ``logger=None`` makes the request quiet even when the
        recommender has a constructor logger.
        """
        if isinstance(k, (bool, np.bool_)) or not isinstance(k, (int, np.integer)):
            raise TypeError("k must be an integer")
        if int(k) < 1:
            raise ValueError(f"k must be >= 1, got {k}")
        if on_insufficient not in {"truncate", "raise"}:
            raise ValueError(
                "on_insufficient must be either 'truncate' or 'raise'"
            )
        if isinstance(histories, (str, bytes)):
            raise TypeError("histories must be a sequence of item-ID sequences")
        exclude_seen = self._effective_exclude_seen(exclude_seen)
        try:
            history_values = list(histories)
        except TypeError as error:
            raise TypeError(
                "histories must be a sequence of item-ID sequences"
            ) from error

        reporter = self._prediction_reporter(logger, show_progress)
        reporting_override = logger is not _INHERIT or not (
            show_progress is _INHERIT or show_progress is None
        )
        use_reporting_path = (
            reporting_override or reporter.active or reporter.show_progress
        )

        source_vocabulary, candidate_vocabulary = self._recommend_vocabularies()
        rows = [
            source_vocabulary.rows_for(history, name=f"histories[{row}]")
            for row, history in enumerate(history_values)
        ]

        eligible = np.zeros(candidate_vocabulary.n_items, dtype=bool)
        eligible[self._scoreable_candidate_rows(candidate_vocabulary)] = True
        if allowlist is not None:
            allowed = np.zeros(candidate_vocabulary.n_items, dtype=bool)
            allowed[
                candidate_vocabulary.rows_for(allowlist, name="allowlist")
            ] = True
            eligible &= allowed
        if blocklist is not None:
            eligible[
                candidate_vocabulary.rows_for(blocklist, name="blocklist")
            ] = False
        candidate_rows = np.flatnonzero(eligible)
        if on_insufficient == "raise" and candidate_rows.size < int(k):
            raise ValueError(f"k must be in [1, {candidate_rows.size}], got {k}")

        selected_rows = set(candidate_rows.tolist())
        available_counts = np.full(
            len(rows),
            candidate_rows.size,
            dtype=np.int64,
        )
        if exclude_seen:
            for row, history_rows in enumerate(rows):
                seen_candidate_rows = {
                    candidate_vocabulary.id_to_row[item_id]
                    for item_id in source_vocabulary.item_ids[
                        history_rows
                    ].tolist()
                    if item_id in candidate_vocabulary.id_to_row
                    and candidate_vocabulary.id_to_row[item_id] in selected_rows
                }
                available_counts[row] -= len(seen_candidate_rows)
                if (
                    on_insufficient == "raise"
                    and available_counts[row] < int(k)
                ):
                    raise ValueError(
                        f"histories[{row}] has only {available_counts[row]} unseen "
                        f"candidates, fewer than k={k}"
                    )

        returned_counts = np.minimum(available_counts, int(k))
        selected_ids = candidate_vocabulary.item_ids[candidate_rows]
        item_ids = np.full((len(rows), int(k)), None, dtype=object)
        scores = np.full((len(rows), int(k)), -np.inf, dtype=np.float64)
        valid_mask = np.zeros((len(rows), int(k)), dtype=bool)

        for count in np.unique(returned_counts):
            count = int(count)
            if count == 0:
                continue
            batch_rows = np.flatnonzero(returned_counts == count)
            source = self._recommendation_source(
                [rows[row] for row in batch_rows],
                vocabulary=source_vocabulary,
            )
            if not use_reporting_path:
                predictions = self._predict_identified(
                    source,
                    k=count,
                    exclude_seen=exclude_seen,
                    candidate_ids=selected_ids,
                )
            else:
                predictions = self._predict_identified_with_reporting(
                    source,
                    k=count,
                    exclude_seen=exclude_seen,
                    candidate_ids=selected_ids,
                    reporter=reporter,
                )
            if (
                predictions.rows != batch_rows.size
                or predictions.cols_total != candidate_vocabulary.n_items
            ):
                raise ValueError(
                    "identified prediction shape does not match the "
                    "recommendation source and candidate catalog"
                )
            columns = predictions.cols.detach().cpu().numpy()
            values = predictions.vals.detach().cpu().numpy()
            item_ids[batch_rows, :count] = candidate_vocabulary.item_ids[columns]
            scores[batch_rows, :count] = values
            valid_mask[batch_rows, :count] = True

        return Recommendations(
            item_ids=item_ids,
            scores=scores,
            valid_mask=valid_mask,
        )
