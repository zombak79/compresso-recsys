from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, is_dataclass, replace
import inspect
from pathlib import Path
import re
import time
from typing import (
    Any,
    ClassVar,
    Hashable,
    Literal,
    Protocol,
    Sequence,
    TypeVar,
    runtime_checkable,
)

import numpy as np
import torch
from scipy.sparse import csr_matrix
from torch import nn

from compresso import SRPTensor
from compresso_recsys._reporting import (
    _INHERIT,
    _Inherit,
    _Reporter,
    _format_duration,
    _resolve_reporter,
)
from compresso_recsys.checkpoint import (
    load_manifest,
    read_checkpoint,
    save_manifest,
    update_checkpoint,
)
from compresso_recsys.persistence import (
    ModelCheckpointReader,
    ModelCheckpointWriter,
)
from compresso_recsys.sequences import ItemSequences
from compresso_recsys.models._validation import canonical_csr
from compresso_recsys.models.identifiers import ItemVocabulary, Recommendations

__all__ = [
    "BasePersistableRecommender",
    "BaseIdentifiedRecommender",
    "BaseCollaborativeRecommender",
    "BaseSequentialRecommender",
    "IdentifiedRecommender",
    "PersistableRecommender",
    "Recommender",
    "SequentialRecommender",
]

_PersistableT = TypeVar("_PersistableT", bound="BasePersistableRecommender")
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


@runtime_checkable
class PersistableRecommender(Protocol):
    """A fitted recommender with the package model-checkpoint API."""

    def to(
        self,
        device: str | torch.device,
    ) -> "PersistableRecommender": ...

    def save(
        self,
        path: str | Path,
        *,
        include_optimizer: bool = False,
    ) -> None: ...

    def save_to_checkpoint(
        self,
        checkpoint_path: str | Path,
        name: str,
        *,
        include_optimizer: bool = False,
    ) -> None: ...

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        device: str | torch.device = "cpu",
        load_optimizer: bool = False,
    ) -> "PersistableRecommender": ...

    @classmethod
    def load_from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        name: str,
        *,
        device: str | torch.device = "cpu",
        load_optimizer: bool = False,
    ) -> "PersistableRecommender": ...


@runtime_checkable
class IdentifiedRecommender(Protocol):
    """A recommender accepting histories and filters as stable item IDs."""

    def recommend(
        self,
        histories: Sequence[Sequence[Hashable]],
        *,
        k: int = 100,
        exclude_seen: bool = False,
        allowlist: Sequence[Hashable] | np.ndarray | None = None,
        blocklist: Sequence[Hashable] | np.ndarray | None = None,
        on_insufficient: Literal["truncate", "raise"] = "truncate",
    ) -> Recommendations: ...


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


class BasePersistableRecommender(BaseIdentifiedRecommender):
    """Common fitted-model persistence workflow.

    The base owns the versioned archive, configuration, Torch state, device
    routing and optional optimizer state. Subclasses describe construction and
    any state that does not naturally live in a Torch ``state_dict``.
    """

    checkpoint_type: ClassVar[str]

    @property
    @abstractmethod
    def is_fitted(self) -> bool:
        """Whether the recommender is ready to save and predict."""

    def _checkpoint_config(self) -> dict[str, Any]:
        config = getattr(self, "cfg", None)
        if config is None or not is_dataclass(config):
            raise NotImplementedError(
                f"{type(self).__name__} must implement _checkpoint_config()"
            )
        return asdict(config)

    @classmethod
    def _from_checkpoint_config(
        cls: type[_PersistableT],
        config: dict[str, Any],
        reader: ModelCheckpointReader,
        *,
        device: torch.device,
    ) -> _PersistableT:
        """Construct the model shape before learned state is installed."""
        del config, reader, device
        raise NotImplementedError(
            f"{cls.__name__} must implement _from_checkpoint_config()"
        )

    def _checkpoint_module(self) -> nn.Module | None:
        """Torch module whose state is learned, if this recommender has one."""
        return self if isinstance(self, nn.Module) else None

    def _checkpoint_optimizer(self) -> torch.optim.Optimizer | None:
        optimizer = getattr(self, "optimizer", None)
        return optimizer if isinstance(optimizer, torch.optim.Optimizer) else None

    def _prepare_checkpoint_module_state(
        self,
        state: dict[str, Any],
    ) -> None:
        """Adjust dynamic module structure before loading its state dictionary."""

    def _save_checkpoint_state(self, writer: ModelCheckpointWriter) -> None:
        """Write non-module fitted state."""

    def _load_checkpoint_state(self, reader: ModelCheckpointReader) -> None:
        """Restore non-module fitted state."""

    def _build_checkpoint_optimizer(self) -> None:
        """Construct the optimizer before optional optimizer state is loaded."""

    def _finish_checkpoint_load(self) -> None:
        """Restore derived inference state after the checkpoint is installed."""

    def _move_checkpoint_state(self, device: torch.device) -> None:
        """Move non-module tensors or clear device-specific caches."""

    @staticmethod
    def _move_optimizer_value(value: Any, device: torch.device) -> Any:
        if isinstance(value, torch.Tensor):
            return value.to(device)
        if isinstance(value, dict):
            return {
                key: BasePersistableRecommender._move_optimizer_value(item, device)
                for key, item in value.items()
            }
        if isinstance(value, list):
            return [
                BasePersistableRecommender._move_optimizer_value(item, device)
                for item in value
            ]
        if isinstance(value, tuple):
            return tuple(
                BasePersistableRecommender._move_optimizer_value(item, device)
                for item in value
            )
        return value

    def to(self: _PersistableT, device: str | torch.device) -> _PersistableT:
        """Move this recommender's Torch state to ``device`` and return ``self``."""
        resolved_device = torch.device(device)
        module = self._checkpoint_module()
        if module is None and not hasattr(self, "device"):
            raise TypeError(
                f"{type(self).__name__} has no device-backed state to move"
            )
        if module is not None:
            nn.Module.to(module, resolved_device)

        self.device = resolved_device
        config = getattr(self, "cfg", None)
        if config is not None and hasattr(config, "device"):
            if is_dataclass(config):
                self.cfg = replace(config, device=str(resolved_device))
            else:
                config.device = str(resolved_device)

        optimizer = self._checkpoint_optimizer()
        if optimizer is not None:
            for state in optimizer.state.values():
                for key, value in state.items():
                    state[key] = self._move_optimizer_value(value, resolved_device)

        self._move_checkpoint_state(resolved_device)
        return self

    def save(
        self,
        path: str | Path,
        *,
        include_optimizer: bool = False,
    ) -> None:
        """Persist this fitted recommender as a safe, versioned ZIP checkpoint."""
        if not self.is_fitted:
            raise RuntimeError(
                f"{type(self).__name__} must be fitted before saving"
            )
        model_type = getattr(type(self), "checkpoint_type", None)
        if not isinstance(model_type, str) or not model_type:
            raise RuntimeError(
                f"{type(self).__name__} does not declare checkpoint_type"
            )
        optimizer = self._checkpoint_optimizer()
        if include_optimizer and optimizer is None:
            raise ValueError(
                f"{type(self).__name__} has no optimizer state to save"
            )

        with ModelCheckpointWriter(
            path,
            model_type=model_type,
            optimizer_included=include_optimizer,
        ) as writer:
            writer.write_json("config.json", self._checkpoint_config())
            self._save_checkpoint_common_state(writer)
            module = self._checkpoint_module()
            if module is not None:
                writer.write_torch(
                    "state/model.pt",
                    _unwrapped_module(module).state_dict(),
                )
            self._save_checkpoint_state(writer)
            if include_optimizer:
                assert optimizer is not None
                writer.write_torch("state/optimizer.pt", optimizer.state_dict())

    def save_to_checkpoint(
        self,
        checkpoint_path: str | Path,
        name: str,
        *,
        include_optimizer: bool = False,
    ) -> None:
        """Save this model under ``models/<name>.zip`` in a data checkpoint."""
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.is_file():
            raise FileNotFoundError(checkpoint_path)
        model_type = getattr(type(self), "checkpoint_type", None)
        if not isinstance(model_type, str) or not model_type:
            raise RuntimeError(
                f"{type(self).__name__} does not declare checkpoint_type"
            )

        with update_checkpoint(checkpoint_path) as root:
            destination = _embedded_model_path(root, name)
            manifest = load_manifest(root)
            models = manifest.setdefault("models", {})
            if not isinstance(models, dict):
                raise ValueError("checkpoint manifest models must be an object")
            existing = models.get(name)
            if existing is not None:
                if not isinstance(existing, dict):
                    raise ValueError(
                        f"checkpoint manifest model {name!r} must be an object"
                    )
                existing_type = existing.get("model_type")
                if existing_type != model_type:
                    raise ValueError(
                        f"checkpoint model {name!r} contains type "
                        f"{existing_type!r}, not {model_type!r}"
                    )
            if destination.exists():
                with ModelCheckpointReader(
                    destination,
                    expected_model_type=model_type,
                ):
                    pass

            self.save(destination, include_optimizer=include_optimizer)
            models[name] = {
                "path": f"{_MODELS_DIR}/{name}.zip",
                "model_type": model_type,
                "optimizer_included": bool(include_optimizer),
            }
            save_manifest(root, manifest)

    @classmethod
    def load(
        cls: type[_PersistableT],
        path: str | Path,
        *,
        device: str | torch.device = "cpu",
        load_optimizer: bool = False,
    ) -> _PersistableT:
        """Load a fitted, prediction-ready recommender on ``device``."""
        model_type = getattr(cls, "checkpoint_type", None)
        if not isinstance(model_type, str) or not model_type:
            raise RuntimeError(f"{cls.__name__} does not declare checkpoint_type")
        resolved_device = torch.device(device)
        with ModelCheckpointReader(
            path,
            expected_model_type=model_type,
        ) as reader:
            if load_optimizer and not reader.optimizer_included:
                raise ValueError("checkpoint does not contain optimizer state")
            config = reader.read_json("config.json")
            model = cls._from_checkpoint_config(
                config,
                reader,
                device=resolved_device,
            )
            module = model._checkpoint_module()
            if module is not None:
                state = reader.read_torch(
                    "state/model.pt",
                    device=resolved_device,
                )
                model._prepare_checkpoint_module_state(state)
                module = model._checkpoint_module()
                if module is None:  # pragma: no cover - defensive invariant
                    raise RuntimeError("checkpoint preparation removed the model")
                _unwrapped_module(module).load_state_dict(state, strict=True)
                _unwrapped_module(module).eval()
            elif reader.exists("state/model.pt"):
                raise ValueError(
                    f"checkpoint contains Torch state but {cls.__name__} did not "
                    "construct a Torch module"
                )
            model._load_checkpoint_state(reader)
            model._load_checkpoint_common_state(reader)
            if load_optimizer:
                model._build_checkpoint_optimizer()
                optimizer = model._checkpoint_optimizer()
                if optimizer is None:
                    raise ValueError(
                        f"{cls.__name__} cannot restore optimizer state"
                    )
                optimizer.load_state_dict(
                    reader.read_torch(
                        "state/optimizer.pt",
                        device=resolved_device,
                    )
                )
            model._finish_checkpoint_load()
            if not model.is_fitted:
                raise ValueError(
                    f"checkpoint did not restore a fitted {cls.__name__}"
                )
            return model

    @classmethod
    def load_from_checkpoint(
        cls: type[_PersistableT],
        checkpoint_path: str | Path,
        name: str,
        *,
        device: str | torch.device = "cpu",
        load_optimizer: bool = False,
    ) -> _PersistableT:
        """Load ``models/<name>.zip`` from a data checkpoint."""
        checkpoint_path = Path(checkpoint_path)
        with read_checkpoint(checkpoint_path) as root:
            source = _embedded_model_path(root, name)
            if not source.is_file():
                raise FileNotFoundError(
                    f"model {name!r} is not stored in checkpoint "
                    f"{str(checkpoint_path)!r}"
                )
            return cls.load(
                source,
                device=device,
                load_optimizer=load_optimizer,
            )


@runtime_checkable
class Recommender(Protocol):
    """A fitted recommender that produces ranked predictions for one batch."""

    def predict_on_batch(
        self,
        source: csr_matrix,
        *,
        k: int,
        exclude_seen: bool = True,
        candidate_ids: Sequence[Hashable] | np.ndarray | None = None,
    ) -> SRPTensor:
        """Return top-``k`` predictions, optionally excluding source items."""


class BaseCollaborativeRecommender(BasePersistableRecommender):
    """Reusable base for fixed-catalog collaborative recommenders.

    Implementors provide :meth:`fit`, :attr:`is_fitted`, :attr:`n_items`, and
    :meth:`predict_on_batch`. The base validates source matrices and supplies a
    memory-bounded :meth:`predict` implementation that concatenates ranked
    batches without materializing a complete score matrix.
    """

    @property
    @abstractmethod
    def is_fitted(self) -> bool:
        """Whether the model is ready for prediction."""

    @property
    @abstractmethod
    def n_items(self) -> int | None:
        """Number of fitted item columns, or ``None`` before fitting."""

    @abstractmethod
    def fit(
        self,
        interactions: csr_matrix,
        *,
        item_ids: Sequence[Hashable] | np.ndarray | None = None,
    ) -> BaseCollaborativeRecommender:
        """Fit the model from a user-item CSR interaction matrix."""

    @abstractmethod
    def predict_on_batch(
        self,
        source: csr_matrix,
        *,
        k: int,
        exclude_seen: bool = True,
        candidate_ids: Sequence[Hashable] | np.ndarray | None = None,
    ) -> SRPTensor:
        """Return ranked predictions for one source batch."""

    def _recommendation_source(
        self,
        rows: list[np.ndarray],
        *,
        vocabulary: ItemVocabulary,
    ) -> csr_matrix:
        lengths = np.fromiter((row.size for row in rows), dtype=np.int64)
        row_indices = np.repeat(np.arange(len(rows), dtype=np.int64), lengths)
        columns = (
            np.concatenate(rows)
            if rows
            else np.empty(0, dtype=np.int64)
        )
        source = csr_matrix(
            (
                np.ones(columns.size, dtype=np.float32),
                (row_indices, columns),
            ),
            shape=(len(rows), vocabulary.n_items),
        )
        source.sum_duplicates()
        source.data.fill(1.0)
        return source

    def _predict_identified(
        self,
        source: csr_matrix | ItemSequences,
        *,
        k: int,
        exclude_seen: bool,
        candidate_ids: np.ndarray,
    ) -> SRPTensor:
        if not isinstance(source, csr_matrix):
            raise TypeError("collaborative recommendations require a CSR source")
        return self.predict(
            source,
            k=k,
            exclude_seen=exclude_seen,
            candidate_ids=candidate_ids,
        )

    def _predict_identified_with_reporting(
        self,
        source: csr_matrix | ItemSequences,
        *,
        k: int,
        exclude_seen: bool,
        candidate_ids: np.ndarray,
        reporter: _Reporter,
    ) -> SRPTensor:
        if not isinstance(source, csr_matrix):
            raise TypeError("collaborative recommendations require a CSR source")
        predict = self.predict
        if not _accepts_reporting_keywords(predict):
            # Fall back to base batching for extensions overriding the released
            # predict() signature without a logger keyword.
            predict = BaseCollaborativeRecommender.predict.__get__(self)
        return predict(
            source,
            k=k,
            exclude_seen=exclude_seen,
            candidate_ids=candidate_ids,
            logger=reporter,
            show_progress=_INHERIT,
        )

    def _prepare_source(self, source: csr_matrix) -> csr_matrix:
        """Validate a source matrix against the fitted item catalog."""
        if not self.is_fitted or self.n_items is None:
            raise RuntimeError(
                f"{type(self).__name__} must be fitted before prediction"
            )
        source = canonical_csr(source, name="source")
        if source.shape[1] != self.n_items:
            raise ValueError(
                f"source has {source.shape[1]} items, but "
                f"{type(self).__name__} was fitted with {self.n_items} items"
            )
        return source

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
        """Predict all source rows by repeatedly calling ``predict_on_batch``."""
        reporter = self._prediction_reporter(logger, show_progress)
        source = self._prepare_source(source)
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        candidate_count = int(self._candidate_rows(candidate_ids).size)
        if not 1 <= int(k) <= candidate_count:
            raise ValueError(f"k must be in [1, {candidate_count}], got {k}")

        columns: list[torch.Tensor] = []
        values: list[torch.Tensor] = []
        starts = range(0, source.shape[0], batch_size)
        steps = len(starts)
        started = time.monotonic()
        reporter.log(
            f"predict@{k} started: {source.shape[0]} rows | "
            f"{steps} batches of {batch_size}"
        )
        for step, start in enumerate(
            reporter.wrap(
                starts,
                total=steps,
                desc=f"{type(self).__name__} predict@{k}",
            ),
            start=1,
        ):
            kwargs = (
                {}
                if candidate_ids is None
                else {"candidate_ids": candidate_ids}
            )
            result = self.predict_on_batch(
                source[start : start + batch_size],
                k=k,
                exclude_seen=exclude_seen,
                **kwargs,
            )
            if result.cols_total != source.shape[1]:
                raise ValueError(
                    "predict_on_batch() item count must match the fitted catalog"
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
            kwargs = (
                {}
                if candidate_ids is None
                else {"candidate_ids": candidate_ids}
            )
            prediction = self.predict_on_batch(
                source,
                k=k,
                exclude_seen=exclude_seen,
                **kwargs,
            )
        else:
            prediction = SRPTensor(
                cols=torch.vstack(columns),
                vals=torch.vstack(values),
                shape=source.shape,
                validate=False,
            )
        reporter.log(
            f"predict@{k} finished: "
            f"{_format_duration(time.monotonic() - started)} total | "
            f"{source.shape[0]} rows"
        )
        return prediction


@runtime_checkable
class SequentialRecommender(Protocol):
    """A fitted recommender that ranks from chronological histories.

    The same contract as :class:`Recommender` with a different source type. Kept
    structural, like its sibling, so a model satisfies it by having the method
    rather than by inheriting anything.
    """

    def predict_on_batch(
        self,
        source: ItemSequences,
        *,
        k: int,
        exclude_seen: bool = True,
        candidate_ids: Sequence[Hashable] | np.ndarray | None = None,
    ) -> SRPTensor:
        """Return top-``k`` predictions, optionally excluding source items."""


class BaseSequentialRecommender(BasePersistableRecommender):
    """Reusable base for recommenders that read chronological histories.

    Parallel to :class:`BaseCollaborativeRecommender` rather than derived from
    it. The two differ only in how a user's history arrives — a CSR row of items
    interacted with, or an ordered history that keeps repeats — and crossing that
    with cold-start capability in the type hierarchy would give four classes for
    two ideas. Candidate capability is composed instead: a model that scores
    unseen items owns a catalog rather than inheriting one.

    Implementors provide :attr:`is_fitted`, :attr:`n_items`, and
    :meth:`predict_on_batch`. ``fit`` is deliberately absent from the contract:
    trainers follow the package's existing shape, where
    ``SomeTrainer(config).fit(data)`` returns a fitted model and the model owes
    only the prediction contract.

    Two properties this base is careful not to assume.

    **The source vocabulary need not equal the candidate catalog.**
    :attr:`n_items` describes what can be *scored*. A history may be expressed
    over a different, usually smaller, vocabulary — a truncated context, a
    hashed one — and a cold-capable model scores candidates that never appear in
    any history at all. Nothing here compares the two.

    **Truncation is not exclusion.** ``exclude_seen=True`` must mask every item
    in the *full* history handed to it, even where the encoder reads only a
    suffix. A model that attends to the last 200 interactions must still refuse
    to recommend the 201st.
    """

    @property
    @abstractmethod
    def is_fitted(self) -> bool:
        """Whether the model is ready for prediction."""

    @property
    @abstractmethod
    def n_items(self) -> int | None:
        """Number of scoreable candidates, or ``None`` before fitting."""

    @abstractmethod
    def predict_on_batch(
        self,
        source: ItemSequences,
        *,
        k: int,
        exclude_seen: bool = True,
        candidate_ids: Sequence[Hashable] | np.ndarray | None = None,
    ) -> SRPTensor:
        """Return ranked predictions for one batch of histories."""

    def _recommendation_source(
        self,
        rows: list[np.ndarray],
        *,
        vocabulary: ItemVocabulary,
    ) -> ItemSequences:
        return ItemSequences.from_rows(rows, n_items=vocabulary.n_items)

    def _predict_identified(
        self,
        source: csr_matrix | ItemSequences,
        *,
        k: int,
        exclude_seen: bool,
        candidate_ids: np.ndarray,
    ) -> SRPTensor:
        if not isinstance(source, ItemSequences):
            raise TypeError(
                "sequential recommendations require an ItemSequences source"
            )
        return self.predict(
            source,
            k=k,
            exclude_seen=exclude_seen,
            candidate_ids=candidate_ids,
        )

    def _predict_identified_with_reporting(
        self,
        source: csr_matrix | ItemSequences,
        *,
        k: int,
        exclude_seen: bool,
        candidate_ids: np.ndarray,
        reporter: _Reporter,
    ) -> SRPTensor:
        if not isinstance(source, ItemSequences):
            raise TypeError(
                "sequential recommendations require an ItemSequences source"
            )
        predict = self.predict
        if not _accepts_reporting_keywords(predict):
            predict = BaseSequentialRecommender.predict.__get__(self)
        return predict(
            source,
            k=k,
            exclude_seen=exclude_seen,
            candidate_ids=candidate_ids,
            logger=reporter,
            show_progress=_INHERIT,
        )

    def _prepare_source(self, source: ItemSequences) -> ItemSequences:
        """Check a batch of histories against the fitted model."""
        if not self.is_fitted or self.n_items is None:
            raise RuntimeError(
                f"{type(self).__name__} must be fitted before prediction"
            )
        if not isinstance(source, ItemSequences):
            raise TypeError(
                f"{type(self).__name__} predicts from ItemSequences, got "
                f"{type(source).__name__}"
            )
        return source

    @staticmethod
    def _check_unseen_capacity(
        source: ItemSequences,
        *,
        n_items: int,
        k: int,
        candidate_rows: np.ndarray | None = None,
    ) -> None:
        """Require every row to contain at least ``k`` scoreable unseen items."""
        selected = (
            np.ones(n_items, dtype=bool)
            if candidate_rows is None
            else np.zeros(n_items, dtype=bool)
        )
        if candidate_rows is not None:
            selected[candidate_rows] = True
        candidate_count = int(selected.sum())
        for row in range(source.n_rows):
            history = source.row(row)
            scoreable = history[history < n_items]
            seen = np.unique(scoreable)
            available = candidate_count - int(selected[seen].sum())
            if available < k:
                raise ValueError(
                    f"source row {row} has only {available} unseen items, "
                    f"fewer than k={k}"
                )

    def predict(
        self,
        source: ItemSequences,
        *,
        k: int = 100,
        batch_size: int = 1024,
        exclude_seen: bool = True,
        candidate_ids: Sequence[Hashable] | np.ndarray | None = None,
        logger: Any | None = _INHERIT,
        show_progress: bool | None | _Inherit = _INHERIT,
    ) -> SRPTensor:
        """Predict all histories by repeatedly calling ``predict_on_batch``."""
        reporter = self._prediction_reporter(logger, show_progress)
        source = self._prepare_source(source)
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        candidate_count = int(self._candidate_rows(candidate_ids).size)
        if not 1 <= int(k) <= candidate_count:
            raise ValueError(f"k must be in [1, {candidate_count}], got {k}")

        columns: list[torch.Tensor] = []
        values: list[torch.Tensor] = []
        starts = range(0, source.n_rows, batch_size)
        steps = len(starts)
        started = time.monotonic()
        reporter.log(
            f"predict@{k} started: {source.n_rows} rows | "
            f"{steps} batches of {batch_size}"
        )
        for step, start in enumerate(
            reporter.wrap(
                starts,
                total=steps,
                desc=f"{type(self).__name__} predict@{k}",
            ),
            start=1,
        ):
            kwargs = (
                {}
                if candidate_ids is None
                else {"candidate_ids": candidate_ids}
            )
            result = self.predict_on_batch(
                source.take_rows(start, start + batch_size),
                k=k,
                exclude_seen=exclude_seen,
                **kwargs,
            )
            if result.cols_total != self.n_items:
                raise ValueError(
                    "predict_on_batch() item count must match the candidate catalog"
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
            kwargs = (
                {}
                if candidate_ids is None
                else {"candidate_ids": candidate_ids}
            )
            prediction = self.predict_on_batch(
                source,
                k=k,
                exclude_seen=exclude_seen,
                **kwargs,
            )
        else:
            prediction = SRPTensor(
                cols=torch.vstack(columns),
                vals=torch.vstack(values),
                shape=(source.n_rows, self.n_items),
            )
        reporter.log(
            f"predict@{k} finished: "
            f"{_format_duration(time.monotonic() - started)} total | "
            f"{source.n_rows} rows"
        )
        return prediction
