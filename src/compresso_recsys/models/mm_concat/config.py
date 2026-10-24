"""Published settings for the concatenation wrapper, as validated dataclasses."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from compresso_recsys.models.core.cold_start import BaseColdStartRecommender


class _FrozenWeights(Mapping[str, float]):
    """Small immutable mapping supporting hashing, copying, and pickling."""

    __slots__ = ("_pairs",)

    def __init__(self, weights: Mapping[str, float]) -> None:
        object.__setattr__(self, "_pairs", tuple(sorted(weights.items())))

    def __getitem__(self, key: str) -> float:
        for name, value in self._pairs:
            if name == key:
                return value
        raise KeyError(key)

    def __iter__(self) -> Iterator[str]:
        return (name for name, _ in self._pairs)

    def __len__(self) -> int:
        return len(self._pairs)

    def __hash__(self) -> int:
        return hash(self._pairs)

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError("weights are immutable")

    def __delattr__(self, name: str) -> None:
        raise AttributeError("weights are immutable")

    def __reduce__(self):
        return type(self), (dict(self._pairs),)

    def __repr__(self) -> str:
        return repr(dict(self._pairs))


@dataclass(frozen=True)
class MMConcatWrapperConfig:
    """Configure feature concatenation and construction of the inner model.

    ``model`` is a cold-start model/trainer class constructed with
    ``model_config`` as its sole positional argument (or no arguments if None).
    ``modalities`` fixes the block order. Unknown keys in features and
    ``modality_masks`` are rejected by default. Set ``extra_modalities="ignore"``
    to select modalities from larger feature and mask mappings; unused keys
    are then ignored, including misspellings.
    ``fit_features_parameter`` identifies the inner fit argument to transform,
    whether supplied positionally or by keyword. Other fit arguments pass through.

    ``missing`` handles explicitly unavailable rows: reject them, fill with
    zeros, or impute the mean of available training rows. Zero vectors alone do
    not imply missingness. Every selected modality must be present at first fit
    to establish its width; later calls may omit entire modalities.

    Imputation precedes optional per-modality row L2 normalization, followed by
    multiplication by ``weights`` (unspecified weights are 1). These are feature
    multipliers, not promised contributions to an inner model's scores.
    ``dtype`` controls concatenation; the inner model controls its own precision.

    Weights are copied into an immutable mapping compatible with ``asdict``,
    copying, and pickling. Those operations still depend on the supplied model
    class/config supporting them; hashing also requires a hashable model config.
    """

    model: type[BaseColdStartRecommender]
    modalities: Sequence[str]
    model_config: Any = None
    fit_features_parameter: str = "item_features"
    missing: Literal["error", "zero", "mean"] = "error"
    normalize: bool = False
    weights: Mapping[str, float] | None = None
    dtype: Literal["float32", "float64"] = "float32"
    extra_modalities: Literal["error", "ignore"] = "error"

    def __post_init__(self) -> None:
        if not isinstance(self.model, type) or not issubclass(
            self.model, BaseColdStartRecommender
        ):
            raise TypeError("model must be a BaseColdStartRecommender class")
        if isinstance(self.modalities, str):
            raise TypeError("modalities must be a sequence of names, not a string")
        names = tuple(self.modalities)
        if not names or any(not isinstance(n, str) or not n.strip() for n in names):
            raise ValueError("modalities must contain non-empty names")
        if len(set(names)) != len(names):
            raise ValueError("modalities must not contain duplicate names")
        if (
            not isinstance(self.fit_features_parameter, str)
            or not self.fit_features_parameter.isidentifier()
            or self.fit_features_parameter in {"modality_masks", "feature_fit_indices"}
        ):
            raise ValueError(
                "fit_features_parameter must be an unreserved parameter name"
            )
        if self.missing not in {"error", "zero", "mean"}:
            raise ValueError("missing must be 'error', 'zero', or 'mean'")
        if self.extra_modalities not in {"error", "ignore"}:
            raise ValueError("extra_modalities must be 'error' or 'ignore'")
        if not isinstance(self.normalize, bool):
            raise TypeError("normalize must be a bool")
        if self.dtype not in {"float32", "float64"}:
            raise ValueError("dtype must be float32 or float64")
        weights = (
            {}
            if self.weights is None
            else {name: float(value) for name, value in self.weights.items()}
        )
        if set(weights) - set(names):
            raise ValueError("weights must refer to selected modalities")
        if any(not np.isfinite(w) or w < 0 for w in weights.values()):
            raise ValueError("weights must be finite and non-negative")
        if not any(weights.get(n, 1.0) > 0 for n in names):
            raise ValueError("at least one modality weight must be positive")
        object.__setattr__(self, "modalities", names)
        object.__setattr__(self, "weights", _FrozenWeights(weights))
