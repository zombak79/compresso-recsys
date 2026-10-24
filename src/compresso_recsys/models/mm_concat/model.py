"""The wrapper itself: one concatenated matrix out of many named ones."""

from __future__ import annotations

import inspect
from collections.abc import Hashable, Mapping, Sequence
from dataclasses import replace
from itertools import pairwise
from pathlib import Path
from types import MappingProxyType
from typing import Any, ClassVar

import numpy as np
import torch
from scipy.sparse import csr_matrix, hstack

from compresso_recsys.models.core.validation import canonical_train_item_indices
from compresso_recsys.models.core.catalog import CandidateCatalog
from compresso_recsys.models.core.mutable_catalog import MutableCandidateCatalog
from compresso_recsys.models.core.cold_start import BaseColdStartRecommender
from compresso_recsys.models.core.features import canonical_item_features
from compresso_recsys.models.core.multimodal import (
    BaseMultiModalRecommender,
    MultiModalItemFeatures,
)
from compresso_recsys.persistence import ModelCheckpointReader, ModelCheckpointWriter
from compresso_recsys.models.mm_concat.config import MMConcatWrapperConfig

_Matrix = csr_matrix | np.ndarray


def _argument(bound: inspect.BoundArguments, name: str) -> tuple[dict, str] | None:
    """Locate one ordinary argument or a named entry inside **kwargs."""
    parameter = bound.signature.parameters.get(name)
    if parameter is not None and parameter.kind not in {
        inspect.Parameter.VAR_POSITIONAL,
        inspect.Parameter.VAR_KEYWORD,
    }:
        return (bound.arguments, name) if name in bound.arguments else None
    for parameter in bound.signature.parameters.values():
        if parameter.kind is inspect.Parameter.VAR_KEYWORD:
            keywords = bound.arguments.get(parameter.name, {})
            if name in keywords:
                return keywords, name
    return None


def _impute_csr_rows(
    matrix: csr_matrix,
    missing: np.ndarray,
    mean: np.ndarray,
) -> csr_matrix:
    """Replace rows of a canonical CSR matrix using only final CSR buffers.

    Copy observed rows in contiguous runs and broadcast the nonzero mean entries
    directly into the destination buffers for missing runs. No repeated filler,
    coordinate matrix, or matrix product is needed. A dense mean still requires
    one stored value per feature in every missing row.
    """
    mean_columns = np.flatnonzero(mean)
    mean_values = mean[mean_columns]
    row_sizes = np.diff(matrix.indptr).astype(np.int64, copy=False)
    row_sizes[missing] = mean_columns.size
    nnz = int(row_sizes.sum())
    index_dtype = (
        np.int64 if max(*matrix.shape, nnz) > np.iinfo(np.int32).max else np.int32
    )
    indptr = np.empty(matrix.shape[0] + 1, dtype=index_dtype)
    indptr[0] = 0
    np.cumsum(row_sizes, dtype=index_dtype, out=indptr[1:])
    indices = np.empty(nnz, dtype=index_dtype)
    values = np.empty(nnz, dtype=matrix.dtype)

    boundaries = np.concatenate(
        ([0], np.flatnonzero(missing[1:] != missing[:-1]) + 1, [matrix.shape[0]])
    )
    for first, last in pairwise(boundaries):
        start, stop = indptr[first], indptr[last]
        if missing[first]:
            if mean_columns.size:
                indices[start:stop].reshape(-1, mean_columns.size)[:] = mean_columns
                values[start:stop].reshape(-1, mean_columns.size)[:] = mean_values
        else:
            source = slice(matrix.indptr[first], matrix.indptr[last])
            indices[start:stop] = matrix.indices[source]
            values[start:stop] = matrix.data[source]

    return csr_matrix((values, indices, indptr), shape=matrix.shape, copy=False)


class MMConcatWrapper(BaseMultiModalRecommender):
    """Concatenate named matrices and delegate learning/scoring to a cold model.

    The inner model owns the only candidate catalog, containing its normal
    stored matrix representation. Use this wrapper's candidate mutation methods
    to preprocess raw modalities. ``inner_model`` exposes model-specific APIs.
    Source/history support, filters, prediction batching, and publication
    semantics remain those of the inner model.

    Refitting constructs a fresh inner model and publishes it only after fitting
    succeeds. Means use ``feature_fit_indices``, otherwise the forwarded
    ``train_item_indices``, otherwise all rows. Callers with a differently named
    training subset must supply ``feature_fit_indices`` explicitly.
    """

    checkpoint_type = "mm_concat_wrapper"
    _registered_models: ClassVar[dict[str, type[BaseColdStartRecommender]]] = {}

    def __init__(self, config: MMConcatWrapperConfig) -> None:
        # This specialization owns no second catalog: the inner model owns it.
        self.cfg = config
        self.inner_model = self._new_inner()
        self._feature_dims: dict[str, int] = {}
        self._means: dict[str, np.ndarray] = {}
        self._fit_sparse_output = False
        self.register_model(config.model)

    def _new_inner(self) -> BaseColdStartRecommender:
        if self.cfg.model_config is None:
            return self.cfg.model()
        return self.cfg.model(self.cfg.model_config)

    @property
    def candidates(self) -> MutableCandidateCatalog:  # type: ignore[override]
        return self.inner_model.candidates

    @property
    def device(self) -> torch.device:
        """The current inner model's device, when the inner model exposes one."""
        return self.inner_model.device

    @property
    def feature_dims_(self) -> Mapping[str, int]:
        """Fitted widths of the selected input modalities."""
        return MappingProxyType(self._feature_dims)

    @property
    def is_fitted(self) -> bool:
        return bool(self._feature_dims) and self.inner_model.is_fitted

    def _inputs(
        self,
        features: MultiModalItemFeatures,
        masks: Mapping[str, np.ndarray] | None,
        *,
        dimensions: Mapping[str, int] | None,
        n_items: int | None = None,
    ) -> tuple[dict[str, _Matrix], dict[str, np.ndarray], int]:
        if not isinstance(features, Mapping):
            raise TypeError(
                "item features must be a mapping of modality names to matrices"
            )
        if masks is not None and not isinstance(masks, Mapping):
            raise TypeError(
                "modality_masks must be a mapping of names to boolean vectors"
            )
        masks = {} if masks is None else masks
        if self.cfg.extra_modalities == "error":
            selected = set(self.cfg.modalities)
            for label, values in (
                ("item_features", features),
                ("modality_masks", masks),
            ):
                unknown = set(values) - selected
                if unknown:
                    names = ", ".join(sorted(repr(name) for name in unknown))
                    raise ValueError(
                        f"{label} contains unknown modality keys: {names}; "
                        f"expected keys from {self.cfg.modalities!r}. "
                        "Set extra_modalities='ignore' to select from larger mappings."
                    )
        matrices, available = {}, {}
        for name in self.cfg.modalities:
            if name not in features:
                if dimensions is None:
                    raise ValueError(
                        f"fit requires modality {name!r} to establish its width"
                    )
                continue
            matrix = canonical_item_features(
                features[name], dtype=np.dtype(self.cfg.dtype)
            )
            values = matrix.data if isinstance(matrix, csr_matrix) else matrix
            if not np.all(np.isfinite(values)):
                raise ValueError(
                    f"modality {name!r} must remain finite in {self.cfg.dtype}"
                )
            if n_items is None:
                n_items = matrix.shape[0]
            if matrix.shape[0] != n_items:
                raise ValueError(f"modality {name!r} must have {n_items} rows")
            if dimensions is not None and matrix.shape[1] != dimensions[name]:
                raise ValueError(
                    f"modality {name!r} must have {dimensions[name]} columns"
                )
            mask = np.asarray(masks.get(name, np.ones(n_items, dtype=bool)))
            if mask.dtype != np.dtype(bool) or mask.shape != (n_items,):
                raise ValueError(
                    f"modality_masks[{name!r}] must be a boolean vector of length {n_items}"
                )
            matrices[name], available[name] = matrix, mask
        if n_items is None or n_items < 1:
            raise ValueError("at least one item and one selected modality are required")
        # Missing blocks must not introduce sparsity into otherwise dense input.
        # An entirely omitted input has no blocks to inspect, so reuse the format
        # of the concatenated features from the last successful fit.
        sparse_output = (
            any(isinstance(matrix, csr_matrix) for matrix in matrices.values())
            if matrices
            else self._fit_sparse_output
        )
        for name in self.cfg.modalities:
            if name not in matrices:
                if name in masks:
                    mask = np.asarray(masks[name])
                    if (
                        mask.dtype != np.dtype(bool)
                        or mask.shape != (n_items,)
                        or mask.any()
                    ):
                        raise ValueError(
                            f"absent modality {name!r} cannot have available rows"
                        )
                shape = (n_items, dimensions[name])
                matrices[name] = (
                    csr_matrix(shape, dtype=self.cfg.dtype)
                    if sparse_output
                    else np.zeros(shape, dtype=self.cfg.dtype)
                )
                available[name] = np.zeros(n_items, dtype=bool)
            if self.cfg.missing == "error" and not available[name].all():
                raise ValueError(
                    f"modality {name!r} has unavailable rows; set missing='zero' or 'mean'"
                )
        return matrices, available, n_items

    def _concatenate(
        self,
        matrices: Mapping[str, _Matrix],
        available: Mapping[str, np.ndarray],
        means: Mapping[str, np.ndarray],
    ) -> _Matrix:
        blocks = []
        for name in self.cfg.modalities:
            block = matrices[name]
            missing = ~available[name]
            if missing.any():
                if isinstance(block, csr_matrix):
                    # Preserve sparsity for zero imputation; means may add nonzeros.
                    if self.cfg.missing == "mean":
                        block = _impute_csr_rows(block, missing, means[name])
                    else:
                        block = block.multiply(available[name][:, None]).tocsr()
                else:
                    # Dense imputation writes in place; other operations allocate.
                    block = block.copy()
                    block[missing] = means[name] if self.cfg.missing == "mean" else 0
            if self.cfg.normalize:
                precise = block.astype(np.float64, copy=False)
                norm = (
                    np.sqrt(np.asarray(precise.multiply(precise).sum(axis=1)).ravel())
                    if isinstance(block, csr_matrix)
                    else np.linalg.norm(precise, axis=1)
                )
                scale = 1 / np.maximum(norm, 1e-12)
                block = (
                    block.multiply(scale[:, None]).tocsr()
                    if isinstance(block, csr_matrix)
                    else block * scale[:, None]
                )
            weight = self.cfg.weights.get(name, 1.0)
            if weight != 1.0:
                block = block * weight
            block = block.astype(self.cfg.dtype, copy=False)
            values = block.data if isinstance(block, csr_matrix) else block
            if not np.isfinite(values).all():
                raise ValueError(f"preprocessed modality {name!r} must be finite")
            blocks.append(block)
        if any(isinstance(b, csr_matrix) for b in blocks):
            result = hstack(blocks, format="csr", dtype=self.cfg.dtype)
            result.eliminate_zeros()
            return result
        return np.concatenate(blocks, axis=1)

    def fit(
        self,
        *args,
        modality_masks: Mapping[str, np.ndarray] | None = None,
        feature_fit_indices: Sequence[int] | np.ndarray | None = None,
        **kwargs,
    ) -> MMConcatWrapper:
        """Bind the inner fit signature, transform its features, and forward it.

        ``modality_masks`` and ``feature_fit_indices`` belong to the wrapper and
        are not forwarded. For an opaque ``fit(*args, **kwargs)``, pass features
        by the configured keyword; unnamed positional arguments cannot be inferred.
        """
        inner = self._new_inner()
        bound = inspect.signature(inner.fit).bind(*args, **kwargs)
        location = _argument(bound, self.cfg.fit_features_parameter)
        if location is None:
            raise TypeError(
                f"supply features using the inner fit parameter {self.cfg.fit_features_parameter!r}"
            )
        arguments, key = location
        matrices, available, n_items = self._inputs(
            arguments[key], modality_masks, dimensions=None
        )
        if feature_fit_indices is None:
            train_location = _argument(bound, "train_item_indices")
            if train_location is not None:
                feature_fit_indices = train_location[0][train_location[1]]
        rows = canonical_train_item_indices(feature_fit_indices, n_items=n_items)
        means = {}
        if self.cfg.missing == "mean":
            for name, matrix in matrices.items():
                observed = rows[available[name][rows]]
                if not observed.size:
                    raise ValueError(
                        f"modality {name!r} has no available training rows for mean imputation"
                    )
                means[name] = (
                    np.asarray(matrix[observed].astype(np.float64).mean(axis=0))
                    .ravel()
                    .astype(self.cfg.dtype)
                )
        concatenated = self._concatenate(matrices, available, means)
        arguments[key] = concatenated
        inner.fit(*bound.args, **bound.kwargs)
        if not inner.is_fitted:
            raise RuntimeError("inner fit returned without a fitted model")
        self.inner_model = inner
        self._feature_dims = {
            name: matrices[name].shape[1] for name in self.cfg.modalities
        }
        self._means = means
        self._fit_sparse_output = isinstance(concatenated, csr_matrix)
        return self

    def transform_features(
        self,
        item_features: MultiModalItemFeatures,
        *,
        modality_masks: Mapping[str, np.ndarray] | None = None,
    ) -> _Matrix:
        """Return concatenated features using the fitted schema and preprocessing."""
        return self._transform(item_features, modality_masks)

    def _transform(
        self,
        features: MultiModalItemFeatures,
        masks: Mapping[str, np.ndarray] | None,
        *,
        n_items: int | None = None,
    ) -> _Matrix:
        if not self.is_fitted:
            raise RuntimeError(
                "MMConcatWrapper must be fitted before transforming features"
            )
        matrices, available, _ = self._inputs(
            features, masks, dimensions=self._feature_dims, n_items=n_items
        )
        return self._concatenate(matrices, available, self._means)

    def build_candidates(
        self,
        *,
        item_ids: Sequence[Hashable] | np.ndarray,
        item_features: MultiModalItemFeatures,
        modality_masks: Mapping[str, np.ndarray] | None = None,
        **kwargs,
    ) -> CandidateCatalog:  # type: ignore[override]
        """Replace inner candidates after applying the fitted transformation."""
        features = self._transform(item_features, modality_masks, n_items=len(item_ids))
        return self.inner_model.build_candidates(
            item_ids=item_ids, item_features=features, **kwargs
        )

    def update_candidates(
        self,
        *,
        item_ids: Sequence[Hashable] | np.ndarray,
        item_features: MultiModalItemFeatures,
        modality_masks: Mapping[str, np.ndarray] | None = None,
        **kwargs,
    ) -> CandidateCatalog:  # type: ignore[override]
        """Register candidates without expanding the inner history vocabulary."""
        features = self._transform(item_features, modality_masks, n_items=len(item_ids))
        return self.inner_model.update_candidates(
            item_ids=item_ids, item_features=features, **kwargs
        )

    def remove_candidates(self, item_ids, **kwargs) -> CandidateCatalog:  # type: ignore[override]
        return self.inner_model.remove_candidates(item_ids, **kwargs)

    def align_source(self, source, **kwargs):
        return self.inner_model.align_source(source, **kwargs)

    def recommend(self, *args, **kwargs):
        return self.inner_model.recommend(*args, **kwargs)

    def predict(self, *args, **kwargs):
        return self.inner_model.predict(*args, **kwargs)

    def predict_on_batch(self, *args, **kwargs):
        return self.inner_model.predict_on_batch(*args, **kwargs)

    def to(self, device: str | torch.device) -> MMConcatWrapper:
        """Move the inner model and retain its config for subsequent refits."""
        self.inner_model = self.inner_model.to(device)
        self.cfg = replace(
            self.cfg,
            model_config=getattr(self.inner_model, "cfg", self.cfg.model_config),
        )
        return self

    @classmethod
    def register_model(cls, model: type[BaseColdStartRecommender]) -> None:
        """Allow a trusted custom model class when loading wrapper checkpoints.

        Built-ins are registered automatically. Construction also registers its
        class; a fresh process must register custom classes explicitly. Checkpoint
        contents never trigger dynamic imports or pickle deserialization.
        """
        if not isinstance(model, type) or not issubclass(
            model, BaseColdStartRecommender
        ):
            raise TypeError("model must be a BaseColdStartRecommender class")
        cls._registered_models[f"{model.__module__}.{model.__qualname__}"] = model

    def _checkpoint_config(self) -> dict[str, Any]:
        """Serialize wrapper settings; the inner checkpoint carries its config."""
        return {
            "model": f"{self.cfg.model.__module__}.{self.cfg.model.__qualname__}",
            "modalities": list(self.cfg.modalities),
            "fit_features_parameter": self.cfg.fit_features_parameter,
            "missing": self.cfg.missing,
            "normalize": self.cfg.normalize,
            "weights": dict(self.cfg.weights),
            "dtype": self.cfg.dtype,
            "extra_modalities": self.cfg.extra_modalities,
        }

    def save(self, path: str | Path, *, include_optimizer: bool = False) -> None:
        """Save preprocessing and an ordinary inner-model checkpoint together."""
        if not self.is_fitted:
            raise RuntimeError("MMConcatWrapper must be fitted before saving")
        with ModelCheckpointWriter(
            path, model_type=self.checkpoint_type, optimizer_included=include_optimizer
        ) as writer:
            writer.write_json("config.json", self._checkpoint_config())
            writer.write_json(
                "preprocessing.json",
                {
                    "feature_dims": self._feature_dims,
                    "fit_sparse_output": self._fit_sparse_output,
                },
            )
            for index, name in enumerate(self.cfg.modalities):
                if name in self._means:
                    writer.write_numpy(f"means/{index}.npy", self._means[name])
            self.inner_model.save(
                writer.root / "inner.zip", include_optimizer=include_optimizer
            )

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        device: str | torch.device = "cpu",
        load_optimizer: bool = False,
    ) -> MMConcatWrapper:
        """Restore preprocessing and the inner model, including candidate changes."""
        from compresso_recsys.models.content import ContentRecommender
        from compresso_recsys.models.teaser import TEASER
        from compresso_recsys.models.teaser_gd import TEASERGDTrainer

        for model in (ContentRecommender, TEASER, TEASERGDTrainer):
            cls.register_model(model)
        with ModelCheckpointReader(
            path, expected_model_type=cls.checkpoint_type
        ) as reader:
            config = reader.read_json("config.json")
            model_id = config.pop("model", None)
            if not isinstance(model_id, str) or not model_id.strip():
                raise ValueError("checkpoint model must be a non-empty registered ID")
            if model_id not in cls._registered_models:
                raise ValueError(
                    f"unregistered inner model {model_id!r}; call MMConcatWrapper.register_model first"
                )
            model_class = cls._registered_models[model_id]
            inner = model_class.load(
                reader.root / "inner.zip", device=device, load_optimizer=load_optimizer
            )
            wrapper = cls.__new__(cls)
            try:
                wrapper.cfg = MMConcatWrapperConfig(
                    model=model_class,
                    model_config=getattr(inner, "cfg", None),
                    **config,
                )
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"invalid MMConcatWrapper checkpoint config: {error}"
                ) from error
            preprocessing = reader.read_json("preprocessing.json")
            dimensions = preprocessing.get("feature_dims")
            if (
                not isinstance(dimensions, dict)
                or set(dimensions) != set(wrapper.cfg.modalities)
                or any(type(d) is not int or d < 1 for d in dimensions.values())
            ):
                raise ValueError("invalid checkpoint modality dimensions")
            wrapper._feature_dims = {
                name: dimensions[name] for name in wrapper.cfg.modalities
            }
            if "fit_sparse_output" in preprocessing:
                fit_sparse_output = preprocessing["fit_sparse_output"]
                if not isinstance(fit_sparse_output, bool):
                    raise ValueError("invalid checkpoint fit_sparse_output")
            else:
                # Older checkpoints did not retain the fitted input format.
                # Their current matrix catalog is the available storage hint.
                fit_sparse_output = isinstance(
                    inner.candidates.snapshot().item_features, csr_matrix
                )
            wrapper._fit_sparse_output = fit_sparse_output
            wrapper._means = {}
            if wrapper.cfg.missing == "mean":
                for index, name in enumerate(wrapper.cfg.modalities):
                    mean = reader.read_numpy(f"means/{index}.npy")
                    if (
                        mean.shape != (dimensions[name],)
                        or mean.dtype != np.dtype(wrapper.cfg.dtype)
                        or not np.isfinite(mean).all()
                    ):
                        raise ValueError(
                            f"invalid checkpoint mean for modality {name!r}"
                        )
                    wrapper._means[name] = mean
            wrapper.inner_model = inner
            if not wrapper.is_fitted:
                raise ValueError("checkpoint inner model is not fitted")
            return wrapper
