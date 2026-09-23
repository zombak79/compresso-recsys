"""Saving and loading a fitted recommender, as a template with model hooks.

The archive layout lives here and nowhere else: a subclass contributes its own
state through ``_save_checkpoint_state`` and friends rather than deciding what
a checkpoint looks like.
"""

from __future__ import annotations


from abc import abstractmethod
from dataclasses import asdict, is_dataclass, replace
from pathlib import Path
from typing import (
    Any,
    ClassVar,
    TypeVar,
)

import torch
from torch import nn

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
from compresso_recsys.models.base.identified import (
    BaseIdentifiedRecommender,
    _MODELS_DIR,
    _embedded_model_path,
    _unwrapped_module,
)

# Bound to the class defined just below, so it lives here rather than in
# identified, which cannot see it.
_PersistableT = TypeVar("_PersistableT", bound="BasePersistableRecommender")


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
