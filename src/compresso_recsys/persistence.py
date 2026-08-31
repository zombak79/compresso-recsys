"""Safe, versioned persistence helpers for fitted recommenders."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from contextlib import AbstractContextManager
from dataclasses import asdict, is_dataclass
import json
import math
from pathlib import Path, PurePosixPath
import tempfile
from typing import Any, Hashable
import zipfile

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, load_npz, save_npz
import torch

MODEL_CHECKPOINT_FORMAT = "compresso.recsys.model"
MODEL_CHECKPOINT_VERSION = 1
MODEL_MANIFEST_NAME = "manifest.json"

__all__ = [
    "MODEL_CHECKPOINT_FORMAT",
    "MODEL_CHECKPOINT_VERSION",
    "ModelCheckpointReader",
    "ModelCheckpointWriter",
]


def _checked_relative_path(relpath: str | Path) -> Path:
    value = PurePosixPath(str(relpath).replace("\\", "/"))
    if value.is_absolute() or not value.parts or ".." in value.parts:
        raise ValueError(f"checkpoint path must be relative, got {relpath!r}")
    return Path(*value.parts)


def _json_value(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return _json_value(asdict(value))
    if isinstance(value, torch.device):
        return str(value)
    if isinstance(value, np.generic):
        return _json_value(value.item())
    if isinstance(value, Mapping):
        out: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError("JSON object keys must be strings")
            out[key] = _json_value(item)
        return out
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("JSON values must contain only finite numbers")
        return value
    raise TypeError(
        f"value of type {type(value).__name__} is not safely JSON serializable"
    )


def _encoded_item_id(value: Hashable) -> dict[str, Any]:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, bool):
        return {"type": "bool", "value": value}
    if isinstance(value, int):
        return {"type": "int", "value": value}
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("item IDs must not contain non-finite floats")
        return {"type": "float", "value": value}
    if isinstance(value, str):
        return {"type": "str", "value": value}
    raise TypeError(
        "item IDs in model checkpoints must be strings, integers, finite "
        f"floats, or booleans; got {type(value).__name__}"
    )


def _decoded_item_id(state: object) -> Hashable:
    if not isinstance(state, dict):
        raise ValueError("encoded item ID must be an object")
    kind = state.get("type")
    value = state.get("value")
    if kind == "bool" and isinstance(value, bool):
        return value
    if kind == "int" and isinstance(value, int) and not isinstance(value, bool):
        return value
    if kind == "float" and isinstance(value, (int, float)) and not isinstance(
        value, bool
    ):
        result = float(value)
        if math.isfinite(result):
            return result
    if kind == "str" and isinstance(value, str):
        return value
    raise ValueError(f"invalid encoded item ID: {state!r}")


def _write_zip(root: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
    try:
        with zipfile.ZipFile(
            temporary,
            "w",
            compression=zipfile.ZIP_DEFLATED,
        ) as archive:
            for file in sorted(path for path in root.rglob("*") if path.is_file()):
                archive.write(file, file.relative_to(root).as_posix())
        temporary.replace(destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def _extract_zip(source: Path, root: Path) -> None:
    if not source.exists():
        raise FileNotFoundError(source)
    if not zipfile.is_zipfile(source):
        raise ValueError(f"model checkpoint is not a ZIP archive: {source}")
    with zipfile.ZipFile(source, "r") as archive:
        for member in archive.infolist():
            path = PurePosixPath(member.filename)
            if path.is_absolute() or ".." in path.parts:
                raise ValueError(
                    f"model checkpoint contains an unsafe path: {member.filename!r}"
                )
        archive.extractall(root)


class _ModelCheckpointFiles:
    def __init__(self, root: Path | None) -> None:
        self.root = root

    def _path(self, relpath: str | Path) -> Path:
        if self.root is None:
            raise RuntimeError("model checkpoint reader or writer is not open")
        return self.root / _checked_relative_path(relpath)

    def exists(self, relpath: str | Path) -> bool:
        """Whether a regular file exists at ``relpath`` in the archive."""
        return self._path(relpath).is_file()


class ModelCheckpointWriter(_ModelCheckpointFiles, AbstractContextManager):
    """Build one model checkpoint while owning its manifest and ZIP layout."""

    def __init__(
        self,
        path: str | Path,
        *,
        model_type: str,
        optimizer_included: bool = False,
    ) -> None:
        if not isinstance(model_type, str) or not model_type:
            raise ValueError("model_type must be a non-empty string")
        self.path = Path(path)
        self.model_type = model_type
        self.optimizer_included = bool(optimizer_included)
        self._temporary: tempfile.TemporaryDirectory[str] | None = None
        super().__init__(None)

    def __enter__(self) -> "ModelCheckpointWriter":
        if self._temporary is not None:
            raise RuntimeError("model checkpoint writer is already open")
        self._temporary = tempfile.TemporaryDirectory()
        self.root = Path(self._temporary.name)
        self.write_json(
            MODEL_MANIFEST_NAME,
            {
                "format": MODEL_CHECKPOINT_FORMAT,
                "version": MODEL_CHECKPOINT_VERSION,
                "model_type": self.model_type,
                "optimizer_included": self.optimizer_included,
            },
        )
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        temporary = self._temporary
        if temporary is None:
            return False
        try:
            if exc_type is None:
                _write_zip(self.root, self.path)
        finally:
            temporary.cleanup()
            self._temporary = None
            self.root = None
        return False

    def _output_path(self, relpath: str | Path) -> Path:
        if self._temporary is None:
            raise RuntimeError("model checkpoint writer is not open")
        path = self._path(relpath)
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def write_json(self, relpath: str | Path, value: Any) -> None:
        """Write dataclasses and JSON-safe values without non-finite numbers."""
        self._output_path(relpath).write_text(
            json.dumps(
                _json_value(value),
                indent=2,
                sort_keys=True,
                allow_nan=False,
            ),
            encoding="utf-8",
        )

    def write_numpy(self, relpath: str | Path, value: np.ndarray) -> None:
        """Write a non-object NumPy array with pickle disabled."""
        array = np.asarray(value)
        if array.dtype == object:
            raise TypeError("object arrays must use write_item_ids()")
        np.save(self._output_path(relpath), array, allow_pickle=False)

    def write_sparse(self, relpath: str | Path, value: csr_matrix) -> None:
        """Write a CSR matrix in SciPy's compressed NPZ representation."""
        if not isinstance(value, csr_matrix):
            raise TypeError("value must be a scipy.sparse.csr_matrix")
        save_npz(self._output_path(relpath), value, compressed=True)

    def write_features(
        self,
        relpath: str | Path,
        value: csr_matrix | np.ndarray,
    ) -> str:
        """Write dense or CSR features and return their storage discriminator."""
        if isinstance(value, csr_matrix):
            self.write_sparse(f"{relpath}.npz", value)
            return "csr"
        self.write_numpy(f"{relpath}.npy", np.asarray(value))
        return "dense"

    def write_dataframe(self, relpath: str | Path, value: pd.DataFrame) -> None:
        """Write candidate metadata as Parquet without an index."""
        if not isinstance(value, pd.DataFrame):
            raise TypeError("value must be a pandas.DataFrame")
        try:
            value.to_parquet(self._output_path(relpath), index=False)
        except (TypeError, ValueError, OverflowError) as error:
            raise TypeError(
                "candidate metadata must contain Parquet-serializable values"
            ) from error

    def write_torch(self, relpath: str | Path, value: Mapping[str, Any]) -> None:
        """Write a Torch state mapping intended for ``weights_only`` loading."""
        if not isinstance(value, Mapping):
            raise TypeError("Torch checkpoint state must be a mapping")
        torch.save(dict(value), self._output_path(relpath))

    def write_item_ids(
        self,
        relpath: str | Path,
        values: Sequence[Hashable] | np.ndarray,
    ) -> None:
        """Write stable scalar item IDs with explicit, pickle-free type tags."""
        array = np.asarray(values, dtype=object)
        if array.ndim != 1:
            raise ValueError("item IDs must be one-dimensional")
        self.write_json(
            relpath,
            [_encoded_item_id(value) for value in array.tolist()],
        )


class ModelCheckpointReader(_ModelCheckpointFiles, AbstractContextManager):
    """Validate and read one fitted-recommender checkpoint."""

    def __init__(
        self,
        path: str | Path,
        *,
        expected_model_type: str,
    ) -> None:
        self.path = Path(path)
        self.expected_model_type = expected_model_type
        self.manifest: dict[str, Any] = {}
        self._temporary: tempfile.TemporaryDirectory[str] | None = None
        super().__init__(None)

    def __enter__(self) -> "ModelCheckpointReader":
        if self._temporary is not None:
            raise RuntimeError("model checkpoint reader is already open")
        self._temporary = tempfile.TemporaryDirectory()
        self.root = Path(self._temporary.name)
        try:
            _extract_zip(self.path, self.root)
            self.manifest = self.read_json(MODEL_MANIFEST_NAME)
            self._validate_manifest()
        except Exception:
            self._temporary.cleanup()
            self._temporary = None
            self.root = None
            raise
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        temporary = self._temporary
        if temporary is not None:
            temporary.cleanup()
            self._temporary = None
            self.root = None
        return False

    @property
    def optimizer_included(self) -> bool:
        """Whether the checkpoint manifest advertises optimizer state."""
        return bool(self.manifest.get("optimizer_included", False))

    def _validate_manifest(self) -> None:
        if self.manifest.get("format") != MODEL_CHECKPOINT_FORMAT:
            raise ValueError(
                "not a Compresso Recsys model checkpoint: expected format "
                f"{MODEL_CHECKPOINT_FORMAT!r}"
            )
        version = self.manifest.get("version")
        if version != MODEL_CHECKPOINT_VERSION:
            raise ValueError(f"unsupported model checkpoint version {version!r}")
        actual = self.manifest.get("model_type")
        if actual != self.expected_model_type:
            raise ValueError(
                f"checkpoint contains model type {actual!r}, not "
                f"{self.expected_model_type!r}"
            )
        if not isinstance(self.manifest.get("optimizer_included", False), bool):
            raise ValueError("manifest optimizer_included must be a bool")

    def _input_path(self, relpath: str | Path) -> Path:
        if self._temporary is None:
            raise RuntimeError("model checkpoint reader is not open")
        path = self._path(relpath)
        if not path.is_file():
            raise ValueError(f"model checkpoint is missing {str(relpath)!r}")
        return path

    def read_json(self, relpath: str | Path) -> dict[str, Any]:
        """Read a JSON object."""
        try:
            value = json.loads(self._input_path(relpath).read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid JSON in model checkpoint: {relpath}") from error
        if not isinstance(value, dict):
            raise ValueError(f"checkpoint JSON object expected at {relpath}")
        return value

    def read_json_value(self, relpath: str | Path) -> Any:
        """Read any JSON value."""
        try:
            return json.loads(self._input_path(relpath).read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid JSON in model checkpoint: {relpath}") from error

    def read_numpy(self, relpath: str | Path) -> np.ndarray:
        """Read a NumPy array with pickle disabled."""
        return np.load(self._input_path(relpath), allow_pickle=False)

    def read_sparse(self, relpath: str | Path) -> csr_matrix:
        """Read a SciPy sparse matrix and canonicalize it to CSR."""
        return load_npz(self._input_path(relpath)).tocsr()

    def read_features(
        self,
        relpath: str | Path,
        *,
        storage: str,
    ) -> csr_matrix | np.ndarray:
        """Read features according to a writer-provided storage discriminator."""
        if storage == "csr":
            return self.read_sparse(f"{relpath}.npz")
        if storage == "dense":
            return self.read_numpy(f"{relpath}.npy")
        raise ValueError(f"unsupported feature storage {storage!r}")

    def read_dataframe(self, relpath: str | Path) -> pd.DataFrame:
        """Read Parquet metadata."""
        return pd.read_parquet(self._input_path(relpath))

    def read_torch(
        self,
        relpath: str | Path,
        *,
        device: str | torch.device = "cpu",
    ) -> dict[str, Any]:
        """Read Torch state as data using ``weights_only=True``."""
        value = torch.load(
            self._input_path(relpath),
            map_location=torch.device(device),
            weights_only=True,
        )
        if not isinstance(value, dict):
            raise ValueError(f"Torch checkpoint mapping expected at {relpath}")
        return value

    def read_item_ids(self, relpath: str | Path) -> np.ndarray:
        """Read explicitly typed stable item IDs into an object array."""
        value = self.read_json_value(relpath)
        if not isinstance(value, list):
            raise ValueError(f"encoded item ID list expected at {relpath}")
        result = np.empty(len(value), dtype=object)
        result[:] = [_decoded_item_id(item) for item in value]
        return result
