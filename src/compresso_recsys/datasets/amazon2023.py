from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import quote
from urllib.error import URLError
from urllib.request import Request, urlopen

import pandas as pd

from .base import RecSysDataset


DEFAULT_TEXT_FIELDS = ("title", "features", "description", "categories")
CATEGORY_ALIASES = {
    "beauty": "All_Beauty",
    "clothing": "Clothing_Shoes_and_Jewelry",
    "electronics": "Electronics",
    "toys": "Toys_and_Games",
    "toys_and_games": "Toys_and_Games",
}


class AmazonReviews2023(RecSysDataset):
    """Amazon Reviews 2023 category dataset loaded from McAuley's files.

    The recommender pipeline only needs compact rating-only interactions plus
    item metadata. Reviews are intentionally not downloaded.
    """

    name = "amazon2023"
    hf_name = "McAuley-Lab/Amazon-Reviews-2023"
    hf_revision = "main"
    source_base_url = "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023"

    def __init__(
        self,
        data_dir: str = "data",
        *,
        category: str = "Toys_and_Games",
        metadata_text_fields: Iterable[str] = DEFAULT_TEXT_FIELDS,
        min_entity_text_words: int = 0,
        include_image_urls: bool = False,
        show_progress: bool = True,
    ) -> None:
        self.category = self.normalize_category(category)
        self.metadata_text_fields = tuple(metadata_text_fields)
        self.min_entity_text_words = int(min_entity_text_words)
        self.include_image_urls = bool(include_image_urls)
        self.show_progress = bool(show_progress)
        super().__init__(data_dir=data_dir)
        self.root = self.data_dir / self.name / self.category
        self.root.mkdir(parents=True, exist_ok=True)

    @property
    def interactions_config(self) -> str:
        return f"0core_rating_only_{self.category}"

    @property
    def metadata_config(self) -> str:
        return f"raw_meta_{self.category}"

    @staticmethod
    def normalize_category(category: str) -> str:
        key = category.strip()
        if not key:
            raise ValueError("Amazon category cannot be empty")
        return CATEGORY_ALIASES.get(key.lower(), key)

    def download(self) -> None:
        # `prepare` triggers the category-specific downloads; this method exists
        # for interface parity with the other dataset loaders.
        self.root.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _stringify(value: Any, *, separator: str = " ") -> str:
        if value is None:
            return ""
        try:
            if pd.isna(value):
                return ""
        except (TypeError, ValueError):
            pass
        if isinstance(value, str):
            return value.strip()
        if isinstance(value, dict):
            parts: list[str] = []
            for key, val in value.items():
                val_text = AmazonReviews2023._stringify(val, separator=separator)
                if val_text:
                    parts.append(f"{key}: {val_text}")
            return separator.join(parts).strip()
        if isinstance(value, (list, tuple, set)):
            parts = [AmazonReviews2023._stringify(v, separator=separator) for v in value]
            return separator.join(p for p in parts if p).strip()
        return str(value).strip()

    @staticmethod
    def _parse_details(value: Any) -> Any:
        if not isinstance(value, str):
            return value
        text = value.strip()
        if not text or text == "{}":
            return ""
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return text

    @staticmethod
    def _parse_images(value: Any) -> list[dict[str, Any]]:
        if value is None:
            return []
        try:
            if pd.isna(value):
                return []
        except (TypeError, ValueError):
            pass
        if isinstance(value, str):
            text = value.strip()
            if not text or text in {"[]", "nan"}:
                return []
            try:
                value = json.loads(text)
            except json.JSONDecodeError:
                return [{"large": text}]
        if isinstance(value, dict):
            return [value]
        if isinstance(value, (list, tuple)):
            return [entry for entry in value if isinstance(entry, dict)]
        return []

    @classmethod
    def extract_image_urls(cls, value: Any) -> list[str]:
        urls: list[str] = []
        seen: set[str] = set()
        for image in cls._parse_images(value):
            for key in ("hi_res", "large", "thumb"):
                url = image.get(key)
                if not isinstance(url, str):
                    continue
                url = url.strip()
                if url and url not in seen:
                    seen.add(url)
                    urls.append(url)
        return urls

    @classmethod
    def best_image_url(cls, value: Any) -> str:
        urls = cls.extract_image_urls(value)
        return urls[0] if urls else ""

    @classmethod
    def build_entity_text(cls, row: pd.Series, fields: Iterable[str]) -> str:
        parts: list[str] = []
        for field in fields:
            if field not in row:
                continue
            value = cls._parse_details(row[field]) if field == "details" else row[field]
            text = cls._metadata_value_to_text(value, separator="\n" if field in {"features", "description"} else " > ")
            if text:
                label = field.replace("_", " ").title()
                parts.append(f"{label}: {text}")
        return "\n\n".join(parts).strip()

    @staticmethod
    def _word_count(text: str) -> int:
        return len(str(text).split())

    def _hf_headers(self) -> dict[str, str]:
        token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
        return {"Authorization": f"Bearer {token}"} if token else {}

    def _headers_for_url(self, url: str) -> dict[str, str]:
        return self._hf_headers() if "://huggingface.co/" in url else {}

    def _hf_resolve_url(self, path: str) -> str:
        quoted_path = quote(path, safe="/")
        return f"https://huggingface.co/datasets/{self.hf_name}/resolve/{self.hf_revision}/{quoted_path}"

    def _hf_tree_url(self, path: str) -> str:
        quoted_path = quote(path, safe="/")
        return f"https://huggingface.co/api/datasets/{self.hf_name}/tree/{self.hf_revision}/{quoted_path}"

    def _hf_files_for_path(self, path: str) -> list[dict[str, Any]]:
        request = Request(self._hf_tree_url(path), headers=self._hf_headers())
        with urlopen(request, timeout=60) as response:
            payload = json.loads(response.read().decode("utf-8"))
        return [entry for entry in payload if entry.get("type") == "file"]

    def _hf_source(self, path: str, *, kind: str, size: int | None = None) -> dict[str, Any]:
        return {
            "url": self._hf_resolve_url(path),
            "local_path": Path("huggingface") / path,
            "kind": kind,
            "size": size,
        }

    def _mirror_source(self, url: str, filename: str, kind: str) -> dict[str, Any]:
        return {
            "url": url,
            "local_path": Path("mcauley") / filename,
            "kind": kind,
            "size": None,
        }

    def _mirror_source_for_config(self, config: str, *, split: str = "full") -> list[dict[str, Any]]:
        if config == self.metadata_config:
            if split != "full":
                raise ValueError(f"Amazon metadata config {config!r} only supports split='full'")
            filename = f"meta_{self.category}.jsonl.gz"
            url = f"{self.source_base_url}/raw/meta_categories/{filename}"
            return [self._mirror_source(url, filename, "jsonl")]

        if config == self.interactions_config:
            if split != "full":
                raise ValueError(f"Amazon rating-only config {config!r} only supports split='full'")
            filename = f"{self.category}.csv.gz"
            url = f"{self.source_base_url}/benchmark/0core/rating_only/{filename}"
            return [self._mirror_source(url, filename, "csv")]

        timestamp_config = f"0core_timestamp_w_his_{self.category}"
        if config == timestamp_config:
            if split not in {"train", "valid", "test"}:
                raise ValueError(f"Amazon timestamp config {config!r} requires split='train', 'valid', or 'test'")
            filename = f"{self.category}.{split}.csv.gz"
            url = f"{self.source_base_url}/benchmark/0core/timestamp_w_his/{filename}"
            return [self._mirror_source(url, filename, "csv")]

        raise ValueError(f"Unsupported Amazon Reviews 2023 config: {config!r}")

    def _hf_source_for_config(self, config: str, *, split: str = "full") -> list[dict[str, Any]]:
        if config == self.metadata_config:
            if split != "full":
                raise ValueError(f"Amazon metadata config {config!r} only supports split='full'")
            files = sorted(
                self._hf_files_for_path(self.metadata_config),
                key=lambda entry: entry["path"],
            )
            sources = [
                self._hf_source(entry["path"], kind="parquet", size=entry.get("size"))
                for entry in files
                if str(entry.get("path", "")).endswith(".parquet")
            ]
            if not sources:
                raise FileNotFoundError(f"No Hugging Face parquet files found for {self.metadata_config!r}")
            return sources

        if config == self.interactions_config:
            if split != "full":
                raise ValueError(f"Amazon rating-only config {config!r} only supports split='full'")
            return [self._hf_source(f"benchmark/0core/rating_only/{self.category}.csv", kind="csv")]

        timestamp_config = f"0core_timestamp_w_his_{self.category}"
        if config == timestamp_config:
            if split not in {"train", "valid", "test"}:
                raise ValueError(f"Amazon timestamp config {config!r} requires split='train', 'valid', or 'test'")
            return [self._hf_source(f"benchmark/0core/timestamp_w_his/{self.category}.{split}.csv", kind="csv")]

        raise ValueError(f"Unsupported Amazon Reviews 2023 config: {config!r}")

    def _source_groups_for_config(self, config: str, *, split: str = "full") -> list[list[dict[str, Any]]]:
        try:
            return [
                self._hf_source_for_config(config, split=split),
                self._mirror_source_for_config(config, split=split),
            ]
        except Exception:
            return [self._mirror_source_for_config(config, split=split)]

    def _download_file(self, url: str, destination: Path, *, size: int | None = None) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        tmp = destination.with_suffix(destination.suffix + ".tmp")
        try:
            request = Request(url, headers=self._headers_for_url(url))
            with urlopen(request, timeout=60) as response, tmp.open("wb") as out:
                total_raw = response.headers.get("Content-Length")
                total = int(total_raw) if total_raw and total_raw.isdigit() else size
                progress = None
                if self.show_progress:
                    print(f"Downloading {destination.name}...", flush=True)
                    try:
                        from tqdm import tqdm
                    except Exception:  # pragma: no cover - optional dependency
                        pass
                    else:
                        progress = tqdm(
                            total=total,
                            unit="B",
                            unit_scale=True,
                            desc=f"Downloading {destination.name}",
                            file=sys.stdout,
                            leave=True,
                        )
                try:
                    while True:
                        chunk = response.read(1024 * 1024)
                        if not chunk:
                            break
                        out.write(chunk)
                        if progress is not None:
                            progress.update(len(chunk))
                finally:
                    if progress is not None:
                        progress.close()
            tmp.replace(destination)
        except (OSError, URLError) as e:  # pragma: no cover - network errors are environment-specific
            if tmp.exists():
                tmp.unlink()
            raise RuntimeError(
                "Failed to download Amazon Reviews 2023 data "
                f"({url}). Please check connectivity and try again."
            ) from e

    def _read_source(self, path: Path, kind: str) -> pd.DataFrame:
        if kind == "parquet":
            return pd.read_parquet(path)
        if kind == "jsonl":
            return pd.read_json(path, lines=True, compression="infer")
        return pd.read_csv(path, compression="infer")

    def _load_hf_dataframe(self, config: str, *, split: str = "full") -> pd.DataFrame:
        """Load a McAuley Amazon 2023 config into a DataFrame.

        Kept under its historical name for compatibility with tests and
        subclasses, but this no longer uses Hugging Face ``datasets``. Recent
        ``datasets`` releases reject repositories that still expose loading
        scripts, so we read direct Hugging Face/McAuley data files instead.
        """
        errors: list[Exception] = []
        for group in self._source_groups_for_config(config, split=split):
            try:
                frames = []
                for source in group:
                    local_path = self.root / source["local_path"]
                    if not local_path.exists():
                        self._download_file(source["url"], local_path, size=source.get("size"))
                    frames.append(self._read_source(local_path, source["kind"]))
                return pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]
            except Exception as e:
                errors.append(e)
                if self.show_progress:
                    print(f"Falling back to alternate Amazon source after: {e}", flush=True)
        raise RuntimeError(f"Failed to load Amazon Reviews 2023 config {config!r}") from errors[-1]

    load_source_dataframe = _load_hf_dataframe

    def load_timestamp_splits_with_history(self) -> dict[str, pd.DataFrame]:
        """Load McAuley's timestamp split with per-row history fields."""
        config = f"0core_timestamp_w_his_{self.category}"
        return {
            "train": self._load_hf_dataframe(config, split="train"),
            "valid": self._load_hf_dataframe(config, split="valid"),
            "test": self._load_hf_dataframe(config, split="test"),
        }

    def prepare(self) -> None:
        self.download()

        meta = self._load_hf_dataframe(self.metadata_config, split="full")
        if "parent_asin" not in meta.columns:
            raise ValueError(f"Amazon metadata config {self.metadata_config!r} has no parent_asin column")
        meta = meta.rename(columns={"parent_asin": "item_id"}).copy()
        meta["item_id"] = meta["item_id"].astype(str)
        meta = meta.drop_duplicates(subset=["item_id"], keep="first")
        meta = self.add_entity_text(
            meta,
            fields=self.metadata_text_fields,
            min_words=self.min_entity_text_words,
        )
        if self.include_image_urls and "images" in meta.columns:
            image_urls = meta["images"].map(self.extract_image_urls)
            meta["image_url"] = image_urls.map(lambda urls: urls[0] if urls else "")
            meta["image_urls"] = image_urls.map(lambda urls: " ".join(urls))

        interactions = self._load_hf_dataframe(self.interactions_config, split="full")
        expected = {"user_id", "parent_asin", "rating", "timestamp"}
        missing = expected.difference(interactions.columns)
        if missing:
            raise ValueError(f"Amazon interactions config {self.interactions_config!r} is missing columns {sorted(missing)}")
        interactions = interactions.rename(columns={"parent_asin": "item_id", "rating": "value"})
        interactions = interactions[["user_id", "item_id", "value", "timestamp"]].copy()
        interactions["user_id"] = interactions["user_id"].astype(str)
        interactions["item_id"] = interactions["item_id"].astype(str)
        interactions["value"] = pd.to_numeric(interactions["value"], errors="coerce")
        interactions["timestamp"] = pd.to_numeric(interactions["timestamp"], errors="coerce")
        interactions = interactions.dropna(subset=["user_id", "item_id", "value"])

        interactions = self.restrict_interactions_to_metadata_items(interactions, meta)

        preferred = [
            "item_id",
            "title",
            "store",
            "main_category",
            "categories",
            "features",
            "description",
            "details",
            "price",
            "average_rating",
            "rating_number",
            "entity_text",
        ]
        if self.include_image_urls:
            preferred.extend(["image_url", "image_urls"])
        keep = [col for col in preferred if col in meta.columns]
        self._item_metadata = meta[keep].reset_index(drop=True)
        self._interactions = interactions
