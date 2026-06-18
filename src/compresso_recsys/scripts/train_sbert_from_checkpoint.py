from __future__ import annotations

import argparse
import random

import numpy as np
import torch

from compresso_recsys.checkpoint import SBERT_DIR, load_recsys_split, save_json, update_checkpoint, update_stage_manifest
from compresso_recsys.retrieval import evaluate_item_embeddings_with_holdout


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint_path", type=str, required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--model_name", type=str, default="sentence-transformers/all-MiniLM-L6-v2")
    p.add_argument("--text_columns", type=str, default=None, help="Comma-separated metadata columns to join and encode.")
    p.add_argument("--text_separator", type=str, default="\\n", help="Separator used when joining --text_columns.")
    p.add_argument("--text_column", type=str, default="description")
    p.add_argument("--fallback_columns", type=str, default="title,authors,genres")
    p.add_argument("--sbert_batch_size", type=int, default=64)
    p.add_argument("--normalize_embeddings", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--device", type=str, default="mps")
    p.add_argument("--eval_batch_size", type=int, default=1024)
    return p.parse_args()


def resolve_device(requested: str) -> str:
    req = requested.lower()
    if req == "cpu":
        return "cpu"
    if req == "mps":
        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"
    if req == "cuda":
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    return req


def select_metrics(metrics: dict) -> dict:
    return {
        "recall@20": metrics.get("recall@20", 0.0),
        "recall@50": metrics.get("recall@50", 0.0),
        "ndcg@100": metrics.get("ndcg@100", 0.0),
    }


def eval_three_metrics(item_embs, source_indices, target_indices, eval_batch_size):
    out = {}
    for k in (20, 50, 100):
        m = evaluate_item_embeddings_with_holdout(
            item_embeddings=item_embs,
            source_indices=source_indices,
            target_indices=target_indices,
            k=k,
            score_batch_size=eval_batch_size,
        )
        out.update({kk: vv for kk, vv in m.items() if kk != "n_eval_users"})
    return select_metrics(out)


def _clean_text(value) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and np.isnan(value):
        return ""
    return str(value).strip()


def build_texts(
    metadata,
    *,
    text_columns: list[str] | None = None,
    text_separator: str = "\n",
    text_column: str = "description",
    fallback_columns: list[str] | None = None,
) -> list[str]:
    if metadata is None:
        raise ValueError("Checkpoint data split has no entity_metadata. Rebuild the checkpoint with metadata first.")
    fallback_columns = fallback_columns or []
    if text_columns:
        missing = [c for c in text_columns if c not in metadata.columns]
        if missing:
            raise ValueError(f"entity_metadata is missing text columns {missing!r}; available columns: {list(metadata.columns)}")
    elif text_column not in metadata.columns:
        raise ValueError(f"entity_metadata has no text column {text_column!r}; available columns: {list(metadata.columns)}")

    texts: list[str] = []
    for _, row in metadata.iterrows():
        if text_columns:
            parts = [_clean_text(row.get(c)) for c in text_columns]
            text = text_separator.join(p for p in parts if p)
        else:
            text = _clean_text(row.get(text_column))
        if not text and fallback_columns:
            parts = [_clean_text(row.get(c)) for c in fallback_columns if c in metadata.columns]
            text = text_separator.join(p for p in parts if p)
        texts.append(text)
    if not any(texts):
        raise ValueError("No non-empty text could be built from metadata.")
    return texts


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = resolve_device(args.device)

    try:
        from sentence_transformers import SentenceTransformer
    except Exception as e:  # pragma: no cover - optional dependency
        raise ImportError(
            "Install sentence-transformers to train SBERT embeddings, e.g. "
            "`pip install sentence-transformers` or `pip install -e .[sbert]`."
        ) from e

    text_columns = [c.strip() for c in args.text_columns.split(",") if c.strip()] if args.text_columns else None
    fallback_columns = [c.strip() for c in args.fallback_columns.split(",") if c.strip()]

    with update_checkpoint(args.checkpoint_path) as root:
        split = load_recsys_split(root)
        texts = build_texts(
            split["entity_metadata"],
            text_columns=text_columns,
            text_separator=args.text_separator,
            text_column=args.text_column,
            fallback_columns=fallback_columns,
        )

        print(f"Encoding {len(texts)} entity descriptions with {args.model_name!r} on {device}")
        model = SentenceTransformer(args.model_name, device=device)
        item_embs = model.encode(
            texts,
            batch_size=args.sbert_batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=args.normalize_embeddings,
        ).astype(np.float32)

        metrics = eval_three_metrics(
            item_embs,
            split["test_source_indices"],
            split["test_target_indices"],
            args.eval_batch_size,
        )
        print("SBERT metrics:", metrics)

        stage_dir = root / SBERT_DIR
        stage_dir.mkdir(parents=True, exist_ok=True)
        np.save(stage_dir / "item_embeddings.npy", item_embs)
        save_json(
            root,
            f"{SBERT_DIR}/metrics.json",
            {
                "test_metrics": metrics,
                "model_name": args.model_name,
                "text_columns": text_columns,
                "text_separator": args.text_separator,
                "text_column": args.text_column,
                "fallback_columns": fallback_columns,
                "normalize_embeddings": bool(args.normalize_embeddings),
                "embedding_dim": int(item_embs.shape[1]),
            },
        )
        update_stage_manifest(
            root,
            "sbert",
            {
                "model_name": args.model_name,
                "text_columns": text_columns,
                "text_separator": args.text_separator,
                "text_column": args.text_column,
                "fallback_columns": fallback_columns,
                "normalize_embeddings": bool(args.normalize_embeddings),
                "embedding_dim": int(item_embs.shape[1]),
                "metrics": metrics,
            },
        )
    print(f"Saved SBERT stage to checkpoint: {args.checkpoint_path}")


if __name__ == "__main__":
    main()
