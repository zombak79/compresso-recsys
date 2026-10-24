"""Evaluate WarpRec's SASRec with compresso-recsys, next to EASE and Popularity.

WarpRec trains the model; compresso-recsys only scores it. The adapter turns the
evaluator's batches of histories into WarpRec's padded sequences and back into
ranked items. Everything after that -- metrics, per-user values, paired
statistics -- is the same path the built-in models take.

Run (WarpRec is not a dependency of compresso-recsys; it needs Python >= 3.12)::

    pip install warprec
    python examples/interop/warprec_sasrec.py --data-dir data --epochs 20

ML-1M, ``leave_last_out``: each user's last item is the test target, scored
against the full catalogue.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import lightning
import narwhals as nw
import numpy as np
import pandas as pd
import torch
from compresso import SRPTensor
from scipy.sparse import csr_matrix
from warprec.data.entities import Sessions
from warprec.recommenders.sequential_recommender.sasrec import SASRec

import compresso_recsys as cr


class WarpRecSequential:
    """Adapter: a trained WarpRec sequential model behind the compresso-recsys prediction contract."""

    def __init__(self, model, max_seq_len):
        self.model, self.max_seq_len = model.eval(), max_seq_len

    def predict_on_batch(self, source, *, k, exclude_seen=True):
        # source is an ItemSequences batch; WarpRec wants right-padded ids, padding id = n_items.
        rows = [source.values[a:b][-self.max_seq_len:] for a, b in zip(source.indptr[:-1], source.indptr[1:])]
        lengths = torch.tensor([max(len(r), 1) for r in rows])
        padded = torch.full((len(rows), int(lengths.max())), self.model.n_items, dtype=torch.long)
        for i, r in enumerate(rows):
            padded[i, : len(r)] = torch.tensor(r, dtype=torch.long)
        with torch.no_grad():
            scores = self.model.predict(padded, lengths)
        if exclude_seen:
            seen = padded < self.model.n_items
            scores[seen.nonzero()[:, 0], padded[seen]] = float("-inf")
        values, cols = scores.topk(k, dim=1)
        return SRPTensor(cols=cols, vals=values, shape=(len(rows), source.n_items))


def train_warprec_sasrec(sequences: cr.ItemSequences, *, epochs: int, seed: int = 0) -> SASRec:
    """Fit WarpRec's SASRec on the checkpoint's training sequences, item ids = catalogue columns."""
    users = np.repeat(np.arange(sequences.n_rows), np.diff(sequences.indptr))
    position = np.arange(sequences.values.size) - np.repeat(sequences.indptr[:-1], np.diff(sequences.indptr))
    frame = pd.DataFrame({"user_id": users, "item_id": np.asarray(sequences.values), "timestamp": position})
    interactions = csr_matrix((np.ones(len(frame), dtype=np.float32), (users, frame["item_id"])),
                              shape=(sequences.n_rows, sequences.n_items))
    sessions = Sessions(nw.from_native(frame), {u: u for u in range(sequences.n_rows)},
                        {i: i for i in range(sequences.n_items)}, interactions)
    params = dict(embedding_size=50, n_layers=2, n_heads=1, inner_size=50, dropout_prob=0.2,
                  attn_dropout_prob=0.2, reg_weight=0.0, weight_decay=0.0, batch_size=128,
                  epochs=epochs, learning_rate=0.001, neg_samples=1, max_seq_len=200)
    model = SASRec(params, {"n_users": sequences.n_rows, "n_items": sequences.n_items}, seed=seed)
    loader = model.get_dataloader(interactions=None, sessions=sessions, seed=seed)
    lightning.Trainer(max_epochs=epochs, logger=False, enable_checkpointing=False,
                      enable_progress_bar=False, enable_model_summary=False).fit(model, loader)
    return model


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--checkpoint", type=Path, default=Path("artifacts/ml1m-leave_last_out.zip"))
    parser.add_argument("--epochs", type=int, default=20)
    args = parser.parse_args()

    if not args.checkpoint.exists():
        cr.build_recsys_checkpoint(dataset="ml1m", split_mode="leave_last_out", data_dir=str(args.data_dir),
                                   checkpoint_path=str(args.checkpoint), min_user_support=5, item_min_support=5,
                                   min_value_to_keep=1.0, min_entity_text_words=0, seed=0,
                                   annotation_source="none", show_progress=False)
    with cr.read_checkpoint(args.checkpoint) as root:
        split = cr.load_recsys_split(root)

    sasrec = WarpRecSequential(train_warprec_sasrec(split["x_train_sequences"], epochs=args.epochs), 200)
    models = {
        "EASE": (cr.EASE().fit(split["x_train"]), split["test_source_matrix"]),
        "Pop": (cr.PopularityBaseline().fit(split["x_train"]), split["test_source_matrix"]),
        "SASRec (WarpRec)": (sasrec, split["test_source_sequences"]),
    }
    results = {
        name: cr.evaluate_recommender(model, source=source, targets=split["test_target_matrix"],
                                      sample_ids=split["test_eval_user_ids"], exclude_seen=True,
                                      metrics=[cr.NDCG([10, 20]), cr.HitRate(10)])
        for name, (model, source) in models.items()
    }
    for name, result in results.items():
        print(f"{name:18s}", {key: round(value, 4) for key, value in result.metrics.items()})
    print(cr.compare_models(results, metrics=["ndcg@20", "hit_rate@10"], reference="EASE").to_markdown())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
