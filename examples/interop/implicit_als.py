"""Evaluate a model from another library: implicit's ALS next to EASE and Popularity.

compresso-recsys evaluates anything with ``predict_on_batch(source, *, k)``
returning ranked items, so a model from another library needs only a small
adapter. Its results then go through the same ``evaluate_recommender`` and
``compare_models`` as the built-in models, on the same held-out users.

Run (``implicit`` is not a dependency of compresso-recsys)::

    pip install implicit
    OPENBLAS_NUM_THREADS=1 python examples/interop/implicit_als.py --data-dir data

ML-1M, ``user_split``: the test users are held out of training entirely, so
every model must fold them in from their test-time history.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from compresso import SRPTensor
from implicit.als import AlternatingLeastSquares

import compresso_recsys as cr


class ImplicitALS:
    """Adapter: implicit's ALS behind the compresso-recsys prediction contract."""

    def __init__(self, **params):
        self.model = AlternatingLeastSquares(**params)

    def fit(self, x):
        self.model.fit(x.tocsr().astype(np.float32), show_progress=False)
        return self

    def predict_on_batch(self, source, *, k, exclude_seen=True, candidate_ids=None):
        source = source.tocsr().astype(np.float32)
        # recalculate_user=True is essential. implicit's default (False) reads
        # the first argument as indices of *training* users, so with held-out
        # users it silently scores the wrong people's factors: on this split
        # ALS then falls below Popularity, and neither library can notice,
        # because targets and histories are both correctly paired.
        ids, scores = self.model.recommend(
            np.arange(source.shape[0]), source, N=k,
            recalculate_user=True, filter_already_liked_items=exclude_seen,
        )
        return SRPTensor(cols=torch.from_numpy(ids).long(), vals=torch.from_numpy(scores).float(),
                         shape=source.shape)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--checkpoint", type=Path, default=Path("artifacts/ml1m-user_split.zip"))
    args = parser.parse_args()

    if not args.checkpoint.exists():
        cr.build_recsys_checkpoint(dataset="ml1m", split_mode="user_split", data_dir=str(args.data_dir),
                                   checkpoint_path=str(args.checkpoint), annotation_source="none",
                                   seed=42, show_progress=False)
    with cr.read_checkpoint(args.checkpoint) as root:
        split = cr.load_recsys_split(root)

    models = {
        "EASE": cr.EASE(),
        "Pop": cr.PopularityBaseline(),
        "ALS (implicit)": ImplicitALS(factors=128, regularization=0.05, alpha=2.0, iterations=15, random_state=0),
    }
    results = {}
    for name, model in models.items():
        model.fit(split["x_train"])
        results[name] = cr.evaluate_recommender(
            model, source=split["test_source_matrix"], targets=split["test_target_matrix"],
            sample_ids=split["test_eval_user_ids"], exclude_seen=True,
        )
        print(f"{name:16s}", {key: round(value, 4) for key, value in results[name].metrics.items()})

    report = cr.compare_models(results, metrics=["calibrated_recall@20", "ndcg@100"], reference="EASE")
    print(report.to_markdown())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
