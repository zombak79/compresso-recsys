# Compresso Recsys

[![PyPI](https://img.shields.io/pypi/v/compresso-recsys.svg)](https://pypi.org/project/compresso-recsys/)
[![Python](https://img.shields.io/pypi/pyversions/compresso-recsys.svg)](https://pypi.org/project/compresso-recsys/)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Docs](https://img.shields.io/badge/docs-GitHub%20Pages-blue.svg)](https://zombak79.github.io/compresso-recsys/)

`compresso-recsys` is the recommender-system companion package for
[Compresso](https://github.com/zombak79/compresso). It provides dataset
loaders, checkpoint builders, checkpoint read/write helpers, and retrieval
metrics for sparse representation learning experiments.

The package distribution name is `compresso-recsys`; the Python import is:

```python
import compresso_recsys as cr
```

## Install

Install from PyPI:

```bash
pip install compresso-recsys
```

Install optional dataset export support:

```bash
pip install "compresso-recsys[datasets]"
```

For local development:

```bash
pip install -e ../compresso
pip install -e ".[dev,datasets]"
```

## Quickstart

Build a MovieLens 1M checkpoint from Python:

```python
import compresso_recsys as cr

checkpoint_path = cr.build_recsys_checkpoint(
    dataset="ml1m",
    checkpoint_path="artifacts/ml1m/exp001.zip",
    annotation_source="genres",
)

with cr.read_checkpoint(checkpoint_path) as root:
    split = cr.load_recsys_split(root)

print(split["x_train"].shape)
```

Build the same kind of checkpoint from the command line:

```bash
compresso-recsys-build-checkpoint \
  --dataset ml1m \
  --checkpoint_path artifacts/ml1m/exp001.zip \
  --annotation_source genres
```

Amazon Reviews 2023 checkpoints can use item metadata for cold-item retrieval
experiments:

```bash
compresso-recsys-build-checkpoint \
  --dataset amazon2023 \
  --amazon_category Toys_and_Games \
  --checkpoint_path artifacts/amazon_toys/temporal_exp001.zip \
  --split_mode temporal \
  --temporal_period_hours 8136 \
  --metadata_text_fields title,features,description,categories \
  --min_entity_text_words 30 \
  --annotation_source none
```

Temporal checkpoints use expanding histories and cumulative mixed warm/cold
catalogs. Their train, validation, and test matrices are aligned by the
corresponding `train_item_ids`, `val_item_ids`, and `test_item_ids` arrays.

## What Is Included

- Dataset utilities for GoodBooks, MovieLens 1M, MovieLens 20M, and Amazon
  Reviews 2023.
- ZIP checkpoint format for source/target splits, embeddings, sparse
  embeddings, metrics, and Compresso cluster-graph stages.
- Calibrated Recall and nDCG defaults, with optional standard Recall,
  Precision, Hit Rate, MRR, and MAP at configurable cutoffs.
- Batched EASE, ADMM and gradient-trained TEASER cold-start models, a
  content-similarity cold-start baseline, dense ELSA, and lottery-ticket
  compressed ELSA with streaming evaluation.
- Sequential recommenders over chronological histories: `SimpleRNN`, a recurrent
  next-item baseline, and `SimpleGPT`, a causal transformer with a tied head.
  Both are built from replaceable parts — an `ItemTokenizer` owning the
  vocabulary and a `SequenceBatcher` owning the context window — so bringing your
  own vocabulary or context length does not mean forking a trainer.
- One evaluation path for both model shapes. A sequential and a matrix model are
  compared through the same `evaluate_recommender` and `compare_models` calls,
  because a recommender is anything implementing `predict_on_batch`.
- One production recommendation path through `model.recommend(histories, k=...)`.
  Histories and optional allow/block filters use stable item IDs; ranked item IDs
  and scores come back in a batch-shaped `Recommendations` result. Rows with
  fewer than `k` valid candidates truncate without weakening the filters.
- Unified fitted-model persistence through `model.save(path)` and
  `ModelClass.load(path)`, using versioned ZIP archives that carry model state,
  configuration, and the vocabulary or candidate catalog needed for serving.
- A checkpoint-building console command:
  `compresso-recsys-build-checkpoint`.

## Documentation

Release documentation is available at:

```text
https://zombak79.github.io/compresso-recsys/
```

The full CLI parameter table, checkpoint split schema, and supported Amazon
Reviews 2023 categories are maintained in the
[Checkpoint CLI Reference](https://zombak79.github.io/compresso-recsys/cli-reference.html).
Academic references and copy-ready BibTeX for EASE, TEASER, ELSA,
large-scale ELSA, and compressed ELSA are available in the
[citation guide](https://zombak79.github.io/compresso-recsys/citing.html).

Build the docs locally:

```bash
pip install -e ".[docs]"
sphinx-build -b html docs/source docs/build/html
```

## License

Apache License 2.0. See [LICENSE](LICENSE).
