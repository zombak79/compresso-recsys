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
  --metadata_text_fields title,features,description,categories \
  --min_entity_text_words 30 \
  --annotation_source none
```

## What Is Included

- Dataset utilities for GoodBooks, MovieLens 1M, MovieLens 20M, and Amazon
  Reviews 2023.
- ZIP checkpoint format for source/target splits, embeddings, sparse
  embeddings, metrics, and Compresso cluster-graph stages.
- Retrieval metrics for Recall and nDCG at common cutoffs including 20, 50,
  and 100.
- A batched EASE collaborative-filtering baseline with streaming evaluation.
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

Build the docs locally:

```bash
pip install -e ".[docs]"
sphinx-build -b html docs/source docs/build/html
```

## License

Apache License 2.0. See [LICENSE](LICENSE).
