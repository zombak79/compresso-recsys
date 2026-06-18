# Compresso Recsys

`compresso-recsys` is the recommender-system companion package for
[Compresso](https://github.com/zombak79/compresso). It contains dataset
loaders, ELSA/CompressedELSA models, checkpointed experiment pipelines, and
retrieval metrics used to demonstrate sparse representation learning.

The package distribution name is `compresso-recsys`; the Python import is:

```python
import compresso_recsys as cr
```

## Install

For local development next to a checkout of `compresso`:

```bash
pip install -e ../compresso
pip install -e ".[test,sbert]"
```

From GitHub, once both repositories are public:

```bash
pip install "compresso-recsys @ git+https://github.com/zombak79/compresso-recsys.git"
```

## What Is Included

- Dataset utilities for GoodBooks, MovieLens 1M, and MovieLens 20M.
- ZIP checkpoint format for splits, embeddings, sparse embeddings, and metrics.
- ELSA and CompressedELSA training helpers.
- SAE and SBERT checkpoint stages.
- Retrieval metrics for Recall@20, Recall@50, and nDCG@100.
- Console commands for the full experiment pipeline.

## Commands

Build a dataset checkpoint:

```bash
compresso-recsys-build-checkpoint \
  --dataset ml1m \
  --checkpoint_path artifacts/ml1m/exp001.zip \
  --annotation_source genres
```

Train ELSA:

```bash
compresso-recsys-train-elsa \
  --checkpoint_path artifacts/ml1m/exp001.zip \
  --elsa_dim 1024 \
  --elsa_epochs 10 \
  --device mps
```

Train CompressedELSA:

```bash
compresso-recsys-train-compressed-elsa \
  --checkpoint_path artifacts/ml1m/exp001.zip \
  --elsa_dim 1024 \
  --sparse_k_target 128 \
  --sparse_num_stages 5 \
  --sparse_ste_alpha 0.01 \
  --device mps
```

Train an SAE on an existing embedding stage:

```bash
compresso-recsys-train-sae \
  --checkpoint_path artifacts/ml1m/exp001.zip \
  --embedding_stage elsa \
  --sae_k 128 \
  --sae_ste_alpha 0.01 \
  --sae_post_norm_l1 \
  --device mps
```

Evaluate all available checkpoint stages:

```bash
compresso-recsys-eval-checkpoint \
  --checkpoint_path artifacts/ml1m/exp001.zip \
  --device mps
```

## Python API

```python
import compresso_recsys as cr

dataset = cr.MovieLens1M(data_dir="data")
interactions = dataset.get_interactions()
```

Subpackages are also available:

```python
from compresso_recsys.datasets import Goodbooks, MovieLens1M, MovieLens20M
from compresso_recsys.models import TorchELSA, CompressedELSA
from compresso_recsys.checkpoint import update_checkpoint, load_recsys_split
```
