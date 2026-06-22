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

- Dataset utilities for GoodBooks, MovieLens 1M, MovieLens 20M, and Amazon
  Reviews 2023.
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

### Amazon Reviews 2023

Amazon checkpoints use compact rating-only interactions plus item metadata:

```text
0core_rating_only_<category>
raw_meta_<category>
```

For temporal checkpoints, Amazon uses McAuley's predefined timestamp split with
history:

```text
0core_timestamp_w_his_<category>
```

The builder also constructs a canonical `entity_text` column from configurable
metadata fields. Downstream SBERT stages can then simply encode `entity_text`.

#### Leave-Last-Out Checkpoint

`leave_last_out` is computed locally from timestamps. For every eligible user,
the latest interaction becomes the target and earlier interactions become the
source profile.

```bash
compresso-recsys-build-checkpoint \
  --dataset amazon2023 \
  --amazon_category Toys_and_Games \
  --checkpoint_path artifacts/amazon_toys/leave_last_out_exp001.zip \
  --split_mode leave_last_out \
  --metadata_text_fields title,features,description,categories \
  --min_entity_text_words 30 \
  --min_user_support 20 \
  --item_min_support 20 \
  --min_value_to_keep 4.0 \
  --set_all_values_to 1.0 \
  --min_source_items 1 \
  --min_target_items 1 \
  --annotation_source none
```

#### Temporal Checkpoint

`temporal` uses the Amazon Reviews 2023 predefined timestamp split when
`--dataset amazon2023` is selected. Targets are kept cold with respect to the
Amazon training split, so this checkpoint is intended for metadata/SBERT-style
cold-item evaluation.

```bash
compresso-recsys-build-checkpoint \
  --dataset amazon2023 \
  --amazon_category Toys_and_Games \
  --checkpoint_path artifacts/amazon_toys/temporal_exp001.zip \
  --split_mode temporal \
  --metadata_text_fields title,features,description,categories \
  --min_entity_text_words 30 \
  --min_user_support 20 \
  --item_min_support 20 \
  --min_value_to_keep 4.0 \
  --set_all_values_to 1.0 \
  --min_source_items 1 \
  --min_target_items 1 \
  --annotation_source none
```

You can use official category names or short aliases:

```bash
--amazon_category Toys_and_Games
--amazon_category toys
--amazon_category Electronics
--amazon_category electronics
--amazon_category Clothing_Shoes_and_Jewelry
--amazon_category clothing
```

Train SBERT embeddings on the resulting Amazon checkpoint:

```bash
compresso-recsys-train-sbert \
  --checkpoint_path artifacts/amazon_toys/temporal_exp001.zip \
  --model_name sentence-transformers/all-MiniLM-L6-v2 \
  --text_columns entity_text \
  --sbert_batch_size 64 \
  --device cuda
```

Train SAE on SBERT embeddings. For checkpoints with cold item indices, SAE fits
only on `train_item_indices` and then transforms all items for evaluation:

```bash
compresso-recsys-train-sae \
  --checkpoint_path artifacts/amazon_toys/temporal_exp001.zip \
  --embedding_stage sbert \
  --sae_k 128 \
  --sae_ste_alpha 0.01 \
  --sae_post_norm_l1 \
  --device cuda
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
