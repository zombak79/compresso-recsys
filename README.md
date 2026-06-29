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

## Documentation

The documentation can be built locally with Sphinx:

```bash
pip install -r docs/requirements.txt
pip install -e .
sphinx-build -b html docs/source docs/build/html
```

After GitHub Pages is enabled for the `gh-pages` branch, release documentation
will be available at:

```text
https://zombak79.github.io/compresso-recsys/
```

## What Is Included

- Dataset utilities for GoodBooks, MovieLens 1M, MovieLens 20M, and Amazon
  Reviews 2023.
- ZIP checkpoint format for source/target splits, embeddings, sparse
  embeddings, and metrics.
- ELSA and CompressedELSA training helpers.
- SAE and SBERT checkpoint stages.
- Retrieval metrics for Recall and nDCG at common cutoffs including 20, 50,
  and 100.
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
history. This is the preferred split when avoiding future-to-past leakage
matters:

```text
0core_timestamp_w_his_<category>
```

The builder also constructs a canonical `entity_text` column from configurable
metadata fields. Downstream SBERT stages can then simply encode `entity_text`.

#### Leave-Last-Out Checkpoint

`leave_last_out` is computed locally from timestamps. For every eligible user,
the latest interaction becomes the target and earlier interactions become the
source profile. This respects time within each user, but it is not globally
future-blind because other users may contribute later interactions to training.

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

Train SAE on SBERT embeddings. For checkpoints with item partitions, SAE fits
only on `train_item_indices` and then transforms all items for evaluation. For
warm `user_split` checkpoints, `train_item_indices` defaults to all items:

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

Checkpoint evaluation stores the common six-metric table:

```text
recall@20, ndcg@20, recall@50, ndcg@50, recall@100, ndcg@100
```

## Python API

```python
import compresso_recsys as cr

dataset = cr.MovieLens1M(data_dir="data")
interactions = dataset.get_interactions()

checkpoint_path = cr.build_recsys_checkpoint(
    dataset="ml1m",
    checkpoint_path="artifacts/ml1m/exp001.zip",
    annotation_source="genres",
)
```

Subpackages are also available:

```python
from compresso_recsys.datasets import Goodbooks, MovieLens1M, MovieLens20M
from compresso_recsys.checkpoint import update_checkpoint, load_recsys_split
```

## Checkpoint split schema

Every checkpoint stores source/target matrices for train, validation, and test:

```text
data/train_source_matrix.npz
data/train_target_matrix.npz
data/val_source_matrix.npz
data/val_target_matrix.npz
data/test_source_matrix.npz
data/test_target_matrix.npz
```

`source` is the profile/input side and `target` is what retrieval metrics try
to recover. The older `data/train_matrix.npz` file is still written as an alias
for `train_source_matrix.npz`.

Depending on the split mode, the checkpoint also stores partition ids:

- `user_split`: stores `train_user_ids.npy`, `val_user_ids.npy`, and
  `test_user_ids.npy`. It does not store explicit item partitions; loaders
  treat all items as train items.
- `item_split`: stores `train_item_indices.npy`, `val_item_indices.npy`, and
  `test_item_indices.npy`.
- `leave_last_out`: stores source/target matrices built from per-user latest
  interactions. It is chronological per user, but not globally future-blind.
- `temporal`: stores source/target matrices from a global timestamp split. For
  Amazon Reviews 2023, this uses McAuley's predefined temporal split.

Validation/test source-target rows also have aligned `val_eval_user_ids.npy`
and `test_eval_user_ids.npy` when user identifiers are available.

## All params for checkpoint builder

`--min_source_items 1` and `--min_target_items 1` mean:

```text
Keep an evaluation user only if they have at least 1 source item and at least 1 target item.
```

For cold-item splits:

```text
source items = warm/train items used as the user profile
target items = cold held-out items we want to recommend
```

So if a user has only cold targets but no warm source items, we cannot build a profile, and the user is dropped.

Here is the full current `compresso-recsys-build-checkpoint` parameter table.

| Parameter | Default | Description |
|---|---:|---|
| `--dataset` | required | Dataset to build. Choices: `goodbooks`, `ml1m`, `ml20m`, `amazon2023`. |
| `--data_dir` | `data` | Directory where raw/downloaded dataset files are stored. |
| `--checkpoint_path` | dataset-specific | Output ZIP checkpoint path. If omitted, uses the dataset default. |
| `--seed` | dataset-specific | Random seed for user/item splitting and reproducibility. |
| `--val_users` | dataset-specific | Number of validation users for `user_split`. |
| `--test_users` | dataset-specific | Number of test users for `user_split`. |
| `--min_user_support` | dataset-specific | Minimum number of interactions per user during iterative pruning. |
| `--item_min_support` | dataset-specific | Minimum number of interactions per item during iterative pruning. |
| `--min_value_to_keep` | dataset-specific | Drop interactions below this value. Usually `4.0`, meaning keep positive ratings only. |
| `--set_all_values_to` | dataset-specific | If set, binarize all remaining interaction values to this value. Usually `1.0`. |
| `--eval_fold` | `0` | Evaluation fold protocol for `user_split`. `0` means stacked 5-fold paper-style behavior; `1` means single fold. |
| `--split_mode` | `user_split` | Split protocol. Choices: `user_split`, `item_split`, `leave_last_out`, `temporal`. |
| `--val_items` | `None` | Exact number of cold validation items for `item_split`. Overrides `--item_val_frac`. |
| `--test_items` | `None` | Exact number of cold test items for `item_split`. Overrides `--item_test_frac`. |
| `--item_val_frac` | `0.05` | Fraction of items held out as cold validation items for `item_split`. |
| `--item_test_frac` | `0.10` | Fraction of items held out as cold test items for `item_split`. |
| `--temporal_test_frac` | `0.10` | For local temporal split, latest global fraction of interactions used as target side. For Amazon `temporal`, McAuley predefined timestamp split is used instead. |
| `--min_source_items` | `1` | Minimum number of source/profile items an eval user must have. For cold-item eval, these are train/warm items. |
| `--min_target_items` | `1` | Minimum number of target/held-out items an eval user must have. For cold-item eval, these are cold items. |
| `--amazon_category` | `Toys_and_Games` | Amazon Reviews 2023 category. Supports official names and aliases like `toys`, `electronics`, `clothing`. |
| `--metadata_text_fields` | `title,features,description,categories` | Metadata columns joined into canonical `entity_text`. Mostly important for Amazon/SBERT. |
| `--min_entity_text_words` | `30` | Drop items whose constructed `entity_text` is shorter than this many words. Mostly useful for Amazon. |
| `--annotation_source` | `genres` | Optional tag source for clustering. Choices: `genres`, `ml20m_tags`, `goodbooks_tags`, `none`. |
| `--annotation_min_count` | `100` | Minimum count threshold for tag annotations when using user-generated tags. |

Dataset-specific defaults:

| Dataset | `checkpoint_path` | `seed` | `val_users` | `test_users` | `min_user_support` | `item_min_support` |
|---|---|---:|---:|---:|---:|---:|
| `goodbooks` | `artifacts/goodbooks/recsys_checkpoint.zip` | `0` | `1000` | `2500` | `5` | `1` |
| `ml1m` | `artifacts/ml1m/recsys_checkpoint.zip` | `42` | `500` | `1000` | `5` | `1` |
| `ml20m` | `artifacts/ml20m/recsys_checkpoint.zip` | `42` | `2500` | `5000` | `5` | `1` |
| `amazon2023` | `artifacts/amazon2023/{amazon_category}/recsys_checkpoint.zip` | `42` | `2500` | `5000` | `20` | `20` |

All datasets currently default to:

| Parameter | Default |
|---|---:|
| `min_value_to_keep` | `4.0` |
| `set_all_values_to` | `1.0` |

## Supported Amazon 2023 datasets

| Official Amazon 2023 category | Alias in `compresso-recsys` | Supported? |
|---|---|---|
| `All_Beauty` | `beauty` | yes |
| `Amazon_Fashion` | none | yes, pass official name |
| `Appliances` | none | yes |
| `Arts_Crafts_and_Sewing` | none | yes |
| `Automotive` | none | yes |
| `Baby_Products` | none | yes |
| `Beauty_and_Personal_Care` | none | yes |
| `Books` | none | yes |
| `CDs_and_Vinyl` | none | yes |
| `Cell_Phones_and_Accessories` | none | yes |
| `Clothing_Shoes_and_Jewelry` | `clothing` | yes |
| `Digital_Music` | none | yes |
| `Electronics` | `electronics` | yes |
| `Gift_Cards` | none | yes |
| `Grocery_and_Gourmet_Food` | none | yes |
| `Handmade_Products` | none | yes |
| `Health_and_Household` | none | yes |
| `Health_and_Personal_Care` | none | yes |
| `Home_and_Kitchen` | none | yes |
| `Industrial_and_Scientific` | none | yes |
| `Kindle_Store` | none | yes |
| `Magazine_Subscriptions` | none | yes |
| `Movies_and_TV` | none | yes |
| `Musical_Instruments` | none | yes |
| `Office_Products` | none | yes |
| `Patio_Lawn_and_Garden` | none | yes |
| `Pet_Supplies` | none | yes |
| `Software` | none | yes |
| `Sports_and_Outdoors` | none | yes |
| `Subscription_Boxes` | none | yes |
| `Tools_and_Home_Improvement` | none | yes |
| `Toys_and_Games` | `toys`, `toys_and_games` | yes |
| `Video_Games` | none | yes |
