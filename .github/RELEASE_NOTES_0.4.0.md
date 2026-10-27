# compresso-recsys 0.4.0

compresso-recsys is a toolkit for practical recommender development, and the
recommender companion of Compresso. 0.4.0 completes its three parts, which you can
use alone or together: **one data checkpoint** (matrix, sequences, item metadata,
embeddings and splits from a single command, now with recorded provenance and room
for results and model checkpoints), a **model base class** that handles the boring
parts around your model's own `fit` (`predict`, `predict_on_batch` batching,
`recommend` with stable item IDs and filters, data handling, save and load), now
with early stopping and a user-identity channel, and **evaluation with
statistical testing** as the final arbiter, where paired comparisons refuse
mismatched protocols.

## Highlights

- **Paired comparison you can trust.** `compare_models` now refuses to pair results
  whose evaluation source (new `EvaluationResult.source_fingerprint`), targets or
  `exclude_seen` setting differ. The seen-item choice is recorded in
  `EvaluationResult.protocol`.
- **Better defaults, fewer arguments.** `evaluate_recommender` needs no `metrics=`:
  it defaults to `CalibratedRecall([20, 50])` and `NDCG(100)`, the
  strong-generalisation protocol of Liang et al. (2018) and Steck (2019).
- **Provenance checkpoints (manifest v2).** Per-stage provenance (library, git commit,
  Python and package versions, never a hostname), resolved build arguments, SHA-256
  of every raw source file, and `content_sha256`, a digest of the data that is
  identical across rebuilds. Read it with `load_provenance`. Checkpoints still load
  in 0.3.x.
- **BERT4Rec** (`Bert4Rec`, `Bert4RecConfig`, `Bert4RecTrainer`): a faithful port of
  the published architecture and Cloze objective. The MovieLens 1M leave-last-out
  notebook records HR@10 0.6965 and NDCG@10 0.4805 against 100 popularity-sampled
  negatives (paper: 0.6970 / 0.4818), labelled an adapted benchmark.
- **Early stopping.** `fit(..., validation=Validation(...), early_stopping=EarlyStopping(metric, patience))`
  on ELSA (dense), MultDAE, MultVAE, SASRec, SimpleGPT, SimpleRNN,
  SimpleBidirectionalTransformer and TEASER-GD. Validation goes through
  `evaluate_recommender` and draws no random numbers; the best weights are restored
  by default and recorded in `trainer.best_`.
- **CLCRec** (`CLCRec`, `CLCRecConfig`, `CLCRecTrainer`): the contrastive cold-start
  model of Wei et al. (ACM MM 2021), implemented from the paper. Warm candidates are
  scored by learned ID embeddings and cold ones by a content encoder; it is the first
  model to use the user-identity channel, with optional fold-in for unseen users.
- **Single import.** Evaluation, comparison, metric and model entry points are
  importable from the top level (`cr.EASE`, `cr.evaluate_recommender`,
  `cr.compare_models`, `cr.NDCG`, ...). `import compresso_recsys` is now lazy and no
  longer loads PyTorch unless you use a model.

## Breaking and behaviour changes

- **`ItemKNNRecommender` no longer normalises scores by default.**
  `ItemKNNConfig(normalize_scores=False)` is the new default: on MovieLens 1M
  (strong generalisation, k=100) the plain weighted sum raised nDCG@100 from 0.329
  to 0.380. Use `ItemKNNConfig(normalize_scores=True)` for the old scores.
  Checkpoints saved before 0.4.0 keep normalised scoring when loaded.
- **Modules reorganised into packages by role** (`builder`, `checkpoint`,
  `evaluation`, `stats`, `retrieval` and the large models). Public import paths are
  unchanged, but the **private modules moved**: `models._batching`, `models._ranking`,
  `models._validation`, `models._schedule` and `models._autoencoder_batching` now
  live under `models.core`. Code importing them must update its imports.
- **`evaluate_recommender` metric default** is new (see above); pass `metrics=` to keep
  your own set.
- **Fix that changes numbers with repeated draws:** `evaluate_item_embeddings` and
  `evaluate_item_embeddings_with_holdout` now pass the holdout's user ids, so with
  `eval_draws > 1` repeated draws of one user are no longer counted as independent
  users in `compare_models` (confidence intervals were too narrow before). Results
  with `eval_draws=1` are unchanged.
- Builder and download progress messages go to **stderr** instead of stdout.
- `tqdm` is now a declared dependency. Development status is Beta.
- Python 3.13 and 3.14 classifiers removed until CI covers them (see below).

## New APIs

- `ComparisonReport.summary()`, `to_latex()` (long or compact booktabs layout) and
  `to_markdown()`; p-values on the Monte Carlo floor print as upper bounds.
- `evaluate_recommender(..., exclude_seen=None)` fixes the seen-item policy for every
  model compared.
- User-identity channel for models with a learned embedding per training user
  (`accepts_user_ids = True`, `predict_on_batch(..., user_ids=)`,
  `recommend(user_ids=...)`, `cfg.unknown_users` = `"error"` or `"fold_in"`).
- `stats.adjust_p_values` (Holm or Bonferroni), `models.batching.normalized_mse`,
  and the public `MeanAtCutoffsMetric` base for custom metrics, with an
  "Adding a metric" guide whose code runs in the test suite.
- `build_recsys_checkpoint(dataset_options=...)` and repeatable
  `--dataset_option KEY=VALUE`; Retailrocket, Music4All-Onion, OTTO and Yambda
  registrations from 0.3.7 in the reorganised `builder` package.
- `examples/interop/`: evaluate models from other libraries (implicit's ALS, WarpRec's
  SASRec) through `evaluate_recommender` and `compare_models` with a short adapter.
- `examples/paper/`: the benchmark runner (`list`, `train`, `select`, `compare`,
  `bundle`), its configuration, a minutes-long ML-1M smoke configuration, and a docs
  page on reproducing the benchmark and verifying its results deposit.

## CI and Python policy

- Tests run on Python 3.10, 3.11 and 3.12 (required) with 3.13 allowed to fail,
  network access disabled, a coverage report and a pyflakes lint gate. A test pins
  the on-disk layout of model checkpoints per model type.
- Supported versions and the add/drop policy are in `CONTRIBUTING.md`. Python >= 3.10.

## Documentation, licences and citation

- New `CONTRIBUTING.md`, issue templates, `CITATION.cff`, BibTeX in the citation guide
  and a dataset licences page. Several datasets restrict redistribution of derived
  checkpoints (for example Netflix Prize and MovieLens); read the licences page before
  sharing checkpoints.
- Documentation: https://zombak79.github.io/compresso-recsys/
- Full list of changes: CHANGELOG.md.

## Install

```bash
pip install compresso-recsys==0.4.0
```

## Cite as

Cite this release (see `CITATION.cff`); the Zenodo DOI is listed on this release
page once archived. A paper describing compresso-recsys is under submission.
