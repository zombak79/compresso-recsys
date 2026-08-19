# Changelog

All notable changes to Compresso Recsys are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and
this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
While the version stays below 1.0, the minor number carries breaking changes.

Entries before 0.2.0 are reconstructed from release history rather than written
at the time, so they summarise what each tag contained rather than listing every
change.

## [Unreleased]

### Fixed

- **`leave_last_out` was not leave-last-out.** It removed from training every
  item that was *anyone's* last interaction, forcing all targets cold: 56% of
  the catalog at MovieLens-1M shape, 86% at MovieLens-20M shape. A warm model
  scored zero on it by construction, since the items it had to rank were exactly
  the ones absent from its training. It now holds out each user's last two
  interactions and leaves the catalog whole; item partitions are observed rather
  than imposed, so they are empty unless an item's every occurrence falls in a
  held-out tail.
- **`leave_last_out` validation was test.** `val_holdout` and `test_holdout`
  were the same object, so any tuning on validation was tuning on test. The
  second-to-last interaction is now the validation target and the last the test
  target, with the third-to-last as the training target.
- `leave_last_out` now populates `train_source_matrix` and
  `train_target_matrix` distinctly, with `x_train` as their union, matching what
  `temporal` already did.

### Added

- `ItemSequences`, a chronological interaction history: catalog indices in order
  with duplicates preserved, and nothing else. No padding, no `MASK`/`PAD`/`BOS`,
  no maximum length, no truncation — tokenisation is a modelling decision that
  models disagree about, so it happens in trainers rather than the checkpoint.
  Rows may be empty and carry no identity, matching how `csr_matrix` sources are
  already aligned. Exported alongside `save_item_sequences` and
  `load_item_sequences`.
- `BaseSequentialRecommender` and the `SequentialRecommender` protocol, parallel
  to their matrix siblings rather than derived from them. `predict_on_batch`
  reads an `ItemSequences` and returns an `SRPTensor`, so everything downstream
  is unchanged. `fit` is deliberately outside the contract: trainers keep the
  existing `SomeTrainer(config).fit(data)` shape and the fitted model owes only
  prediction. The base assumes neither that the history vocabulary equals the
  candidate catalog, nor that a truncating encoder may recommend what it did not
  read — `exclude_seen` masks the whole history either way.
- `evaluate_recommender` accepts either source type. Its batching loop asks a
  source only for its row count and a row slice, so one adapter covers
  `csr_matrix` and `ItemSequences` answers natively. Nothing downstream of
  `predict_on_batch` knows which was given, which is what lets a sequential model
  and a matrix model appear in one `compare_models` call with no
  statistics-side changes.
- Sequence views are persisted and loaded. A checkpoint from a chronological
  split mode now stores `x_train_sequences` and `{stage}_source_sequences`
  alongside the matrices, and the manifest lists only the ones that mode
  produced. Loading a checkpoint without them yields `None`, so anything built
  before sequences existed — or by a non-chronological mode — still loads
  complete for every matrix model. Saving refuses a sequence whose catalog
  disagrees with its own stage's item IDs, since the two views must share a
  column space — per stage, because temporal windows each have their own
  catalog.
- The chronological split modes build sequence and matrix views of the same
  events in one pass. `leave_last_out` and `temporal` payloads now carry
  `x_train_sequences`, `train_source_sequences`, `val_source_sequences` and
  `test_source_sequences`; `user_split` and `item_split` carry none, having no
  ordering to preserve. Each sequence addresses the same rows and columns as the
  matrix it accompanies.
- `SequenceBatcher`, the encoding step shared by sequential architectures:
  where special tokens live in the vocabulary, how a ragged batch becomes a
  dense tensor, which positions are real, and how far back to look. Catalog ids
  are the identity and special tokens are appended, so adding a second special
  token cannot shift items out from under an already-trained model.
  `pad_side="right"` suits an RNN reading to each row's own final position and
  `"left"` a causal transformer reading position `-1`; `max_length` truncates to
  the most recent interactions. It deliberately owns no training objective —
  next-item shift, masked positions and sampled negatives differ per
  architecture and stay in trainers.
- `SimpleRNN`, `SimpleRNNConfig` and `SimpleRNNTrainer`: a GRU or LSTM trained
  on next-item cross entropy at every position, one example per user. The
  smallest model that actually uses order, and so the baseline a transformer has
  to beat before its extra machinery has earned anything. Prediction reads each
  row's own final state rather than the last column, which under right padding
  would score most users from a pad embedding; `exclude_seen` masks the whole
  history including the part `max_length` truncated away.
- `ItemSequences.select_rows`, the non-contiguous counterpart to `take_rows`,
  which is what shuffling a training set needs.
- `save_recsys_split` enforces `x_train = train_source_matrix ∪
  train_target_matrix` and refuses a checkpoint whose training keys disagree.
  The relationship was already true of every split mode but nothing checked it,
  so a new mode could partition its training data inconsistently and be written
  out. Its docstring now states the relationship and the per-mode partition
  rule: `temporal` divides by time, `leave_last_out` by position, and the
  non-chronological modes not at all.

### Changed

- **Breaking.** `leave_last_out` derives its minimum history from the support
  arguments rather than hardcoding it: the structural floor of four, raised by
  `min_user_support`, and by `min_source_items` plus the three stage targets.
  `min_target_items` above 1 is refused, since each stage holds out exactly one
  item. The rewritten protocol had briefly ignored all three.
- **Breaking.** `leave_last_out` requires at least four interactions per user,
  up from two, so every stage has a non-empty source and target. Results from
  this split mode change; checkpoints built before this release are not
  comparable to ones built after.
- **Breaking.** `build_leave_last_out_holdout` takes `stage` and `min_history`
  in place of `min_source_items` and `min_target_items`, and returns one stage at
  a time.

## [0.2.0] — 2026-08-17

Paired statistical comparison of evaluation results, the per-user retention it
depends on, and two evaluation-protocol corrections found while validating it.

### Added

- `compresso_recsys.stats`, with `compare_pair` and `compare_models`. Comparison
  works from the paired per-user difference rather than aggregate means: a
  paired bootstrap over users gives the confidence interval, and a paired
  sign-flip randomization test gives the p-value.
- `test_method` selects the test — `"randomization"` (default, exact under
  paired-label exchangeability), `"bootstrap"`, or `"t"` for a paired t-test run
  as a one-sample test on the same differences.
- Multiplicity correction across every pair and metric one call produces, since
  those hypotheses are dependent through overlapping users. Holm by default,
  Bonferroni and `None` available.
- `EvaluationResult`, returned by both evaluation entry points in place of a
  plain dict. It remains a `Mapping`, so `result["ndcg@20"]` and `dict(result)`
  keep working, and additionally carries per-user metric values and the
  `sample_ids` that pair them. `collect_per_user=False` opts out.
- `EvaluationResult.target_fingerprint`, a canonical digest of the target matrix
  that is independent of `batch_size`. Comparison refuses two results scored
  against different targets, which matching identifiers cannot detect — two
  evaluations on unrelated datasets both number their rows from zero.
- Unit-aware resampling. Repeated `sample_ids` mean one user produced several
  rows; each user is reduced to the mean of their rows and every procedure runs
  on those, so a user evaluated five times counts once. `n_units` reports how
  many independent units there were.
- `eval_holdout_frac` on `build_recsys_checkpoint`, `build_eval_holdout` and
  `evaluate_item_embeddings`: the share of each held-out user's history scored
  against, the rest being the fold-in history the model sees.
- `show_progress` on `compare_pair` and `compare_models`, drawing a tqdm bar
  when the package is installed. Cost is linear in units and hypotheses, so a
  large evaluation compared across several metrics and models runs for minutes.
  The bar counts hypotheses but advances within each one, so it still moves when
  a call produces a single slow comparison.
- `docs/statistical-comparison`, a guide assuming no statistics background,
  built around a reproducible EASE versus ELSA comparison on GoodBooks.
- Citations for the four statistical methods and for the evaluation protocol
  itself — Liang et al. (2018), previously uncited despite every `user_split`
  number being produced under it.

### Changed

- **Breaking.** `RankingMetric.update()` must now return the per-row values it
  computed, shape `(rows, len(result_keys))`, rather than only folding them into
  a running sum. Without this the evaluator would compute every metric twice to
  retain per-user observations. Third-party metric implementations must be
  updated.
- **Breaking.** `EvaluationResult.n_eval_users` is now `n_scored_rows`, which is
  what it counts: rows carrying at least one target. `n_units` reports the
  distinct identifiers among them. Both appear in the mapping view.
- **Breaking.** `eval_fold` is replaced by `eval_draws`, which means the count it
  always looked like. The old parameter was a two-state switch — the
  implementation branched on `fold != 1`, so `0`, `5` and `10` all produced the
  five draws the protocol stacks. Verified byte-identical: `eval_draws=5`
  reproduces `eval_fold=0` and `eval_draws=1` reproduces `eval_fold=1`.
- **Breaking.** Checkpoint metadata records `n_val_eval_rows` and
  `n_test_eval_rows` for row counts, and reserves `n_val_eval_users` and
  `n_test_eval_users` for distinct users. They previously held row counts under
  the user name, overstating the evaluation by the draw count.
- `compare_models` refuses a one-sided `alternative` without a `reference`.
  Direction otherwise came from mapping insertion order, which is cosmetic for a
  two-sided test and the entire hypothesis for a directional one.
- Sphinx reads the version from `pyproject.toml` before falling back to
  installed metadata, so the documentation is no longer labelled with whatever
  happens to be installed in the build environment.

### Removed

- **Breaking.** `--eval_fold` from the CLI, replaced by `--eval_draws` and
  `--eval_holdout_frac`. No deprecation shim: the package is alpha and nothing
  read the value back — it was written into checkpoint metadata and never
  consulted.
- **Breaking.** `min_items_per_user` from `evaluate_item_embeddings`, which
  accepted and ignored it. Its `holdout_frac` becomes `eval_holdout_frac` and is
  now honoured; it too was previously ignored, so passing `0.5` held out 20%
  exactly as `0.2` did.

### Fixed

- Metric accumulation cast to `float64` on the source device, so evaluating any
  model held on an Apple GPU raised `TypeError`. The host copy now precedes the
  widening.
- `eval_holdout_frac` is honoured rather than hardcoded at 0.2 in three
  signatures that accepted it.

## [0.1.2] — 2026-08-14

### Added

- Cold-start recommenders: TEASER and its gradient-trained variant, with
  `BaseColdStartRecommender`, `CandidateCatalog` and a warm-catalog adapter.
- Temporal and user split protocols alongside the existing item split.

## [0.1.1] — 2026-07-29

### Added

- Ranking evaluation: `RankingEvaluator`, `evaluate_recommender`,
  `evaluate_ranked_predictions`, and the metric set — `CalibratedRecall`,
  `Recall`, `Precision`, `HitRate`, `MRR`, `MAP`, `NDCG`.
- Retrieval evaluation over item embeddings, and worked examples for the
  collaborative filtering models.

## [0.1.0] — 2026-07-09

Initial release. EASE, ELSA and CompressedELSA, checkpoint building and the
dataset loaders.

[0.2.0]: https://github.com/zombak79/compresso-recsys/releases/tag/v0.2.0
[0.1.2]: https://github.com/zombak79/compresso-recsys/releases/tag/v0.1.2
[0.1.1]: https://github.com/zombak79/compresso-recsys/releases/tag/v0.1.1
[0.1.0]: https://github.com/zombak79/compresso-recsys/releases/tag/v0.1.0
