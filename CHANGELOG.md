# Changelog

All notable changes to Compresso Recsys are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and
this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
While the version stays below 1.0, the minor number carries breaking changes.

Entries before 0.2.0 are reconstructed from release history rather than written
at the time, so they summarise what each tag contained rather than listing every
change.

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
