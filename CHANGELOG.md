# Changelog

All notable changes to Compresso Recsys are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and
this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
While the version stays below 1.0, the minor number carries breaking changes.

Entries before 0.2.0 are reconstructed from release history rather than written
at the time, so they summarise what each tag contained rather than listing every
change.

## [Unreleased]

## [0.3.0] — 2026-08-26

Sequential recommendation as four replaceable parts -- tokenizer, batcher,
model, trainer -- with `SimpleGPT` as the worked example and `SimpleRNN`
rebuilt on the same layers. The cold-start catalog becomes an owned object
rather than inherited state, and an evaluation defect that fabricated
adjacency in cold-item sequences is removed.

### Fixed

- The `SimpleRNN` example in `docs/api/models` passed `max_length` to
  `SimpleRNNConfig`, which stopped existing when that field moved to the batcher
  in this release. The documented snippet raised `TypeError`; Sphinx does not
  execute code blocks, so building with `-W` never caught it. Every snippet in
  that section is now run rather than only rendered.

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
- `ItemTokenizer` and the `Tokenizer` protocol, splitting the vocabulary out of
  `SequenceBatcher`. Specials now come **first** and the catalog is offset after
  them, because catalog growth is what actually happens: stage catalogs nest by
  prefix, cold-start catalogs append, and an incremental fit extends the
  embedding table at the end — where a `cat` splices optimizer state correctly
  instead of permuting it, and a wrong permutation would attach one item's
  momentum to a special token without raising. `n_reserved` removes
  front-loading's one cost, so a special added later lands in a reserved slot and
  no trained id moves. The offset stops at the tokenizer: a model's head is
  `n_items` wide and catalog-indexed, so predictions leave in the same space as
  the target matrix and the metrics.
  An item outside the catalog becomes `unk` and keeps its position, which is the
  fix for a real defect — dropping it instead joins its neighbours as though they
  had been consecutive, fabricating 21% of all adjacencies on a temporal
  MovieLens-1M split, across every row, at a cost of about 9% of ndcg@20. A
  vocabulary built without `unk` raises rather than guesses. IDs are an optional
  presentation layer: `encode_indices`/`encode_ids`/`decode_indices`/`decode_ids`
  are explicit, `encode` dispatches on dtype, and integer `item_ids` are refused
  because they would make that dispatch ambiguous. `to_dict`/`from_dict` persist
  the vocabulary with the weights, including `item_ids` by default, since serving
  is why they exist.
- `SequenceBatcher` now takes a tokenizer and owns only ragged-to-dense: padding,
  truncation, and reading the result. The two are separate because they have
  different lifetimes — a vocabulary belongs to the dataset, while `max_length`
  is `block_size` under another name and belongs to the model, so one tokenizer
  can serve two models that read different amounts of history. `pad_side` still
  defaults to `"right"`, now documented as the better default for a *causal*
  model too: pads sit after every real token, so a causal mask already excludes
  them and training needs no attention mask, only a loss mask. `catalog_logits`
  is gone, unnecessary once the head is catalog-width.
- `SimpleGPT`, `TransformerConfig`, `SimpleGPTConfig` and `SimpleGPTTrainer`: a
  causal transformer over the same histories `SimpleRNN` reads — nanoGPT with two
  recommendation-shaped adjustments, and the model the tokenizer/batcher split was
  made for.
  A `CLS` prefix replaces the next-item shift: position 0 holds a learned vector,
  so `states[:, i]` has read `CLS` plus `tokens[:, :i]` and predicts
  `tokens[:, i]`. The alignment becomes a property of the input rather than
  arithmetic in the trainer, every real position becomes a target where a left
  shift skips the first, a single-interaction history becomes a usable example,
  and an empty history gets a *defined* input instead of the state after reading
  one pad. `CLS` is an `nn.Parameter` rather than a vocabulary entry so it can be
  conditioned later — a user or global vector added into position 0 per row, which
  a lookup cannot express.
  There is no attention mask. Right padding means a causal mask already excludes
  the padding, so the attention module accepts no mask argument and the trainer
  *refuses* a `pad_side="left"` batcher rather than quietly building one. It also
  refuses `max_length=None`, since that value sizes the positional table —
  `block_size` is derived rather than configured, so the two cannot disagree.
  `save_simple_gpt` / `load_simple_gpt` carry the vocabulary with the weights,
  because a served model that cannot say what column 41 means is not much use.
  Loading is self-contained and reads with `weights_only=True`, parsing the file
  as data rather than executing it as a pickle.
  Measured `ndcg@20` on two datasets, as a mean over three training seeds with
  its standard deviation, every sequential budget selected on the **validation**
  split from the grid `1, 2, 3, 5, 10, 20` and only then scored on test, against
  a popularity floor. On MovieLens-1M `leave_last_out` (6,033 users) SimpleGPT
  scores 0.1520 ± 0.0017 against SimpleRNN's 0.1471 ± 0.0008 — a gap of
  `+0.0049`, about three times the larger of the two deviations — and both are far
  past ELSA's 0.0562 and popularity's 0.0176. On Amazon Office_Products
  `leave_last_out` the same three models land within 0.005 of each other
  (ELSA 0.0327 ± 0.0011, SimpleGPT 0.0310 ± 0.0013, SimpleRNN 0.0282 ± 0.0018),
  and on its `temporal` split **nothing beats popularity's 0.0094**: SimpleRNN
  0.0091, SimpleGPT 0.0088, ELSA 0.0076. Office targets average exactly 1.0 items
  per user with no repeats in the first two thousand — a purchase history is
  nearly a set, so there is little order to exploit — which is why any
  sequential-versus-matrix claim has to name its dataset.
  Two cautions attach to those numbers. Validation picked the *edge* of the grid
  on MovieLens and both curves were still rising, so 0.1520 and 0.1471 are lower
  bounds. And the budget mattered more than the architecture: it moved Office
  temporal by 43% (0.0111 at three epochs against 0.0063 at twenty), overfitting
  after five epochs there while MovieLens still improved at twenty. Neither
  trainer reads the `val_source_sequences` and `val_target_matrix` that every
  chronological checkpoint already carries, so `epochs` remains a guess — select
  it on validation, and never on test, which on Office temporal would have read
  0.0111 instead of 0.0088.
- `SimpleRNN`, `SimpleRNNConfig` and `SimpleRNNTrainer`: a GRU or LSTM trained
  on next-item cross entropy at every position, one example per user. The
  smallest model that actually uses order, and so the baseline a transformer has
  to beat before its extra machinery has earned anything. Prediction reads each
  row's own final state rather than the last column, which under right padding
  would score most users from a pad embedding; `exclude_seen` masks the whole
  history including the part truncation dropped. The encoder is a constructor
  parameter rather than something `fit` invents, so the context window, the
  padding side and the vocabulary are all replaceable — and `max_length` left
  `SimpleRNNConfig`, since it describes what the encoder reads rather than the
  network. Its next-item objective decodes targets with `tokens - n_reserved` and
  excludes `unk` from them, because "predict the item I cannot identify" is not a
  question with an answer.
- `MutableCandidateCatalog`, the candidate-catalog lifecycle as an owned object:
  the lock, the current snapshot, the fitted source vocabulary, and the
  operations that publish, extend, shrink and align against them.
  `CandidateCatalog` was always a standalone immutable snapshot; what was stuck
  inside `BaseColdStartRecommender` was the lifecycle around it, which made
  "cold-capable" and "inherits that base" the same statement. Adding the
  sequential axis would then have forced multiple inheritance or a fourth base
  class for two independent ideas. Composition rather than a mixin, because the
  state is what decides it: a mixin would install nine attributes on whatever
  class it is mixed into, six of them public, and two stateful mixins
  initialising through `super().__init__()` is where MRO pain lives. Reads go
  through `snapshot()` rather than forwarded properties, so several reads cannot
  straddle a concurrent republish. `on_publish` notifies the owner while the lock
  is held, which is how a model drops caches derived from the previous snapshot.
- `SimpleRNNConfig.unk_dropout`, replacing that fraction of *input* positions
  with the tokenizer's `unk` token during training. Non-zero by default, because
  otherwise `unk` is never trained at all: the training vocabulary *is* the
  training window, so an out-of-catalog item cannot occur until evaluation and
  the embedding row would still sit at its initialisation when a quarter of a
  temporal test history needs it. Applied after the next-item shift and never to
  the targets, so a corrupted position teaches "an item was here you cannot
  identify, predict the next one anyway" rather than costing a training example.
  Padding is never eligible, or `unk` and `pad` would come to mean the same
  thing. Measured on a temporal MovieLens-1M stage with 26% out-of-catalog test
  histories: `ndcg@20` is 0.090 at rate zero, 0.113 at 0.05 and 0.121 at 0.25 —
  and rate zero scores no better than deleting the unknown items outright, so
  representing them and training them are one change, not two.
- `WarmCatalogAdapter` accepts an `ItemSequences` source, unaligned, and widens
  its prediction columns. It no longer *projects* histories: `align_source`
  refuses a sequence and says why, since a model's tokenizer maps an
  out-of-catalog index to `unk` in place, and dropping those items instead joined
  their neighbours as though they had been consecutive. The matrix path still
  projects, because a CSR row has no adjacency to corrupt and a set model has no
  `unk` to fall back on. Only the output widening is shared, and it stays
  necessary: the evaluator requires prediction width to match the targets. Without it the `temporal` split
  mode and sequential models could not meet: every temporal stage has its own
  expanding catalog, so a model fitted on the training window cannot even accept
  a test source, and the matrix side had an adapter while the sequence side had
  none. One class rather than two, because the projection is one operation on two
  views — keep the fitted columns, or keep each history's fitted items in order —
  and the vocabulary validation, index mapping, identity fast path and
  device-cached remap are shared verbatim. Rows survive either way, so a user
  whose history is entirely cold becomes an empty row rather than disappearing,
  which is what keeps an aligned source row-aligned with its targets.
  Also documented is when to reach for it outside `temporal`: under
  `leave_last_out` the catalogs match, but items whose every occurrence falls in
  a held-out tail are still absent from training, and the model families do not
  bury such columns alike. A softmax next-item objective pushes down every
  non-target logit, and a never-trained item is never a target, so SimpleRNN
  ranks them at the 95th percentile on MovieLens-1M where ELSA leaves them at the
  60th. Neither figure is about recommendation quality, so a cross-family
  comparison is sounder with both models wrapped — when the data warrants it,
  which on MovieLens-1M it does not: 3 of 6,033 rows.
- `save_recsys_split` enforces two invariants the sequence and matrix views rest
  on, both of which were already true of every split mode and neither of which
  anything checked. **Stage catalogs must nest by prefix**, so a warm item keeps
  its column index in every later stage — which is what lets a model fitted on
  the training catalog read a later stage's indices directly, treating anything
  at or above its own item count as unseen. `temporal` grows the catalog window
  by window and the other modes hold it fixed, but a mode that re-sorted IDs per
  stage would have made an index mean different items in different stages without
  failing. **And a sequence must describe the same events as the view beside it**
  — sharing a column space is not enough, since two views built from different
  filter passes can agree on shape and disagree on contents, which trains a
  matrix model and a sequential model on different data while every shape check
  passes. The comparison is per row and set-wise, because order and duplicates
  are the sequence view's whole purpose and a CSR row can express neither.
- `ItemSequences.select_rows`, the non-contiguous counterpart to `take_rows`,
  which is what shuffling a training set needs.
- An end-to-end test spanning one `leave_last_out` build, a checkpoint round
  trip, both model families and one `compare_models` call. The signal is one
  only order carries: users read windows of a cycle, so the two candidates
  outside a window are indistinguishable by co-occurrence and only the direction
  of time separates them. EASE places the target in its top 5 for every user and
  then puts the *past* neighbour first for 108 of 120 — the sequential model is
  credited with fixing exactly that, and nothing else.
- `docs/api/sequences` and a sequential section of `docs/api/models`, covering
  what a sequence holds and deliberately does not, which split modes produce
  one, the sequential contract and base, the vocabulary layout, and why
  prediction reads a gathered final position rather than the last column.
- `save_recsys_split` enforces `x_train = train_source_matrix ∪
  train_target_matrix` and refuses a checkpoint whose training keys disagree.
  The relationship was already true of every split mode but nothing checked it,
  so a new mode could partition its training data inconsistently and be written
  out. Its docstring now states the relationship and the per-mode partition
  rule: `temporal` divides by time, `leave_last_out` by position, and the
  non-chronological modes not at all.

### Changed

- **Breaking.** The item-partition keys are renamed to say what they hold:
  `train_item_indices` → `warm_item_indices`, `val_item_indices` →
  `val_cold_item_indices`, `test_item_indices` → `test_cold_item_indices`. The
  old spelling promised a relationship to `{phase}_item_ids` that does not
  exist — the two answer different questions, and they agree only by coincidence
  and only under `temporal` and `user_split`, where the catalogs already encode
  the partition. Under `leave_last_out` and `item_split` all three phases share
  one catalog and the partition is *observed*, so mirroring the catalogs would
  have reported every item warm in exactly the two modes where the cold items
  are the point. The warm partition is exactly the columns present in `x_train`
  in all four modes, which is what every cold-start `fit` consumes.
  Checkpoints written before the rename are read under their old filenames
  first. That fallback is load-bearing rather than courteous: a missing warm file
  defaults to the whole catalog and a missing cold file to nothing, so reading
  only the new names would turn an older checkpoint into a confident wrong
  answer that nothing downstream could detect.
  The `train_item_indices` argument to `TEASER.fit`, `TEASERGD.fit` and CSELSA is
  **unchanged** — it names what the model does with the columns rather than what
  they are, so a call now reads
  `fit(..., train_item_indices=split["warm_item_indices"])`.

- **Breaking.** The six fitted `source_*_` attributes moved onto the owned
  catalog: `model.source_item_ids_` is now `model.candidates.source_item_ids`,
  and likewise for `source_vocabulary_`, `source_id_to_row_`,
  `source_popularity_`, `feature_space_id_` and `n_input_features_`. No
  forwarding properties — that would give one piece of state two names for the
  sake of a convention.
- **Breaking.** `model.candidates` is the `MutableCandidateCatalog`, not a
  snapshot. Reading a field off it becomes `model.candidates.snapshot().n_items`.
  `n_items` is the one convenience kept directly on the holder, because it must
  answer before a catalog is installed. `n_candidates_` is removed in its favour.
- **Breaking.** `_install_feature_catalog` and `_resolve_candidate_selection` are
  gone from `BaseColdStartRecommender`; a subclass calls
  `self.candidates.install(...)` and `self.candidates.resolve_selection(...)`.
  `build_candidates`, `update_candidates`, `remove_candidates` and
  `align_source` are unchanged on the model, now as a facade over the owned
  catalog.
- **Breaking.** The `ColdStartRecommender` protocol no longer requires a
  `source_vocabulary_` member, and its `candidates` is now a
  `MutableCandidateCatalog`. Structural checks against it are unaffected
  otherwise.
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

[0.3.0]: https://github.com/zombak79/compresso-recsys/releases/tag/v0.3.0
[0.2.0]: https://github.com/zombak79/compresso-recsys/releases/tag/v0.2.0
[0.1.2]: https://github.com/zombak79/compresso-recsys/releases/tag/v0.1.2
[0.1.1]: https://github.com/zombak79/compresso-recsys/releases/tag/v0.1.1
[0.1.0]: https://github.com/zombak79/compresso-recsys/releases/tag/v0.1.0
