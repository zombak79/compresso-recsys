## [Unreleased]

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

- A production-facing ``model.recommend(...)`` API inherited by collaborative,
  sequential, and cold-start recommenders. It accepts a batch of histories as
  stable item IDs, applies batch-wide ``allowlist`` and ``blocklist`` filters
  before top-k selection, and returns
  immutable ranked IDs and scores through ``Recommendations``. Unknown IDs
  raise. Rows with fewer than ``k`` eligible candidates are truncated
  independently (after removing seen items when requested) and exposed through
  ``valid_mask`` and ``valid_counts``;
  padded positions are ``None``/``-inf`` and ``to_dicts()`` omits them. Strict
  callers can select ``on_insufficient="raise"``. Fixed-catalog fits accept
  optional ``item_ids`` and use positional integer IDs when omitted;
  fitted-model checkpoints preserve the identity mapping.
  ``exclude_seen`` defaults to ``False`` for production use, where repeated
  recommendations are legal and callers can block particular seen IDs;
  evaluation-style exclusion remains an explicit option.
  ``WarmCatalogAdapter`` supports the same API while keeping appended cold
  items unreachable to its wrapped transductive model.
- Unified fitted-model persistence through `model.save(path)` and
  `ModelClass.load(path, device="cpu")` on the collaborative, sequential,
  and cold-start bases. EASE, dense and compressed ELSA, SimpleRNN, SimpleGPT,
  ContentRecommender, TEASER, and TEASERGD now round-trip as prediction-ready
  models in a versioned ZIP format. Configurations, safe item IDs, Torch state,
  sparse and dense arrays, metadata, histories, tokenizers, mappings, and the
  current mutable candidate catalog travel with the fitted model. Loading is
  CPU-first and rejects the wrong model class or unsupported format version.
  Torch-backed recommenders can be moved after loading with ``model.to(device)``;
  the shared operation keeps model, configuration, optimizer tensors, and
  device-specific caches synchronized.
  Optimizer state is excluded by default and can be included and restored
  explicitly; scheduler, RNG, partial-epoch, compiled-wrapper, and cache state
  remain runtime concerns rather than an exact-resumption promise.
- `ModelCheckpointWriter`, `ModelCheckpointReader`,
  `BasePersistableRecommender`, and the structural
  `PersistableRecommender` protocol. The reader and writer own the manifest
  and safe typed storage operations so third-party models do not invent a ZIP
  layout. `WarmCatalogAdapter` remains a stage-specific projection and is
  rebuilt around its loaded nested model and dataset catalog IDs.
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
- `SequenceBatcher`, the encoding step shared by sequential architectures. It
  uses a supplied tokenizer to map catalog indices, then owns how a ragged batch
  becomes a dense tensor, which positions are real, and how far back to look.
  Padding is always on the right: an RNN reads each row's own final position,
  while a causal transformer cannot attend to later pad positions. `max_length`
  truncates to the most recent interactions. The batcher deliberately owns no
  training objective — next-item shift, masked positions and sampled negatives
  differ per architecture and stay in trainers.
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
  can serve two models that read different amounts of history. Padding is fixed
  on the right: pads sit after every real token, so a causal mask already
  excludes them and training needs no attention mask, only a loss mask.
  `catalog_logits` is gone, unnecessary once the head is catalog-width.
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
  the padding, so the attention module accepts no padding-mask argument. The
  batcher enforces that invariant rather than exposing an unsupported left-pad
  mode. The trainer also refuses `max_length=None`, since that value sizes the
  positional table —
  `block_size` is derived rather than configured, so the two cannot disagree.
  The unified fitted-model checkpoint carries its vocabulary with the weights,
  because a served model that cannot say what column 41 means is not much use.
  Measured figures for this model are under **Changed** below rather than here:
  the defaults moved twice inside this release — a tied head, then the GPT-2 init
  with a cosine schedule — and one release section quoting three generations of
  numbers would be worse than none.
- `SimpleRNN`, `SimpleRNNConfig` and `SimpleRNNTrainer`: a GRU or LSTM trained
  on next-item cross entropy at every position, one example per user. The
  smallest model that actually uses order, and so the baseline a transformer has
  to beat before its extra machinery has earned anything. Prediction reads each
  row's own final state rather than the last column, which under right padding
  would score most users from a pad embedding; `exclude_seen` masks the whole
  history including the part truncation dropped. The encoder is a constructor
  parameter rather than something `fit` invents, so the context window and
  vocabulary are replaceable — and `max_length` left
  `SimpleRNNConfig`, since it describes what the encoder reads rather than the
  network. Its next-item objective decodes targets with `tokens - n_reserved` and
  excludes `unk` from them, because "predict the item I cannot identify" is not a
  question with an answer. Training refuses a dataset with no two-item history
  left after context-window truncation, rather than returning an untrained model
  with a NaN loss.
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

- `SimpleGPTConfig.tie_embeddings`, scoring with the input embedding's item rows
  instead of a separate head. **On by default**, because it won on every split
  measured, and re-checked against the init and schedule below rather than left
  standing on a configuration that no longer exists: ML-1M `leave_last_out` by
  `+0.0171` (0.1740 ± 0.0011 against 0.1569 ± 0.0009, sixteen seed deviations),
  Office `leave_last_out` by `+0.0068` (six), and Office `temporal` by `+0.0006`
  (under one). A capacity sweep
  separated perfectly — every tied configuration beat every untied one on Office
  `leave_last_out`, and on ML-1M a tied `d_model=64, n_layers=1` model at 277k
  parameters beat an untied `d_model=128, n_layers=2` at 1,267k. Set it `False`
  to reproduce figures recorded before this was the default.
  The margin tracks how far the head outweighs the data, which is what a
  regularizer should do: largest where 573k of 1,143k parameters faced 6,313
  training users, smallest on the split with the fewest of both. It also buys
  stability — on Office `temporal` the untied model lost 32% of its score
  between its best budget and twenty epochs while the tied one stayed flat, and
  the seed deviation on Office `leave_last_out` fell sixfold.
  Front loading makes the tied weight a slice rather than the whole embedding,
  so `pad` and `unk` sit below it, never enter the head, and never take
  output-side gradient — which is what we want, since neither is ever a target.
  The offset is derived rather than passed, making it the same number the
  objective already uses to decode targets. The bias survives tying, because
  tying is a claim about the weight alone.
  Tying converges *later*, not faster: `nn.Linear` starts near
  `±1/sqrt(d_model)` while the embedding starts at `std=0.02`, so the tied head
  begins with a flatter softmax. On ML-1M it trails untied at ten epochs and
  passes it by twenty. Compare the two at a validated budget, never a fixed one,
  or the slower start reads as a worse model.

### Changed

- **The init and the schedule interact, and the sign flips.** Under a constant
  learning rate the GPT-2 init is *worse* on Office `leave_last_out` (0.0341
  against 0.0389); under cosine it is *better* (0.0422 against 0.0380). They were
  designed together and using one without the other is the mismatch, so both are
  on or neither should be. Change one of `tie_embeddings`, the init or
  `lr_schedule` and re-measure the others: a one-variable comparison inverted
  twice while these figures were being taken, once for tying between ten and
  twenty epochs and once here.
- **Breaking (numerically).** `SimpleGPT` now initialises the way the
  architecture it claims to be does. Every weight starts at `std=0.02` — PyTorch's
  `nn.Linear` default is roughly 2.5× wider at `d_model=128` — and each block's two
  residual-stream projections start at `0.02 / sqrt(2 * n_layers)`, GPT-2's scaled
  init. Only the embeddings were initialised before; every `Linear` was left at the
  PyTorch default, which was a silent departure from nanoGPT rather than a
  decision. A block adds to the residual stream twice, so without the scaling the
  stream's variance grows with depth. Existing checkpoints load unchanged; only
  fresh runs move.
- `lr_schedule` on both sequential configs, with `warmup_fraction` and
  `min_lr_ratio`. `"cosine"` gives linear warmup then cosine decay to a floor,
  with that floor used by the final optimizer update. The curve is measured in
  optimizer steps so its shape is invariant to batch size. **Default for
  `SimpleGPT`**, where it is worth `+0.0080` (six seed deviations) on Office
  `leave_last_out` and `+0.0052` on ML-1M; `"constant"` stays the default for
  `SimpleRNN`, which it does not help on any split — under a seed deviation on all
  three. That split is the behaviour to want: warmup addresses a failure mode a
  transformer has and a recurrence does not. `SimpleRNN` gets the option anyway,
  because the two share a comparison table and a knob offered to one model but not
  the other converts a tuning difference into an apparent architectural one. Warmup is there because the first steps of a transformer are the ones
  most able to wreck it, and neither half is expressible through the optimizer
  alone. `fit` records the rate it trained at in `history`, so a schedule is
  visible in the log it should explain. The schedule advances on batches the
  objective declines, so the curve is exactly the configured shape rather than one
  whose floor depends on how many batches carried targets.

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

- The recorded `ndcg@20` figures are re-measured at the current defaults, with
  the epoch budget *and* the schedule selected on validation and every grid
  widened until the validation curve turned over, so no published number sits at
  a grid edge. SimpleGPT reads 0.1740 ± 0.0011 on ML-1M `leave_last_out` (peak at
  forty epochs), 0.0422 ± 0.0011 on Office `leave_last_out` (twenty), and
  0.0102 ± 0.0007 on Office `temporal` (twenty). SimpleRNN is unchanged at
  0.1471 / 0.0282 / 0.0091: it was offered the same schedule and validation
  declined it. The
  earlier "still improving at the edge of the grid, treat as lower bounds" caveat
  is retired: those peaks are measured.
  SimpleRNN was re-run on the same widened grids so the comparison is not tilted
  by tuning one model further than the other. Its figures did not move — it peaks
  at twenty epochs on ML-1M and *declines* at forty and eighty, so its earlier
  edge-of-grid selection turned out to be its optimum.
  Office `leave_last_out` changes conclusion, not just magnitude: SimpleGPT now
  scores 0.0384 against ELSA's 0.0327, where the untied model read 0.0310. The
  previous finding that a shallow matrix model beat the transformer on set-like
  purchase data was a fact about the untied head.

### Documentation

- Office `temporal` now carries the warning it always needed. 85% of its test
  targets are items absent from training and 75% of its users have at least one,
  so a fixed-width lookup head cannot reach most of the answers: an oracle
  restricted the same way scores 0.3252, not 1.0. Every model in that column is
  competing for a third of the available ndcg, which is why they all land within
  thousandths of a popularity baseline. It is a cold-start diagnostic, not a
  ranking of these models.

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

[Unreleased]: https://github.com/zombak79/compresso-recsys/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/zombak79/compresso-recsys/releases/tag/v0.2.0
[0.1.2]: https://github.com/zombak79/compresso-recsys/releases/tag/v0.1.2
[0.1.1]: https://github.com/zombak79/compresso-recsys/releases/tag/v0.1.1
[0.1.0]: https://github.com/zombak79/compresso-recsys/releases/tag/v0.1.0
