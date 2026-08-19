Models API
==========

Recommender Contract
--------------------

Models evaluated by :func:`compresso_recsys.evaluation.evaluate_recommender`
implement the small :class:`compresso_recsys.models.Recommender` protocol.
The required ``predict_on_batch(source, *, k, exclude_seen=True)`` method
returns ranked top-``k`` predictions as an :class:`compresso.SRPTensor`.
EASE and ELSA exclude items already present in the source interactions by
default. Pass ``exclude_seen=False`` to inspect rankings that may contain
previously interacted items.

.. autoclass:: compresso_recsys.models.Recommender
   :members:

Feature-based cold-start models additionally implement
:class:`compresso_recsys.models.ColdStartRecommender`. Its source vocabulary
is fixed by training, while its identified candidate catalog can be rebuilt or
updated independently.

.. autoclass:: compresso_recsys.models.ColdStartRecommender
   :members:

.. autoclass:: compresso_recsys.models.ItemVocabulary
   :members:

Models that read chronological histories implement
:class:`compresso_recsys.models.SequentialRecommender` instead. It is the same
one-method contract, differing only in what a source is: an
:class:`compresso_recsys.ItemSequences` rather than a ``csr_matrix``.
:func:`compresso_recsys.evaluation.evaluate_recommender` accepts either, asking a
source only for its row count and a row slice, so a sequential model and a matrix
model can appear in one :func:`compresso_recsys.stats.compare_models` call with
no statistics-side changes.

.. autoclass:: compresso_recsys.models.SequentialRecommender
   :members:

Implementing New Models
-----------------------

The protocols above are sufficient when a model only needs to work with the
evaluation API. Model authors who want Compresso RecSys validation, batched
prediction, and catalog management can instead inherit one of the abstract
bases.

Use :class:`compresso_recsys.models.BaseCollaborativeRecommender` for a model
whose fitted source and candidate catalog has one fixed positional item space.
Implement ``fit``, ``is_fitted``, ``n_items``, and ``predict_on_batch``. Call
the inherited ``_prepare_source`` at the start of ``predict_on_batch``; the
base then supplies ``predict`` with bounded batching and optional progress.

.. autoclass:: compresso_recsys.models.BaseCollaborativeRecommender
   :members:
   :private-members: _prepare_source

Use :class:`compresso_recsys.models.BaseColdStartRecommender` when source items
are fixed by fitting but identified candidates can be rebuilt or updated from
features, and the source is a ``csr_matrix``. Subclass constructors must call
``super().__init__()``. After fitting the source encoder, call
``self.candidates.install(...)`` once with the fitted source IDs and initial
candidate features. Later changes then validate features against that feature
space and preserve stable IDs. ``predict_on_batch`` can call
``self.candidates.resolve_selection(...)`` to resolve an optional candidate
allowlist while keeping returned :class:`compresso.SRPTensor` columns in the
complete catalog space.

The catalog lifecycle is *owned* rather than inherited -- see
:ref:`the-owned-candidate-catalog` below -- which is why this base is only about
reading a matrix. The methods on it are a facade over
``self.candidates``, kept because they are the documented model surface.

.. autoclass:: compresso_recsys.models.BaseColdStartRecommender
   :members:
   :private-members: _prepare_source

Use :class:`compresso_recsys.models.BaseSequentialRecommender` for a model that
reads ordered histories. It is parallel to
:class:`compresso_recsys.models.BaseCollaborativeRecommender` rather than derived
from it, because crossing the two source representations with cold-start
capability in the type hierarchy would give four classes for two ideas. Candidate
capability is composed instead: a model that scores unseen items owns a catalog
rather than inheriting one. Implement ``is_fitted``, ``n_items`` and
``predict_on_batch``; the base supplies ``predict`` with bounded batching over
row slices.

``fit`` is deliberately outside the contract. Trainers keep the package's
existing ``SomeTrainer(config).fit(data)`` shape and a fitted model owes only
prediction.

The base is careful not to assume two things. ``n_items`` describes what can be
*scored*, which need not be the vocabulary a history is expressed over — a
truncated or hashed context, or a cold-capable model scoring items that appear in
no history at all. And truncation is not exclusion: ``exclude_seen=True`` must
mask every item in the *full* history handed to it, even where the encoder reads
only a suffix. A model attending to the last 200 interactions must still refuse
to recommend the 201st.

.. autoclass:: compresso_recsys.models.BaseSequentialRecommender
   :members:
   :private-members: _prepare_source

Training Interaction Batches
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:class:`compresso_recsys.models.InteractionBatchSampler` provides the compact
source-prefix batching used by ELSA and TEASERGD. It keeps the interaction
matrix sparse until the training step chooses to densify the selected active
columns. ``batch.x`` uses local compact columns; ``batch.sources`` maps those
columns to global fitted item rows.

When ``max_output`` is an integer, ``batch.candidates`` begins with exactly
``batch.sources`` and appends items absent from the whole batch. It is a soft
limit because active sources are never dropped. With ``max_output=None``,
``batch.candidates`` is ``None`` and the model should score its complete output
catalog. Call ``on_epoch_end()`` after each epoch to advance shuffling and
negative sampling reproducibly.

.. code-block:: python

   from compresso_recsys.models import (
       InteractionBatchSampler,
       dense_training_target,
   )

   sampler = InteractionBatchSampler(
       interactions,
       device="cuda",
       batch_size=1024,
       shuffle=True,
       max_output=5000,
       seed=0,
   )

   for batch_index in range(len(sampler)):
       batch = sampler[batch_index]
       x = batch.x.to_dense()
       targets = dense_training_target(
           x,
           sources=batch.sources,
           candidates=batch.candidates,
           input_dim=interactions.shape[1],
       )
       predictions = model(
           x,
           sources=batch.sources,
           candidates=batch.candidates,
       )

   sampler.on_epoch_end()

.. autoclass:: compresso_recsys.models.InteractionBatch
   :members:

.. autoclass:: compresso_recsys.models.InteractionBatchSampler
   :members:

.. autofunction:: compresso_recsys.models.dense_training_target

Transductive Models in Expanding Catalogs
-----------------------------------------

:class:`compresso_recsys.models.WarmCatalogAdapter` evaluates a fixed-catalog
model such as EASE or ELSA against a larger validation or test catalog. It
selects the fitted warm columns from the stage source matrix and remaps warm
prediction indices into the expanded target space. Cold targets remain part of
the metric calculation, but the wrapped transductive model cannot recommend
them. This makes the result directly comparable with a cold-start model on the
same users and targets while preserving the transductive model's limitation.

The item IDs supplied to the adapter define both column orders, so evaluation
does not depend on warm items occupying a catalog prefix:

.. code-block:: python

   from compresso_recsys.evaluation import evaluate_recommender
   from compresso_recsys.models import WarmCatalogAdapter

   adapted_elsa = WarmCatalogAdapter(
       elsa,
       train_item_ids=split["train_item_ids"],
       catalog_item_ids=split["test_item_ids"],
   )

   result = evaluate_recommender(
       adapted_elsa,
       source=adapted_elsa.align_source(split["test_source_matrix"]),
       targets=split["test_target_matrix"],
       metrics=metrics,
       batch_size=1024,
   )

The input to :meth:`~compresso_recsys.models.WarmCatalogAdapter.align_source`
must use the exact column order declared by ``catalog_item_ids``. Construct a
separate adapter for validation when its catalog differs from the test catalog.

.. autoclass:: compresso_recsys.models.WarmCatalogAdapter
   :members:

EASE
----

EASE is a closed-form collaborative-filtering model. Fitting creates a dense
item-by-item coefficient matrix, so its memory use grows quadratically with
the number of items. ``float32`` is the memory-efficient default. Select
``float64`` explicitly when additional numerical precision is more important
than fit and prediction speed. See :doc:`../citing` for the original EASE
paper and copy-ready BibTeX.

.. autoclass:: compresso_recsys.models.EASEConfig
   :members:

.. autoclass:: compresso_recsys.models.EASE
   :members:

ELSA
----

ELSA learns a low-rank matrix of normalized item embeddings with a shallow
linear autoencoder objective. Unlike EASE, its model size grows linearly with
the number of items, making it suitable for larger catalogs and GPU training.
See :doc:`../citing` for citations covering standard ELSA, large-scale
candidate sampling, and compressed ELSA.

During training, ``max_output`` can limit each batch's output candidates. All
items with positive interactions in the batch are always retained, and the
remaining candidate budget is sampled without replacement from items absent
from the whole batch. This makes ``max_output`` a soft upper bound when a
batch contains more distinct positive items than the configured limit. Use
``None`` to score the complete catalog during training.

Compressed ELSA uses Compresso's lottery-ticket schedule to retain a fixed
number of latent values per item. During mask search, each stable stage rewinds
the item factors to their original initialization under the new mask and
restarts the optimizer. After the final mask stabilizes, the factors are
converted to a row-packed sparse parameter and only its values are trained for
``ELSAConfig.epochs``. Learning-rate decay, when enabled, applies only to this
final sparse fine-tuning phase.

By default, a mask-search stage advances only after its mask change stays
within ``change_threshold`` for ``stability_window`` updates. Set
``max_epochs_per_stage`` to force a stage to accept its latest proposed mask
after a fixed number of epochs. This bounds training time but may select a less
stable ticket. Training checkpoints during this phase are not currently
resumable, and ``torch.compile`` is not supported. When ``max_output`` limits
the candidate set, mask search projects only those ``MaskedParam`` rows and
sparse fine-tuning selects only those gradient-connected ``SRPParam`` rows.
The default dense fine-tuning backend densifies that selection, while the COO
backend keeps it sparse through differentiable sparse matrix multiplications.
COO can reduce both memory and runtime for highly sparse tickets, while dense
matrix multiplication can win as the retained ``k`` grows. The crossover is
hardware- and workload-dependent. ``max_output=None`` scores the complete
catalog during training.

Sparse inference defaults to cached CSR full-catalog scoring and densifies only
the selected source rows. The dense inference backend instead caches one full
normalized factor matrix and can be faster for less sparse tickets. Configure
the normal backend in ``ELSACompressionConfig`` or override it per
``predict`` or ``predict_on_batch`` call without retraining.

ContentRecommender
------------------

ContentRecommender is a cold-start baseline that learns nothing. A user profile
is the sum of the feature vectors of the items they interacted with, and
candidates are ranked by similarity to that profile, so items are recommendable
as soon as they are registered on the catalog. ``fit`` takes item features
alone; unlike TEASER there is no encoder for an interaction matrix to train.

With its default configuration it reproduces the scoring in
:func:`compresso_recsys.retrieval.evaluate_item_embeddings_with_holdout`
exactly, which makes it the reference point for judging whether a learned
cold-start model beats plain feature similarity on the same embeddings. That
function is an ELSA-forward recommender fused with an evaluator rather than a
neutral evaluator, so matching it needs L2-normalized item vectors, the
self-subtraction and ReLU of ELSA-forward, and seen-item masking. ``normalize``
and ``elsa_forward`` expose the first and the middle two; masking always
follows ``exclude_seen``.

``elsa_forward`` does not change the ranking while ``exclude_seen`` is set,
because the self-subtraction only touches entries that masking then sets to
``-inf``. It matters only when predicting with ``exclude_seen=False``.

Every matrix product runs through torch, so ``device`` moves scoring onto a
GPU; only the score matrix returns to the host.

.. autoclass:: compresso_recsys.models.ContentRecommenderConfig
   :members:

.. autoclass:: compresso_recsys.models.ContentRecommender
   :members:

TEASER
------

TEASER learns item-to-feature encoder weights from binary implicit interactions
while keeping the supplied item-feature matrix fixed as its decoder. The
reference implementation uses the original ADMM updates. It accepts item
features as a SciPy CSR matrix, :class:`compresso.SRPTensor`, NumPy array, or
PyTorch tensor. Binary tags reproduce the original model, while real-valued
dense or sparse embeddings provide the same fixed-decoder abstraction without
necessarily retaining human-readable feature explanations.

TEASER separates two item spaces for production cold start. The source
vocabulary is fixed at fit time: only items with fitted encoder rows may appear
in user history. The candidate catalog is an immutable, versioned snapshot that
can be rebuilt or updated without retraining. New candidates receive a decoder
row from their features and can be recommended immediately, but cannot appear
in source history until the model is retrained.

When an external interaction matrix uses a larger or differently ordered item
space, :meth:`~compresso_recsys.models.TEASER.align_source` projects it into the
fitted source vocabulary by stable ID. The operation uses sparse CSR column
indexing, never densifies user histories, and returns an already aligned matrix
unchanged.

For an item cold-start split, pass the checkpoint's ``train_item_indices`` to
``fit``. Only those interaction columns and feature rows participate in ADMM,
but the decoder keeps feature rows for every item. Validation and test items can
therefore be ranked from metadata without being treated as zero-valued training
targets. Source histories must contain fitted training items.

Pass stable ``item_ids`` and optional aligned ``metadata`` to ``fit`` to build
the initial catalog. :meth:`~compresso_recsys.models.TEASER.update_candidates`
appends new IDs and can replace existing rows;
:meth:`~compresso_recsys.models.TEASER.build_candidates` atomically replaces
the entire catalog; and
:meth:`~compresso_recsys.models.TEASER.remove_candidates` removes IDs. When an
embedding model supplies the features, set ``feature_space_id`` during fit so
later updates can reject an explicitly different model or revision.

``predict`` and ``predict_on_batch`` accept ``candidate_ids`` as a shared
allowlist for the batch. They gather and score only those registered rows. The
result remains an :class:`compresso.SRPTensor` over the complete current
catalog, so its columns can be resolved with
:meth:`~compresso_recsys.models.CandidateCatalog.ids_for`. Unknown and duplicate
allowlist IDs are rejected.

The original solver forms a dense warm-item Gram matrix and eigendecomposes it,
so fit memory grows quadratically and fit time cubically with the number of warm
items. It also forms a dense feature Gram matrix. Sparse item features reduce
input storage and can accelerate prediction, but do not remove those training
costs. ``float64`` is the parity-oriented default.

See :doc:`../citing` for the original TEASER paper.

.. autoclass:: compresso_recsys.models.TEASERConfig
   :members:

.. autoclass:: compresso_recsys.models.TEASER
   :members:
   :inherited-members:

.. autoclass:: compresso_recsys.models.CandidateCatalog
   :members:

.. _the-owned-candidate-catalog:

The Owned Candidate Catalog
~~~~~~~~~~~~~~~~~~~~~~~~~~~

:class:`compresso_recsys.models.CandidateCatalog` is an immutable snapshot.
:class:`compresso_recsys.models.MutableCandidateCatalog` is the lifecycle around
it: the lock, the current snapshot, the fitted source vocabulary, and the
operations that publish, extend, shrink and align against them.

While that lifecycle lived on a base class, "cold-capable" and "inherits
:class:`compresso_recsys.models.BaseColdStartRecommender`" were the same
statement. Adding a second axis -- a model that reads ordered histories rather
than a matrix -- would then have forced a choice between multiple inheritance and
a fourth base class, for two independent ideas. An owned object removes the
choice: a model holds one, whichever base it derives from.

Composition rather than a mixin, because the state is what decides it. A mixin
would not encapsulate these attributes; it would install them on whatever class
it is mixed into, six of them public. Two stateful mixins both initialising
through ``super().__init__()`` is where MRO ordering and private-name collisions
live. An owned object has its own ``__init__``, its own lock and its own tests,
and a model could hold two if that ever made sense.

.. code-block:: python

   class SequentialContentRNN(BaseSequentialRecommender):
       def __init__(self) -> None:
           self.candidates = MutableCandidateCatalog()

       def fit(self, sequences, item_features, *, item_ids):
           ...
           self.candidates.install(...)
           return self

       def predict_on_batch(self, source, *, k, exclude_seen=True):
           catalog = self.candidates.snapshot()

Reads go through :meth:`~compresso_recsys.models.MutableCandidateCatalog.snapshot`
rather than through forwarded properties, deliberately. A snapshot is a
consistent view: several reads off one snapshot cannot straddle a concurrent
republish, which forwarding ``item_ids``, ``rows_for`` and ``ids_for``
separately would silently allow. ``n_items`` is the one convenience, because
"how many candidates are there" needs an answer before installation, which a
snapshot cannot give.

``on_publish`` is called with each new snapshot while the lock is held, and is
how an owner drops caches derived from the previous one. The six fitted
``source_*`` attributes live on the catalog, so a model's fitted source
vocabulary is ``model.candidates.source_item_ids`` rather than
``model.source_item_ids_``.

.. autoclass:: compresso_recsys.models.MutableCandidateCatalog
   :members:

TEASERGD
--------

TEASERGD keeps TEASER's fixed item-feature decoder but learns the encoder with
PyTorch instead of the reference ADMM solver. It never materializes the dense
item-by-item coefficient matrix. A batch first forms user feature profiles and
then scores candidate feature rows, so model storage grows with
``warm_items * feature_dim`` rather than ``warm_items ** 2``.

Training shares ELSA's scalable controls: ``max_output`` retains every source
item appearing in a batch as the candidate prefix and fills the remaining
budget with sampled negatives, cosine-normalized reconstruction is optimized
with NAdam or AdamW, and optional cosine learning-rate decay and
``torch.compile`` are available. ``diagonal_scale`` multiplies the self
coefficient removed using ``(encoder * item_features).sum(-1)`` for each source
item. The default ``1.0`` removes it completely, ``0.0`` leaves it untouched,
and intermediate values subtract that fraction. No dense coefficient matrix is
needed for the correction. Values below one intentionally relax TEASER's
anti-identity constraint and apply consistently during training and inference.

``loss="normalized_mse"`` is the default and preserves the ELSA-style
row-normalized reconstruction objective. ``loss="teaser"`` instead optimizes
the original TEASER objective divided by the number of training users:
per-user Frobenius reconstruction plus the complete off-diagonal coefficient
norm and encoder norm. Dividing every term by the same constant does not change
the minimizer. With sampled output candidates, negative reconstruction errors
are importance-weighted to estimate the complete output error. Set
``use_relu=False`` with this mode for parity with the paper; enabling ReLU is an
optional modification of the original model. Paper parity also requires
``diagonal_scale=1.0``.

The encoder defaults to Xavier initialization. Set
``encoder_init="features"`` to initialize each warm encoder row from its fixed
decoder feature row, making the initial coefficient matrix a scaled-diagonal
variant of the metadata-similarity model ``S @ S.T``. At the default
``diagonal_scale=1.0`` its diagonal is removed. This is particularly useful for
sparse SAE codes because active feature dimensions receive a meaningful signal
before the first gradient update. Dense and CSR features are both supported;
the CSR path fills the allocated encoder directly without constructing another
dense feature matrix.

``normalize_encoder=True`` applies row-wise L2 normalization to the effective
encoder in every training and inference path, as ELSA does for its item
factors. The underlying trainable parameter remains unnormalized. Explicit
``l2_encoder`` and optimizer ``weight_decay`` still regularize that underlying
parameter, so start with both set to zero when evaluating encoder
normalization.

The TEASER coefficient penalty is estimated from random off-diagonal item pairs
and, when ``diagonal_scale < 1``, matching sampled residual-diagonal entries.
Its cost is
``coefficient_regularization_samples * feature_dim`` per batch; set the sample
count to zero to disable it. TEASER loss mode scales the estimate to the full
coefficient-matrix norm, while normalized-MSE mode preserves the previous mean
penalty. :meth:`~compresso_recsys.models.TEASERGD.exact_coefficient_squared_norm`
is provided for diagnostics on small problems, but deliberately materializes
the coefficient matrix and should not be used in large training loops.

Dense feature matrices are cached on the training device and indexed there.
CSR feature matrices remain sparse for candidate scoring and only selected
source rows are densified. Full candidate tensors are cached for prediction and
invalidated when the catalog changes. TEASERGD uses the same stable-ID catalog,
``align_source``, cold-candidate updates, metadata handling, and candidate
allowlists as the ADMM implementation above.

See :doc:`../citing` for the original TEASER paper.

.. autoclass:: compresso_recsys.models.TEASERGDConfig
   :members:

.. autoclass:: compresso_recsys.models.TEASERGD
   :members:

.. autoclass:: compresso_recsys.models.TEASERGDTrainer
   :members:

Backend Performance
~~~~~~~~~~~~~~~~~~~

Sparse backends are not necessarily slower. In one representative benchmark
with ``latent_dim=4096``, their speed advantage disappeared between
``k_target=16`` and ``k_target=32``:

.. list-table:: Relative throughput compared with the corresponding dense backend
   :header-rows: 1
   :widths: 15 28 28

   * - ``k_target``
     - COO fine-tuning
     - CSR inference
   * - 8
     - 32% faster
     - 9% faster
   * - 16
     - 13% faster
     - 4% faster
   * - 32
     - 4% slower
     - 2.6% slower

This table is illustrative rather than a universal selection rule. Sparse
kernel overhead, device characteristics, batch size, candidate count, catalog
size, and ``latent_dim`` all affect the crossover. Even above it, COO or CSR
may be preferable because they avoid the much larger dense factor
representation. For a new workload, benchmark both backends with the same
trained ticket; ``sparse_inference_backend`` can be overridden per prediction
call without retraining. Tiny metric differences between backends can occur
because floating-point reductions use a different order.

Compressed ELSA uses Compresso's exact ``MaskedParam.to_srp_param()``
conversion, which preserves the final selected mask even for tied or
zero-valued entries. Compresso currently moves its initialization copy with the
model, so mask search temporarily retains an additional dense factor buffer on
the training device.

.. autoclass:: compresso_recsys.models.ELSAConfig
   :members:

.. autoclass:: compresso_recsys.models.ELSACompressionConfig
   :members:

.. autoclass:: compresso_recsys.models.ELSA
   :members:

.. autoclass:: compresso_recsys.models.CompressedELSA
   :members:

.. autoclass:: compresso_recsys.models.ELSATrainer
   :members:

Sequential Models
-----------------

Batching Histories
~~~~~~~~~~~~~~~~~~

:class:`compresso_recsys.ItemSequences` holds no padding, no special tokens and no
length limit, because those are modelling decisions. Several of them are the
*same* across sequential architectures, though, and re-deriving them per model is
how off-by-one bugs get in.
:class:`compresso_recsys.models.SequenceBatcher` owns exactly the shared part:
where special tokens live in the vocabulary, how a ragged batch becomes a dense
tensor, which positions are real, and how far back to look.

The vocabulary puts the catalog first and appends special tokens::

   catalog index i  ->  token i          (the identity)
   "pad"            ->  token n_items
   "mask"           ->  token n_items + 1

That ordering exists so **catalog token ids never move**. Reserving ids at the
front, as text models do, would mean introducing a second special token shifts
every item by one and invalidates any model already trained. Appending instead
keeps ``logits[..., :n_items]`` the catalog scores under any future vocabulary,
which :meth:`~compresso_recsys.models.SequenceBatcher.catalog_logits` states
where it is relied on.

``pad_side`` is the setting architectures genuinely disagree about. With
``"right"`` the content comes first: an RNN reads to each row's own final
position, so trailing padding costs nothing. With ``"left"`` the newest
interaction sits at a fixed index, which is what a causal transformer wants so
that prediction always reads position ``-1``. ``max_length`` truncates to the
**most recent** interactions, the only sensible direction, since a context window
is a claim about recency rather than about where a history happened to start.

:meth:`~compresso_recsys.models.SequenceBatcher.final_positions` and
:meth:`~compresso_recsys.models.SequenceBatcher.gather_final` exist for the single
easiest thing to get wrong. Under right padding the last *column* is padding for
every row shorter than the batch maximum, so reading ``states[:, -1]`` silently
scores most users from a pad embedding — and agrees with itself at
``batch_size=1``, where every row fills its own batch, which is what makes the bug
survive casual testing. Empty rows report position 0, which is padding; pair the
position with :meth:`~compresso_recsys.models.SequenceBatcher.has_history` rather
than trusting it alone.

What the batcher deliberately does not own is a training objective. A next-item
shift, a masked-position target and sampled negatives differ between
architectures, and a component with three mutually exclusive modes is not an
abstraction. Those stay in trainers.

.. code-block:: python

   from compresso_recsys.models import SequenceBatcher

   batcher = SequenceBatcher(n_items=3295, max_length=200, pad_side="right")
   tokens, mask = batcher.encode(split["test_source_sequences"], device="cuda")
   states = model(tokens)                        # (rows, length, hidden)
   final = batcher.gather_final(states, mask)    # (rows, hidden)

.. autoclass:: compresso_recsys.models.SequenceBatcher
   :members:

SimpleRNN
~~~~~~~~~

SimpleRNN is a GRU or LSTM trained on next-item cross entropy at every position,
one training example per user. It is the smallest model that actually uses order,
which makes it the baseline a transformer has to beat before its extra machinery
has earned anything.

Training reads each history left to right and predicts the following item::

   tokens   [a, b, c, PAD, PAD]      mask   [T, T, T, F, F]
   inputs   [a, b, c, PAD]
   targets  [b, c, PAD, PAD]         valid  [T, T, F, F]

Under right padding, ``mask[:, 1:]`` is exactly the set of positions whose target
is a real item, so no arithmetic over lengths is needed and padding can never
become a target. The head scores ``n_items`` rather than the full vocabulary: a
special token is never a target, so an output column for it could only learn to be
wrong, and a shift bug raises an index error instead of scoring plausibly.

A history of one interaction yields no training example, since a next-item target
needs a preceding item. Such rows remain predictable, and an entirely empty
history yields the state after a single pad — identical for every empty row, so
effectively a learned prior.

``history`` records the mean loss per epoch alongside the number of positions it
was averaged over. That count is worth reading rather than assuming: it is
``sum(min(length, max_length) - 1)``, so it reports what truncation costs. On
MovieLens-1M under ``leave_last_out``, the default ``max_length=200`` puts 697 of
6,033 users over the window and drops 80k of 543k training positions.

Tied embeddings, learning-rate schedules, early stopping and sampled softmax are
all deliberately absent. This is a baseline, and each of those is a separate
claim that deserves to be measured on its own.

.. code-block:: python

   from compresso_recsys.evaluation import evaluate_recommender
   from compresso_recsys.metrics import CalibratedRecall, NDCG
   from compresso_recsys.models import SimpleRNNConfig, SimpleRNNTrainer

   model = SimpleRNNTrainer(
       SimpleRNNConfig(
           rnn_type="gru",
           embedding_dim=64,
           hidden_dim=128,
           epochs=8,
           batch_size=256,
           lr=3e-3,
           max_length=200,
       )
   ).fit(split["x_train_sequences"])

   result = evaluate_recommender(
       model,
       source=split["test_source_sequences"],
       targets=split["test_target_matrix"],
       metrics=[CalibratedRecall(20), NDCG(20)],
       batch_size=512,
   )

On MovieLens-1M with ``min_value_to_keep=4.0`` under ``leave_last_out``, that
configuration reaches ``ndcg@20 = 0.132`` against ``0.070`` for EASE at
``l2=200``. The gap is expected rather than impressive: the protocol scores one
held-out final interaction per user, and EASE has no notion of recency to bring to
it. The loss is still falling at the eighth epoch, so the number is a
verification that the wiring works, not a tuned result.

.. autoclass:: compresso_recsys.models.SimpleRNNConfig
   :members:

.. autoclass:: compresso_recsys.models.SimpleRNN
   :members:

.. autoclass:: compresso_recsys.models.SimpleRNNTrainer
   :members:
