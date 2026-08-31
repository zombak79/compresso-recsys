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

Production Recommendations
--------------------------

The low-level ``predict`` methods consume catalog-shaped matrices or
:class:`compresso_recsys.ItemSequences` and return catalog column indices. For
serving, every built-in model also inherits one identified interface that works
with stable item IDs:

.. code-block:: python

   model = EASE().fit(interactions, item_ids=item_ids)

   ranked = model.recommend(
       [["item-14", "item-87"], ["item-32"]],
       k=20,
       exclude_seen=True,
       allowlist=eligible_item_ids,
       blocklist=unavailable_item_ids,
       on_insufficient="truncate",
   )

   ranked.item_ids   # shape: (2, 20)
   ranked.scores     # shape: (2, 20)
   payload = ranked.to_dicts()

``histories`` is always a batch. To recommend for one user, pass one nested
history such as ``[["item-14", "item-87"]]``. Collaborative models interpret
a history as binary interactions, while sequential models preserve its order
and repeated IDs. ``allowlist`` and ``blocklist`` are optional batch-wide
candidate filters; both apply before top-k and the blocklist wins when an ID is
present in both. Unknown history or filter IDs raise.

Unlike the evaluation-oriented ``predict`` methods, ``recommend`` defaults to
``exclude_seen=False``. Recommending a previously seen item is legal in serving,
and a caller such as a worker can put whichever seen IDs are inappropriate into
``blocklist``. Pass ``exclude_seen=True`` to apply the conventional offline
evaluation policy to the complete history.

``k`` is a requested maximum. By default, a row with fewer than ``k`` eligible
candidates is truncated independently; when ``exclude_seen=True``, eligibility
is calculated after removing seen items. Other users in the same batch still
receive all available results up to ``k``. The arrays remain ``(users, k)``:
unused item positions contain ``None``, their scores are ``-inf``, and
``valid_mask`` plus ``valid_counts`` distinguish them from recommendations.
``to_dicts()`` omits those positions. Pass ``on_insufficient="raise"`` when a
short row should instead fail the complete request. Neither policy reintroduces
blocked items, or seen items when their exclusion was requested.

:class:`compresso_recsys.models.Recommendations` holds immutable ``(users, k)``
arrays. :meth:`~compresso_recsys.models.Recommendations.to_dicts` returns one
insertion-ordered ``item_id: score`` dictionary per user, so dictionary order
is rank order and short rows become shorter dictionaries. Fixed-catalog models
use positional integer IDs when ``item_ids`` is omitted from fitting. The
mapping is stored in fitted-model checkpoints.

.. autoclass:: compresso_recsys.models.IdentifiedRecommender
   :members:

.. autoclass:: compresso_recsys.models.Recommendations
   :members:

Fitted Model Persistence
------------------------

Built-in fitted recommenders inherit
:class:`compresso_recsys.models.BasePersistableRecommender` and expose
``model.save(path)`` plus ``ModelClass.load(path, device="cpu")``. The archive
is self-contained and loading produces a prediction-ready model. Optimizer state
is optional and exact training resumption is outside this contract. See
:doc:`persistence` for the format, device behavior, extension helpers, and the
reason a :class:`compresso_recsys.models.WarmCatalogAdapter` is rebuilt rather
than persisted.

.. autoclass:: compresso_recsys.models.PersistableRecommender
   :members:

.. autoclass:: compresso_recsys.models.BasePersistableRecommender
   :members:
   :private-members: _checkpoint_config, _from_checkpoint_config, _checkpoint_module, _checkpoint_optimizer, _save_checkpoint_state, _load_checkpoint_state, _build_checkpoint_optimizer, _finish_checkpoint_load

Implementing New Models
-----------------------

The protocols above are sufficient when a model only needs to work with the
evaluation API. Model authors who want Compresso RecSys validation, batched
prediction, and catalog management can instead inherit one of the abstract
bases.

All three bases derive from
:class:`compresso_recsys.models.BaseIdentifiedRecommender`, which supplies the
complete ``recommend`` workflow. A fixed-catalog model records identities with
``self._set_item_ids(item_ids, n_items=...)`` during fitting and supports the
optional ``candidate_ids`` selection in ``predict_on_batch``. The source base
then handles ID validation, history conversion, candidate filters, seen-item
capacity checks, and result decoding automatically.

.. autoclass:: compresso_recsys.models.BaseIdentifiedRecommender
   :members:
   :private-members: _set_item_ids, _recommendation_source, _predict_identified

Use :class:`compresso_recsys.models.BaseCollaborativeRecommender` for a model
whose fitted source and candidate catalog has one fixed positional item space.
Implement ``fit``, ``is_fitted``, ``n_items``, and candidate-aware
``predict_on_batch``. Call the inherited ``_prepare_source`` at the start of
``predict_on_batch``; the base then supplies ``predict`` with bounded batching
and optional progress, plus ``recommend`` for stable IDs.

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
candidate-aware ``predict_on_batch``; the base supplies ``predict`` with bounded
batching over row slices and ``recommend`` with order-preserving ID histories.

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

.. autoclass:: compresso_recsys.models.CandidateCatalog
   :members:

.. autoclass:: compresso_recsys.models.MutableCandidateCatalog
   :members:

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
model such as EASE, ELSA or SimpleRNN against a larger validation or test
catalog. It projects the stage source into the fitted item space and remaps warm
prediction indices into the expanded target space. Cold targets remain part of
the metric calculation, but the wrapped transductive model cannot recommend
them. This makes the result directly comparable with a cold-start model on the
same users and targets while preserving the transductive model's limitation.

Both source representations work because checkpoint stage catalogs grow by
appending: ``train_item_ids`` is an exact ordered prefix of every later catalog.
A ``csr_matrix`` is projected to that fitted prefix. An
:class:`compresso_recsys.ItemSequences` is passed through whole: its warm indices
keep their meaning, while the sequential model's tokenizer turns appended cold
indices into ``unk`` without deleting their positions or inventing adjacency.
Rows survive either way, which keeps the source aligned with the targets. The
adapter validates the prefix invariant when it is constructed rather than
silently interpreting a reordered catalog.

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

A sequential model is wrapped the same way, but its history is passed directly
because dropping cold positions would change the sequence:

.. code-block:: python

   adapted_rnn = WarmCatalogAdapter(
       rnn,
       train_item_ids=split["train_item_ids"],
       catalog_item_ids=split["test_item_ids"],
   )

   result = evaluate_recommender(
       adapted_rnn,
       source=split["test_source_sequences"],
       targets=split["test_target_matrix"],
       metrics=metrics,
   )

The input to :meth:`~compresso_recsys.models.WarmCatalogAdapter.align_source`
must be a matrix using the exact column order declared by ``catalog_item_ids``;
sequences go straight to ``predict_on_batch`` or ``evaluate_recommender``.
Construct a separate adapter for validation when its catalog differs from the
test catalog.

When to Reach for It Outside ``temporal``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Under ``temporal`` the adapter is mandatory: stage catalogs expand, so a model
fitted on the training window cannot even accept a test source. Under
``leave_last_out`` the catalogs match and nothing forces the issue -- but items
whose every occurrence falls in a held-out tail are still absent from training,
and **the model families do not treat such columns alike.**

A softmax next-item objective pushes down every non-target logit at every step.
An item that never appears in training is never a target, so it collects only
downward pressure. A reconstruction objective has no such term and leaves the
item near its initialization. Measured on MovieLens-1M under ``leave_last_out``,
ranking the full catalog for 300 test users:

.. list-table:: Rank percentile of never-trained items
   :header-rows: 1
   :widths: 30 30

   * - Model
     - Percentile
   * - ELSA
     - 60th
   * - SimpleRNN
     - 95th

A random item would sit at the 50th. Neither figure says anything about
recommendation quality, so a comparison spanning both families is sounder with
the cold items made unreachable for each -- which is what wrapping both models
does.

Whether it matters is a question about the data rather than the protocol. On
MovieLens-1M only 3 of 6,033 test users have a never-trained target, so the bias
cannot move a metric. On a sparse catalog the share grows. Count first:

.. code-block:: python

   never_trained = np.flatnonzero(
       np.asarray(split["x_train"].sum(axis=0)).ravel() == 0
   )
   cold = set(never_trained.tolist())
   targets = split["test_target_matrix"]
   affected = sum(
       1
       for row in range(targets.shape[0])
       if cold & set(targets[row].indices.tolist())
   )
   print(f"{affected} of {targets.shape[0]} rows have a never-trained target")

Note that ``leave_last_out`` sets ``train_item_ids`` to the whole catalog, since
that mode does not partition items. The warm subset is
``split["item_ids"][split["warm_item_indices"]]``, and passing it is what makes
the adapter do anything at all in that mode.

.. autoclass:: compresso_recsys.models.WarmCatalogAdapter
   :members:

Collaborative Filtering Models
------------------------------

EASE
~~~~

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
~~~~

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
resumable, although a fitted ELSA model can be saved and loaded after training;
``torch.compile`` is not supported. When ``max_output`` limits the candidate
set, mask search projects only those ``MaskedParam`` rows and sparse
fine-tuning selects only those gradient-connected ``SRPParam`` rows.
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

Configuration
^^^^^^^^^^^^^

.. autoclass:: compresso_recsys.models.ELSAConfig
   :members:

.. autoclass:: compresso_recsys.models.ELSACompressionConfig
   :members:

Models and Trainer
^^^^^^^^^^^^^^^^^^

.. autoclass:: compresso_recsys.models.ELSA
   :members:

.. autoclass:: compresso_recsys.models.CompressedELSA
   :members:

.. autoclass:: compresso_recsys.models.ELSATrainer
   :members:

Backend Performance
^^^^^^^^^^^^^^^^^^^

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

Cold-Start Models
-----------------

ContentRecommender
~~~~~~~~~~~~~~~~~~

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
~~~~~~

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

For an item cold-start split, pass the checkpoint's ``warm_item_indices`` to
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

TEASERGD
~~~~~~~~

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

Sequential Models
-----------------

Tokenizing Histories
~~~~~~~~~~~~~~~~~~~~

:class:`compresso_recsys.ItemSequences` holds catalog indices and nothing else —
no padding, no special tokens, no length limit — because those are modelling
decisions. Two components add them back, and they are separate on purpose.

:class:`compresso_recsys.models.ItemTokenizer` owns the **vocabulary**: which
token an item is, and what an item the model has never seen becomes.
:class:`compresso_recsys.models.SequenceBatcher` owns **ragged-to-dense**: how
far back to read, how right padding forms a dense tensor, and which positions
are real.

They are split because they have different lifetimes. A vocabulary is a property
of the dataset and outlives any model. ``max_length`` is a property of the
*model* — it is ``block_size`` under another name, and it sizes a transformer's
positional embedding — so one tokenizer can serve two models that read different
amounts of history. Fusing them gives one object two owners, which shows up as
the same number written into two configs and a runtime check to keep them honest.

Vocabulary layout puts the specials first::

   0 .. n_reserved-1        specials -- named, or reserved for later
   n_reserved .. vocab-1    catalog item i  ->  token  i + n_reserved

That ordering is chosen because **catalog growth appends**. Stage catalogs nest
by prefix, a cold-start catalog grows by appending, and an incremental fit
extends the embedding table at the end — where a ``cat`` splices the optimizer
state correctly. Reserving ids at the back instead would place each new item
exactly where the specials sit, turning every extension into a permutation of the
parameter *and* its momentum; a wrong permutation attaches one item's history to
a special token without raising.

Front-loading has one cost: introducing a special later would shift every item.
``n_reserved`` removes it. Name the specials you use, reserve a few more, and a
token added later lands in the reserve while every trained id stays put — for the
price of a few embedding rows that never receive a gradient.

The offset stops at the tokenizer. A model's head is ``n_items`` wide and indexed
by catalog position, so predictions leave in the same space as the target matrix,
the metrics and the item IDs, and nothing downstream of a model sees a token id.
The one other place it appears is a next-item objective, which decodes its
targets back with ``tokens - n_reserved``.

An item outside the catalog becomes ``unk``, keeping its position. That matters
more than it sounds: a later split stage genuinely contains items the model was
not fitted on, and *dropping* them instead would join their neighbours as though
they had been consecutive. On a temporal MovieLens-1M split that fabricates 21%
of all adjacencies across every row and costs about 9% of ndcg@20. A vocabulary
built without ``unk`` cannot express such an item and raises rather than guesses.

:class:`~compresso_recsys.models.SequenceBatcher` always pads on the right. For
an RNN, the final-state helpers read each row before its trailing pads. For a
causal transformer, real tokens cannot attend to padding that follows them, so
training needs no padding mask — only a loss mask. Left padding is deliberately
not configurable: a raw GRU or LSTM would process the leading pad steps, while a
transformer would need an additional key-padding mask. In either case it could
make behavior depend on the other histories in the batch.

``max_length`` truncates to the **most recent** interactions, the only sensible
direction, since a context window is a claim about recency rather than about
where a history happened to start.

Those two reading helpers exist for the single easiest thing to get wrong. Under
right padding the last *column* is padding for every row shorter than the batch
maximum, so reading ``states[:, -1]`` silently scores most users from a pad
embedding — and agrees with itself at ``batch_size=1``, where every row fills its
own batch, which is what lets the bug survive casual testing. Empty rows report
position 0, which is padding; pair the position with
:meth:`~compresso_recsys.models.SequenceBatcher.has_history`.

Neither component owns a training objective. A next-item shift, a masked-position
target and sampled negatives differ between architectures, and a component with
three mutually exclusive modes is not an abstraction. Nor does either apply
*corruption*: injecting ``unk`` or selecting ``mask`` positions is stochastic, and
keeping ``encode`` a pure function of its input is what lets a test assert that
batching cannot change a prediction. Both live in trainers.

.. code-block:: python

   from compresso_recsys.models import ItemTokenizer, SequenceBatcher

   tokenizer = ItemTokenizer(
       n_items=1085,
       special_tokens={"pad": 0, "unk": 1},
       n_reserved=4,                       # room for mask, cls, later
       item_ids=split["train_item_ids"],   # optional; enables the ID path
   )
   batcher = SequenceBatcher(tokenizer, max_length=200)

   # A later stage may be wider than the tokenizer; unknown items become unk.
   tokens, mask = batcher.encode(split["test_source_sequences"], device="cuda")
   states = model(tokens)                        # (rows, length, hidden)
   final = batcher.gather_final(states, mask)    # (rows, hidden)

Bring Your Own Vocabulary
^^^^^^^^^^^^^^^^^^^^^^^^^

:class:`compresso_recsys.models.ItemTokenizer` is a convenience. What
:class:`compresso_recsys.models.SequenceBatcher` actually depends on is
:class:`compresso_recsys.models.Tokenizer`, a two-member structural protocol —
``pad_id`` and ``encode_indices`` — so a custom vocabulary qualifies by having
them rather than by inheriting anything, exactly as
:class:`compresso_recsys.models.Recommender` works.

``encode_indices`` must return one token per value, in order. The batcher
computes each destination before it maps anything, so a vocabulary that expands
one item into several tokens — semantic IDs from a residual-quantised autoencoder,
say — cannot be used with it and needs its own batcher.

.. autoclass:: compresso_recsys.models.Tokenizer
   :members:

.. autoclass:: compresso_recsys.models.ItemTokenizer
   :members:

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

A history retaining one interaction after truncation yields no training example,
since a next-item target needs a preceding item. ``fit`` raises if truncation
leaves every history this short. Such rows remain predictable, and an entirely
empty history yields the state after a single pad — identical for every empty
row, so effectively a learned prior.

``history`` records the mean loss per epoch alongside the number of positions it
was averaged over. That count is worth reading rather than assuming: it is
``sum(max(min(length, max_length) - 1, 0))``, so it reports what truncation
costs.

The trainer currently runs a fixed epoch budget and rebuilds the model on every
``fit`` call. It does not provide validation-based early stopping or incremental
training. Tied embeddings and sampled softmax are also not implemented.

.. code-block:: python

   from compresso_recsys.evaluation import evaluate_recommender
   from compresso_recsys.metrics import CalibratedRecall, NDCG
   from compresso_recsys.models import (
       ItemTokenizer,
       SequenceBatcher,
       SimpleRNNConfig,
       SimpleRNNTrainer,
   )

   model = SimpleRNNTrainer(
       SimpleRNNConfig(
           rnn_type="gru",
           embedding_dim=64,
           hidden_dim=128,
           epochs=8,
           batch_size=256,
           lr=3e-3,
       ),
       # The window belongs to the encoder, not to the network.
       SequenceBatcher(ItemTokenizer(split["x_train_sequences"].n_items),
                       max_length=200),
   ).fit(split["x_train_sequences"])

   result = evaluate_recommender(
       model,
       source=split["test_source_sequences"],
       targets=split["test_target_matrix"],
       metrics=[CalibratedRecall(20), NDCG(20)],
       batch_size=512,
   )

.. autoclass:: compresso_recsys.models.SimpleRNNConfig
   :members:

.. autoclass:: compresso_recsys.models.SimpleRNN
   :members:

.. autoclass:: compresso_recsys.models.SimpleRNNTrainer
   :members:

SimpleGPT
~~~~~~~~~

SimpleGPT is a causal transformer over the same histories `SimpleRNN` reads. The
architecture is nanoGPT — pre-norm blocks, fused QKV attention, a learned
absolute position per slot — with the recommendation-shaped adjustments below,
and it
is the model :class:`compresso_recsys.models.ItemTokenizer` and
:class:`compresso_recsys.models.SequenceBatcher` were split apart for.

**A `CLS` prefix replaces the shift.** Position 0 holds a learned vector, so
``states[:, i]`` has read `CLS` plus ``tokens[:, :i]`` and therefore predicts
``tokens[:, i]``. The next-item alignment becomes a property of the input rather
than arithmetic in the trainer::

   tokens   [a, b, c, PAD]      mask   [T, T, T, F]
   input    [CLS, a, b, c, PAD]
   targets  [a,   b, c, PAD]    valid  [T, T, T, F]

Two things follow. Every real position is a target, where a left shift makes
every position but the first one — so a history of a single interaction is a
usable training example here, and `CLS` buys back one example per user. And an
empty history has a *defined* input: it reads `CLS` alone and scores from the
learned prefix, rather than from the state after reading one pad.

`CLS` is an ``nn.Parameter`` rather than a vocabulary entry, which is the more
complicated of the two options and the deliberate one. A parameter can be
*conditioned* — a user embedding or a global feature added into position 0 per
row, as ``rstar`` does — and a vocabulary lookup cannot express that. Nothing in
this library has user features yet, so today it does the job `BOS` would.

**There is no attention mask.** The batcher always pads on the right, so a causal
mask already excludes padding: a real token at position ``i`` attends only to
``<= i``, all of which are real. Pad positions do compute garbage and nothing
reads it — the loss is masked and prediction reads each row's last real
position. This invariant is why the attention module needs no padding-mask
argument.

**The head is tied to the input embedding, and scores the catalog rather than
the vocabulary.** Items occupy the last ``n_items`` rows of the vocabulary, so
the output weight is a slice of the embedding and ``pad`` and ``unk`` fall below
it — which is what we want, since neither is ever a prediction target. Tying
halves the parameters and is the default; set ``tie_embeddings=False`` to use a
separate output head. A tied head starts with a flatter softmax because
``nn.Linear`` initialises near ``±1/sqrt(d_model)`` while an embedding starts at
``std=0.02``, so tying can also change the training curve rather than only the
parameter count.

**Initialisation follows GPT-2, including the depth-scaled residual init.**
Every weight starts at ``std=0.02`` — PyTorch's ``nn.Linear`` default is roughly
2.5× wider at ``d_model=128`` — and the two projections in each block that write
into the residual stream start at ``0.02 / sqrt(2 * n_layers)`` instead. Each
block adds to the stream twice, so without that scaling its variance grows with
depth and a deeper model starts further from anything usable.

**A cosine learning-rate schedule is on by default.** It gives linear warmup over
``warmup_fraction`` of the run, then cosine decay to ``min_lr_ratio × lr``,
with the final optimizer update using that floor. The curve is measured in
optimizer steps so its shape does not move with batch size. Set
``lr_schedule="constant"`` to disable both. Warmup exists because the earliest
steps of a transformer are the ones most able to wreck it — attention has
learned nothing, so gradients are large and badly aimed. Neither half is
expressible through the optimizer alone, which is why they arrive as one option
rather than two.

**The context window is derived, not configured.** ``max_length`` on the batcher
sizes the positional table, so ``SimpleGPTConfig`` carries no ``block_size`` and
the two cannot disagree. The consequence is that ``max_length=None`` is an error
for this model — learned absolute positions need a bound — which is a real
difference from `SimpleRNN`, where the window only decides how much history is
read.

.. code-block:: python

   from compresso_recsys.models import (
       ItemTokenizer,
       SequenceBatcher,
       SimpleGPTConfig,
       SimpleGPTTrainer,
       TransformerConfig,
   )

   tokenizer = ItemTokenizer(split["x_train_sequences"].n_items)
   batcher = SequenceBatcher(tokenizer, max_length=200)   # required, and it
                                                          # sizes the positions

   model = SimpleGPTTrainer(
       SimpleGPTConfig(
           transformer=TransformerConfig(
               d_model=128, n_heads=4, n_layers=2, dropout=0.1
           ),
           # Select the fixed budget on validation data.
           epochs=10,
           batch_size=128,
           lr=1e-3,
       ),
       batcher,
   ).fit(split["x_train_sequences"])

   result = evaluate_recommender(
       model,
       source=split["test_source_sequences"],
       targets=split["test_target_matrix"],
       metrics=[CalibratedRecall(20), NDCG(20)],
   )

The trainer currently runs a fixed epoch budget and rebuilds the model on every
``fit`` call. It does not provide validation-based early stopping or incremental
training, so select ``epochs`` using validation data. Sampled softmax, a logit
temperature, and pooling strategies other than reading the final real position
are also not implemented.

Saving carries the vocabulary with the weights, because a served model that
cannot say what column 41 means is not much use. It uses the same persistence
API as every other fitted recommender:

.. code-block:: python

   from compresso_recsys.models import SimpleGPTTrainer

   model.save("artifacts/simple_gpt.ckpt")
   restored = SimpleGPTTrainer.load("artifacts/simple_gpt.ckpt")

.. autoclass:: compresso_recsys.models.TransformerConfig
   :members:

.. autoclass:: compresso_recsys.models.SimpleGPTConfig
   :members:

.. autoclass:: compresso_recsys.models.SimpleGPT
   :members:

.. autoclass:: compresso_recsys.models.SimpleGPTTrainer
   :members:
