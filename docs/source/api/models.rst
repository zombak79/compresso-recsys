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

LEMSA
-----

LEMSA (Language Embeddings Meet Shallow Autoencoders) is an inductive shallow
autoencoder for recommending cold items to warm users. It learns one encoder
row per warm item while keeping supplied item embeddings fixed as its decoder.
New candidates therefore need only an embedding in the same feature space;
they do not require interactions or retraining.

LEMSA uses diagonal gating rather than TEASER's global zero-diagonal
constraint. When encoder row ``i`` is updated, only users who interacted with
item ``i`` are used, and item ``i`` is removed from both their reconstruction
targets and the fixed decoder. This blocks the direct self-copy shortcut for
the row being learned while preserving the semantic contribution of every
other item in each user's history. The resulting encoder-decoder diagonal is
intentionally unconstrained.

The default ``solver="eigen"`` implementation is exact. It rotates training
into the eigenspace of the feature Gram matrix, where every closed-form row
system is diagonal minus rank one and can be solved with the Sherman-Morrison
identity. The encoder is rotated back to the original input feature space after
training, so candidate embeddings,
:meth:`~compresso_recsys.models.LEMSA.user_profiles`, and catalog updates retain
their original language-embedding coordinates. ``solver="direct"`` evaluates
the same row systems with dense solves and is intended for small reference
checks.

Rows are updated in snapshot-based blocks, matching the parallel update
strategy described by the paper's appendix. Every row in a block is solved
against the same frozen encoder and user profiles, then all changes are
committed together. ``update_batch_size=1`` recovers sequential Gauss-Seidel
updates, while ``update_batch_size=None`` performs one full Jacobi update per
epoch. The paper reports small parallel batches but does not publish the exact
batch size. Because simultaneous updates can be unstable when many items
co-occur, the correctness-first default is ``1``; larger values must be
selected on validation data.

``update_rate`` optionally damps each closed-form proposal before updating the
encoder and cached user profiles. If ``e*`` is the solved row, the committed
row is ``e + update_rate * (e* - e)``. The default ``1.0`` preserves the exact
paper-compatible update; smaller values retain the same fixed points while
using a more conservative optimization path. When ``tolerance`` is configured,
convergence is tested against the full undamped proposal so a small update rate
cannot cause premature stopping. Both applied and proposed changes are exposed
in ``fit_history_``.

Set ``shuffle_updates=True`` to draw a reproducible item permutation from
``seed`` before every epoch. Sequential updates then avoid a persistent catalog
order bias, while snapshot blocks are formed from consecutive rows in the
permutation. Shuffling has no numerical effect when ``update_batch_size=None``
because the complete item set is still solved from one frozen Jacobi snapshot.
The default is ``False`` to preserve the original deterministic row order.

Fitting maintains a dense user-by-feature profile matrix and dense
item-by-feature encoder. Its main storage is therefore
``(users + warm_items) * feature_dim`` rather than a dense item-by-item matrix.
``precompute_batch_size`` bounds the temporary user buffer used to form fixed
semantic target sums. The feature Gram matrix and eigendecomposition still
cost ``feature_dim ** 2`` memory.

LEMSA shares TEASER's stable-ID candidate catalog, sparse ``align_source``
operation, atomic candidate replacement and updates, metadata handling, and
prediction-time candidate allowlists. Source histories remain restricted to
items with fitted encoder rows.

See :doc:`../citing` for the original LEMSA paper.

.. autoclass:: compresso_recsys.models.LEMSAConfig
   :members:

.. autoclass:: compresso_recsys.models.LEMSA
   :members:
   :inherited-members:

LEMSAGD
-------

LEMSAGD trains the same unconstrained ``E @ S.T`` architecture through
held-out reconstruction rather than the paper's item-wise ALS objective. Its
default ``training_mode="leave_one_out"`` creates one virtual example for
every observed interaction of every eligible user. The designated interaction
is removed from the source history and used as a one-hot target, so each
interaction is predicted exactly once during a complete epoch.

Virtual examples are generated directly from the original CSR matrix. The
trainer stores an ordering of its observed entries but never materializes the
much larger expanded source or target CSR matrices. The default
``loo_batch_order="round_robin"`` takes one target from every active user,
shuffles those users, and divides that round into batches before moving to the
next target per user. A batch therefore contains at most one example from any
user. The target order within each user and the user order within each round
are reshuffled each epoch. ``loo_batch_order="grouped"`` retains the older
user-contiguous ordering when batch locality is preferred. Round-robin batches
usually span a wider union of source items, so grouped ordering can be faster
when source-prefix candidate sampling would otherwise exceed ``max_output``.

Round boundaries are preserved, so some batches can be smaller than
``batch_size``. That setting counts virtual interaction examples rather than
unique users, and a complete epoch is substantially longer than one epoch of
ordinary user-level reconstruction.

The active source history is removed from its loss mask. Source interactions
are therefore not mislabeled as negative outputs, and the encoder-decoder
diagonal is neither subtracted nor explicitly penalized. Identity copying
cannot reduce the objective because the target interaction is absent from its
source. Prediction uses the full unconstrained scores and applies ordinary
seen-item masking afterward.

Only users retaining at least two warm interactions can leave out a target
while preserving non-empty context; other users are ignored during fitting but
remain fully supported during prediction. ``training_mode="symmetric"`` retains
the earlier denoising
ablation: histories are randomly divided into two non-empty views and both
``x -> y`` and ``y -> x`` are optimized. ``split_probability`` applies only to
this mode. All ordering, splitting, and negative sampling is reproducible from
``seed``.

The trainer reuses TEASERGD's fixed dense or CSR feature decoder, stable-ID
candidate catalog, cold candidate updates, optional popularity feature,
Xavier or feature initialization, row-normalized encoder option, NAdam/AdamW,
cosine decay, ``torch.compile``, and source-prefix ``max_output`` sampling.
LEMSAGD is an experimental held-out reconstruction variant inspired by LEMSA;
it is not the gradient equivalent of the paper's gated ALS algorithm.

.. autoclass:: compresso_recsys.models.LEMSAGDConfig
   :members:

.. autoclass:: compresso_recsys.models.LEMSAGD
   :members:

.. autoclass:: compresso_recsys.models.LEMSAGDTrainer
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
