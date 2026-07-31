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

.. autoclass:: compresso_recsys.models.CandidateCatalog
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
