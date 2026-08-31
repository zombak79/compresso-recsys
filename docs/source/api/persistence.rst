Fitted Model Persistence
========================

Every built-in fitted recommender exposes the same self-contained API:

.. code-block:: python

   from compresso_recsys.models import EASE

   model = EASE().fit(interactions, item_ids=item_ids)
   model.save("artifacts/ease.ckpt")

   restored = EASE.load("artifacts/ease.ckpt")

The loaded object is fitted and ready to predict. Checkpoints contain the model
configuration, learned state, and the vocabulary, mappings, or candidate catalog
needed to interpret its prediction columns. Loading defaults to CPU regardless
of the device used during training:

.. code-block:: python

   on_cpu = ELSATrainer.load("artifacts/elsa.ckpt")
   on_gpu = ELSATrainer.load("artifacts/elsa.ckpt", device="cuda")

Calling ``load`` through a different model class, reading an unsupported format
version, or saving an unfitted model raises rather than guessing.

Stable Item IDs
---------------

Fitted item identities are part of the model checkpoint, so the production
interface is unchanged after loading:

.. code-block:: python

   before = model.recommend([["item-14", "item-87"]], k=20)
   restored = EASE.load("artifacts/ease.ckpt")
   after = restored.recommend([["item-14", "item-87"]], k=20)

Checkpoint item IDs use an explicit pickle-free encoding for strings, integers,
finite floats, and booleans. A fixed-catalog model fitted without ``item_ids``
uses positional integer IDs. Cold-start checkpoints preserve both the fitted
source vocabulary and the current mutable candidate catalog.

Optimizer State
---------------

Optimizer state is excluded by default because prediction does not need it. A
trainer can include and restore it explicitly:

.. code-block:: python

   model.save("artifacts/elsa-with-optimizer.ckpt", include_optimizer=True)
   restored = ELSATrainer.load(
       "artifacts/elsa-with-optimizer.ckpt",
       load_optimizer=True,
   )

This preserves optimizer continuity for training APIs that can use it. It does
not promise exact training resumption: scheduler, random-number-generator,
data-order, and partial-epoch state are not stored. A model without an optimizer
rejects ``include_optimizer=True``.

Checkpoint Format
-----------------

Model checkpoints are atomic, versioned ZIP files, separate from dataset
checkpoints. The framework owns their manifest and layout. Configurations and
IDs use explicit JSON-safe encodings, NumPy arrays load with
``allow_pickle=False``, candidate metadata uses Parquet, sparse matrices use
SciPy ``.npz``, and Torch state is loaded with ``weights_only=True``.

The public reader and writer helpers give third-party model implementations the
same typed storage operations without requiring them to invent a manifest or
archive convention.

.. autoclass:: compresso_recsys.ModelCheckpointWriter
   :members:

.. autoclass:: compresso_recsys.ModelCheckpointReader
   :members:

Compiled Torch wrappers and runtime caches are not serialized. Torch models are
stored through their eager ``state_dict`` and loaded uncompiled. A direct
``nn.Module`` recommender can use the inherited Torch behavior after declaring
how its JSON configuration reconstructs the module; wrappers add only the
tokenizer, catalog, or other state outside that module.

Extending Persistence
---------------------

A model base subclass declares a stable ``checkpoint_type`` and implements
``_from_checkpoint_config`` so loading can reconstruct its fitted shape. A
dataclass stored as ``self.cfg`` supplies the default JSON configuration. For a
recommender that is itself an ``nn.Module``, those pieces are sufficient for
configuration and learned weights; the inherited implementation saves and
strictly reloads its ``state_dict``.

Trainer wrappers return their nested module from ``_checkpoint_module``.
Non-Torch learned state belongs in ``_save_checkpoint_state`` and
``_load_checkpoint_state``, using :class:`ModelCheckpointWriter` and
:class:`ModelCheckpointReader`. A trainer that supports optional optimizer
restoration also implements ``_build_checkpoint_optimizer``. Derived caches and
other runtime-only state are rebuilt in ``_finish_checkpoint_load``.

Warm Catalog Adapters
---------------------

:class:`compresso_recsys.models.WarmCatalogAdapter` is a projection onto a
particular evaluation-stage catalog rather than learned model state. Save its
nested fitted model and rebuild the adapter from the dataset checkpoint:

.. code-block:: python

   model.save("artifacts/rnn.ckpt")
   restored = SimpleRNNTrainer.load("artifacts/rnn.ckpt")
   restored = WarmCatalogAdapter(
       restored,
       train_item_ids=split["train_item_ids"],
       catalog_item_ids=split["test_item_ids"],
   )
