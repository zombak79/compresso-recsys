Getting Started
===============

Compresso Recsys provides recommender-system experiments and pipeline utilities
around Compresso sparse representation learning.

The public API is centered on:

* dataset loaders for GoodBooks, MovieLens, and Amazon Reviews 2023
* ZIP checkpoint helpers for storing splits, embeddings, sparse embeddings, and
  metrics
* retrieval holdout builders for user split, item cold-start split,
  leave-last-out, and temporal evaluation
* lightweight model helpers for ELSA, CompressedELSA, SBERT, and SAE stages

Basic Python Usage
------------------

.. code-block:: python

   import compresso_recsys as cr

   dataset = cr.MovieLens1M(data_dir="data")
   interactions = dataset.get_interactions()

   print(interactions.head())

Datasets expose interactions with a canonical schema:

* ``user_id`` as a string
* ``item_id`` as a string
* ``value`` as a float
* ``timestamp`` when available

Checkpoint Workflow
-------------------

Use :func:`compresso_recsys.build_recsys_checkpoint` to create the ZIP
checkpoint used as the exchange format between stages:

.. code-block:: python

   import compresso_recsys as cr

   checkpoint_path = cr.build_recsys_checkpoint(
       dataset="ml1m",
       checkpoint_path="artifacts/ml1m/exp001.zip",
       annotation_source="genres",
   )

Checkpoint helpers can then read or update that checkpoint:

.. code-block:: python

   import compresso_recsys as cr

   with cr.update_checkpoint(checkpoint_path) as root:
       split = cr.load_recsys_split(root)

Pipeline scripts and adapters can add embeddings, sparse representations,
metrics, or cluster graphs to the same checkpoint.

What Goes Into a Checkpoint
---------------------------

The dataset builder writes one portable ZIP file. The core split contains:

* ``item_ids``: item-id order used by every embedding matrix
* ``train_source_matrix`` / ``train_target_matrix``:
  sparse matrices defining the training input and target
* ``val_source_matrix`` / ``val_target_matrix`` and
  ``test_source_matrix`` / ``test_target_matrix``:
  sparse matrices defining validation/test retrieval inputs and targets
* ``val_source_indices`` / ``val_target_indices`` and
  ``test_source_indices`` / ``test_target_indices``:
  list-of-index views of the same validation/test holdouts
* ``train_user_ids`` and ``val_eval_user_ids`` / ``test_eval_user_ids`` when
  user ids are meaningful for the split
* ``train_item_indices`` / ``val_item_indices`` / ``test_item_indices`` when
  the protocol partitions items, especially for cold-item experiments
* optional ``entity_metadata``, ``entity_tag_matrix``, and ``tag_names``

For compatibility with older scripts, ``x_train`` loaded by
:func:`compresso_recsys.load_recsys_split` is an alias for
``train_source_matrix``.

Training stages append their own directories to the same checkpoint, for
example ``elsa/``, ``sbert/``, ``sae/``, or ``compressed_elsa/``. Each stage
can save embeddings, sparse representations, model files, and metrics.

Split Modes
-----------

``user_split``
   Holds out validation/test users and builds source/target folds from those
   users. This mirrors the original ELSA-style protocol. The checkpoint stores
   ``train_user_ids``, ``val_user_ids``, and ``test_user_ids``. It is not a
   future-blind temporal protocol.

``item_split``
   Holds out cold validation/test items. Training stages should fit only on
   ``train_item_indices`` and then transform all items before evaluation. The
   checkpoint stores item partitions rather than user partitions.

``leave_last_out``
   Uses each user's latest timestamped interaction as the target and earlier
   interactions as source. This requires timestamps and respects order within
   each user, but it is not globally future-blind: interactions from other
   users may occur after a given user's held-out target.

``temporal``
   Uses a global timestamp split. For Amazon Reviews 2023, this uses the
   predefined ``0core_timestamp_w_his`` split with item histories. This is the
   recommended split mode when future-to-past leakage must be avoided.

Retrieval Metrics
-----------------

Low-level evaluation functions return ``recall@K`` and ``ndcg@K`` for the
single ``K`` requested. The command-line evaluation pipeline calls them for
``K = 20, 50, 100`` and stores the six common metrics:

* ``recall@20``
* ``ndcg@20``
* ``recall@50``
* ``ndcg@50``
* ``recall@100``
* ``ndcg@100``
