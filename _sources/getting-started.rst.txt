Getting Started
===============

Compresso Recsys provides recommender-system experiments and pipeline utilities
around Compresso sparse representation learning.

The public API is centered on:

* dataset loaders for GoodBooks, MovieLens, and Amazon Reviews 2023
* ZIP checkpoint helpers for storing splits, embeddings, sparse embeddings, and
  metrics

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
