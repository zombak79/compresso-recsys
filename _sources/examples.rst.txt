Examples
========

Dataset Loader
--------------

.. code-block:: python

   import compresso_recsys as cr

   dataset = cr.MovieLens1M(data_dir="data")
   interactions = dataset.get_interactions()
   x_train, user_ids, item_ids = dataset.to_sparse_matrix(interactions)

   print(x_train.shape)

Building Checkpoints
--------------------

For programmatic checkpoint creation, call
:func:`compresso_recsys.build_recsys_checkpoint`.

MovieLens 1M:

.. code-block:: python

   import compresso_recsys as cr

   checkpoint_path = cr.build_recsys_checkpoint(
       dataset="ml1m",
       checkpoint_path="artifacts/ml1m/exp001.zip",
       annotation_source="genres",
   )

GoodBooks with item tags:

.. code-block:: python

   checkpoint_path = cr.build_recsys_checkpoint(
       dataset="goodbooks",
       checkpoint_path="artifacts/goodbooks/item_split_exp001.zip",
       split_mode="item_split",
       annotation_source="goodbooks_tags",
       annotation_min_count=100,
   )

Amazon Reviews 2023 with metadata text:

.. code-block:: python

   checkpoint_path = cr.build_recsys_checkpoint(
       dataset="amazon2023",
       amazon_category="Toys_and_Games",
       checkpoint_path="artifacts/amazon_toys/item_split_exp001.zip",
       split_mode="item_split",
       metadata_text_fields=["title", "features", "description", "categories"],
       min_entity_text_words=20,
       min_user_support=10,
       item_min_support=10,
       min_value_to_keep=1.0,
       set_all_values_to=1.0,
       min_source_items=1,
       min_target_items=1,
       annotation_source="none",
   )

Checkpoint Read/Write
---------------------

.. code-block:: python

   import compresso_recsys as cr

   checkpoint_path = "artifacts/ml1m/exp001.zip"

   with cr.read_checkpoint(checkpoint_path) as root:
       split = cr.load_recsys_split(root)
       print(split["x_train"].shape)


