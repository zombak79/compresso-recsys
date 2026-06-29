Quickstart
==========

Build a dataset checkpoint from Python:

.. code-block:: python

   import compresso_recsys as cr

   checkpoint_path = cr.build_recsys_checkpoint(
       dataset="ml1m",
       checkpoint_path="artifacts/ml1m/exp001.zip",
       annotation_source="genres",
   )

   print(checkpoint_path)

Read the generated checkpoint:

.. code-block:: python

   with cr.read_checkpoint(checkpoint_path) as root:
       split = cr.load_recsys_split(root)

   print(split["x_train"].shape)

Create an Amazon Reviews 2023 item-split checkpoint:

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
