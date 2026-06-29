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

Choose a Split Mode
-------------------

Use ``user_split`` for classic warm-item recommender experiments:

.. code-block:: bash

   compresso-recsys-build-checkpoint \
     --dataset ml1m \
     --checkpoint_path artifacts/ml1m/user_split_exp001.zip \
     --split_mode user_split \
     --annotation_source genres

Use ``item_split`` when you want cold-item evaluation. Models should train on
``train_item_indices`` only, then transform all items before evaluation:

.. code-block:: bash

   compresso-recsys-build-checkpoint \
     --dataset amazon2023 \
     --amazon_category Video_Games \
     --checkpoint_path artifacts/amazon_video_games/item_split_exp001.zip \
     --split_mode item_split \
     --metadata_text_fields title,features,description,categories \
     --min_entity_text_words 20 \
     --annotation_source none

Use ``leave_last_out`` or ``temporal`` when timestamps should define the
evaluation target. Prefer ``temporal`` when you need a future-blind split:

.. code-block:: bash

   compresso-recsys-build-checkpoint \
     --dataset amazon2023 \
     --amazon_category Toys_and_Games \
     --checkpoint_path artifacts/amazon_toys/temporal_exp001.zip \
     --split_mode temporal \
     --metadata_text_fields title,features,description,categories \
     --min_entity_text_words 30 \
     --annotation_source none

Evaluate a Checkpoint
---------------------

After adding one or more embedding stages, evaluate everything stored in the
checkpoint:

.. code-block:: bash

   compresso-recsys-eval-checkpoint \
     --checkpoint_path artifacts/amazon_toys/temporal_exp001.zip \
     --device cuda

The full checkpoint-level metric set is ``recall@20``, ``ndcg@20``,
``recall@50``, ``ndcg@50``, ``recall@100``, and ``ndcg@100``.
