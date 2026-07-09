Checkpoint CLI Reference
========================

The ``compresso-recsys-build-checkpoint`` command builds ZIP checkpoints for
the supported recommender-system datasets.

Basic usage:

.. code-block:: bash

   compresso-recsys-build-checkpoint \
     --dataset ml1m \
     --checkpoint_path artifacts/ml1m/exp001.zip \
     --annotation_source genres

Amazon Reviews 2023
-------------------

Amazon checkpoints use compact rating-only interactions plus item metadata:

.. code-block:: text

   0core_rating_only_<category>
   raw_meta_<category>

For temporal checkpoints, Amazon uses McAuley's predefined timestamp split with
history. This is the preferred split when avoiding future-to-past leakage
matters:

.. code-block:: text

   0core_timestamp_w_his_<category>

The builder also constructs a canonical ``entity_text`` column from
configurable metadata fields, so downstream code can encode item descriptions
consistently.

Leave-Last-Out Checkpoint
~~~~~~~~~~~~~~~~~~~~~~~~~

``leave_last_out`` is computed locally from timestamps. For every eligible
user, the latest interaction becomes the target and earlier interactions
become the source profile. This respects time within each user, but it is not
globally future-blind because other users may contribute later interactions to
training.

.. code-block:: bash

   compresso-recsys-build-checkpoint \
     --dataset amazon2023 \
     --amazon_category Toys_and_Games \
     --checkpoint_path artifacts/amazon_toys/leave_last_out_exp001.zip \
     --split_mode leave_last_out \
     --metadata_text_fields title,features,description,categories \
     --min_entity_text_words 30 \
     --min_user_support 20 \
     --item_min_support 20 \
     --min_value_to_keep 4.0 \
     --set_all_values_to 1.0 \
     --min_source_items 1 \
     --min_target_items 1 \
     --annotation_source none

Temporal Checkpoint
~~~~~~~~~~~~~~~~~~~

``temporal`` uses the Amazon Reviews 2023 predefined timestamp split when
``--dataset amazon2023`` is selected. Targets are kept cold with respect to the
Amazon training split, so this checkpoint is intended for metadata/SBERT-style
cold-item evaluation.

.. code-block:: bash

   compresso-recsys-build-checkpoint \
     --dataset amazon2023 \
     --amazon_category Toys_and_Games \
     --checkpoint_path artifacts/amazon_toys/temporal_exp001.zip \
     --split_mode temporal \
     --metadata_text_fields title,features,description,categories \
     --min_entity_text_words 30 \
     --min_user_support 20 \
     --item_min_support 20 \
     --min_value_to_keep 4.0 \
     --set_all_values_to 1.0 \
     --min_source_items 1 \
     --min_target_items 1 \
     --annotation_source none

Checkpoint evaluation commonly reports this six-metric table:

.. code-block:: text

   recall@20, ndcg@20, recall@50, ndcg@50, recall@100, ndcg@100

Checkpoint Split Schema
-----------------------

Every checkpoint stores source/target matrices for train, validation, and
test:

.. code-block:: text

   data/train_source_matrix.npz
   data/train_target_matrix.npz
   data/val_source_matrix.npz
   data/val_target_matrix.npz
   data/test_source_matrix.npz
   data/test_target_matrix.npz

``source`` is the profile/input side and ``target`` is what retrieval metrics
try to recover. The older ``data/train_matrix.npz`` file is still written as an
alias for ``train_source_matrix.npz``.

Depending on the split mode, the checkpoint also stores partition ids:

``user_split``
   Stores ``train_user_ids.npy``, ``val_user_ids.npy``, and
   ``test_user_ids.npy``. It does not store explicit item partitions; loaders
   treat all items as train items.

``item_split``
   Stores ``train_item_indices.npy``, ``val_item_indices.npy``, and
   ``test_item_indices.npy``.

``leave_last_out``
   Stores source/target matrices built from per-user latest interactions. It is
   chronological per user, but not globally future-blind.

``temporal``
   Stores source/target matrices from a global timestamp split. For Amazon
   Reviews 2023, this uses McAuley's predefined temporal split.

Validation/test source-target rows also have aligned ``val_eval_user_ids.npy``
and ``test_eval_user_ids.npy`` when user identifiers are available.

Builder Parameters
------------------

``--min_source_items 1`` and ``--min_target_items 1`` mean:

.. code-block:: text

   Keep an evaluation user only if they have at least 1 source item and at least 1 target item.

For cold-item splits:

.. code-block:: text

   source items = warm/train items used as the user profile
   target items = cold held-out items we want to recommend

If a user has only cold targets but no warm source items, the builder cannot
construct a profile, and the user is dropped.

Full ``compresso-recsys-build-checkpoint`` parameter table:

.. list-table::
   :header-rows: 1
   :widths: 22 18 60

   * - Parameter
     - Default
     - Description
   * - ``--dataset``
     - required
     - Dataset to build. Choices: ``goodbooks``, ``ml1m``, ``ml20m``,
       ``amazon2023``.
   * - ``--data_dir``
     - ``data``
     - Directory where raw/downloaded dataset files are stored.
   * - ``--checkpoint_path``
     - dataset-specific
     - Output ZIP checkpoint path. If omitted, uses the dataset default.
   * - ``--seed``
     - dataset-specific
     - Random seed for user/item splitting and reproducibility.
   * - ``--val_users``
     - dataset-specific
     - Number of validation users for ``user_split``.
   * - ``--test_users``
     - dataset-specific
     - Number of test users for ``user_split``.
   * - ``--min_user_support``
     - dataset-specific
     - Minimum number of interactions per user during iterative pruning.
   * - ``--item_min_support``
     - dataset-specific
     - Minimum number of interactions per item during iterative pruning.
   * - ``--min_value_to_keep``
     - dataset-specific
     - Drop interactions below this value. Usually ``4.0``, meaning keep
       positive ratings only.
   * - ``--set_all_values_to``
     - dataset-specific
     - If set, binarize all remaining interaction values to this value. Usually
       ``1.0``.
   * - ``--eval_fold``
     - ``0``
     - Evaluation fold protocol for ``user_split``. ``0`` means stacked 5-fold
       paper-style behavior; ``1`` means single fold.
   * - ``--split_mode``
     - ``user_split``
     - Split protocol. Choices: ``user_split``, ``item_split``,
       ``leave_last_out``, ``temporal``.
   * - ``--val_items``
     - ``None``
     - Exact number of cold validation items for ``item_split``. Overrides
       ``--item_val_frac``.
   * - ``--test_items``
     - ``None``
     - Exact number of cold test items for ``item_split``. Overrides
       ``--item_test_frac``.
   * - ``--item_val_frac``
     - ``0.05``
     - Fraction of items held out as cold validation items for ``item_split``.
   * - ``--item_test_frac``
     - ``0.10``
     - Fraction of items held out as cold test items for ``item_split``.
   * - ``--temporal_test_frac``
     - ``0.10``
     - For local temporal split, latest global fraction of interactions used as
       target side. For Amazon ``temporal``, McAuley's predefined timestamp
       split is used instead.
   * - ``--min_source_items``
     - ``1``
     - Minimum number of source/profile items an eval user must have. For
       cold-item eval, these are train/warm items.
   * - ``--min_target_items``
     - ``1``
     - Minimum number of target/held-out items an eval user must have. For
       cold-item eval, these are cold items.
   * - ``--amazon_category``
     - ``Toys_and_Games``
     - Amazon Reviews 2023 category. Supports official names and aliases like
       ``toys``, ``electronics``, ``clothing``.
   * - ``--metadata_text_fields``
     - ``title,features,description,categories``
     - Metadata columns joined into canonical ``entity_text``. Mostly important
       for Amazon/SBERT.
   * - ``--min_entity_text_words``
     - ``30``
     - Drop items whose constructed ``entity_text`` is shorter than this many
       words. Mostly useful for Amazon.
   * - ``--include_image_urls``
     - ``False``
     - For Amazon, include ``image_url`` and ``image_urls`` columns in
       ``entity_metadata`` without adding them to ``entity_text``.
   * - ``--annotation_source``
     - ``genres``
     - Optional tag source for clustering. Choices: ``genres``,
       ``ml20m_tags``, ``goodbooks_tags``, ``none``.
   * - ``--annotation_min_count``
     - ``100``
     - Minimum count threshold for tag annotations when using user-generated
       tags.

Dataset-specific defaults:

.. list-table::
   :header-rows: 1
   :widths: 24 46 10 10 10 12 12

   * - Dataset
     - ``checkpoint_path``
     - ``seed``
     - ``val_users``
     - ``test_users``
     - ``min_user_support``
     - ``item_min_support``
   * - ``goodbooks``
     - ``artifacts/goodbooks/recsys_checkpoint.zip``
     - ``0``
     - ``1000``
     - ``2500``
     - ``5``
     - ``1``
   * - ``ml1m``
     - ``artifacts/ml1m/recsys_checkpoint.zip``
     - ``42``
     - ``500``
     - ``1000``
     - ``5``
     - ``1``
   * - ``ml20m``
     - ``artifacts/ml20m/recsys_checkpoint.zip``
     - ``42``
     - ``2500``
     - ``5000``
     - ``5``
     - ``1``
   * - ``amazon2023``
     - ``artifacts/amazon2023/{amazon_category}/recsys_checkpoint.zip``
     - ``42``
     - ``2500``
     - ``5000``
     - ``20``
     - ``20``

All datasets currently default to:

.. list-table::
   :header-rows: 1

   * - Parameter
     - Default
   * - ``min_value_to_keep``
     - ``4.0``
   * - ``set_all_values_to``
     - ``1.0``

Supported Amazon Reviews 2023 Datasets
--------------------------------------

.. list-table::
   :header-rows: 1
   :widths: 45 30 25

   * - Official Amazon 2023 category
     - Alias in ``compresso-recsys``
     - Supported?
   * - ``All_Beauty``
     - ``beauty``
     - yes
   * - ``Amazon_Fashion``
     - none
     - yes, pass official name
   * - ``Appliances``
     - none
     - yes
   * - ``Arts_Crafts_and_Sewing``
     - none
     - yes
   * - ``Automotive``
     - none
     - yes
   * - ``Baby_Products``
     - none
     - yes
   * - ``Beauty_and_Personal_Care``
     - none
     - yes
   * - ``Books``
     - none
     - yes
   * - ``CDs_and_Vinyl``
     - none
     - yes
   * - ``Cell_Phones_and_Accessories``
     - none
     - yes
   * - ``Clothing_Shoes_and_Jewelry``
     - ``clothing``
     - yes
   * - ``Digital_Music``
     - none
     - yes
   * - ``Electronics``
     - ``electronics``
     - yes
   * - ``Gift_Cards``
     - none
     - yes
   * - ``Grocery_and_Gourmet_Food``
     - none
     - yes
   * - ``Handmade_Products``
     - none
     - yes
   * - ``Health_and_Household``
     - none
     - yes
   * - ``Health_and_Personal_Care``
     - none
     - yes
   * - ``Home_and_Kitchen``
     - none
     - yes
   * - ``Industrial_and_Scientific``
     - none
     - yes
   * - ``Kindle_Store``
     - none
     - yes
   * - ``Magazine_Subscriptions``
     - none
     - yes
   * - ``Movies_and_TV``
     - none
     - yes
   * - ``Musical_Instruments``
     - none
     - yes
   * - ``Office_Products``
     - none
     - yes
   * - ``Patio_Lawn_and_Garden``
     - none
     - yes
   * - ``Pet_Supplies``
     - none
     - yes
   * - ``Software``
     - none
     - yes
   * - ``Sports_and_Outdoors``
     - none
     - yes
   * - ``Subscription_Boxes``
     - none
     - yes
   * - ``Tools_and_Home_Improvement``
     - none
     - yes
   * - ``Toys_and_Games``
     - ``toys``, ``toys_and_games``
     - yes
   * - ``Video_Games``
     - none
     - yes
