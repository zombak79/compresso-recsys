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

Temporal checkpoints are built from the timestamped rating data. They use
equal-width train, validation, and test target windows with expanding source
histories:

.. code-block:: text

   0core_timestamp_w_his_<category>

The builder also constructs a canonical ``entity_text`` column from
configurable metadata fields, so downstream code can encode item descriptions
consistently.

Leave-Last-Out Checkpoint
~~~~~~~~~~~~~~~~~~~~~~~~~

``leave_last_out`` is computed locally from timestamps. Each user's latest
interaction is the test target, the one before it the validation target, and the
one before that the training target; sources are the corresponding prefixes, so
each stage sees everything up to its own target. A user needs at least four
interactions to contribute to all three stages.

The catalog is left whole — nothing is withheld from training merely for being
someone's target. This respects time within each user, but it is not globally
future-blind, because another user's training interactions may post-date this
user's test target.

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

``temporal`` uses three equal target windows ending at the latest interaction.
The default period is 339 days, following the scale of the official Amazon
Reviews 2023 absolute-timestamp validation interval. Each split ranks a mixed
catalog of previously available warm items and newly supported cold items.

For period ``w`` and latest timestamp ``T``, the target windows are
``[T-3w, T-2w)``, ``[T-2w, T-w)``, and ``[T-w, T]``. Their corresponding
sources contain every eligible interaction before the start of that target
window.

.. code-block:: bash

   compresso-recsys-build-checkpoint \
     --dataset amazon2023 \
     --amazon_category Toys_and_Games \
     --checkpoint_path artifacts/amazon_toys/temporal_exp001.zip \
     --split_mode temporal \
     --temporal_period_hours 8136 \
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

   calibrated_recall@20, ndcg@20, calibrated_recall@50, ndcg@50, calibrated_recall@100, ndcg@100

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
try to recover. The older ``data/train_matrix.npz`` file stores ``x_train``;
for temporal checkpoints this is the train source/target union.

Every checkpoint also stores:

.. code-block:: text

   data/train_item_ids.npy
   data/val_item_ids.npy
   data/test_item_ids.npy

Each array defines the columns of both matrices in that split. Temporal item
spaces are cumulative, so source and target shapes match within a split while
the number and order of columns may differ between splits.

For temporal checkpoints, ``warm_item_indices`` maps the training catalog into
the global ``item_ids`` array, while ``val_cold_item_indices`` and
``test_cold_item_indices`` identify items newly admitted in those stages. The
stage-specific ``*_item_ids`` arrays, not these index subsets, define matrix
columns.

Depending on the split mode, the checkpoint also stores partition ids:

``user_split``
   Stores ``train_user_ids.npy``, ``val_user_ids.npy``, and
   ``test_user_ids.npy``. It does not store explicit item partitions; loaders
   treat all items as train items.

``item_split``
   Stores ``warm_item_indices.npy``, ``val_cold_item_indices.npy``, and
   ``test_cold_item_indices.npy``.

``leave_last_out``
   Stores source/target matrices built from per-user latest interactions. It is
   chronological per user, but not globally future-blind.

``temporal``
   Stores equal-width tail windows with expanding histories and cumulative
   mixed warm/cold item catalogs.

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

Temporal support filtering is iterative. A row must satisfy
``min_source_items``, ``min_target_items``, and ``min_user_support`` over the
boolean source/target union. Newly introduced items need
``item_min_support`` distinct retained users. Items admitted in an earlier
window remain warm candidates in later catalogs even when they are uncommon in
that later evaluation population.

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
   * - ``--eval_draws``
     - ``5``
     - How many independent fold-in/scored splits to draw per held-out user in
       ``user_split``, stacked one row per draw. More draws sharpen each user's
       score without adding independent users; ``1`` gives one row per user.
   * - ``--eval_holdout_frac``
     - ``0.2``
     - Share of each held-out user's history scored against, the rest being the
       fold-in history the model sees.
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
   * - ``--temporal_period_hours``
     - ``8136``
     - Width of each temporal target window in hours. ``8136`` is 339 days.
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
