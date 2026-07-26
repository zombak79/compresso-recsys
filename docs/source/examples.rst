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
       include_image_urls=True,
       min_user_support=10,
       item_min_support=10,
       min_value_to_keep=1.0,
       set_all_values_to=1.0,
       min_source_items=1,
       min_target_items=1,
       annotation_source="none",
   )

The same configuration can be executed from the command line:

.. code-block:: bash

   compresso-recsys-build-checkpoint \
     --dataset amazon2023 \
     --amazon_category Toys_and_Games \
     --checkpoint_path artifacts/amazon_toys/item_split_exp001.zip \
     --split_mode item_split \
     --metadata_text_fields title,features,description,categories \
     --min_entity_text_words 20 \
     --min_user_support 10 \
     --item_min_support 10 \
     --min_value_to_keep 1.0 \
     --set_all_values_to 1.0 \
     --min_source_items 1 \
     --min_target_items 1 \
     --annotation_source none

Checkpoint Read/Write
---------------------

.. code-block:: python

   import compresso_recsys as cr

   checkpoint_path = "artifacts/ml1m/exp001.zip"

   with cr.read_checkpoint(checkpoint_path) as root:
       split = cr.load_recsys_split(root)
       print(split["x_train"].shape)

Evaluate Embeddings From Python
-------------------------------

The fixed holdouts are plain arrays of item indices, so you can evaluate a
manually computed embedding matrix directly:

.. code-block:: python

   import numpy as np
   import compresso_recsys as cr
   from compresso_recsys.retrieval import evaluate_item_embeddings_with_holdout

   with cr.read_checkpoint("artifacts/amazon_toys/item_split_exp001.zip") as root:
       split = cr.load_recsys_split(root)

   rng = np.random.default_rng(0)
   item_embeddings = rng.normal(size=(len(split["item_ids"]), 64)).astype("float32")

   metrics_100 = evaluate_item_embeddings_with_holdout(
       item_embeddings=item_embeddings,
       source_indices=split["test_source_indices"],
       target_indices=split["test_target_indices"],
       k=100,
       score_batch_size=1024,
       show_progress=True,
   )

   print(metrics_100)

Evaluate Model Predictions
--------------------------

The generic recommender evaluator requests one prediction batch at a time and
immediately compares it with the corresponding CSR target rows. This keeps
memory bounded during evaluation:

.. code-block:: python

   import compresso_recsys as cr
   from compresso_recsys.evaluation import evaluate_recommender
   from compresso_recsys.metrics import CalibratedRecall, NDCG
   from compresso_recsys.models import EASE, EASEConfig

   with cr.read_checkpoint("artifacts/ml20m-elsa.zip") as root:
       split = cr.load_recsys_split(root)

   model = EASE(EASEConfig(l2=500.0))
   model.fit(split["x_train"])

   result = evaluate_recommender(
       model,
       source=split["test_source_matrix"],
       targets=split["test_target_matrix"],
       metrics=[
           CalibratedRecall([20, 50, 100]),
           NDCG([20, 50, 100]),
       ],
       batch_size=1024,
       show_progress=True,
   )

   print(result)

For a model that already produced one complete :class:`compresso.SRPTensor`,
use :func:`compresso_recsys.evaluation.evaluate_ranked_predictions` instead.
