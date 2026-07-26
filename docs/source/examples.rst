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

A collaborative-filtering model can be evaluated directly after it produces
ranked scores for the source interactions. Mask source items before selecting
the top recommendations so already-seen items cannot be counted:

.. code-block:: python

   import torch
   import compresso_recsys as cr
   from compresso import SRPTensor
   from compresso_recsys.evaluation import evaluate_ranked_predictions
   from compresso_recsys.metrics import CalibratedRecall, NDCG

   with cr.read_checkpoint("artifacts/ml20m-elsa.zip") as root:
       split = cr.load_recsys_split(root)

   source = split["test_source_matrix"]
   targets = split["test_target_matrix"]

   # Replace this with batched scores from your model.
   scores = model(source)
   source_rows, source_cols = source.nonzero()
   source_rows = torch.from_numpy(source_rows).to(scores.device)
   source_cols = torch.from_numpy(source_cols).to(scores.device)
   scores[source_rows, source_cols] = -torch.inf

   values, columns = torch.topk(scores, k=100, dim=1, sorted=True)
   predictions = SRPTensor(
       cols=columns,
       vals=values,
       shape=(scores.shape[0], scores.shape[1]),
   )

   result = evaluate_ranked_predictions(
       predictions=predictions,
       targets=targets,
       metrics=[
           CalibratedRecall([20, 50, 100]),
           NDCG([20, 50, 100]),
       ],
       batch_size=4096,
   )

   print(result)

For large evaluations, use :class:`compresso_recsys.evaluation.RankingEvaluator`
directly and call ``update`` for each prediction batch. This avoids retaining
all user scores or predictions in memory.
