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

Additional metrics can be opted into without changing the defaults:

.. code-block:: python

   from compresso_recsys.metrics import HitRate, MAP, MRR, Precision, Recall

   result = evaluate_recommender(
       model,
       source=split["test_source_matrix"],
       targets=split["test_target_matrix"],
       metrics=[
           Recall([20, 50, 100]),
           Precision([20, 50, 100]),
           HitRate([20, 50, 100]),
           MRR([20, 50, 100]),
           MAP([20, 50, 100]),
       ],
       batch_size=1024,
   )

For a model that already produced one complete :class:`compresso.SRPTensor`,
use :func:`compresso_recsys.evaluation.evaluate_ranked_predictions` instead.

Train and Evaluate ELSA
-----------------------

ELSA follows the same ``fit`` and evaluation interface as EASE. Set
``max_output`` to train against sampled negative candidates instead of the
entire item catalog:

.. code-block:: python

   from compresso_recsys.evaluation import evaluate_recommender
   from compresso_recsys.metrics import CalibratedRecall, NDCG
   from compresso_recsys.models import ELSAConfig, ELSATrainer

   model = ELSATrainer(
       ELSAConfig(
           latent_dim=1024,
           batch_size=1024,
           max_output=10_000,
           epochs=10,
           lr=0.1,
           device="cuda",
       )
   )
   model.fit(split["x_train"])

   result = evaluate_recommender(
       model,
       source=split["test_source_matrix"],
       targets=split["test_target_matrix"],
       metrics=[
           CalibratedRecall([20, 50]),
           NDCG(100),
       ],
       batch_size=1024,
       show_progress=True,
   )

To search for a lottery-ticket ELSA with 64 retained values per item, add a
nested compression configuration:

.. code-block:: python

   from compresso_recsys.models import (
       ELSACompressionConfig,
       ELSAConfig,
       ELSATrainer,
   )

   model = ELSATrainer(
       ELSAConfig(
           latent_dim=1024,
           batch_size=1024,
           max_output=10_000,
           epochs=10,
           lr=0.1,
           decay=True,
           device="cuda",
           compression=ELSACompressionConfig(
               k_target=64,
               num_stages=10,
               stability_window=5,
               change_threshold=0.01,
               mask_update_interval=10,
               max_epochs_per_stage=20,
               sparse_finetune_backend="dense",
               sparse_inference_backend="csr",
           ),
       )
   )
   model.fit(split["x_train"])

``epochs`` controls the final fixed-SRP fine-tuning, not mask search. A
mask-search stage advances when it stabilizes or reaches
``max_epochs_per_stage``; either transition rewinds and restarts its optimizer.
Use ``None`` for an unlimited stability search. The sampler continues its
random sequence across stages, so rewinds see new sampled negatives.
Because this example sets ``max_output``, both mask search and sparse
fine-tuning gather only the batch's candidate rows. Setting ``max_output=None``
scores the complete item catalog during training.
The default ``sparse_finetune_backend="dense"`` densifies those selected rows
and is normally faster. Set it to ``"coo"`` to keep the fixed factors sparse
through two differentiable sparse matrix multiplications, reducing
fine-tuning memory at the cost of speed.

After fitting, the normalized sparse item factors can be exported without
densifying:

.. code-block:: python

   item_factors = model.elsa.export_item_embeddings()
   payload = item_factors.to_dict()

Both recommenders provide batch and full-matrix prediction methods. Seen items
are excluded by default:

.. code-block:: python

   batch_predictions = model.predict_on_batch(
       split["test_source_matrix"][:1024],
       k=100,
   )

   all_predictions = model.predict(
       split["test_source_matrix"],
       k=100,
       batch_size=1024,
   )

Compressed ELSA uses the configured ``sparse_inference_backend`` for both
methods. ``"csr"`` is the memory-efficient default; ``"dense"`` caches one
full dense normalized factor matrix and can be faster when the retained
``k_target`` is relatively large. Override the backend for one prediction
without retraining:

.. code-block:: python

   dense_predictions = model.predict(
       split["test_source_matrix"],
       k=100,
       batch_size=1024,
       sparse_inference_backend="dense",
   )

To allow previously interacted items in a diagnostic ranking, pass
``exclude_seen=False`` to either method.
