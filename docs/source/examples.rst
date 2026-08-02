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

Amazon temporal checkpoint with three five-day target windows:

.. code-block:: python

   checkpoint_path = cr.build_recsys_checkpoint(
       dataset="amazon2023",
       amazon_category="Pet_Supplies",
       checkpoint_path="artifacts/pets-temporal.zip",
       split_mode="temporal",
       temporal_period_hours=24 * 5,
       metadata_text_fields=["title", "features", "description", "categories"],
       min_entity_text_words=20,
       min_user_support=10,
       item_min_support=10,
       min_value_to_keep=1.0,
       annotation_source="none",
   )

Temporal source and target matrices share columns within each stage. Catalogs
expand between stages, so always use the matching item-ID array:

.. code-block:: python

   with cr.read_checkpoint(checkpoint_path) as root:
       split = cr.load_recsys_split(root)

   assert split["val_source_matrix"].shape == split["val_target_matrix"].shape
   assert split["val_source_matrix"].shape[1] == len(split["val_item_ids"])
   assert split["test_source_matrix"].shape[1] == len(split["test_item_ids"])

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

Train TEASER for Cold Items
---------------------------

TEASER consumes the checkpoint's item-feature matrix directly. With an
``item_split`` checkpoint, fit only encoder rows for warm items while retaining
all item features in the fixed decoder:

.. code-block:: python

   from compresso_recsys.evaluation import evaluate_recommender
   from compresso_recsys.metrics import CalibratedRecall, NDCG
   from compresso_recsys.models import TEASER, TEASERConfig

   model = TEASER(
       TEASERConfig(
           l2_coefficients=0.05,
           l2_encoder=0.05,
           rho=0.05,
           max_iterations=10,
       )
   )
   model.fit(
       split["x_train"],
       item_features=split["entity_tag_matrix"],
       train_item_indices=split["train_item_indices"],
       item_ids=split["item_ids"],
       metadata=split["entity_metadata"],
       feature_names=split["tag_names"],
       show_progress=True,
   )

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

``item_features`` may instead be a dense NumPy or PyTorch embedding matrix, a
SciPy CSR matrix, or an :class:`compresso.SRPTensor`. Its rows must follow
``split["item_ids"]``. Interactions must be binary implicit feedback. The
original ADMM solver is intended as a reproducible baseline and can require
substantial memory for large warm-item catalogs.

Train LEMSA with Language Embeddings
------------------------------------

LEMSA learns warm-item encoder directions in the same space as fixed language
embeddings. For an item cold-start split, train encoder rows only for warm
items while retaining every embedding in the initial candidate catalog:

.. code-block:: python

   from compresso_recsys.evaluation import evaluate_recommender
   from compresso_recsys.metrics import CalibratedRecall, MRR, NDCG
   from compresso_recsys.models import LEMSA, LEMSAConfig

   model = LEMSA(
       LEMSAConfig(
           l2_encoder=0.05,
           epochs=10,
           solver="eigen",
           update_batch_size=1,
           update_rate=0.1,
           shuffle_updates=True,
           seed=0,
           dtype="float32",
       )
   )
   model.fit(
       split["x_train"],
       item_features=item_embeddings,
       train_item_indices=split["train_item_indices"],
       item_ids=split["item_ids"],
       metadata=split["entity_metadata"],
       feature_space_id="Qwen/Qwen3-Embedding-0.6B",
       show_progress=True,
   )

   result = evaluate_recommender(
       model,
       source=split["test_source_matrix"],
       targets=split["test_target_matrix"],
       metrics=[
           CalibratedRecall([20, 50]),
           NDCG(100),
           MRR([20, 50, 100]),
       ],
       batch_size=1024,
       show_progress=True,
   )

``update_rate`` moves partway toward each exact closed-form row solution and
``shuffle_updates`` removes persistent catalog-order bias between epochs. Use
``update_rate=1.0`` and ``shuffle_updates=False`` for the original full-update,
fixed-order behavior. Smaller update rates generally require more epochs and
should be selected together with ``l2_encoder`` on validation data.

``solver="eigen"`` performs exact rank-one row solves. Rows are computed in
snapshot-based blocks controlled by ``update_batch_size`` and committed only
after the whole block has been solved. Use ``1`` for sequential Gauss-Seidel
updates or ``None`` for a full Jacobi sweep. The appendix does not report its
batch size, so validation comparisons across several values are appropriate
for strict replication. ``solver="direct"`` solves the same systems densely
and is mainly useful for reproducing the equations on small datasets. New item
embeddings can subsequently be published through
``update_candidates`` or a complete catalog can be replaced with
``build_candidates`` without fitting new encoder rows.

Train LEMSAGD with Leave-One-Out Targets
----------------------------------------

LEMSAGD leaves ``E @ S.T`` unconstrained and prevents identity training by
predicting interactions that have been removed from their source histories:

.. code-block:: python

   from compresso_recsys.models import LEMSAGDConfig, LEMSAGDTrainer

   model = LEMSAGDTrainer(
       LEMSAGDConfig(
           device="cuda",
           batch_size=1024,
           max_output=10_000,
           epochs=10,
           lr=1e-3,
           decay=True,
           training_mode="leave_one_out",
           loo_batch_order="round_robin",
           use_relu=True,
           encoder_init="xavier",
           normalize_encoder=False,
           l2_encoder=0.0,
       )
   )
   model.fit(
       split["x_train"],
       item_features=item_embeddings,
       train_item_indices=split["train_item_indices"],
       item_ids=split["item_ids"],
       metadata=split["entity_metadata"],
       feature_space_id="Qwen/Qwen3-Embedding-0.6B",
   )

   result = evaluate_recommender(
       model,
       source=split["test_source_matrix"],
       targets=split["test_target_matrix"],
       metrics=[CalibratedRecall([20, 50]), NDCG(100)],
       batch_size=1024,
       show_progress=True,
   )

The virtual sampler visits every eligible observed interaction exactly once per
epoch. For each example it removes that interaction from its original CSR user
row and predicts it from the remaining history. It stores no expanded source
or target matrix. Round-robin ordering contributes at most one example per user
to each batch; target order within users and user order within rounds are
reshuffled every epoch. Source entries are excluded from the loss rather than
treated as negative labels. Users with fewer than two warm interactions do not
contribute training examples. Set ``loo_batch_order="grouped"`` to keep each
user's virtual examples contiguous instead.

Set ``training_mode="symmetric"`` to use complementary random history views
instead; ``split_probability`` controls that mode's split. Leave-one-out has
approximately ``x_train.nnz`` examples per epoch, so it is intentionally much
more expensive than symmetric user-level batches. This model is experimental,
not an exact reproduction of LEMSA's ALS optimizer.

Train TEASER with Gradient Descent
----------------------------------

Use TEASERGD when the ADMM solver's dense warm-item Gram matrix is too costly.
The production catalog API is the same, while training supports ELSA-style
output candidate sampling:

.. code-block:: python

   from compresso_recsys.models import TEASERGDConfig, TEASERGDTrainer

   model = TEASERGDTrainer(
       TEASERGDConfig(
           device="cuda",
           batch_size=1024,
           max_output=10_000,
           epochs=10,
           lr=1e-3,
           decay=True,
           loss="normalized_mse",
           use_relu=False,
           encoder_init="features",
           normalize_encoder=True,
           diagonal_scale=1.0,
           l2_encoder=0.0,
           coefficient_regularization_samples=4096,
       )
   )
   model.fit(
       split["x_train"],
       item_features=item_embeddings,
       train_item_indices=split["train_item_indices"],
       item_ids=split["item_ids"],
       metadata=split["entity_metadata"],
       feature_space_id="Qwen/Qwen3-Embedding-0.6B",
   )

   result = evaluate_recommender(
       model,
       source=split["test_source_matrix"],
       targets=split["test_target_matrix"],
       metrics=[CalibratedRecall([20, 50]), NDCG(100)],
       batch_size=1024,
       show_progress=True,
   )

``max_output=None`` trains against every warm item. With a finite value, every
positive item represented in the current user batch is retained and sampled
negatives fill the remaining output budget. The source-prefix rule is what
makes exact diagonal removal possible in the sampled output. TEASER loss mode
importance-weights sampled negative errors. Use ``loss="normalized_mse"`` for
the ELSA-style row-normalized objective instead. Set ``diagonal_scale=0.0`` to
disable self-coefficient subtraction or use a value between zero and one for a
partial correction; ``1.0`` preserves standard TEASER behavior.

Serve New TEASER Candidates
---------------------------

Stable item IDs separate the fixed source vocabulary from the mutable output
catalog for both TEASER solvers. Fit a production-style model only on warm source items, then publish
the complete initial candidate catalog. ``feature_space_id`` is an optional
caller-provided label used to reject explicitly incompatible updates:

.. code-block:: python

   train_items = split["train_item_indices"]

   model.fit(
       interactions=split["x_train"][:, train_items],
       item_features=item_embeddings[train_items],
       item_ids=split["item_ids"][train_items],
       metadata=split["entity_metadata"].iloc[train_items].reset_index(drop=True),
       feature_space_id="Qwen/Qwen3-Embedding-0.6B@revision",
   )

   model.build_candidates(
       item_ids=split["item_ids"],
       item_features=item_embeddings,
       metadata=split["entity_metadata"],
       feature_space_id="Qwen/Qwen3-Embedding-0.6B@revision",
   )

Checkpoint source matrices still use the complete checkpoint item space. Align
one to the fitted warm source vocabulary before prediction or evaluation:

.. code-block:: python

   result = evaluate_recommender(
       model,
       source=model.align_source(
           split["test_source_matrix"],
           item_ids=split["item_ids"],
       ),
       targets=split["test_target_matrix"],
       metrics=[
           CalibratedRecall([20, 50]),
           NDCG(100),
       ],
       batch_size=1024,
       show_progress=True,
   )

Register new decoder-only candidates without retraining:

.. code-block:: python

   catalog = model.update_candidates(
       item_ids=new_item_ids,
       item_features=new_item_embeddings,
       metadata=new_metadata,
       on_conflict="error",
       feature_space_id="Qwen/Qwen3-Embedding-0.6B@revision",
   )

Use :meth:`~compresso_recsys.models.TEASER.build_candidates` when publishing a
complete catalog snapshot is simpler than incremental changes. Both operations
validate everything before atomically swapping the catalog.

Prediction can score the complete catalog or a request-specific allowlist. An
allowlist contains registered IDs; it does not rebuild or copy the catalog:

.. code-block:: python

   catalog = model.candidates
   predictions = model.predict(
       model.align_source(source, item_ids=source_item_ids),
       k=100,
       candidate_ids=eligible_item_ids,
   )
   recommended_item_ids = catalog.ids_for(predictions.cols)

The prediction columns remain global rows in ``catalog`` rather than positions
inside ``eligible_item_ids``. Resolve them against the same catalog snapshot.
New items cannot be present in ``source`` until a retrained encoder gives them
source rows.

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
and uses dense matrix multiplication. Set it to ``"coo"`` to keep the fixed
factors sparse through two differentiable sparse matrix multiplications. COO
uses less fine-tuning memory and can also be faster for very small
``k_target`` values; dense operations tend to become more competitive as the
ticket grows. The crossover depends on the device, batch, and candidate sizes.

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
