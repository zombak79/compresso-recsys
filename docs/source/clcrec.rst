CLCRec for Item Cold Start
==========================

:class:`~compresso_recsys.models.CLCRecTrainer` implements CLCRec, *Contrastive
Learning for Cold-Start Recommendation* (Wei et al., ACM MM 2021), with the
paper's matrix-factorization encoder. It was implemented from the paper
(arXiv:2107.05315v3); no third-party code was consulted.

What it learns
--------------

Each user and each training item gets an ID embedding, and an MLP with one hidden
layer maps an item's feature vector into the same space. The loss is
``λ·L_RE + (1−λ)·L_UI + η‖Θ‖²`` (Eq. 13 of the paper):

- ``L_UI`` contrasts every observed user-item pair with ``n_negatives`` items
  the user has not interacted with, scoring ``z_u·v/τ``. With probability
  ``hybrid_prob`` each item vector ``v`` is the item's content representation
  instead of its ID embedding, which places content representations in the
  space the user embeddings score.
- ``L_RE`` contrasts an item's ID embedding with its own content representation
  against other items' content representations, scoring cosine similarity over
  ``τ``.

Warm candidates (with an ID embedding) are scored by that embedding and cold
candidates by their content representation, so one ranking mixes both.
Candidates registered after fitting with
:meth:`~compresso_recsys.models.CLCRecTrainer.update_candidates` are scored by
content immediately, without refitting.

Training on an item split
-------------------------

.. code-block:: python

   from compresso_recsys.evaluation import evaluate_recommender
   from compresso_recsys.metrics import CalibratedRecall, NDCG
   from compresso_recsys.models import CLCRecConfig, CLCRecTrainer

   # split: a loaded item cold-start split; features: one row per item,
   # aligned with the split's item_ids (cold items included).
   model = CLCRecTrainer(CLCRecConfig(epochs=100, device="cuda"))
   model.fit(
       split["x_train"],
       features,
       train_item_indices=split["warm_item_indices"],
       item_ids=split["item_ids"],
       user_ids=split["train_user_ids"],
   )
   result = evaluate_recommender(
       model,
       source=split["val_source_matrix"],
       targets=split["val_target_matrix"],
       sample_ids=split["val_eval_user_ids"],
       metrics=[CalibratedRecall(20), NDCG(100)],
   )

Pass the checkpoint's ``train_user_ids`` to ``fit`` and its evaluation user
IDs as ``sample_ids``: the evaluator forwards them to the model, which looks
up each row's learned user embedding. On item cold-start splits every
evaluation user is a training user. Rows whose user is unknown raise by
default; ``unknown_users="fold_in"`` instead fits an embedding for the history
with all item parameters frozen. Fold-in is an extension, not part of the
paper's protocol.

For several modalities, wrap the trainer in
:class:`~compresso_recsys.models.MMConcatWrapper`. Its ``missing`` and
``normalize`` options handle items without a modality and modalities of
different scale. See :doc:`multimodal-concat-wrapper`.

Choices where the paper is silent
---------------------------------

The paper reports the embedding size (64), Xavier initialization, Adam, the
learning-rate and regularization grids and early stopping on validation
recall@10 with patience 10. It does not report ``temperature``, the hidden
width and activation of the feature encoder, or the batch size. These are
configuration fields to select on validation. Further choices, all
configurable or documented on :class:`~compresso_recsys.models.CLCRecConfig`:

- The hybrid swap is drawn independently for every item slot (the positive and
  each negative).
- The R-E term reuses the U-I negatives, as the paper's Figure 1 draws one set
  for both. ``re_n_negatives`` draws a separate set instead.
- ``η‖Θ‖²`` penalizes the embedding rows a batch uses plus the encoder weights,
  spread evenly over an epoch.
- Only the matrix-factorization encoder is implemented, not the LightGCN one.

Numbers obtained with this class on compresso-recsys checkpoints are not
comparable with the paper's tables, which use different data and features.
See :doc:`citing` for the reference.
