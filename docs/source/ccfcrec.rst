CCFCRec for Item Cold Start
===========================

:class:`~compresso_recsys.models.CCFCRecTrainer` implements CCFCRec,
*Contrastive Collaborative Filtering for Cold-Start Item Recommendation* (Zhou,
Zhang and Yang, WWW 2023). It was implemented from the paper
(arXiv:2302.02151v2); no third-party code was consulted.

What it learns
--------------

CCFCRec learns two embeddings for every training item and one for every user:

- a content-based embedding ``q_v``: a two-layer LeakyReLU MLP over the item's
  content vector (Eq. 1);
- a co-occurrence embedding ``z_v``: the sum of learned columns over the users
  who interacted with the item (Eq. 2);
- a user embedding ``s_u``: the sum of learned columns over the items in the
  user's history (Eq. 3).

The loss is ``L_q + L_z + λ·L_c`` (Eq. 9):

- ``L_q`` and ``L_z`` are BPR losses (Eq. 7-8). For each observed
  ``(item, user)`` pair a user who did not interact with the item is sampled,
  and the item must score the observed user higher, once through ``q_v`` and
  once through ``z_v``. Both share ``s_u``.
- ``L_c`` (Eq. 4) contrasts ``q_v`` with the co-occurrence embeddings of items
  that some user interacted with together with ``v`` (positives) against items
  that never co-occur with it (negatives), with inner products over ``τ``.

The contrastive term teaches the content encoder what the co-occurrence
embeddings know, so content alone ranks well when an item has no interactions.

Users are not looked up by ID: ``s_u`` is computed from whatever history is
passed, so any user can be scored, including users unseen in training. Items in
a history that were not trained on contribute nothing.

Inference
---------

The paper scores every item by its content embedding, ``⟨q_v, s_u⟩``. That is
the default, ``inference="content"``. ``inference="hybrid"`` is an extension:
items with training interactions are scored by their co-occurrence embedding,
cold items by content. Candidates registered after fitting with
:meth:`~compresso_recsys.models.CCFCRecTrainer.update_candidates` are scored by
content immediately, without refitting.

``"hybrid"`` is not a better default for a catalog that mixes warm and cold
items. A co-occurrence embedding sums one learned column per interacting user,
so popular trained items get large scores that cold items, scored by content,
cannot match. On a local ML-1M item split, where every test target is a cold
item, hybrid scoring lowered full-catalog nDCG@100 from 0.101 to 0.037; cold-only
rankings are unaffected because they contain no trained items.

Training on an item split
-------------------------

.. code-block:: python

   from compresso_recsys.evaluation import evaluate_recommender
   from compresso_recsys.metrics import CalibratedRecall, NDCG
   from compresso_recsys.models import CCFCRecConfig, CCFCRecTrainer
   from compresso_recsys.training import EarlyStopping, Validation

   # split: a loaded item cold-start split; features: one row per item,
   # aligned with the split's item_ids (cold items included).
   model = CCFCRecTrainer(CCFCRecConfig(epochs=100, device="cuda"))
   model.fit(
       split["x_train"],
       features,
       train_item_indices=split["warm_item_indices"],
       item_ids=split["item_ids"],
       validation=Validation(
           split["val_source_matrix"], split["val_target_matrix"], [NDCG(100)]
       ),
       early_stopping=EarlyStopping("ndcg@100", patience=10),
   )
   result = evaluate_recommender(
       model,
       source=split["test_source_matrix"],
       targets=split["test_target_matrix"],
       metrics=[CalibratedRecall(20), NDCG(100)],
   )

For several modalities, wrap the trainer in
:class:`~compresso_recsys.models.MMConcatWrapper`. See
:doc:`multimodal-concat-wrapper`.

Choices and deviations
----------------------

The paper reports a batch size of 1024, ``τ = 0.1``, Adam, and per dataset the
embedding size, learning rate, ``λ`` and the numbers of contrastive positives
and negatives. The defaults of :class:`~compresso_recsys.models.CCFCRecConfig`
take those values; they are listed there. Not reported, and configurable:

- the hidden width of the content encoder (``hidden_dim``);
- the scale of the ``‖Θ‖`` term of Eq. 9, applied as Adam weight decay
  (``weight_decay``).

Deviation: the paper builds the content vector by concatenating per-attribute
embeddings and learns lookup tables for one-hot and multi-hot attributes such
as genre or director. Here the content vector is the supplied dense feature
vector; learnable attribute lookups are not implemented.
:class:`~compresso_recsys.models.MMConcatWrapper` concatenates several
modalities into that vector.

Further implementation choices:

- An item that co-occurs with no other item uses itself as its contrastive
  positive; the count is kept by the sampler.
- Contrastive negatives are drawn exactly and uniformly from the items that
  never co-occur with ``v``, not by rejection: on dense data a popular item
  co-occurs with most of the catalog, and redraws would keep landing on
  co-occurring items. An item that co-occurs with every other item has no
  negatives and gets uniform items instead.
- Negative users for the BPR terms are drawn by rejection. One that still
  collides after ``max_negative_attempts`` redraws is kept. Both fallbacks are
  counted per epoch in :attr:`~compresso_recsys.models.CCFCRecTrainer.history`.
- The item-item co-occurrence matrix of the training items is held in memory
  during fitting, which is quadratic in the number of training items in the
  worst case.

Numbers obtained with this class on compresso-recsys checkpoints are not
comparable with the paper's tables, which use different data, features and an
item-centric evaluation. See :doc:`citing` for the reference.
