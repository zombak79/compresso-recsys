Ranking Evaluation API
======================

Predictions and Targets
-----------------------

Ranked predictions use :class:`compresso.SRPTensor`: each row stores exactly
the top ``K`` item indices and their scores in descending score order. Targets
use a SciPy CSR matrix so each row can contain a different number of relevant
items. Target values are treated as binary relevance; only nonzero locations
matter.

The evaluator creates one boolean hit tensor of shape ``(batch_size, K)`` and
shares it across all configured metrics. It does not densify model scores or
the complete target matrix. Rows without target items are excluded from metric
means and from ``n_eval_users``.

.. autofunction:: compresso_recsys.evaluation.evaluate_recommender

.. autofunction:: compresso_recsys.evaluation.evaluate_ranked_predictions

.. autoclass:: compresso_recsys.evaluation.RankingEvaluator
   :members:

Metrics
-------

Both built-in metrics accept one cutoff or a sequence of cutoffs. For
compatibility with the embedding evaluator, ``CalibratedRecall`` is reported
as ``recall@K`` and divides hits by ``min(K, number_of_targets)``.

.. autoclass:: compresso_recsys.metrics.RankingMetric
   :members:

.. autoclass:: compresso_recsys.metrics.RankingBatch
   :members:

.. autoclass:: compresso_recsys.metrics.CalibratedRecall
   :members:

.. autoclass:: compresso_recsys.metrics.NDCG
   :members:
