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

Prediction validation is enabled by default for both evaluation entry points.
It checks item bounds, duplicate recommendations, NaN scores, and score order.

.. autofunction:: compresso_recsys.evaluation.evaluate_recommender

``evaluate_recommender`` requires matching source and target row counts, but
their column counts may differ. This supports a fixed history vocabulary with a
separately managed candidate catalog. Every prediction batch must use the same
column space as its corresponding target matrix.

.. autofunction:: compresso_recsys.evaluation.evaluate_ranked_predictions

.. autoclass:: compresso_recsys.evaluation.RankingEvaluator
   :members:

Metrics
-------

Every built-in metric accepts one cutoff or a sequence of cutoffs. Rows without
target items are excluded from all metric means.

Default Metrics
~~~~~~~~~~~~~~~

When metrics are not supplied to prediction or embedding evaluation, the
defaults remain:

* ``CalibratedRecall``, reported as ``calibrated_recall@K``. It divides hits by
  ``min(K, number_of_targets)``.
* ``NDCG``, reported as ``ndcg@K`` with binary relevance.

Optional Metrics
~~~~~~~~~~~~~~~~

The following metrics are available only when explicitly included in the
``metrics`` argument:

* ``Recall``, reported as ``recall@K``. It divides hits by the total number of
  target items, which is the usual definition and the one to compare against
  published numbers. It cannot exceed ``K / number_of_targets``, so users with
  many targets cap below one; ``calibrated_recall@K`` truncates the denominator
  instead and is greater than or equal to it for every user.
* ``Precision``, reported as ``precision@K``. It divides hits by ``K``.
* ``HitRate``, reported as ``hit_rate@K``. It is one when at least one target
  occurs in the top ``K``, otherwise zero.
* ``MRR``, reported as ``mrr@K``. It is the reciprocal rank of the first hit,
  or zero when no hit occurs by ``K``.
* ``MAP``, reported as ``map@K``. Average precision sums precision at each hit
  and divides by ``min(K, number_of_targets)``.

For example:

.. code-block:: python

   from compresso_recsys.metrics import HitRate, MAP, MRR, Precision, Recall

   optional_metrics = [
       Recall([20, 50, 100]),
       Precision([20, 50, 100]),
       HitRate([20, 50, 100]),
       MRR([20, 50, 100]),
       MAP([20, 50, 100]),
   ]

.. autoclass:: compresso_recsys.metrics.RankingMetric
   :members:

.. autoclass:: compresso_recsys.metrics.RankingBatch
   :members:

.. autoclass:: compresso_recsys.metrics.CalibratedRecall
   :members:

.. autoclass:: compresso_recsys.metrics.NDCG
   :members:

.. autoclass:: compresso_recsys.metrics.Recall
   :members:

.. autoclass:: compresso_recsys.metrics.Precision
   :members:

.. autoclass:: compresso_recsys.metrics.HitRate
   :members:

.. autoclass:: compresso_recsys.metrics.MRR
   :members:

.. autoclass:: compresso_recsys.metrics.MAP
   :members:
