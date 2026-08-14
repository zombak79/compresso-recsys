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

Evaluation Results
------------------

Both entry points return an
:class:`~compresso_recsys.evaluation.EvaluationResult` rather than a plain
dictionary. It behaves as a mapping over the aggregate metrics plus
``n_eval_users``, so ``result["ndcg@20"]``, iteration and ``dict(result)``
continue to work, and :meth:`~compresso_recsys.evaluation.EvaluationResult.to_dict`
is available where an actual ``dict`` is required.

Beyond the aggregates it carries the **per-user value behind every metric**,
together with the ``sample_ids`` that identify the rows those values came from:

.. code-block:: python

   result = evaluate_recommender(
       model, source=source, targets=targets,
       metrics=[NDCG(20)], sample_ids=user_ids,
   )

   result["ndcg@20"]            # aggregate, as before
   result.per_user["ndcg@20"]   # one float32 per evaluable row
   result.sample_ids            # aligned identifiers
   result.n_eval_users          # rows with at least one target
   result.n_rows                # rows supplied

Those per-user values are what make paired statistical comparison possible; see
:doc:`../statistical-comparison`. Retaining them costs roughly
``4 * n_users * n_metrics`` bytes, small beside model parameters, so collection
is enabled by default. Pass ``collect_per_user=False`` for deployment-style
monitoring that only needs aggregates, at the cost of being unable to compare
the result against another.

Identifiers default to global input row indices. Supply ``sample_ids`` when the
rows have stable identities of their own, so that comparison across evaluations
fails loudly rather than silently pairing different users.

.. autoclass:: compresso_recsys.evaluation.EvaluationResult
   :members:

Custom Metrics
--------------

:meth:`compresso_recsys.metrics.RankingMetric.update` returns the per-row values
it computed, with shape ``(rows, len(result_keys))``, rather than only folding
them into a running sum. Without that the evaluator would have to compute every
metric twice to retain per-user observations.

Third-party metric implementations must therefore return their values. When
collection is enabled the evaluator validates the returned tensor: two
dimensions, one row per prediction row, one column per result key,
floating-point dtype, and finite values for rows with targets.

Values must be produced for every row, including rows with no targets. Those
rows are excluded from the aggregate but must still occupy a position so that
values stay aligned with ``sample_ids`` before filtering.

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
