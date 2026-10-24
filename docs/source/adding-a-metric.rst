Adding a Metric
===============

A metric in compresso-recsys reads one batch of ranked predictions at a time
and returns a value per row. The evaluator streams batches through it, averages
the values over rows with at least one target, and keeps the per-row values.
Paired statistical comparison (:doc:`statistical-comparison`) runs on those
per-row values. A new metric therefore works everywhere a built-in one does —
``evaluate_recommender``, ``evaluate_ranked_predictions``, ``compare_models`` —
as soon as it follows the contract below. There is no registry: a metric is
used by passing an instance, and its result keys are its names.

The contract
------------

Most ranking metrics are a mean over users of a value computed at one or more
cutoffs. For those, subclass :class:`~compresso_recsys.metrics.MeanAtCutoffsMetric`
and implement one method:

- ``result_prefix``: the name; results are reported as ``<prefix>@<k>``.
- ``batch_values(batch)``: a float tensor of shape ``(rows, len(self.cutoffs))``.
  Column ``i`` is the row's value at ``self.cutoffs[i]``.

The base class handles the rest: normalising cutoffs (``MyMetric(10)`` or
``MyMetric([10, 20])``), ``required_k`` (the evaluator asks the model for that
many items), accumulation in float64, dropping rows without targets from the
mean, and returning per-row values to the evaluator.

The batch, a :class:`~compresso_recsys.metrics.RankingBatch`, gives you:

- ``batch.hits``: boolean ``(rows, ranks)``, True where the item at that rank
  is relevant.
- ``batch.target_counts``: the number of relevant items per row.
- ``batch.predictions``: the ranked predictions; ``cols`` holds the item
  columns and ``vals`` the scores.

Rows with no targets are excluded from the mean but still need a finite value,
so guard every denominator. Keep the computation on ``batch.hits.device``:
evaluation may run on a GPU.

Example: popularity of what is recommended
------------------------------------------

Accuracy metrics read only ``hits``. A beyond-accuracy metric can read the
recommended items themselves. This one reports the mean training popularity
percentile of the top-k items. 1.0 means the model recommends only the most
popular items; around 0.5 means it recommends the catalogue at large.

.. code-block:: python

   import numpy as np
   import torch

   from compresso_recsys.metrics import MeanAtCutoffsMetric, RankingBatch


   class PopularityPercentile(MeanAtCutoffsMetric):
       """Mean training-popularity percentile of the top-k recommended items."""

       result_prefix = "popularity_percentile"

       def __init__(self, cutoffs, *, item_counts):
           counts = np.asarray(item_counts, dtype=np.float64)
           # Percentile rank of each item's count: 0 for the least, 1 for the most popular.
           order = counts.argsort(kind="stable")
           percentile = np.empty_like(counts)
           percentile[order] = np.arange(counts.size) / max(counts.size - 1, 1)
           self.percentile = torch.from_numpy(percentile).to(torch.float32)
           super().__init__(cutoffs)

       def batch_values(self, batch: RankingBatch) -> torch.Tensor:
           cols = batch.predictions.cols[:, : self.required_k]
           percentile = self.percentile.to(cols.device)[cols]   # (rows, required_k)
           running_mean = percentile.cumsum(dim=1) / torch.arange(
               1, self.required_k + 1, device=cols.device, dtype=torch.float32
           )
           columns = torch.tensor([k - 1 for k in self.cutoffs], device=cols.device)
           return running_mean.index_select(dim=1, index=columns)

The constructor takes what the metric needs beyond the batch, here the
training item counts, and calls ``super().__init__(cutoffs)`` last, because
the base's ``__init__`` resets the accumulators.

Using it
--------

Pass an instance like any other metric. Its keys appear in the result, in the
per-user values, and in comparisons:

.. code-block:: python

   import compresso_recsys as cr

   counts = np.asarray(split["x_train"].sum(axis=0)).ravel()
   metrics = [cr.NDCG(20), PopularityPercentile([10, 20], item_counts=counts)]

   results = {
       name: cr.evaluate_recommender(
           model,
           source=split["test_source_matrix"],
           targets=split["test_target_matrix"],
           metrics=metrics,
           sample_ids=split["test_eval_user_ids"],
       )
       for name, model in {"EASE": ease, "Pop": popularity}.items()
   }
   results["EASE"].metrics["popularity_percentile@20"]
   report = cr.compare_models(results, metrics=["ndcg@20", "popularity_percentile@20"], reference="EASE")

Testing it
----------

Compare against a hand computation on a batch small enough to check by eye.
``evaluate_ranked_predictions`` takes fixed rankings, so no model is involved:

.. code-block:: python

   from compresso import SRPTensor
   from scipy.sparse import csr_matrix

   from compresso_recsys.evaluation import evaluate_ranked_predictions

   # Four items with training counts 1, 2, 3, 4: percentiles 0, 1/3, 2/3, 1.
   metric = PopularityPercentile([1, 2], item_counts=[1, 2, 3, 4])
   predictions = SRPTensor(
       cols=torch.tensor([[3, 2], [0, 1], [1, 3]]),
       vals=torch.tensor([[2.0, 1.0], [2.0, 1.0], [2.0, 1.0]]),
       shape=(3, 4),
   )
   targets = csr_matrix(np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 0]], dtype=np.float32))
   result = evaluate_ranked_predictions(predictions=predictions, targets=targets, metrics=[metric])

   # The third row has no target, so the mean is over the first two.
   assert np.isclose(result.metrics["popularity_percentile@1"], (1 + 0) / 2)
   assert np.isclose(result.metrics["popularity_percentile@2"], ((1 + 2 / 3) / 2 + (0 + 1 / 3) / 2) / 2)
   assert result.per_user["popularity_percentile@2"].shape == (2,)

Also check the edge cases the evaluator relies on: a row without targets yields
a finite value, cutoffs larger than one batch's ranks raise, and the metric
gives the same aggregate at any ``batch_size``.

Metrics that are not a mean at cutoffs
--------------------------------------

Implement :class:`~compresso_recsys.metrics.RankingMetric` directly:
``required_k``, ``result_keys``, ``reset``, ``update`` (accumulate and return the
per-row values) and ``compute``. A catalogue-level quantity such as aggregate
diversity has no per-row value. It can still return per-row values that sum to
it, but it is not a candidate for paired per-user comparison, and
``compare_models`` should not be asked to test it.
