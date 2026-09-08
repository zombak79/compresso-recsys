Dataset statistics and baseline sweep
=====================================

Download :download:`dataset_sweep.py <../../examples/validation/dataset_sweep.py>`
or run it from the repository root after installing the package. The first pass
defaults to **statistics plus popularity**, without KNN or embedding downloads.
It covers every registered dataset and split mode; unsupported combinations are
recorded explicitly. Amazon uses one category (``Toys_and_Games`` by default),
not every Amazon category.

Parallel server run
--------------------

On a 64-core server, this command requests ten dataset processes with six
numerical-library threads each. Choose fewer workers if RAM is limited:

.. code-block:: bash

   nohup python -u examples/validation/dataset_sweep.py \
     --workers 10 --threads-per-worker 6 \
     --output artifacts/dataset-sweep \
     > dataset-sweep.log 2>&1 &

Workers own different datasets and run their splits sequentially. With the ten
current datasets, at most ten workers run; ``--workers 20`` will not create twenty
simultaneous jobs for those same sources. This avoids concurrent downloads and
cache writes for one dataset. Spawned processes inherit BLAS/OpenMP thread limits;
PyTorch and Arrow thread pools are also capped. CPU limits are not memory limits:
large datasets can still require substantial RAM, disk space, and download time.
KNN, if enabled, has a separate ``--knn-jobs`` setting (default 1).

Do not run multiple sweeps against the same data/output directories concurrently.
For manually scheduled server jobs, use disjoint dataset lists and separate
output directories. This script's built-in worker pool handles that coordination
within one run.

Useful variants
----------------

.. code-block:: bash

   # Start small; evaluation sampling does not limit downloads or preprocessing.
   python examples/validation/dataset_sweep.py --datasets lfm2k --splits user_split item_split --max-eval-users 200 --output artifacts/sweep-first

   # Statistics only: the empty --baselines list disables all fitting/evaluation.
   python examples/validation/dataset_sweep.py --workers 10 --threads-per-worker 6 --baselines --output artifacts/stats-only

   # Resume the original run with the same settings.
   python examples/validation/dataset_sweep.py --workers 10 --threads-per-worker 6 --output artifacts/dataset-sweep --resume

KNN is optional. Install the ``knn`` extra (``scikit-learn``) and pass
``--baselines popularity itemknn`` when ready. It uses 100 item-item cosine
neighbors by default. Exact neighbor search can be expensive; catalogs above
30,000 items are skipped unless ``--knn-max-items`` is raised deliberately.
Changing baseline settings creates a new run fingerprint, rather than reusing
the old scores; it also gets a separate checkpoint build in this version.

Collected statistics
---------------------

* Loaded and pre-split users, items, event rows, unique user-item pairs, repeat
  event fraction, sparsity, timestamp coverage, and timestamp range in source units.
* User/item activity distributions: min, mean, median, p90, p99, and max.
* Training matrix counts, active versus catalog items, and source/target statistics
  for every stage, including unique users versus repeated evaluation rows.
* Cold-candidate counts, cold-target fraction, source/target overlap, and sequence
  event counts when available.
* Text/image-URL coverage, checkpoint size/hash, settings, provenance, and timings.

For interaction tables, sparsity is ``1 - unique_pairs / (users * items)``;
for matrices, it is ``1 - nnz / (rows * columns)``. Ratings and play-count sums
are not interaction counts. Loaded data has already passed the adapter's metadata
filtering; it is not an untouched-source-file count. Defaults disable minimum
text length, use one evaluation draw and seed 42, and otherwise use each dataset's
builder rating/support/held-out user defaults. No annotations or embeddings are
downloaded by the sweep.

Temporal support filtering happens per stage: its pre-split counts are not the
final filtered total. For DBbook's official mode, pre-split counts describe supplied
training data only. Source histories overlap across stages and evaluation draws
can repeat users, so never sum those matrix counts as a whole-dataset total.

Evaluation protocol
--------------------

Baselines fit the checkpoint's binary ``x_train`` once. There is no tuning,
validation/test training data, or automatic test-time refit. Metrics are Recall,
CalibratedRecall, NDCG, and HitRate at 10 and 20; use ``--cutoffs`` to change them.
Seen items are excluded and the full phase-specific candidate catalog is used.
Item IDs are aligned across temporal vocabularies; future stages' candidates
are not offered in earlier stages.

Popularity and collaborative KNN give cold items zero learned signal. Their
cold-start scores can reflect ties, not a content-based cold-start capability.
The report includes cold-target coverage to make this visible. Rows with no
targets or fewer unseen candidates than the largest requested cutoff are excluded
and counted, without silently changing the cutoff. Seeded ``--max-eval-users``
sampling is shared across baselines and keeps all draws of a selected user together.
Scores from different split protocols are not directly interchangeable.

Outputs and recovery
--------------------

* ``summary.md`` contains readable dataset, training, evaluation-stage and
  baseline tables, with unsupported combinations and errors.
* ``results.jsonl`` contains the full records from this invocation.
* Each ``<dataset>-<split>-<fingerprint>/`` contains ``result.json`` and a
  ``checkpoint.zip`` for successful builds.

Detailed records are saved after each combination; combined summaries update
as dataset workers finish. Fingerprints include settings and code provenance.
``--resume`` skips complete runs and retries failed ones, reusing only checkpoints
whose recorded hashes match. Without it, existing runs are not overwritten.
Do not edit the generated report files by hand if you intend to resume into the
same directory. Keep the log and per-run JSON files if a worker is interrupted
or runs out of memory.

Build/model errors are recorded and other work continues; a failure produces a
nonzero final exit code. Unsupported timestamp combinations and non-DBbook
``official`` modes are marked explicitly, without downloads. Empty temporal
windows are failures, not a reason to silently adjust the protocol. Use
``--temporal-period-hours`` when your chosen window needs changing.

Dataset-specific overrides
----------------------------

``--amazon-category`` selects a category. For other dataset-specific builder
changes, pass ``--builder-overrides settings.json``, for example:

.. code-block:: json

   {
     "ml1m": {
       "val_users": 500,
       "test_users": 1000,
       "min_user_support": 5,
       "item_min_support": 5
     }
   }

Dataset identity, split mode, paths, and embedding downloads cannot be changed
through this file. All effective settings and the checkpoint manifest are saved
with the statistics. These are descriptive benchmarks, not claims of matching
the paper targets in :doc:`dataset-validation`.
