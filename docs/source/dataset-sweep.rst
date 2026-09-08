Dataset statistics and baseline sweep
=====================================

Download :download:`dataset_sweep.py <../../examples/validation/dataset_sweep.py>`
or run it from the repository root after installing the package. The first pass
defaults to **statistics plus popularity**, without KNN or embedding downloads.
It covers every registered dataset and split mode; unsupported combinations are
recorded explicitly. Amazon runs **per category**, with ``Toys_and_Games`` as the
only default. Selected categories are never combined into a full-Amazon dataset.

Parallel server run
--------------------

On a 64-core server, this command requests ten dataset processes with six
numerical-library threads each. Choose fewer workers if RAM is limited:

.. code-block:: bash

   nohup python -u examples/validation/dataset_sweep.py \
     --workers 10 --threads-per-worker 6 \
     --output artifacts/dataset-sweep \
     > dataset-sweep.log 2>&1 &

Workers own different datasets or Amazon categories and run their splits
sequentially. With the ten current datasets and one Amazon category, at most ten
workers run; additional selected Amazon categories create additional independent
jobs. ``--workers`` caps simultaneous jobs, not the total number of jobs. This
avoids concurrent downloads and cache writes for the same source.
Spawned processes inherit BLAS/OpenMP thread limits;
PyTorch and Arrow thread pools are also capped. CPU limits are not memory limits:
large datasets can still require substantial RAM, disk space, and download time.
KNN, if enabled, has a separate ``--knn-jobs`` setting (default 1).

Do not run multiple sweeps against the same data/output directories concurrently.
For manually scheduled server jobs, use disjoint dataset/category lists and
separate output directories. This script's built-in worker pool handles that
coordination within one run.

Amazon subsets
---------------

Select one or more categories explicitly. Each has its own worker job, source
cache, checkpoints, statistics, and popularity evaluation:

.. code-block:: bash

   python -u examples/validation/dataset_sweep.py \
     --datasets amazon2023 \
     --amazon-categories Toys_and_Games Office_Products All_Beauty \
     --workers 3 --threads-per-worker 2 \
     --output artifacts/amazon-sweep

Omit ``--datasets amazon2023`` to include the other datasets in the same sweep.
The existing ``--amazon-category Toys_and_Games`` spelling still works.
Logs and tables show labels such as ``amazon2023[Toys_and_Games]``; JSON records
include ``amazon_category``. Duplicate categories (including recognized aliases)
are rejected to prevent two workers writing the same cache. There is no implicit
"all Amazon" expansion or cross-category concatenation. Individual categories
can still be large; reduce ``--workers`` to limit simultaneous memory use.

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
* Separate observed-item and full-catalog sparsities, with explicit denominators.
* Cold-candidate counts, cold-target fraction, source/target overlap, and sequence
  event counts when available.
* Text/image-URL coverage, checkpoint size/hash, settings, provenance, and timings.

For interaction tables, sparsity is ``1 - unique_pairs / (users * items)``.
Matrix statistics distinguish:

* **Catalog columns** (JSON ``n_items``): the matrix width, including zero-only
  columns for held-out or otherwise unobserved items.
* **Observed items** (JSON ``active_items``): columns with at least one interaction
  in that particular training, source, or target matrix. An assigned split item
  with no retained interactions is not counted as observed.
* **Observed-item sparsity** (JSON ``observed_item_sparsity``):
  ``1 - nnz / (n_rows * active_items)``.
* **Catalog sparsity** (JSON ``catalog_sparsity``):
  ``1 - nnz / (n_rows * n_items)``. The existing JSON ``sparsity`` field retains
  this meaning for compatibility.

Both matrix sparsities keep every row, including empty histories and repeated
evaluation users; they do not substitute unique-user or active-row counts.
Zero-sized denominators produce JSON null and a dash in the report. An all-zero
matrix with nonzero dimensions has catalog sparsity 100%, but undefined
observed-item sparsity. Candidate catalog counts are before per-user seen-item
exclusion and do not imply that every candidate occurs in the targets.

For example, the default seed-42 Goodbooks item split has 10,000 catalog columns,
but 8,500 observed training items, 500 observed validation target items, and 1,000
observed test target items. The report shows those counts separately. Its 100%
cold target pairs are expected: every held-out target item is absent from training,
while users can occur in both training and evaluation.

Ratings and play-count sums are not interaction counts. Loaded data has already passed the adapter's metadata
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
  ``checkpoint.zip`` for successful builds. Amazon paths include the category:
  ``amazon2023-<category>-<split>-<fingerprint>/``.

Detailed records are saved after each combination; combined summaries update
as dataset/category workers finish. Fingerprints include settings and code provenance.
``--resume`` skips complete runs and retries failed ones, reusing only checkpoints
whose recorded hashes match. Without it, existing runs are not overwritten.
Do not edit the generated report files by hand if you intend to resume into the
same directory. Keep the log and per-run JSON files if a worker is interrupted
or runs out of memory.

To correct tables from an existing sweep without downloading data, rebuilding
checkpoints, or rerunning baselines:

.. code-block:: bash

   python examples/validation/dataset_sweep.py \
     --report-only --output artifacts/dataset-sweep

This regenerates only ``summary.md`` from every record in that directory's
``results.jsonl``, including records produced by older script versions. It derives
the sparsities from saved row, observed-item, catalog, and pair counts. It leaves
JSON files, checkpoints, scores, and provenance unchanged. No dataset/split
selection or evaluation options are applied in this mode. A missing
``results.jsonl`` is an error; report-only does not start a new sweep or discover
additional results from unfinished workers.

Updating the script or package source changes the fingerprint, so ``--resume``
does not reuse results from an older code fingerprint. A copied script uses the
imported package's Git provenance when available; wheel installations without
tracked source record a null commit without printing a Git error. Package version
and script hash are recorded in either case.

Build/model errors are recorded and other work continues; a failure produces a
nonzero final exit code. Unsupported timestamp combinations and non-DBbook
``official`` modes are marked explicitly, without downloads. Empty temporal
windows are failures, not a reason to silently adjust the protocol. Use
``--temporal-period-hours`` when your chosen window needs changing.

Dataset-specific overrides
----------------------------

``--amazon-categories`` selects independent subsets. For other dataset-specific
builder changes, pass ``--builder-overrides settings.json``, for example:

.. code-block:: json

   {
     "ml1m": {
       "val_users": 500,
       "test_users": 1000,
       "min_user_support": 5,
       "item_min_support": 5
     }
   }

Dataset identity, Amazon category, split mode, paths, and embedding downloads
cannot be changed through this file. Amazon overrides apply to each selected
category. All effective settings and the checkpoint manifest are saved
with the statistics. These are descriptive benchmarks, not claims of matching
the paper targets in :doc:`dataset-validation`.
