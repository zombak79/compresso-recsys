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

Amazon preprocessing treats all valid ratings as binary interactions, with no
rating cutoff (``min_value_to_keep=None``, ``set_all_values_to=1.0``). User/item
support remains 5/1, with no minimum text length; user splits reserve 100
validation users and 200 test users. Support and split-eligibility filters still
apply, so retaining low-star ratings does not bypass those filters.
The previous iterative 20/20 filter can delete an entire category, and its
2,500/5,000 user holdouts do not fit smaller retained populations. These are
starting settings for exploratory runs, not paper-reproduction settings. There
is no automatic threshold relaxation. Tiny categories may still need smaller
holdouts, and user-split evaluation can retain fewer users after restricting
their histories to the training item catalog.

Update the installed package as well as the script to obtain the new Amazon
builder defaults. With an older package, the same settings can be supplied using
``--builder-overrides``:

.. code-block:: json

   {
     "amazon2023": {
       "min_value_to_keep": 0.0,
       "min_user_support": 5,
       "item_min_support": 1,
       "val_users": 100,
       "test_users": 200,
       "min_entity_text_words": 0
     }
   }

The zero threshold in this older-package workaround retains all valid Amazon
ratings (1–5); passing null to an older builder would restore its old dataset
default instead of disabling the cutoff. With the updated package, omitting the
threshold uses no rating filter. Explicit rating-threshold overrides still work.

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
Changing baseline settings creates a new evaluation fingerprint and recomputes
scores. Compatible checkpoints in the same output root are reused.

.. _dataset-default-table-reproduction:

Reproducing the documentation default tables
--------------------------------------------

The :ref:`tables in each dataset subsection <dataset-default-table-guide>` use
the **installed builder defaults**, not this sweep's exploratory seed/text/window
overrides. From the repository, the helper can measure available full caches and
render each non-Amazon dataset's table:

.. code-block:: bash

   python examples/validation/dataset_default_tables.py measure \
     --cache-root data --output artifacts/default-table-measurements.json

   # Explicitly permit missing source and official feature downloads.
   python examples/validation/dataset_default_tables.py measure \
     --datasets ml1m dbbook lfm2k --features \
     --cache-root data --download-root data \
    --output artifacts/default-feature-measurements.json

   python examples/validation/dataset_default_tables.py render \
     artifacts/default-table-measurements.json --dataset ml1m

Repeat ``--cache-root`` to search additional cache directories. MovieLens and
Goodbooks require their extracted interaction/metadata files and description
cache. Supply **full sources, never smoke-test samples**. Missing caches are
recorded as ``not_measured``; unsupported protocols are recorded separately.
Downloads require explicit ``--download-root``; measurement is otherwise offline.
``--features`` measures every supported encoder in the official SWAP JSON archive
against the union of measured item catalogs, including dimensions and availability
after Last.fm artist pooling. It verifies the release MD5 and records SHA-256;
it does not attach features to a checkpoint. No fitting or reusable checkpoint
archives are produced, although adapters may create canonical interaction caches.
The script uses the same split implementation as checkpoint construction and
records code/source fingerprints, including any existing canonical input caches.
Choose a new output path for each run; it refuses to overwrite an existing audit.
Each completed split is saved atomically as progress, and each completed dataset
is retained in the output. ``--datasets`` limits a worker to selected datasets;
use separate output files for parallel workers. Combine disjoint, completed runs
with ``dataset_default_tables.py merge worker-a.json worker-b.json --output combined.json``;
this retains each run's provenance and rejects duplicate datasets, incomplete
runs or parameters that differ from current installed defaults. Large datasets still
require sufficient RAM and time to construct their splits. Rendering and
measurement do not edit the docs or installed defaults.

The published non-Amazon audit combines full-cache measurements on the local
machine and abaddon. The archive preserves each run's code fingerprints and
source paths/checksums. A failed installed configuration is kept visible, rather
than silently replaced with a sweep override. These measurements are not a claim
that every dataset's defaults have been size-tuned. Amazon uses the
separate complete :doc:`support <amazon-profiling>` and :doc:`metadata <amazon-metadata>`
audits.

.. _dataset-default-table-details:

Interpreting the documentation tables
----------------------------------------

Every dataset subsection has an installed-defaults table. **Support** is written
as minimum user/item interactions (for example, ``5/1``). The user/item totals
describe the graph after preprocessing for random and LLO splits; temporal
totals describe distinct users and catalog items across its filtered stages.
**Train** reports training users separately. **Val/Test** report distinct
eligible users, distinct target items, and the percentage of those target items
absent from nonzero training interactions. They are not candidate-catalog sizes
or percentages of interactions. A dagger (†) flags fewer than 1,000 evaluation
users; it is a warning, not an automatic change to a dataset's defaults.
DBbook additionally has an **Official** column, whose preprocessing totals refer
only to supplied training data. Other datasets have no supported official mode.
For sources with repeated events (such as Steam and Gowalla), random splits
deduplicate user/item pairs before support filtering, whereas ordered splits
retain events. Their preprocessed user/item totals can therefore differ even
with the same numerical support thresholds.

The final column counts distinct items across the **union of measured splits**,
including non-temporal preprocessed items. It shows independent counts for at
least one image URL and at least ten words in the adapter's constructed
``entity_text``. URLs are not fetched or validated. **Image URLs: not exposed**
does not mean images do not exist upstream; optional precomputed image embeddings
are not counted as raw images. MovieLens 1M, DBbook and Last.fm-2K also report
**precomputed features**: available items, percentage of the same union catalog,
and vector dimension (``d``) for each encoder. These are measured from the
checksum-verified official JSON releases with the checkpoint importer's exact ID
matching and validation. Missing vectors stay unavailable; valid zero vectors
still count as available. Last.fm media IDs are pooled by artist, and its
interaction-derived tag embeddings are identified separately from content text.
The ten-word statistic does not change ``min_entity_text_words``.

The non-Amazon tables use full source caches on this machine and abaddon,
with the installed builder defaults. Unlike the exploratory
sweep, these use the dataset's own seed and text-length default (30 words for
MovieLens/Goodbooks); annotations are disabled because they do not change the
interaction or text counts. **Build failed** records a real attempt with the
installed configuration and its error, not a missing cache. **Not supported**
denotes a protocol the adapter cannot supply.
Amazon has a separate :doc:`category/protocol audit <amazon-profiling>`.

The :download:`non-Amazon measurement archive <_static/dataset-default-measurements.json>`
records parameters, source-file checksums, code fingerprints and status for every
dataset/split. See :ref:`dataset-default-table-reproduction` to regenerate the
tables. The missing optional feature releases were downloaded for the audit;
no preprocessing defaults were changed and no embeddings were attached to user
checkpoints.

.. _dbbook-official-protocol:

DBbook official protocol
------------------------

``DBbook.get_official_split()`` returns the supplied train and test DataFrames,
including negative feedback. The combined interaction table retains
``source_split``. Random user/item split modes create new partitions.

For the supplied test boundary, use:

.. code-block:: console

   compresso-recsys-build-checkpoint --dataset dbbook --split_mode official --eval_draws 1 --checkpoint_path artifacts/dbbook/official.zip

This mode filters support using training data only, withholds validation edges
from supplied training interactions, and removes those edges from model training.
Test histories use the full retained positive training history. The model is
not refit automatically. Test positives outside the training user/item vocabulary
or below evaluation support are excluded and counted in the manifest. Overlapping
supplied train/test pairs are rejected. It always uses one validation draw;
``val_users`` and ``test_users`` do not control this mode. This preserves the
test boundary, not an exact published training/evaluation recipe.

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

The sweep shares the builder's temporal defaults: 30-day (720-hour) target windows
for Gowalla and 339-day (8,136-hour) windows otherwise.
``--temporal-period-hours`` overrides this; a dataset-specific
``temporal_period_hours`` builder override takes precedence over that flag.
Windows and support settings are printed in the log and saved in the
resolved parameters; insufficient history or empty filtered windows still fail
explicitly instead of triggering automatic changes to the evaluation protocol.

Evaluation protocol
--------------------

Baselines fit the checkpoint's binary ``x_train`` once. There is no tuning,
validation/test training data, or automatic test-time refit. Metrics are Recall,
CalibratedRecall, NDCG, and HitRate at 10 and 20; use ``--cutoffs`` to change them.
Seen items are excluded by default (``--exclude-seen``). Use
``--no-exclude-seen`` for repeat-interaction prediction, such as Gowalla LLO.
The full phase-specific candidate catalog is used.
Item IDs are aligned across temporal vocabularies; future stages' candidates
are not offered in earlier stages.

Popularity and collaborative KNN give cold items zero learned signal. Their
cold-start scores can reflect ties, not a content-based cold-start capability.
The report includes cold-target coverage to make this visible. Rows with no
targets or fewer allowed candidates than the largest requested cutoff are excluded
and counted, without silently changing the cutoff. Seeded ``--max-eval-users``
sampling is shared across baselines and keeps all draws of a selected user together.
Scores from different split protocols are not directly interchangeable.

The flag is recorded in JSON and the report's **Exclude seen** column, including
failed/skipped evaluations. The report also counts repeat targets and targets
made unreachable by seen-item exclusion. Such cases produce a warning; targets
are never silently dropped. Preprocessing, checkpoints, model defaults, and
training data are unchanged. User/item splits should normally keep exclusion
enabled. Scores with different settings are not directly comparable.

.. code-block:: bash

   python examples/validation/dataset_sweep.py \
     --datasets gowalla --splits leave_last_out temporal \
     --no-exclude-seen --output artifacts/gowalla-repeat-evaluation

To score an existing checkpoint, including one from an older sweep, without
loading raw datasets or building a new split:

.. code-block:: bash

   python examples/validation/dataset_sweep.py \
     --checkpoint /path/to/checkpoint.zip \
     --no-exclude-seen --output artifacts/repeat-evaluation

This reads dataset/split identity and build metadata from the checkpoint and
records its SHA-256. The original file is unchanged. Dataset/split selections,
if supplied, must include that checkpoint; build settings are not applied.
Both modes still fit the requested baselines for each new evaluation run; fitted
models are not cached. A changed candidate policy also changes which users have
enough candidates at the requested cutoff, so evaluated-row counts can differ.

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
whose recorded hashes match. Completed runs are also checked: a missing/corrupt
checkpoint (or missing recorded hash) triggers recovery and fresh evaluation.
The previous result and any invalid archive are preserved in a ``recovery-*``
subdirectory; stale statistics and scores are discarded from the active result.
Recovery reuses another verified compatible checkpoint if available, otherwise
rebuilds it. If rebuilding fails, the run is reported as failed and can be retried
with ``--resume``. Without ``--resume``, existing results are not overwritten.

New sweeps give builds a separate fingerprint based on builder arguments and
package source. Changing only evaluation settings (including ``exclude_seen``)
creates a new result directory but reuses a matching, checksum-verified
checkpoint in the same output root, without reloading raw data. Archives are
hard-linked when possible, otherwise atomically copied. Earlier results remain
intact; the combined summary describes the current invocation. Old sweep records
without build fingerprints are not guessed compatible: use ``--checkpoint`` to
explicitly evaluate those archives.
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
JSON files, checkpoints, scores, and provenance unchanged. Legacy records without
an explicit ``exclude_seen`` value display **unknown (legacy)**. No dataset/split
selection or evaluation options are applied in this mode. A missing
``results.jsonl`` is an error; report-only does not start a new sweep or discover
additional results from unfinished workers.

Updating the script or package source changes the evaluation fingerprint, so
``--resume`` does not reuse results from an older code fingerprint. Unchanged
package source and builder arguments can still reuse the data checkpoint.
A copied script uses the
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
