Amazon default-setting audit
============================

``examples/validation/amazon_profile.py`` searches **support thresholds**, not
model hyperparameters, independently for all 33 named Amazon categories and the
four supported protocols: ``user_split``, ``item_split``, ``leave_last_out`` and
``temporal``. The uncategorized ``Unknown`` bucket is not included.

This is an empirical, bounded search. Its result is the best measured setting
under the objective below, not a claim of a globally optimal preprocessing
protocol or the best recommendation scores. Never select settings using model
test scores.

Installed results
-----------------

All 33 named categories have all four protocol profiles installed (132 profiles).
The original support search and the subsequent holdout, evaluation-detail and
Clothing temporal reviews are complete. The defaults are verified practical
profiles, not a claim that every category meets every preferred size target.
The :ref:`Amazon dataset section <dataset-amazon2023>` lists the selected support
thresholds and requested user-split holdouts. The
:download:`machine-readable measurement archive <_static/amazon-default-profiles.json>`
records per-profile parameters, eligible evaluation users, retained sizes,
warnings, source-file fingerprints and the frozen audit-code fingerprints.
The :download:`complete Markdown results table <_static/amazon-defaults-summary.md>`
includes all 132 profiles with training pairs and validation/test item details.
Entries below 1,000 eligible validation or test users remain explicitly flagged;
verification means the split was reproduced, not that it met every preferred
size target. Explicit builder arguments override installed profiles.

The later :doc:`metadata audit <amazon-metadata>` also installed category-specific
text fields, including selected nested details. These text recipes are separate
from the support/split registry. ``min_entity_text_words=0`` is unchanged, so
the enriched text does not change the measured interaction graphs or split sizes.

All Subscription Boxes profiles fall below the evaluation-user target. In
particular, leave-last-out has just 24 eligible users per phase. Treat this tiny
category as a pipeline check rather than evidence for reliable model comparisons.

For Magazine Subscriptions, a separate user-holdout review raised the requested
validation and test partitions from 680 to 1,200 each without changing its 2/1
support setting. This reaches 1,115 eligible validation and 1,093 test users,
while retaining 4,408 training users. Its archive entry records the original
result, tested alternatives and refinement-code fingerprint. The frozen support
audit is not overwritten; this reviewed refinement takes precedence in the
installed defaults and subsequent reported tables.

Clothing temporal uses the user-approved **120,000-user / 25,000-item exception**
at support **10/17**; its other three splits remain **10/35**, with 5,000 requested
validation and test users for user split. The full temporal rebuild matched the
screening counts: 113,384 unique users across stages, 24,824 catalog items,
29,993 training users, 10,351 observed training items and 428,731 training pairs.
Validation has 76,251 users and 20,663 target items (54.84% cold); test has 59,629
users and 19,421 target items (57.30% cold).

This was a manual review of training-set usefulness, not a model-score-based
selection. The original automatic ranking favored 2/279 because it met the
evaluation-user target, but it left only four observed training items and was
rejected. A within-cap 19/8 refinement improved the training catalog but left
only 636 validation users. The archive preserves both alternatives, the 10/17
screening result, approved limits and independent full-rebuild fingerprints.
The frozen search/ranking implementation and original measurements remain intact;
its automatic ranking alone is not a sufficient training-quality check.

Evaluation item details
-----------------------

``examples/validation/amazon_evaluation_details.py`` rebuilds the selected profiles
from checksum-verified prepared inputs without downloading or searching support
thresholds. It checks the original counts for unchanged profiles and checks that
the preprocessed graph is unchanged when refining Books/Electronics holdouts.
The archive records its separate code and input fingerprints. Run this audit
against the frozen profiler/builder that produced the archived measurements;
their fingerprints must match. Keep the detail runner outside that frozen source
directory and expose the frozen modules through ``PYTHONPATH``. The run uses its
own locked output directory and never overwrites the original measurements.
An explicitly reviewed profile can provide ``size_limits`` with positive integer
``max_users`` and ``max_items``. These bounds are fingerprinted with that profile
and preserved in its result; they do not alter other profiles or the frozen
search policy. Clothing temporal is the only such per-profile exception in this
archive. This check uses preprocessed sizes for random/LLO splits and cumulative
sizes for temporal, so held-out users/items cannot hide an oversized graph.

Validation/test sizes count distinct eligible users and distinct target items,
not the full candidate catalog. Cold-item percentages count distinct target items
with no nonzero interaction in ``x_train``; they are not weighted by interaction
frequency. Item IDs are aligned across phase-specific vocabularies. A temporal
test item introduced during validation is still cold relative to the original
training matrix. Extra evaluation draws do not add distinct users or items unless
they actually expose different target items.

All 132 installed profiles have completed this supplementary detail audit.
For user splits, Books retains 388,352 training users and Electronics 454,089;
both have exactly 20,000 eligible validation and 20,000 eligible test users at
seed 42. Their preprocessed graphs remain 428,352 users / 94,864 items and
494,089 users / 92,180 items respectively. The other categories' holdout
allocations are unchanged. Clothing's separately reviewed temporal support is
10/17; the other categories' support settings are unchanged.

``examples/validation/amazon_profile_table.py`` renders the support, size,
evaluation-detail and item-metadata table to standard output. It reads the
profile archive and the sibling ``amazon-metadata-coverage.json`` by default;
``--metadata-archive`` can specify another metadata audit path. The final column
uses distinct-item union counts, source image URLs and the installed curated
recipe's ten-word threshold. Regenerate the table when updating either archive;
regression tests check every cell against the recorded measurements.

Constraints and selection
-------------------------

* Keep every valid rating as a binary interaction, including 1-star ratings.
* Require item metadata to exist, but impose no minimum text length.
* Use iterative support filtering only: **no user/item subsampling or top-N
  truncation**.
* Ordinary categories: at most 100,000 users and 20,000 items. Prefer at least
  50,000 users and 10,000 items when the graph allows it.
* Books and Electronics: fewer than 500,000 users and at most 100,000 items.
  No other category receives an automatic large-dataset exception.
* Clothing temporal only: a user-approved exception of at most 120,000 users
  and 25,000 items. Its user, item and leave-last-out profiles retain the ordinary
  100,000-user / 20,000-item caps.
* Prefer more users than items, without shrinking a catalog merely to maximize
  that ratio. Aim for at least 1,000 distinct eligible validation users and
  1,000 test users after all filtering. Smaller results carry explicit warnings;
  this practical target is not a statistical power guarantee.
* Count users/items **before splitting** for the random and leave-last-out
  protocols, and also check the resulting checkpoint. For temporal, count the
  unique union of users across stages and the final cumulative catalog, not
  merely the smaller training matrix.

The user-support search is ``2, 3, 4, 5, 6, 7, 8, 10, 12, 15, 20, 25, 30, 40, 50``.
Leave-last-out only searches values of at least four, because it needs one
source plus separate training, validation and test targets. For user splits,
two distinct in-training-catalog items suffice for one source and one target
at the default holdout fraction.
For each user threshold, an expanding/binary search finds the smallest item
threshold satisfying both upper bounds. On small graphs, item thresholds
2, 3, 4 and 5 are also checked to assess evaluation vocabulary coverage. A few
stronger neighboring thresholds (boundary + 1, 1.25x, 1.5x and 2x, rounded up)
are measured to explore better item sharing and user/item balance.

Every feasible measured candidate is passed to the real split builder.
Selection first prefers at least 1,000 eligible validation and test users.
Within that group, it favors histories of at least five when they already yield
50,000 users, 10,000 items and at least as many users as items. Otherwise shorter
histories can rescue sparse categories. It then scores capped size attainment
``min(users / 50000, 1) * min(items / 10000, 1)``, multiplied by the balance
``min(users / items, 1)`` and evaluation coverage
``min(min(val_users, test_users) / 1000, 1)``. Retained training pairs break ties.
Thus a huge ratio from a tiny catalog receives no extra reward.
Small or weak-evaluation results carry explicit warnings; empty splits are not
promoted as successful settings.

The original support search requested 10% of preprocessed users per user-split
holdout, capped at 5,000. A user-requested holdout refinement now requests
**20,000 users each for Books and Electronics**; their pruning thresholds and
preprocessed graphs are unchanged. Other allocations remain unchanged, including
the separately reviewed Magazine Subscriptions refinement. These are evaluation partitions,
not preprocessing subsampling. Item-split fractions remain 5% validation and
10% test; temporal windows remain 339 days. The seed is 42. The audit and normal
checkpoint builder both default to one evaluation draw. Explicitly requesting
more draws does not change unique-user counts.

Temporal support remains stage-wise. The fast search reuses sparse time windows
but calls the builder's own support-filter implementation, then checks its counts
against the full temporal builder. The loader safely skips histories with fewer
than two total events while retaining the actual earliest/latest
metadata-eligible events, so this optimization does not shift time boundaries.
It does not globally prefilter item support before temporal splitting. Earlier
five-event input caches must be regenerated from raw sources; their omitted users
cannot be recovered by lowering a threshold during search.

Running and resuming
--------------------

The complete source download is large and can take hours. No images, embeddings
or full review text are downloaded. Preparation stores user counts and metadata
IDs in a temporary disk-backed SQLite database and writes parquet in chunks;
the original source metadata remains in the normal cache. The hybrid coordinator
does not reload the full prepared frame locally. Heavy profiling still loads
the category on the server. Profile sequentially on memory-constrained machines.

Download with four independent connections (completed files and partial
transfers are reused):

.. code-block:: bash

   python -u examples/validation/amazon_download.py --workers 4

In another terminal, process complete categories as they become available:

.. code-block:: bash

   python -u examples/validation/amazon_profile.py \
     --cached-only --wait-for-sources

Do not run another downloader for the same category at the same time. Without
``--cached-only``, the profiler can also download its own inputs. To audit only
already cached categories:

.. code-block:: bash

   python -u examples/validation/amazon_profile.py --cached-only \
     --categories All_Beauty Digital_Music Office_Products \
       Baby_Products Grocery_and_Gourmet_Food

The default output is ``artifacts/amazon-profiles``. Each category has source
hashes, an input cache, tested thresholds, candidate results, selected parameters
and warnings. ``summary.md`` summarizes measured results and
``run-status.json`` records progress. Missing downloads are not successful
profiling results. The input cache validates source sizes and modification times;
the audit records SHA-256 source hashes. Changed search code or builder code
invalidates verified-result reuse. Repeating a finished run otherwise reuses
the verified split results.

Hybrid local/server execution
-----------------------------

Keep raw downloads on a host with enough disk space and send compact prepared
inputs to a host with more RAM. ``amazon_hybrid.py`` prepares categories
sequentially, transfers only ``input.parquet`` and its source/checksum manifest,
then publishes a ready marker. It collects reports and logs back locally.
It leaves at least 20 GiB free on the remote filesystem and does not remove
existing files to make space.

On the server, deploy an **isolated snapshot** of the same package source and
audit scripts. Do not overwrite a repository running other experiments. The
server needs the package dependencies, but no raw dataset files. For example,
start the prepared-input queue in that snapshot:

.. code-block:: bash

   python -u examples/validation/amazon_remote_worker.py \
     --input-root /path/to/audit/inputs \
     --output /path/to/audit/results --workers 4 --detach

Then start the local coordinator:

.. code-block:: bash

   python -u examples/validation/amazon_hybrid.py \
     --host rtx --remote-root /path/to/audit \
     --remote-python /path/to/python

The remote worker verifies the entire Parquet checksum, category, schema,
preprocessing policy and binary values before use. It does not assume the
original raw sources are present remotely. A ready marker is written only after
both transfers complete. File locks prevent duplicate coordinators or worker
managers from sharing the same output directory. Start with a conservative
worker count: the large categories use more RAM than their compressed file
sizes suggest. The queue also defaults to an 80 GiB admission budget, estimating
1 KiB of working memory per prepared row across active categories. This is a
conservative scheduling estimate, not a hard RSS guarantee. A category exceeding
the entire budget is flagged for review instead of being launched or truncated;
``--memory-budget-gib`` can be adjusted for the host. The GPU is not used here.

Detached remote workers survive SSH disconnections and wait for new prepared
inputs. The local machine still needs to stay awake for downloads, preparation,
transfers and result collection. State is recorded in ``hybrid-state.json``
locally and ``remote-state.json`` remotely; category logs and reports are copied
into ``remote-results`` locally. Results already measured locally remain intact.
Use ``--prepared-root`` with ``amazon_profile.py`` to profile a transferred
input directly without any raw-source access.

Measured settings must be reviewed and applied to the category/split default
registry only after verification. Explicit builder arguments must continue to
override defaults. This audit does not silently rewrite source-code defaults,
commit changes, or push the repository.
