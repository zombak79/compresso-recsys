:orphan:

Provisional Sequential Benchmark Notes
======================================

.. warning::

   These are development notes preserved from the Models API, not a
   reproducible model comparison. They are intentionally absent from the
   documentation menu. Do not cite the figures until the datasets, checkpoints,
   tuning grids, seeds, commands, and raw results are published together.

The notes below record why several sequential-model defaults were chosen and
what should be checked by a future benchmark. They remain useful as hypotheses,
but they have already drifted in places as implementations and defaults changed.

SimpleRNN Smoke Check
---------------------

On MovieLens-1M with ``min_value_to_keep=4.0`` under ``leave_last_out``, the
documented SimpleRNN example reached ``ndcg@20 = 0.132`` against ``0.070`` for
EASE at ``l2=200``. The loss was still falling at the eighth epoch, so this was
a wiring check rather than a tuned comparison.

The same split put 697 of 6,033 users over the default ``max_length=200``
window, dropping roughly 80,000 of 543,000 training positions.

Unknown-Token Dropout
---------------------

One temporal MovieLens-1M stage had 26% out-of-catalog items in its test
histories. Recorded ``ndcg@20`` values for SimpleRNN were 0.090 with no unknown
token dropout, 0.113 at 0.05, and 0.121 at 0.25. This suggests matching the
dropout rate to expected unknown-item exposure, but it needs to be rerun under
the final benchmark protocol.

Sequential Model Comparison
---------------------------

The following table was recorded as mean ``ndcg@20`` over three training seeds.
Epoch budgets and learning-rate schedules were selected on validation data
before test scoring, and the grids were reportedly widened until validation
performance turned over.

.. list-table:: Recorded ndcg@20, validation-selected budget and schedule
   :header-rows: 1
   :widths: 22 20 20 20

   * - model
     - ML-1M ``leave_last_out``
     - Office ``leave_last_out``
     - Office ``temporal``
   * - popularity
     - 0.0176
     - 0.0185
     - 0.0094
   * - ELSA
     - 0.0562 +/- 0.0013
     - 0.0327 +/- 0.0011
     - 0.0076 +/- 0.0004
   * - SimpleRNN
     - 0.1471 +/- 0.0008
     - 0.0282 +/- 0.0018
     - 0.0091 +/- 0.0007
   * - SimpleGPT
     - 0.1740 +/- 0.0011
     - 0.0422 +/- 0.0011
     - 0.0102 +/- 0.0007

The recorded peak budgets were 40, 20, and 20 epochs for SimpleGPT, and 20,
10, and 5 for SimpleRNN. Cosine scheduling was recorded as adding 0.0080 to
SimpleGPT on Office ``leave_last_out`` and less than one seed deviation to
SimpleRNN on all three splits.

Recorded Ablations
------------------

One version of the tying ablation reported improvements of 0.0171, 0.0068, and
0.0006 across the three columns above. A later ``SimpleGPTConfig`` docstring
recorded 0.0127, 0.0074, and 0.0009 instead. The disagreement is one reason these
figures were removed from the API reference; the benchmark must establish which
configuration and result artifact each set describes.

The GPT-2 initialization and cosine schedule were also recorded as interacting:
on Office ``leave_last_out``, the narrower initialization scored 0.0341 against
0.0389 under a constant rate, and 0.0422 against 0.0380 under cosine. Tied
embeddings were observed to converge later than an untied head, trailing at ten
epochs on ML-1M and passing by twenty.

A capacity sweep under older defaults reportedly found that tying dominated
width and depth: every tied configuration beat every untied configuration, one
layer matched two, and a tied model with about a third of the parameters beat
the largest untied model. The original figures were not retained with the API
documentation and the sweep predates the current initialization and schedule.

Dataset Observations
--------------------

MovieLens favored both sequential models strongly in the recorded comparison.
On Amazon Office_Products the margins were much smaller and ELSA remained
competitive with SimpleRNN. Office targets averaged one item per user, and none
of the first two thousand inspected users repeated an item, making those
purchase histories closer to sets than ordered consumption histories.

The recorded Office ``temporal`` split was dominated by cold start: 85% of test
targets never appeared in training and 75% of users had at least one such
target. An oracle restricted to the trained catalog scored 0.3252 rather than
1.0. This column is therefore more useful as a cold-start diagnostic than as a
ranking of fixed-vocabulary recommenders.

Open Questions
--------------

Tying couples input-embedding scale to logit scale. A tied head starts with a
flatter softmax under the current initialization, which may matter more for
these short training runs than it does for language-model runs lasting hundreds
of thousands of steps. A logit temperature has not been evaluated.

A proper comparison should publish, at minimum:

* the repository commit and dependency versions;
* checkpoint-building arguments and immutable dataset identifiers;
* complete model configurations and tuning grids;
* the validation metric and selection rule;
* all random seeds and per-seed results;
* executable commands and machine-readable result artifacts; and
* hardware and timing information where performance is discussed.
