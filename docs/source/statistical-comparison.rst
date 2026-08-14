Comparing Models Statistically
==============================

Two models evaluated on the same users can be compared far more precisely than
their aggregate scores suggest. This guide shows how, and — more importantly —
how to read and report the result without saying something a reviewer will
object to.

No statistics background is assumed. The concepts are introduced where they are
needed rather than up front.

Why the aggregates are not enough
---------------------------------

Suppose EASE scores nDCG@100 of 0.4862 on GoodBooks and ELSA scores 0.4916.
Is ELSA better?

The honest answer is that you cannot tell from those two numbers. They are
averages over thousands of users, and averages hide how much the two models
actually disagreed. Perhaps one model won for nearly every user by a
hair. Perhaps they were identical for 90% of users and the gap comes from a
handful of outliers. Those two situations warrant very different conclusions,
and the aggregate cannot distinguish them.

What settles it is the **per-user difference**: for every user, how much better
did the candidate do than the baseline? That is what
:mod:`compresso_recsys.stats` works with, and it is why evaluation retains
per-user values rather than only their mean.

Do the comparison
-----------------

Evaluate every model on identical users and targets, then compare:

.. code-block:: python

   from compresso_recsys.evaluation import evaluate_recommender
   from compresso_recsys.metrics import CalibratedRecall, NDCG
   from compresso_recsys.stats import compare_models

   metrics = [CalibratedRecall([20, 50]), NDCG(100)]

   ease_result = evaluate_recommender(
       ease, source=test_source, targets=test_targets,
       metrics=metrics, sample_ids=test_user_ids,
   )
   elsa_result = evaluate_recommender(
       elsa, source=test_source, targets=test_targets,
       metrics=metrics, sample_ids=test_user_ids,
   )

   report = compare_models(
       {"EASE": ease_result, "ELSA": elsa_result},
       metrics=["ndcg@100", "calibrated_recall@20"],
       reference="EASE",
   )
   print(report.to_frame().to_string(index=False))

Two requirements, both enforced rather than assumed:

* **The same users, in the same order.** Comparison is *paired*: each user is
  compared against themselves under the other model. Passing ``sample_ids``
  makes that explicit, and mismatched or reordered identifiers raise an error
  instead of quietly comparing different people.
* **Per-user values retained.** They are collected by default. A result
  evaluated with ``collect_per_user=False`` cannot be compared.

For a single hypothesis, use :func:`~compresso_recsys.stats.compare_pair`.

.. _stats-walkthrough:

A complete worked example
-------------------------

Everything below runs end to end on a laptop in about three minutes, including
the dataset download. The numbers in this guide come from this script, not from
illustration.

The question: on GoodBooks, does ELSA beat EASE?

Build the split
~~~~~~~~~~~~~~~

.. code-block:: python

   import compresso_recsys as cr

   cr.build_recsys_checkpoint(
       dataset="goodbooks",
       data_dir="data",
       checkpoint_path="artifacts/goodbooks/comparison.zip",
       split_mode="user_split",
       eval_draws=1,
       seed=42,
   )

Roughly 40 seconds, most of it downloading. A user split holds out whole users,
so both models see the same 9,975 items and are evaluated on 2,500 unseen users.

``eval_draws=1`` gives each held-out user a single fold-in/scored split, so one
row means one user, which keeps this walkthrough simple. The default of 5 splits
each user five times and stacks the rows — a more precise protocol, and the one
the ELSA papers use, but it gives each user several rows. See
:ref:`stats-repeated-rows`.

Train both models
~~~~~~~~~~~~~~~~~

.. code-block:: python

   import torch
   from compresso_recsys.models import EASE, EASEConfig, ELSAConfig, ELSATrainer

   with cr.read_checkpoint("artifacts/goodbooks/comparison.zip") as root:
       split = cr.load_recsys_split(root)

   x_train = split["x_train"]        # (49865, 9975), 3.85M interactions
   device = "cuda" if torch.cuda.is_available() else "cpu"

   ease = EASE(EASEConfig(l2=700.0)).fit(x_train)

   elsa = ELSATrainer(
       ELSAConfig(latent_dim=3250, batch_size=2048, epochs=10,
                  lr=0.05, device=device, seed=0)
   ).fit(x_train)

EASE takes about 12 seconds — it is a closed-form solve. ELSA takes about 100
seconds for 10 epochs on a GPU.

Evaluate both on identical users
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This is the step the comparison depends on. Both models must see the same
source rows, the same targets and the same identifiers:

.. code-block:: python

   from compresso_recsys.evaluation import evaluate_recommender
   from compresso_recsys.metrics import MRR, CalibratedRecall, NDCG, Recall

   source = split["test_source_matrix"]
   targets = split["test_target_matrix"]
   user_ids = split["test_eval_user_ids"]
   metrics = [CalibratedRecall([20, 50]), Recall([20, 50]), NDCG(100), MRR(20)]

   results = {
       name: evaluate_recommender(
           model, source=source, targets=targets, metrics=metrics,
           sample_ids=user_ids, batch_size=1024,
       )
       for name, model in (("EASE", ease), ("ELSA", elsa))
   }

   for name, result in results.items():
       print(name, {k: round(v, 4) for k, v in result.metrics.items()})

.. code-block:: text

   EASE {'calibrated_recall@20': 0.3320, 'calibrated_recall@50': 0.4752,
         'recall@20': 0.3222, 'recall@50': 0.4752,
         'ndcg@100': 0.4862, 'mrr@20': 0.6851}
   ELSA {'calibrated_recall@20': 0.3429, 'calibrated_recall@50': 0.4796,
         'recall@20': 0.3329, 'recall@50': 0.4796,
         'ndcg@100': 0.4916, 'mrr@20': 0.7091}

ELSA is ahead everywhere. Whether that means anything is the next step — but
first, one thing in that output is worth pausing on.

.. note::

   ``recall@20`` and ``calibrated_recall@20`` differ (0.3222 against 0.3320),
   while ``recall@50`` and ``calibrated_recall@50`` are *identical* (0.4752).

   Calibrated recall divides by :math:`\min(k, |\mathcal{R}_u|)` rather than
   :math:`|\mathcal{R}_u|`, so the two coincide exactly when no user has more
   targets than the cutoff. Here users hold between 2 and 35 targets, 15.8 on
   average: 17% exceed 20, and none exceed 50.

   This is why the metric keys were separated. Quoting a
   ``calibrated_recall@20`` of 0.3320 against a published Recall@20 would
   overstate the result by 3%. The absolute gap is bounded — both metrics live
   in :math:`[0, 1]` — but the *relative* overstatement is not, and on a denser
   holdout it grows without bound.

Compare
~~~~~~~

.. code-block:: python

   from compresso_recsys.stats import compare_models

   report = compare_models(
       results,
       metrics=["ndcg@100", "calibrated_recall@20", "recall@20", "mrr@20"],
       reference="EASE",
       n_resamples=9999,
       random_state=0,
   )
   print(report.to_frame().to_string(index=False))

.. code-block:: text

   metric                n_samples  n_nonzero  tie_rate  baseline_mean  candidate_mean  difference  relative    ci_low   ci_high  adj_p   direction
   ndcg@100                   2500       2495      0.2%       0.486191        0.491611    0.005420     1.11%  0.002953  0.007977  0.0004  better
   calibrated_recall@20       2500       1418     43.3%       0.332034        0.342880    0.010846     3.27%  0.007751  0.013895  0.0004  better
   recall@20                  2500       1418     43.3%       0.322212        0.332898    0.010686     3.32%  0.007629  0.013792  0.0004  better
   mrr@20                     2500       1009     59.6%       0.685141        0.709079    0.023938     3.49%  0.013721  0.034306  0.0004  better

Reading it
~~~~~~~~~~

**ELSA beats EASE, and the margin is small but solid.** Every interval sits well
clear of zero, and every adjusted p-value is at the floor for 9,999 resamples.

**The relative gains disagree, and that is informative.** nDCG@100 improves by
1.1%, recall@20 by 3.3%, MRR@20 by 3.5%. ELSA is noticeably better near the top
of the list; across the full hundred ranks the two are much closer. A paper
reporting only nDCG@100 would understate what changed, and one reporting only
MRR@20 would overstate it.

**Look at the tie rate.** For nDCG@100 the models differ for 2,495 of the 2,500
users — essentially everyone, because a metric reading 100 ranks deep notices
almost any reordering. For recall@20 they differ for 1,418, and for MRR@20 only
1,009. So **60% of users get an identical MRR from both models**, because their
first relevant book lands at the same rank either way.

That does not mean the MRR comparison rests on 1,009 users. All 2,500
contribute to the estimated difference and to its interval; a tied user is an
observation that the two systems agreed, not a missing one. What ``n_nonzero``
governs is the *randomization test*: flipping the sign of a zero changes
nothing, so only those 1,009 differences can move the null distribution, and a
small count there makes it coarse. :ref:`stats-trap-tied` returns to this.

**What this does not establish, yet.** Both models were trained once. These
intervals say the advantage would survive a different sample of users; they say
nothing about whether it survives retraining ELSA with a different seed. EASE has
no seed at all, being closed-form, so only one side of the comparison can move.

To claim ELSA is the better *method* rather than that this ELSA beat this EASE,
that has to be measured rather than assumed. :ref:`stats-seeds` does exactly
that, on these models: five seeds, and the gap turns out to be about nine times
the spread they produce.

Reading the output
------------------

One row of the ``report.to_frame()`` above, laid out vertically:

.. code-block:: text

   metric                    ndcg@100
   baseline                  EASE
   candidate                 ELSA
   n_samples                 2500
   n_units                   2500
   n_nonzero                 2495
   tie_rate                  0.002000
   baseline_mean             0.486191
   candidate_mean            0.491611
   difference                0.005420
   relative_difference       0.011148
   ci_low                    0.002953
   ci_high                   0.007977
   confidence_level          0.95
   bootstrap_standard_error  0.001275
   p_value                   0.0001
   adjusted_p_value          0.0004
   significant               True
   direction                 better
   alternative               two-sided
   test_method               randomization
   interval_method           percentile
   n_resamples               9999
   random_state              0

The effect
~~~~~~~~~~

``difference`` is always **candidate minus baseline**, so positive favours the
candidate. Here ELSA gained 0.0050 nDCG@100 over EASE, which
``relative_difference`` expresses as 1.0% of the baseline.

This is the number to lead with. It is computed directly from the data, so it
does not depend on any of the resampling settings.

How precisely you know it
~~~~~~~~~~~~~~~~~~~~~~~~~

``ci_low`` and ``ci_high`` give a range of plausible values for the true
difference — here 0.0030 to 0.0080 at ``confidence_level`` 0.95.

That range comes from a **bootstrap**. Take your 2,500 users and draw 2,500 of
them *with replacement* — meaning the same user can be drawn more than once.

That last phrase is doing all the work. Drawing 2,500 from 2,500 sounds like it
must return the same set, and it would if each user could be drawn only once.
With replacement, a typical draw looks like this:

.. code-block:: text

    1,581 distinct users (63.2%)
      919 never picked
      661 picked two or more times, one as often as six

So every draw is a different dataset — a plausible alternative version of your
evaluation, in which some users happened to be over-represented and about a
third are missing. Recompute the mean difference on it, repeat 9,999 times, and
the spread of those 9,999 numbers shows how much your answer would move if you
had happened to evaluate on a different sample of users from the same
population. ``bootstrap_standard_error`` is that spread as a single number, and
the interval is the middle 95% of it.

An interval that excludes zero says the direction is stable. An interval that
straddles zero says you cannot tell which model is better, whatever the point
estimate suggests.

How often the two models tied
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``n_samples`` is the rows evaluated. ``n_nonzero`` is the rows whose score
actually *differed* between the two models, and ``tie_rate`` is the share that
did not.

That distinction matters more in recommendation than almost anywhere else,
because ranking metrics produce enormous numbers of exact ties. Across the
EASE-versus-ELSA comparison, from the same 2,500 users:

.. code-block:: text

   ndcg@100               2,495 of 2,500   (99.8%)
   calibrated_recall@20   1,418 of 2,500   (56.7%)
   recall@20              1,418 of 2,500   (56.7%)
   mrr@20                 1,009 of 2,500   (40.4%)

A metric reading 100 ranks deep notices almost any reordering, so nearly every
user counts. MRR@20 depends only on where the *first* relevant book lands, so
for 60% of users it lands in the same place under both models.

Report the tie rate. It is a finding in its own right — these two models are
indistinguishable for most users at rank one — and it tells a reader how coarse
the randomization test's null distribution was. What it is *not* is a smaller
sample size: all 2,500 users contribute to the difference and to the interval.
:ref:`stats-trap-tied` works through why that distinction matters.

Whether it could be chance
~~~~~~~~~~~~~~~~~~~~~~~~~~

``p_value`` answers a single question: *if the two models were genuinely
equivalent, how often would chance alone produce a difference this large?*

The default test makes that concrete. Suppose the two models really are
interchangeable. Then for any given user, which model came out ahead is a coin
flip — swapping the two labels for that user alone would produce data just as
plausible as what you saw. Flipping the sign of that user's difference is
exactly that swap.

So the test flips a coin for every user, negates that user's difference when the
coin says so, recomputes the mean, and repeats 9,999 times. Those 9,999 numbers
are what the difference would look like in a world where the models are
equivalent. ``p_value`` is the fraction of them that came out at least as
extreme as the difference you actually measured.

If your result sits comfortably inside that crowd, chance explains it. If almost
nothing in the crowd reaches it, chance does not.

Small p means chance rarely reproduces your result. Large p means it easily
does.

``adjusted_p_value`` corrects for testing several hypotheses at once — see
:ref:`stats-trap-adjusted` below.

The verdict
~~~~~~~~~~~

``significant`` is ``adjusted_p_value <= alpha``, where ``alpha`` is
``1 - confidence_level``.

``direction`` combines that with the sign:

* ``better`` — significant and the candidate is ahead
* ``worse`` — significant and the candidate is behind
* ``inconclusive`` — not significant, **whatever the sign of the difference**

A negative difference that is not significant reads ``inconclusive``, not
``worse``. The sign of an estimate you cannot distinguish from noise is not
information, and a table that prints a direction for it invites a misreading.

.. _stats-traps:

Five traps
----------

These are the mistakes that cost papers, in rough order of how often they occur.

p-values have a floor
~~~~~~~~~~~~~~~~~~~~~

With ``n_resamples`` set to ``B``, the smallest achievable p-value is
``1 / (B + 1)``. At the default 9,999 that is 0.0001.

So the ``p_value`` of 0.0001 in the worked example does **not** mean one in ten
thousand. It means *not one* of 9,999 random sign assignments was as extreme as
what ELSA achieved.

It is tempting to conclude the true p is therefore below :math:`10^{-4}` and to
write ``p < 10^-4``. **That does not follow.** If the exhaustive p were exactly
:math:`10^{-4}`, you would still see zero exceedances in 9,999 draws about 37%
of the time — :math:`(1 - 10^{-4})^{9999} \approx 0.37`. Nothing here rules that
out. Applying the rule of three to 0 exceedances in 9,999 draws puts the 95%
upper bound nearer :math:`3 \times 10^{-4}`, three times the value the ``<``
would be claiming as a hard ceiling.

Report the value the procedure actually produces, and say what it rests on:

   *The paired randomization test gave* :math:`p_{\mathrm{MC}} = 0.0001`, *the
   minimum attainable with 9,999 sampled sign assignments; none of the sampled
   assignments was at least as extreme as the observed difference.*

The ``(1 + extreme) / (B + 1)`` form exists precisely so this number is a valid
p-value on its own terms rather than an estimate that has to be qualified — see
Phipson and Smyth in :ref:`stats-references`.

The deeper point is that **you almost never need a smaller p-value.** If your
argument depends on the difference between :math:`10^{-4}` and :math:`10^{-11}`,
the argument is resting on the wrong quantity. Report the effect and its
interval; the p only has to clear the bar. If you genuinely need finer
resolution, raise ``n_resamples``, or use ``test_method="t"``, which has no
floor — bearing in mind :ref:`what that trades away <stats-t-test>`.

The floor also interacts with the correction. With ``J`` hypotheses the
smallest reachable adjusted p is ``J / (B + 1)``. At ``B = 100`` and six
hypotheses that is 0.059, so **nothing could be significant at 0.05 no matter
what the data said**. Never set ``n_resamples`` below 999.

.. _stats-trap-tied:

Ties are not missing data
~~~~~~~~~~~~~~~~~~~~~~~~~

Ranking metrics tie constantly. In the worked example 60% of users get the same
MRR@20 from both models, and that is ordinary rather than a defect: it says the
two systems put the user's first relevant book at the same rank.

The tempting move is to treat ``n_nonzero`` as the real sample size and say the
MRR comparison "rests on 1,009 users, not 2,500". **It does not.** Consider:

* 30 users, every one differing by :math:`+1`. Mean difference 1.0.
* 10,000 users, 30 differing by :math:`+1` and the rest tied. Mean difference
  0.003.

Both have ``n_nonzero = 30``. They describe completely different systems, and
their bootstrap distributions look nothing alike. A tied user is an observation
that the two models agreed — evidence, not an absence of it.

So read the three numbers as three different things:

* ``n_samples`` — everything the estimate and the interval are computed over.
* ``n_nonzero`` — the rows that can move the randomization test, since flipping
  the sign of a zero does nothing. This bounds how *discrete* the null
  distribution is.
* ``tie_rate`` — how often the two models were indistinguishable, which is a
  finding worth reporting in its own right.

When ``n_nonzero`` falls below 30 the comparison warns, and the warning is about
that discreteness: the resampled differences land on few distinct values, so the
percentile interval is coarse. The estimate still uses every sample.

.. _stats-trap-adjusted:

The interval and the p-value can disagree
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Testing four hypotheses at 5% each gives roughly an 18.5% chance of at least one
false positive when nothing is really different. The correction holds that back
to 5% across the whole family, at the cost of making each individual test
harder to pass.

The family is *everything one* :func:`~compresso_recsys.stats.compare_models`
*call produces* — every pair, every metric.

This creates a disagreement that looks like a bug and is not. Run the same
EASE-versus-ELSA comparison on the first 400 test users instead of all 2,500,
and the four metrics split:

.. code-block:: text

   metric                difference    ci_low   ci_high  p_value  adjusted_p  significant
   ndcg@100                0.006523  0.000391  0.012691   0.0399      0.0676        False
   calibrated_recall@20    0.010172  0.002118  0.018378   0.0123      0.0492         True
   recall@20               0.009912  0.001800  0.017684   0.0159      0.0492         True
   mrr@20                  0.027903  0.002771  0.052912   0.0338      0.0676        False

**Every one of those four intervals excludes zero. Only two are significant.**

Both readings are correct. **Intervals are never adjusted** — each describes one
comparison on its own, and on its own each of these would clear 0.05.
**p-values are adjusted.** Holm ranks the four raw p-values and scales each by
how many hypotheses remain: the smallest, 0.0123, is multiplied by four to give
0.0492 and just survives; ``ndcg@100``'s 0.0399 is multiplied by two to give
0.0676 and does not.

Note what that means for ``ndcg@100``: its interval excludes zero, and it is
still not significant. The row is not broken. It is telling you the evidence for
that particular metric is not strong enough to survive being one of four
questions asked at once.

The same models on all 2,500 users are significant on every metric. Nothing
about the models changed; the evidence did.

Never describe these intervals as simultaneous or family-wise. They are not.

``confidence_level`` also moves the significance bar
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

One parameter controls both, because ``alpha = 1 - confidence_level``. Setting
``confidence_level=0.99`` widens the interval **and** tightens the test from
0.05 to 0.01, which will make results disappear that were significant before.

They are coupled deliberately: a 99% interval printed beside a 5% test would
produce rows that look contradictory for no reason other than mismatched levels.
But it does mean 0.99 is a bigger change than it looks.

Coupling them does not *guarantee* the interval and the verdict agree. They come
from different procedures — a percentile bootstrap interval and a sign-flip test
are not inverses of one another — so even at matching levels, and before any
multiplicity adjustment, they can occasionally disagree. Matching the levels
removes one avoidable source of confusion, not all of them.

.. _stats-seeds:

This conditions on one training run
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Everything here resamples **users**. It answers whether an advantage is stable
across different samples of users from this population.

It says nothing about whether the advantage survives retraining with a different
random seed. If your training is stochastic — and gradient-trained models are —
a reviewer will ask, and "we bootstrapped over users" is not an answer.

So measure it. Retrain the stochastic model under matched seeds and evaluate
every one on identical users and targets:

.. code-block:: python

   seeds = range(5)
   ease_result = evaluate_recommender(
       EASE(EASEConfig(l2=700.0)).fit(x_train),
       source=source, targets=targets, metrics=metrics, sample_ids=user_ids,
   )

   differences = {}
   for seed in seeds:
       elsa = ELSATrainer(
           ELSAConfig(latent_dim=3250, batch_size=2048, epochs=10,
                      lr=0.05, device=device, seed=seed)
       ).fit(x_train)
       result = evaluate_recommender(
           elsa, source=source, targets=targets, metrics=metrics,
           sample_ids=user_ids,
       )
       differences[seed] = compare_pair(
           ease_result, result, metric="ndcg@100", random_state=0,
       ).difference

Only ELSA is retrained. EASE is a closed-form solve with no seed at all, so the
spread below is ELSA's training variance alone rather than a symmetric wobble in
both models. On GoodBooks, five seeds give:

.. code-block:: text

   metric                  seed 0    seed 1    seed 2    seed 3    seed 4
   ndcg@100              0.005420  0.005698  0.006920  0.005982  0.005559
   calibrated_recall@20  0.010846  0.010942  0.011201  0.009937  0.010876
   recall@20             0.010686  0.010889  0.010913  0.009822  0.010734
   mrr@20                0.023938  0.025372  0.025198  0.025392  0.027660

Set the spread beside the interval, because they answer different questions —
*would this hold on other users?* against *would this hold if I retrained?*

.. code-block:: text

   metric                    mean   seed sd   half-CI   ratio   combined
   ndcg@100              0.005916  0.000598  0.002510    0.24   0.002581
   calibrated_recall@20  0.010760  0.000481  0.003083    0.16   0.003120
   recall@20             0.010609  0.000450  0.003062    0.15   0.003095
   mrr@20                0.025512  0.001344  0.010391    0.13   0.010478

**The ratio is the number to read.** Seed variation is a sixth to a quarter of
the user-sampling uncertainty, so the per-user interval was already describing
most of what is uncertain here. Treating the two as independent sources and
adding their variances gives ``combined``, which is 1 to 3% wider than the
interval alone — reseeding barely moves the answer.

Had the ratio come out above 1, the reading would reverse: the interval would be
precise about *this* ELSA while saying little about ELSA, and the honest report
would lead with the seed spread.

**Then answer the blunter question.** A reviewer rarely asks for a standard
deviation; they ask whether the result could go the other way.

.. code-block:: text

   ndcg@100              range [0.0054, 0.0069]   all positive
   calibrated_recall@20  range [0.0099, 0.0112]   all positive
   recall@20             range [0.0098, 0.0109]   all positive
   mrr@20                range [0.0239, 0.0277]   all positive

Twenty seed-metric combinations, every one favouring ELSA. That is a stronger
claim than any interval, and it either holds or it does not.

.. warning::

   **Five seeds cannot resolve a close call.** A standard deviation from five
   runs carries roughly 35% relative error, so the 0.24 above is really "somewhere
   near a quarter". That is fine for concluding *clearly smaller*, and would be
   fine for *clearly larger*. It is not enough to distinguish a ratio of 0.9 from
   1.1. If yours lands near 1, run more seeds rather than reporting the number as
   though it settled anything.

   Note also that ``ndcg@100`` has both the smallest relative gain and the
   largest ratio. A metric reading a hundred ranks deep is the most sensitive to
   the fine ordering that reseeding perturbs, so it is where seed noise shows up
   first — and where a small effect deserves the most suspicion.

Choosing the settings
---------------------

``n_resamples``
~~~~~~~~~~~~~~~

Controls only the p-value's resolution and the interval's Monte Carlo noise. It
never touches ``difference``, ``baseline_mean``, ``candidate_mean`` or
``n_nonzero``, which are computed from the data. So you can iterate cheaply
and pay for precision once.

At the worked example's scale — 2,500 users, one model pair, four metrics:

.. list-table::
   :header-rows: 1

   * - ``n_resamples``
     - randomization
     - bootstrap
     - p resolution
   * - 999
     - 0.2 s
     - 0.1 s
     - 0.001
   * - 9999
     - 1.7 s
     - 1.1 s
     - 0.0001

Cost is linear in ``B``, in users, and in the number of hypotheses. Use 999
while iterating and the default 9999 for anything reported.

Values of the form ``10^k - 1`` are conventional because the p-value is
``(1 + extreme) / (B + 1)``: with ``B + 1`` a power of ten, p-values land on a
clean decimal grid and the test achieves its nominal level exactly.

``test_method``
~~~~~~~~~~~~~~~

* ``"randomization"`` (default) — the sign-flipping test described above. Exact
  under its null and the expected choice in the retrieval literature, so it is
  one fewer thing to defend.
* ``"bootstrap"`` — a null-centred bootstrap. Valid for large samples rather
  than exact, and roughly a third faster because it reuses the interval's
  replicates.

They agree closely. Use ``"bootstrap"`` while iterating if speed matters, and
``"randomization"`` for the reported numbers.

.. _stats-t-test:

``test_method``: why not a t-test?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

It is a fair question, and the answer is that you can — ``test_method="t"``
runs a paired t-test, as a one-sample test on the same per-user differences.
Smucker, Allan and Carterette compared both on retrieval data and found they
agree closely, so this is not a case of the resampled test being obviously
right and the familiar one wrong.

The randomization test is the default for two reasons. It is **exact** under
paired label exchangeability, where the t-test relies on the central limit
theorem holding well enough; and ranking differences are exactly the kind of
data — heavily tied, sharply peaked at zero, occasional large outliers — where
that reliance is least comfortable.

Run both. On the worked example:

.. code-block:: text

   metric                n_nonzero  tie_rate  randomization_p        t_p
   ndcg@100                   2495      0.2%           0.0001   2.11e-05
   calibrated_recall@20       1418     43.3%           0.0001   1.17e-11
   recall@20                  1418     43.3%           0.0001   1.18e-11
   mrr@20                     1009     59.6%           0.0001   6.82e-06

Every randomization p-value is pinned at its floor; the t-test, having no floor,
prints numbers far below it. **Do not read those small numbers as more precise.**
They are the far tail of a normal approximation, where the guarantees are
weakest. What the table really says is that all four differences clear any
threshold either test could set, by a wide margin.

Agreement between the two is reassuring. Disagreement means one of the
approximations is strained, and which one is a question about the tie rate and
the skew of the differences — not a reason to reach for ``n_resamples``.

.. note::

   How far you can trust the t-test here is governed by ``n_nonzero``, not
   ``n_samples``. The Berry–Esseen bound on the normal approximation's error
   scales like :math:`1/\sqrt{n_{\text{nonzero}}}` for a difference
   distribution that is mostly zeros, and never better. Small ``n_nonzero``
   therefore makes the approximation hard to justify from that bound — which is
   not proof it is wrong, only that the usual reassurance is unavailable. A
   large ``n_nonzero`` is not sufficient either: a badly skewed nonzero part can
   still spoil it.

``alternative``
~~~~~~~~~~~~~~~

* ``"two-sided"`` (default) — is there any difference?
* ``"greater"`` — is the candidate better?
* ``"less"`` — is the candidate worse?

One-sided tests are more powerful, but you must commit to the direction **before
seeing the data**. Choosing ``"greater"`` after noticing your model won is not a
statistical test. Use ``"two-sided"`` unless you have a pre-registered reason.

The interval follows the alternative, so a one-sided test reports a one-sided
interval and the two cannot contradict each other.

``correction``
~~~~~~~~~~~~~~

* ``"holm"`` (default) — sequential, uniformly more powerful than Bonferroni
  with the same guarantee.
* ``"bonferroni"`` — multiplies every p-value by the number of hypotheses.
  Simpler to explain, more conservative.
* ``None`` — no adjustment. Honest only when you genuinely have one hypothesis,
  in which case :func:`~compresso_recsys.stats.compare_pair` is the better call.

Holm is the default because these hypotheses are **dependent**: they are
computed over overlapping users, and several metrics on one pair of models
measure closely related things. Holm and Bonferroni control the
family-wise error rate under arbitrary dependence. Procedures that assume
independence, such as Benjamini–Hochberg, do not apply here without further
argument and are deliberately not offered.

Reducing the number of hypotheses is usually better statistics than reducing the
correction. Nominating one primary metric in advance and reporting the rest
descriptively costs nothing and pre-empts the accusation that you tested until
something worked.

.. _stats-repeated-rows:

When one user owns several rows
-------------------------------

Everything above assumes one row per independent unit. Some evaluation
protocols give a user more than one.

``build_recsys_checkpoint(split_mode="user_split", eval_draws=5)`` splits each
held-out user's history into fold-in and scored parts five separate times, so
2,500 users produce 12,500 rows. That is deliberate and worth doing: averaging a
user's score over several draws removes the noise of *which* items were held
out, and on GoodBooks it gives intervals 35 to 42 percent narrower than a single
draw. What it does not give is more independent observations. Five draws of one
reader are still one reader.

**You do not have to do anything about it.** Comparison groups rows by
``sample_ids``, so repeated identifiers mean one unit produced several rows: it
resamples whole users, assigns one sign per user in the randomization test, and
runs the t-test on user means. ``n_units`` reports how many independent units
there were, and equals ``n_samples`` when every row is its own.

Had those rows been resampled as independent, every interval would have come out
27 to 44 percent too narrow, with p-values understated to match.

The one thing this needs from you is **real identifiers**. Without
``sample_ids`` the rows are numbered positionally, every row looks like its own
user, and the repetition becomes invisible. Pass the identifiers you have, and
read ``n_units`` against ``n_samples`` to see what the data actually contained.

.. note::

   The statistics literature calls this cluster sampling, and the references use
   that word. This guide says "unit" because :mod:`compresso.clustering` means
   something unrelated — grouping items into cluster graphs.

What the methods do, formally
-----------------------------

For users :math:`\mathcal{U}` with at least one relevant target, and per-user
metric values :math:`m_u^{(a)}` for model :math:`a`, the paired difference is

.. math::

   d_u = m_u^{(b)} - m_u^{(a)},
   \qquad
   \widehat{\Delta} = \frac{1}{n} \sum_{u \in \mathcal{U}} d_u
   = \widehat{M}^{(b)} - \widehat{M}^{(a)},

so the mean paired difference equals the difference of the reported aggregates,
provided both models were evaluated on the same users.

**Interval.** For replicate :math:`t`, draw indices
:math:`i_{t1}, \ldots, i_{tn}` uniformly with replacement and compute
:math:`\widehat{\Delta}^{*(t)} = n^{-1} \sum_j d_{i_{tj}}`. The reported
interval is the empirical
:math:`[\alpha/2,\, 1-\alpha/2]` quantile range of those replicates.

**Test.** Under the null that the two models are interchangeable for each user,
:math:`d_u` and :math:`-d_u` are equally likely. Draw signs
:math:`\varepsilon_{tu} \in \{-1, +1\}` uniformly, compute
:math:`\widehat{\Delta}^{0*(t)} = n^{-1} \sum_u \varepsilon_{tu} d_u`, and report

.. math::

   p = \frac{1 + \#\{t : |\widehat{\Delta}^{0*(t)}| \ge |\widehat{\Delta}|\}}{B + 1}.

The added one in numerator and denominator counts the observed arrangement among
the possibilities, which makes the test valid at finite :math:`B` and is why
:math:`p` is never zero.

**Correction.** For :math:`J` hypotheses with sorted p-values
:math:`p_{(1)} \le \cdots \le p_{(J)}`, Holm reports

.. math::

   \widetilde{p}_{(i)} = \min\!\left(1,\;
   \max_{1 \le j \le i} \left[(J - j + 1)\, p_{(j)}\right]\right),

restored to the original order. The running maximum ensures a hypothesis is
never easier to reject than one with a smaller p-value.

Reporting
---------

Methods
~~~~~~~

   For each user with at least one relevant test item we computed a per-user
   ranking metric and report its macro average over those users. All models
   were evaluated on identical users and target sets.

   To compare a candidate model against a baseline we formed paired per-user
   differences and report their mean. Statistical significance was assessed
   with a paired two-sided randomization test: under the null hypothesis that
   the two systems are interchangeable for each user, the sign of every paired
   difference is arbitrary, so we drew B uniform sign assignments, recomputed
   the mean difference under each, and report the Monte Carlo p-value given by
   one plus the number of assignments at least as extreme as the observed
   difference, divided by B plus one (Smucker et al., 2007; Davison and
   Hinkley, 1997).

   Uncertainty in the effect size is reported as a percentile confidence
   interval from a paired user-level bootstrap using the same B replicates
   (Efron, 1979). When testing several model and metric combinations together,
   p-values were adjusted with Holm's sequential procedure, which controls the
   family-wise error rate under arbitrary dependence between hypotheses (Holm,
   1979), and significance was assessed at the family-wise level 0.05.

   These intervals quantify variation across evaluation users conditional on
   the fitted model instances. Variation across training seeds is reported
   separately.

Results
~~~~~~~

Report the effect size first, and give the reader everything needed to judge
it without reading the code:

   ELSA improved nDCG@100 over EASE by 0.0054 (95% paired-bootstrap confidence
   interval [0.0030, 0.0080], a relative gain of 1.1%; paired randomization
   test with 9,999 resamples, Holm-adjusted p = 0.0004, the minimum attainable
   at that resample count; n = 2,500 users, of whom 2,495 scored differently
   under the two models).

Every number there does work. The effect size says how much. The interval says
how precisely. The relative gain makes it comparable across metrics. The test
and its resample count say how the p-value was produced and bound how small it
could have been — which is why the sentence reports the adjusted value and names
it as the floor, rather than writing ``p < 0.001`` and implying a bound the run
cannot support. The two counts separate how many users were evaluated from how
many the two models actually disagreed about.

Checklist
~~~~~~~~~

State all of these, every time:

#. the metric and cutoff, and its exact definition where more than one
   convention exists — ``calibrated_recall@k`` and ``map@k`` both normalize by
   :math:`\min(k, |\mathcal{R}_u|)`, which is not what every paper means;
#. the direction and magnitude of the difference, before any p-value;
#. the confidence interval and its level;
#. the sampling unit, with ``n_samples``, ``n_nonzero`` and — when one user
   owns several rows — ``n_units``;
#. the test, the number of resamples, and whether p-values were adjusted;
#. the seed spread for any stochastic model, beside the interval and named as a
   different quantity — see :ref:`stats-seeds`;
#. that inference is conditional on the fitted run, plus separate seed
   variability;
#. which metric was primary, declared in advance.

Avoid "the models are significantly different" without at least the first five.

.. _stats-references:

References
----------

The methods here are standard rather than novel, and citing them is part of
making a comparison defensible. Copy-ready BibTeX for all four is in
:doc:`citing`.

**The bootstrap.** Efron introduced resampling with replacement as a general
way to estimate the sampling distribution of a statistic. The confidence
interval reported by :mod:`compresso_recsys.stats` is the percentile form of
that idea, applied to the mean paired difference.

   Efron, B. (1979). Bootstrap Methods: Another Look at the Jackknife.
   *The Annals of Statistics*, 7(1), 1–26.
   `doi:10.1214/aos/1176344552 <https://doi.org/10.1214/aos/1176344552>`_

**The evaluation protocol.** Held-out users are scored under strong
generalization: part of each user's history is folded in to build their
representation and the rest is scored against, in the 80/20 proportion Liang et
al. describe. ``eval_holdout_frac`` is that scored share.

   Liang, D., Krishnan, R. G., Hoffman, M. D., & Jebara, T. (2018). Variational
   Autoencoders for Collaborative Filtering. *Proceedings of the 2018 World Wide
   Web Conference (WWW '18)*, 689–698.
   `doi:10.1145/3178876.3186150 <https://doi.org/10.1145/3178876.3186150>`_

**The randomization test in retrieval evaluation.** Smucker, Allan and
Carterette compared the randomization test, the bootstrap, the t-test, the
Wilcoxon signed-rank test and the sign test on retrieval data. They recommended
the randomization test, which is why it is the default here — but they also
found that the t-test **agrees closely** with it, and that the tests to avoid
are Wilcoxon signed-rank and the sign test, which discard the magnitude of each
difference. That is the citation behind both the default and the availability of
``test_method="t"``; see :ref:`stats-t-test`.

   Smucker, M. D., Allan, J., & Carterette, B. (2007). A Comparison of
   Statistical Significance Tests for Information Retrieval Evaluation.
   *Proceedings of the Sixteenth ACM Conference on Information and Knowledge
   Management (CIKM '07)*, 623–632.
   `doi:10.1145/1321440.1321528 <https://doi.org/10.1145/1321440.1321528>`_

**Why a Monte Carlo p-value is never zero.** Phipson and Smyth show that
computing a permutation p-value as the plain proportion of extreme resamples is
invalid — it can report zero, which no p-value may be — and derive the
``(1 + extreme) / (B + 1)`` form used here, which is exact rather than a
conservative patch. This is the reference for :ref:`stats-traps` on the floor.

   Phipson, B., & Smyth, G. K. (2010). Permutation P-values Should Never Be
   Zero: Calculating Exact P-values When Permutations Are Randomly Drawn.
   *Statistical Applications in Genetics and Molecular Biology*, 9(1), Article 39.
   `doi:10.2202/1544-6115.1585 <https://doi.org/10.2202/1544-6115.1585>`_

**Multiple comparisons.** Holm's sequential procedure controls the family-wise
error rate under arbitrary dependence between hypotheses, which is what makes
it appropriate when comparisons are computed over overlapping users and several
metrics on one pair of models measure closely related things.

   Holm, S. (1979). A Simple Sequentially Rejective Multiple Test Procedure.
   *Scandinavian Journal of Statistics*, 6(2), 65–70.

**Monte Carlo p-values.** Davison and Hinkley are the standard reference for
the ``(1 + extreme) / (B + 1)`` form and for choosing ``B`` so that
``alpha * (B + 1)`` is an integer — the reason the defaults are 9,999 rather
than 10,000.

   Davison, A. C., & Hinkley, D. V. (1997). *Bootstrap Methods and their
   Application*. Cambridge University Press.
   `doi:10.1017/CBO9780511802843 <https://doi.org/10.1017/CBO9780511802843>`_

See also
--------

* :doc:`api/statistics` — the API reference for
  :mod:`compresso_recsys.stats`.
* :doc:`api/evaluation` — :class:`~compresso_recsys.evaluation.EvaluationResult`
  and the per-user values this guide depends on.
* :doc:`citing` — BibTeX for the works above and for the models being compared.
