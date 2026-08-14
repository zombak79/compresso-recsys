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

Suppose EASE scores nDCG@100 of 0.4837 on GoodBooks and ELSA scores 0.4887.
Is ELSA better?

The honest answer is that you cannot tell from those two numbers. They are
averages over tens of thousands of users, and averages hide how much the two
models actually disagreed. Perhaps one model won for nearly every user by a
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
       seed=42,
   )

Roughly 40 seconds, most of it downloading. A user split holds out whole users,
so both models see the same 9,975 items and are evaluated on 12,500 unseen
users.

Train both models
~~~~~~~~~~~~~~~~~

.. code-block:: python

   import torch
   from compresso_recsys.models import EASE, EASEConfig, ELSAConfig, ELSATrainer

   with cr.read_checkpoint("artifacts/goodbooks/comparison.zip") as root:
       split = cr.load_recsys_split(root)

   x_train = split["x_train"]        # (49865, 9975), 3.8M interactions
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

   EASE {'calibrated_recall@20': 0.3288, 'calibrated_recall@50': 0.4728,
         'recall@20': 0.3191, 'recall@50': 0.4728,
         'ndcg@100': 0.4837, 'mrr@20': 0.6842}
   ELSA {'calibrated_recall@20': 0.3402, 'calibrated_recall@50': 0.4778,
         'recall@20': 0.3304, 'recall@50': 0.4778,
         'ndcg@100': 0.4887, 'mrr@20': 0.7043}

ELSA is ahead everywhere. Whether that means anything is the next step — but
first, one thing in that output is worth pausing on.

.. note::

   ``recall@20`` and ``calibrated_recall@20`` differ (0.3191 against 0.3288),
   while ``recall@50`` and ``calibrated_recall@50`` are *identical* (0.4728).

   Calibrated recall divides by :math:`\min(k, |\mathcal{R}_u|)` rather than
   :math:`|\mathcal{R}_u|`, so the two coincide exactly when no user has more
   targets than the cutoff. Here users hold between 2 and 35 targets: 17% exceed
   20, and none exceed 50.

   This is why the metric keys were separated. Quoting a
   ``calibrated_recall@20`` of 0.3288 against a published Recall@20 would
   overstate the result by 3%, and on a denser holdout the gap grows without
   bound.

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

   metric                n_effective  baseline_mean  candidate_mean  difference  relative    ci_low   ci_high  adj_p   direction
   ndcg@100                    12480       0.483718        0.488702    0.004985     1.03%  0.003808  0.006139  0.0004  better
   calibrated_recall@20         7126       0.328782        0.340230    0.011448     3.48%  0.010031  0.012853  0.0004  better
   recall@20                    7126       0.319140        0.330426    0.011286     3.54%  0.009921  0.012652  0.0004  better
   mrr@20                       5227       0.684200        0.704307    0.020107     2.94%  0.015345  0.024781  0.0004  better

Reading it
~~~~~~~~~~

**ELSA beats EASE, and the margin is small but solid.** Every interval sits well
clear of zero, and every adjusted p-value is at the floor for 9,999 resamples.

**The relative gains disagree, and that is informative.** nDCG@100 improves by
1.0%, recall@20 by 3.5%, MRR@20 by 2.9%. ELSA is noticeably better near the top
of the list; across the full hundred ranks the two are much closer. A paper
reporting only nDCG@100 would understate what changed, and one reporting only
recall@20 would overstate it.

**Look at ``n_effective``.** For nDCG@100 the models differ for 12,480 of the
12,500 users — essentially everyone, because a metric reading 100 ranks deep
notices almost any reordering. For recall@20 they differ for 7,126, and for
MRR@20 only 5,227. So **58% of users get an identical MRR from both models**,
because their first relevant book lands at the same rank either way. That
comparison rests on 5,227 observations, not 12,500.

**What this does not establish.** Both models were trained once. A 1% nDCG
difference is well inside what a different random seed could move for ELSA,
which is gradient-trained; EASE has no seed at all, being closed-form. To claim
ELSA is the better *method* rather than that this ELSA beat this EASE, train
several seeded runs and report their spread alongside these intervals.

Reading the output
------------------

One row of the ``report.to_frame()`` above, laid out vertically:

.. code-block:: text

   metric                    ndcg@100
   baseline                  EASE
   candidate                 ELSA
   n_samples                 12500
   n_effective               12480
   baseline_mean             0.483718
   candidate_mean            0.488702
   difference                0.004985
   relative_difference       0.010305
   ci_low                    0.003808
   ci_high                   0.006139
   confidence_level          0.95
   bootstrap_standard_error  0.000589
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
difference — here 0.0038 to 0.0061 at ``confidence_level`` 0.95.

That range comes from a **bootstrap**. Take your 12,500 users and draw 12,500
of them *with replacement* — meaning the same user can be drawn more than once.

That last phrase is doing all the work. Drawing 12,500 from 12,500 sounds like
it must return the same set, and it would if each user could be drawn only once.
With replacement, a typical draw looks like this:

.. code-block:: text

    7,907 distinct users (63.3%)
    4,593 never picked
    3,252 picked two or more times, one as often as seven

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

How many users the comparison rests on
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``n_samples`` is the users evaluated. ``n_effective`` is the users whose score
actually *differed* between the two models.

That distinction matters more in recommendation than almost anywhere else,
because ranking metrics produce enormous numbers of exact ties. Across the
EASE-versus-ELSA comparison, from the same 12,500 users:

.. code-block:: text

   ndcg@100               12,480 of 12,500  (100%)
   calibrated_recall@20    7,126 of 12,500   (57%)
   recall@20               7,126 of 12,500   (57%)
   mrr@20                  5,227 of 12,500   (42%)

A metric reading 100 ranks deep notices almost any reordering, so nearly every
user counts. MRR@20 depends only on where the *first* relevant book lands, so
for 58% of users it lands in the same place under both models and they tell you
nothing.

The MRR comparison rests on 5,227 observations, not 12,500. Report both
numbers, because ``n_samples`` alone will overstate your evidence by more than
a factor of two.

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
thousand. It means *not one* of 9,999 random sign assignments matched what ELSA
achieved — the true p is below this run's resolution. Write ``p < 10^-4``, not
``p = 0.0001``.

The floor also interacts with the correction. With ``J`` hypotheses the
smallest reachable adjusted p is ``J / (B + 1)``. At ``B = 100`` and six
hypotheses that is 0.059, so **nothing could be significant at 0.05 no matter
what the data said**. Never set ``n_resamples`` below 999.

``n_effective`` is your real sample size
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Covered above, and worth repeating because it is invisible in any aggregate
table. When ``n_effective`` falls below 30, the comparison emits a warning:
percentile intervals need a reasonable number of informative observations, and
below that the interval is indicative rather than trustworthy.

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
EASE-versus-ELSA comparison on the first 400 test users instead of all 12,500,
and four metrics land here at once:

.. code-block:: text

   metric                difference    ci_low   ci_high  p_value  adjusted_p  significant
   ndcg@100                0.006569  0.000586  0.012794   0.0307      0.0784        False
   calibrated_recall@20    0.009656  0.001624  0.017774   0.0196      0.0784        False
   recall@20               0.009425  0.001566  0.017467   0.0208      0.0784        False
   mrr@20                  0.027745  0.002900  0.052719   0.0313      0.0784        False

Every interval excludes zero. Not one is significant.

Both are correct. **Intervals are never adjusted** — each describes one
comparison on its own, and on its own each of these would clear 0.05.
**p-values are adjusted.** Testing four of them together multiplies the
smallest by four, and 0.0196 becomes 0.0784.

The same models on all 12,500 users are significant on every metric. Nothing
about the models changed; the evidence did.

Never describe these intervals as simultaneous or family-wise. They are not.

``confidence_level`` also moves the significance bar
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

One parameter controls both, because ``alpha = 1 - confidence_level``. Setting
``confidence_level=0.99`` widens the interval **and** tightens the test from
0.05 to 0.01, which will make results disappear that were significant before.

They are coupled deliberately: a 99% interval printed beside a 5% test would
produce rows where the interval and the verdict contradict each other by
construction. But it does mean 0.99 is a bigger change than it looks.

This conditions on one training run
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Everything here resamples **users**. It answers whether an advantage is stable
across different samples of users from this population.

It says nothing about whether the advantage survives retraining with a different
random seed. If your training is stochastic — and gradient-trained models are —
a reviewer will ask, and "we bootstrapped over users" is not an answer.

The standard response is to train each model under five or more matched seeds,
evaluate every seed on identical users and targets, and report the mean and
standard deviation across seeds alongside the per-user interval. State plainly
that the two quantify different things.

This matters most when the effect is small. A difference of 0.4% may be
statistically distinguishable across users and still well inside what reseeding
would move.

Choosing the settings
---------------------

``n_resamples``
~~~~~~~~~~~~~~~

Controls only the p-value's resolution and the interval's Monte Carlo noise. It
never touches ``difference``, ``baseline_mean``, ``candidate_mean`` or
``n_effective``, which are computed from the data. So you can iterate cheaply
and pay for precision once.

At the worked example's scale — 12,500 users, one model pair, four metrics:

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

Holm is the default because these hypotheses are **dependent**: they share
resampling draws and overlapping users. Holm and Bonferroni control the
family-wise error rate under arbitrary dependence. Procedures that assume
independence, such as Benjamini–Hochberg, do not apply here without further
argument and are deliberately not offered.

Reducing the number of hypotheses is usually better statistics than reducing the
correction. Nominating one primary metric in advance and reporting the rest
descriptively costs nothing and pre-empts the accusation that you tested until
something worked.

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

   ELSA improved nDCG@100 over EASE by 0.0050 (95% paired-bootstrap confidence
   interval [0.0038, 0.0061], a relative gain of 1.0%; paired randomization
   test with 9,999 resamples, Holm-adjusted p < 0.001; n = 12,500 users, of
   whom 12,480 had a nonzero difference).

Every number there does work. The effect size says how much. The interval says
how precisely. The relative gain makes it comparable across metrics. The test
and its resample count say how the p-value was produced and bound how small it
could have been. And the two counts distinguish how many users were evaluated
from how many carried any information.

Checklist
~~~~~~~~~

State all of these, every time:

#. the metric and cutoff, and its exact definition where more than one
   convention exists — ``calibrated_recall@k`` and ``map@k`` both normalize by
   :math:`\min(k, |\mathcal{R}_u|)`, which is not what every paper means;
#. the direction and magnitude of the difference, before any p-value;
#. the confidence interval and its level;
#. the sampling unit, with both ``n_samples`` and ``n_effective``;
#. the test, the number of resamples, and whether p-values were adjusted;
#. that inference is conditional on the fitted run, plus separate seed
   variability;
#. which metric was primary, declared in advance.

Avoid "the models are significantly different" without at least the first five.

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

**The randomization test in retrieval evaluation.** Smucker, Allan and
Carterette compared the randomization test, the bootstrap, the t-test, the
Wilcoxon signed-rank test and the sign test on retrieval data, and recommended
the randomization test. That is why it is the default here, and citing it
answers the reviewer question of why this test rather than another.

   Smucker, M. D., Allan, J., & Carterette, B. (2007). A Comparison of
   Statistical Significance Tests for Information Retrieval Evaluation.
   *Proceedings of the Sixteenth ACM Conference on Information and Knowledge
   Management (CIKM '07)*, 623–632.
   `doi:10.1145/1321440.1321528 <https://doi.org/10.1145/1321440.1321528>`_

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
