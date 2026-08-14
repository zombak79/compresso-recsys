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

Suppose two models score nDCG@100 of 0.2542 and 0.2538. Is the first better?

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

   elsa_result = evaluate_recommender(
       elsa, source=test_source, targets=test_targets,
       metrics=metrics, sample_ids=test_user_ids,
   )
   cselsa_result = evaluate_recommender(
       cselsa, source=test_source, targets=test_targets,
       metrics=metrics, sample_ids=test_user_ids,
   )

   report = compare_models(
       {"ELSA": elsa_result, "CSELSA": cselsa_result},
       metrics=["ndcg@100", "calibrated_recall@20"],
       reference="ELSA",
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

Reading the output
------------------

A row of ``report.to_frame()``, from a real comparison of two sparse
autoencoder configurations:

.. code-block:: text

   metric                    ndcg@100
   baseline                  SAE_f2048_k8
   candidate                 SAE_f4096_k8
   n_samples                 50000
   n_effective               48843
   baseline_mean             0.243466
   candidate_mean            0.254167
   difference                0.010701
   relative_difference       0.043952
   ci_low                    0.010162
   ci_high                   0.011225
   confidence_level          0.95
   bootstrap_standard_error  0.000272
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
candidate. Here the candidate gained 0.0107 nDCG@100, which
``relative_difference`` expresses as 4.4% of the baseline.

This is the number to lead with. It is computed directly from the data, so it
does not depend on any of the resampling settings.

How precisely you know it
~~~~~~~~~~~~~~~~~~~~~~~~~

``ci_low`` and ``ci_high`` give a range of plausible values for the true
difference — here 0.0102 to 0.0112 at ``confidence_level`` 0.95.

That range comes from a **bootstrap**. Take your 50,000 users and draw 50,000
of them *with replacement* — meaning the same user can be drawn more than once.

That last phrase is doing all the work. Drawing 50,000 from 50,000 sounds like
it must return the same set, and it would if each user could be drawn only once.
With replacement, a typical draw looks like this:

.. code-block:: text

   31,630 distinct users (63.3%)
   18,370 never picked
   13,223 picked two or more times, one as often as seven

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
because ranking metrics produce enormous numbers of exact ties. In the run
above, ``ndcg@100`` differed for 98% of users, but ``calibrated_recall@20``
differed for only 44% — for the other 56%, both models retrieved the same
number of relevant items in the top 20, usually zero. Those users contribute
nothing.

So a comparison with ``n_samples`` of 50,000 and ``n_effective`` of 300 is a
comparison resting on 300 observations. Report both.

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

So ``p_value`` of 0.0001 does **not** mean one in ten thousand. It means *not
one* of 9,999 random rearrangements matched your result — the true p is below
this run's resolution. Write ``p < 10^-4``, not ``p = 0.0001``.

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

This creates a disagreement that looks like a bug and is not:

.. code-block:: text

   ndcg@100:  ci = [-0.000753, -0.000018]   excludes zero
              adjusted_p_value = 0.0990      not significant

Both are correct. **Intervals are never adjusted** — each describes one
comparison on its own. **p-values are adjusted.** Here the raw p was 0.0495,
just under the threshold, and correcting for four hypotheses pushed it to
0.0990.

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

At 50,000 users and four hypotheses:

.. list-table::
   :header-rows: 1

   * - ``n_resamples``
     - randomization
     - bootstrap
     - p resolution
   * - 999
     - 0.7 s
     - 0.5 s
     - 0.001
   * - 9999
     - 7.1 s
     - 4.7 s
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

Methods paragraph
~~~~~~~~~~~~~~~~~

Adapt this rather than paraphrasing it, so the description stays matched to what
the code does:

.. code-block:: latex

   For each user $u$ with at least one relevant test item we computed a
   per-user ranking metric $m_u^{(a)}@k$ for model $a$ and report the macro
   average $\widehat{M}^{(a)}@k = |\mathcal{U}|^{-1}\sum_{u\in\mathcal{U}}
   m_u^{(a)}@k$. All models were evaluated on identical users and target
   sets. To compare a candidate $b$ against a baseline $a$ we formed paired
   per-user differences $d_u = m_u^{(b)}@k - m_u^{(a)}@k$ and report their
   mean $\widehat{\Delta}_{b,a}$. Statistical significance was assessed with
   a paired two-sided randomization test: under the null that the two
   systems are interchangeable for each user, we drew $B$ uniform sign
   assignments $\varepsilon_u\in\{-1,+1\}$, recomputed
   $|\mathcal{U}|^{-1}\sum_u \varepsilon_u d_u$, and report the Monte Carlo
   p-value $(1+\#\{|\cdot|\ge|\widehat{\Delta}_{b,a}|\})/(B+1)$. Uncertainty
   in the effect size is reported as a percentile interval from a paired
   user-level bootstrap with the same $B$ replicates. When testing multiple
   model--metric combinations, p-values were adjusted with Holm's sequential
   procedure, which is valid under arbitrary dependence between hypotheses;
   significance was assessed at the family-wise level $\alpha=0.05$. These
   intervals quantify variation across evaluation users conditional on the
   fitted model instances; variation across training seeds is reported
   separately.

Result sentence
~~~~~~~~~~~~~~~

.. code-block:: text

   SAE_f4096_k8 improved nDCG@100 over SAE_f2048_k8 by 0.0107
   (95% paired-bootstrap CI [0.0102, 0.0112], +4.4% relative;
   paired randomization test, B = 9,999, Holm-adjusted p < 10^-3;
   n = 50,000 users, 48,843 with a nonzero difference).

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

See also
--------

* :doc:`api/statistics` — the API reference for
  :mod:`compresso_recsys.stats`.
* :doc:`api/evaluation` — :class:`~compresso_recsys.evaluation.EvaluationResult`
  and the per-user values this guide depends on.
