Statistics API
==============

Paired comparison of evaluations produced by
:mod:`compresso_recsys.evaluation`. For an introduction, how to read the output
and how to report it, see :doc:`../statistical-comparison`; this page is the
reference.

Every function here works on the paired per-user difference between two models
evaluated on identical users, and requires results carrying per-user values.

Comparison Functions
--------------------

.. autofunction:: compresso_recsys.stats.compare_models

.. autofunction:: compresso_recsys.stats.compare_pair

Results
-------

.. autoclass:: compresso_recsys.stats.PairwiseComparison
   :members:

.. autoclass:: compresso_recsys.stats.ComparisonReport
   :members:

Parameter Values
----------------

``alternative``
   ``"two-sided"``, ``"greater"`` or ``"less"``. Selects the question being
   asked and the orientation of the reported interval.

``correction``
   ``"holm"``, ``"bonferroni"`` or ``None``. Applied across every pair and
   metric produced by one :func:`~compresso_recsys.stats.compare_models` call.

``test_method``
   ``"randomization"`` for the paired sign-flip test, or ``"bootstrap"`` for the
   null-centred bootstrap.

``n_resamples``
   Number of resampling replicates. The smallest achievable p-value is
   ``1 / (n_resamples + 1)``; with ``J`` hypotheses the smallest achievable
   adjusted p-value is ``J / (n_resamples + 1)``.

``confidence_level``
   Interval level, and through ``alpha = 1 - confidence_level`` the significance
   threshold as well.
