"""Paired statistical comparison of recommender evaluations.

Two models evaluated on the same users can be compared far more precisely than
their aggregate means suggest, because most of the variation between users is
shared. Everything here works on the paired per-user difference

.. math::

    d_u = m_u^{(b)} - m_u^{(a)},

and never resamples the two models independently.

Two procedures, each doing the job it is best at:

* **Effect size** — a paired bootstrap over users gives a confidence interval
  for the mean difference. This is the primary output; report it.
* **Hypothesis test** — a paired sign-flip randomization test gives the
  p-value. Its null is *paired label exchangeability*: swapping which model
  produced which score, independently for each user, leaves the joint
  distribution unchanged. Under it the sign of every paired difference is
  arbitrary, so the test is exact up to Monte Carlo error. That assumes no
  parametric family, which is not the same as assuming nothing — exchangeability
  is a real assumption, and it requires that users are the independent units
  being resampled. It is the default in the information-retrieval evaluation
  literature.

``test_method="t"`` runs a paired t-test instead, as a one-sample test on the
same differences. Smucker, Allan and Carterette found the two agree closely on
retrieval data, so it is a familiar cross-check.

The randomization test is the default because it avoids the normal
approximation entirely wherever paired-label exchangeability is defensible.
Choose the t-test when its null is the one you want -- a zero population mean
difference, without the symmetry that exchangeability additionally implies --
and when enough untied units make the normal approximation credible. Not
because it prints a smaller number: having no Monte Carlo floor is a property
of the procedure, not evidence about the models.

Every procedure works on one difference per independent unit. When a protocol
gives a user several evaluation rows, that user is first reduced to the mean of
their rows, so one user is one observation however many rows they produced.
``n_samples`` counts rows, ``n_units`` counts users, and they are equal whenever
each row is its own unit.

Ranking differences are dominated by exact ties: for most users both models
return the same items and the difference is zero. Every comparison therefore
reports ``n_nonzero`` and ``tie_rate`` over units.

Tied users are not spare. They carry no *sign* information — flipping the sign
of a zero changes nothing, so ``n_nonzero`` alone governs the combinatorial
support of the randomization test. But they are part of the empirical
population, and the mean difference and the paired bootstrap interval are
computed over all ``n_units`` of them. Thirty users who all differ by +1 and
ten thousand users of whom thirty differ by +1 share an ``n_nonzero`` and
describe entirely different systems.

Inference here is conditional on the fitted models. It answers whether an
advantage is stable across resampled users, not whether it survives retraining
with a different seed. Report seed variation separately.
"""

from compresso_recsys.stats.results import (
    Alternative,
    ComparisonReport,
    Correction,
    PairwiseComparison,
    TestMethod,
)
from compresso_recsys.stats.compare import compare_models, compare_pair

__all__ = [
    "ComparisonReport",
    "PairwiseComparison",
    "compare_models",
    "compare_pair",
]
