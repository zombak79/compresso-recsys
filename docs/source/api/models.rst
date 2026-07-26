Models API
==========

Recommender Contract
--------------------

Models evaluated by :func:`compresso_recsys.evaluation.evaluate_recommender`
implement the small :class:`compresso_recsys.models.Recommender` protocol.
The required ``predict_on_batch(source, *, k)`` method returns ranked
top-``k`` predictions as an :class:`compresso.SRPTensor`. Implementations must
exclude items already present in the source interactions.

.. autoclass:: compresso_recsys.models.Recommender
   :members:

EASE
----

EASE is a closed-form collaborative-filtering model. Fitting creates a dense
item-by-item coefficient matrix, so its memory use grows quadratically with
the number of items. ``float32`` is the memory-efficient default. Select
``float64`` explicitly when additional numerical precision is more important
than fit and prediction speed.

.. autoclass:: compresso_recsys.models.EASEConfig
   :members:

.. autoclass:: compresso_recsys.models.EASE
   :members:
