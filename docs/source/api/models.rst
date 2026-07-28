Models API
==========

Recommender Contract
--------------------

Models evaluated by :func:`compresso_recsys.evaluation.evaluate_recommender`
implement the small :class:`compresso_recsys.models.Recommender` protocol.
The required ``predict_on_batch(source, *, k, exclude_seen=True)`` method
returns ranked top-``k`` predictions as an :class:`compresso.SRPTensor`.
EASE and ELSA exclude items already present in the source interactions by
default. Pass ``exclude_seen=False`` to inspect rankings that may contain
previously interacted items.

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

ELSA
----

ELSA learns a low-rank matrix of normalized item embeddings with a shallow
linear autoencoder objective. Unlike EASE, its model size grows linearly with
the number of items, making it suitable for larger catalogs and GPU training.

During training, ``max_output`` can limit each batch's output candidates. All
items with positive interactions in the batch are always retained, and the
remaining candidate budget is sampled without replacement from items absent
from the whole batch. This makes ``max_output`` a soft upper bound when a
batch contains more distinct positive items than the configured limit. Use
``None`` to score the complete catalog during training.

Compressed ELSA uses Compresso's lottery-ticket schedule to retain a fixed
number of latent values per item. During mask search, each stable stage rewinds
the item factors to their original initialization under the new mask and
restarts the optimizer. After the final mask stabilizes, the factors are
converted to a row-packed sparse parameter and only its values are trained for
``ELSAConfig.epochs``. Learning-rate decay, when enabled, applies only to this
final sparse fine-tuning phase.

Mask search has no epoch limit: a stage advances only after its mask change
stays within ``change_threshold`` for ``stability_window`` updates. Training
checkpoints during this phase are not currently resumable, and
``torch.compile`` is not supported. Mask search materializes the complete
masked factor matrix before row selection; sparse fine-tuning materializes the
complete SRP factor matrix before row selection. Inference does not densify the
complete matrix and instead uses normalized SRP factors through a CSR matrix
multiplication.

The current Compresso conversion can be ambiguous when boundary values are
tied or zero. Compressed ELSA verifies the exported structure against the final
mask and raises an error instead of silently changing the ticket. Compresso
also currently moves its initialization copy with the model, so mask search
temporarily retains an additional dense factor buffer on the training device.

.. autoclass:: compresso_recsys.models.ELSAConfig
   :members:

.. autoclass:: compresso_recsys.models.ELSACompressionConfig
   :members:

.. autoclass:: compresso_recsys.models.ELSA
   :members:

.. autoclass:: compresso_recsys.models.CompressedELSA
   :members:

.. autoclass:: compresso_recsys.models.ELSATrainer
   :members:
