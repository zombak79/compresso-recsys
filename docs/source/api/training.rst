Training API
============

Validation between epochs and early stopping on it, for the trainers that run
their epochs through the shared progress loop: ``MultDAETrainer``,
``MultVAETrainer``, ``SASRecTrainer``, ``SimpleGPTTrainer``,
``SimpleRNNTrainer``, ``SimpleBidirectionalTransformerTrainer`` and
``TEASERGDTrainer``. Each accepts ``validation=`` and ``early_stopping=`` in
``fit``::

    from compresso_recsys import EarlyStopping, NDCG, Validation

    trainer.fit(
        split["x_train"],
        validation=Validation(val_source, val_targets, [NDCG(100)],
                              sample_ids=val_user_ids),
        early_stopping=EarlyStopping("ndcg@100", patience=5),
    )
    trainer.best_   # {"epoch": ..., "val_ndcg@100": ...}

The configured ``epochs`` becomes the maximum. Validation metrics are stored in
``trainer.history`` as ``val_<metric>`` keys, and validation draws no random
numbers, so a run with validation but without early stopping trains exactly as
one without validation. Learning-rate schedules that span the configured epochs
are not shortened when a run stops early.

``ELSATrainer`` and ``Bert4RecTrainer`` run their own epoch loops and do not
accept these arguments yet.

.. autoclass:: compresso_recsys.training.Validation

.. autoclass:: compresso_recsys.training.EarlyStopping
