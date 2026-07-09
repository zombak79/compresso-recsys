Core API
========

Datasets
--------

.. autoclass:: compresso_recsys.SplitBundle
   :members:

.. autoclass:: compresso_recsys.RecSysDataset
   :members:

.. autoclass:: compresso_recsys.Goodbooks
   :members:

.. autoclass:: compresso_recsys.MovieLens1M
   :members:

.. autoclass:: compresso_recsys.MovieLens20M
   :members:

.. autoclass:: compresso_recsys.AmazonReviews2023
   :members:

Checkpoint Helpers
------------------

.. autofunction:: compresso_recsys.build_recsys_checkpoint

.. autofunction:: compresso_recsys.update_checkpoint

.. autofunction:: compresso_recsys.read_checkpoint

.. autofunction:: compresso_recsys.load_manifest

.. autofunction:: compresso_recsys.save_manifest

.. autofunction:: compresso_recsys.update_stage_manifest

.. autofunction:: compresso_recsys.save_json

.. autofunction:: compresso_recsys.load_json

.. autofunction:: compresso_recsys.save_recsys_split

.. autofunction:: compresso_recsys.load_recsys_split

.. autofunction:: compresso_recsys.save_cluster_graph_stage

.. autofunction:: compresso_recsys.load_cluster_graph_stage

Retrieval API
-------------

.. autofunction:: compresso_recsys.retrieval.build_eval_holdout

.. autofunction:: compresso_recsys.retrieval.build_item_cold_holdout

.. autofunction:: compresso_recsys.retrieval.build_leave_last_out_holdout

.. autofunction:: compresso_recsys.retrieval.build_temporal_holdout

.. autofunction:: compresso_recsys.retrieval.evaluate_item_embeddings

.. autofunction:: compresso_recsys.retrieval.evaluate_item_embeddings_with_holdout

