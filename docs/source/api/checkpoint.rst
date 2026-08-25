Checkpoint API
==============

Checkpoint Contexts
-------------------

.. autofunction:: compresso_recsys.checkpoint.update_checkpoint
   :no-index:

.. autofunction:: compresso_recsys.checkpoint.read_checkpoint
   :no-index:

Manifest and JSON Helpers
-------------------------

.. autofunction:: compresso_recsys.checkpoint.load_manifest
   :no-index:

.. autofunction:: compresso_recsys.checkpoint.save_manifest
   :no-index:

.. autofunction:: compresso_recsys.checkpoint.update_stage_manifest
   :no-index:

.. autofunction:: compresso_recsys.checkpoint.save_json
   :no-index:

.. autofunction:: compresso_recsys.checkpoint.load_json
   :no-index:

Split and Cluster Stages
------------------------

Item partitions
~~~~~~~~~~~~~~~

``warm_item_indices``, ``val_cold_item_indices`` and ``test_cold_item_indices`` are
positions into ``item_ids`` naming the items **each phase introduces**, not the
items it may score:

.. list-table::
   :header-rows: 1
   :widths: 20 40 40

   * - ``split_mode``
     - Train partition
     - Val / test partitions
   * - ``user_split``
     - Full catalog range
     - Empty; no items are held out
   * - ``item_split``
     - Warm items
     - The disjoint cold items held out of training
   * - ``leave_last_out``
     - Every item; nothing is withheld from the catalog
     - Only items whose every occurrence falls in a held-out tail
   * - ``temporal``
     - Items in the first window
     - Items first seen in each later window

An empty partition means *this phase introduces no new items*, which is not the
same as *this phase has no candidates*. A user split scores the whole catalog in
every phase; it simply adds nothing new. The candidate space of a phase is
``{phase}_item_ids``, which equals ``item_ids`` unless the split gives each
phase its own item space (only ``temporal`` does, flagged by
``has_stage_item_spaces`` in the split metadata).

So to select feature or metadata rows for a phase, index with that phase's
``*_item_ids``, not by mirroring ``warm_item_indices``: for splits that hold no
items out, the latter yields an empty selection that fails much later and far
from its cause. ``has_item_partitions`` in the split metadata tells you whether
a split partitions items at all.

Chronological split modes additionally store sequence views —
``x_train_sequences`` and ``{stage}_source_sequences`` — holding the same events
as the matrices in order, with duplicates preserved. They load as ``None`` for
``user_split`` and ``item_split``, and for any checkpoint built before sequences
existed.

.. autofunction:: compresso_recsys.checkpoint.save_recsys_split
   :no-index:

.. autofunction:: compresso_recsys.checkpoint.load_recsys_split
   :no-index:

.. autofunction:: compresso_recsys.checkpoint.save_cluster_graph_stage
   :no-index:

.. autofunction:: compresso_recsys.checkpoint.load_cluster_graph_stage
   :no-index:
