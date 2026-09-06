.. _dataset-validation:

Dataset and model validation matrix
========================================

This is a validation workbench, **not a claim of reproduced results**. Published
scores below are reference targets; no full-data training runs have been measured
for this matrix yet. Small numerical differences are expected, but different
architectures, candidate sets, or split protocols are not rounding differences.

Runnable code
-------------

Download :download:`benchmark.py <../../examples/validation/benchmark.py>`
or run ``examples/validation/benchmark.py`` from the repository root after
installing the package and its model dependencies. Its ``build`` function builds
the checkpoint; ``train`` fits candidates, selects by validation NDCG@100, and
evaluates the selected model on test only once. JSON output records configurations,
validation/test scores, matrix dimensions, package version, seed, and checkpoint
SHA-256. It does not save trained model weights.

.. code-block:: bash

   python examples/validation/benchmark.py build --dataset ml20m --checkpoint artifacts/validation/ml20m.zip
   python examples/validation/benchmark.py train --checkpoint artifacts/validation/ml20m.zip --model ease --l2 200 500 1000 --output artifacts/validation/ml20m-ease.json
   python examples/validation/benchmark.py train --checkpoint artifacts/validation/ml20m.zip --model multvae --epochs 20 50 100 --output artifacts/validation/ml20m-multvae.json
   python examples/validation/benchmark.py train --checkpoint artifacts/validation/ml20m.zip --model multdae --epochs 20 50 100 --output artifacts/validation/ml20m-multdae.json

Replace ``ml20m`` in the build command and artifact paths with ``netflix`` or
``taste-profile`` for the other rows. Epoch candidates are independent fits from
the same seed, not resumed runs. These grids are starting points, not the papers'
complete hyperparameter searches. Extend them using validation results only.
Use ``--device cuda`` when available for neural models; CPU is the default.

Published reference results
---------------------------

All nine rows use full-catalog ranking with observed source items excluded.
R20/R50 mean the papers' Recall@20/50: hits divided by
``min(cutoff, number of relevant items)``. The runner therefore uses
``CalibratedRecall``, **not** ``Recall``. N100 is binary-relevance NDCG@100.
Every row uses the builder and training code linked above.

.. list-table:: Reference targets, not locally measured results
   :header-rows: 1
   :widths: 18 15 10 10 10 20

   * - Paper / model
     - Build dataset
     - R20
     - R50
     - N100
     - Train model / status
   * - EASE_, Table 1
     - ml20m
     - 0.391
     - 0.521
     - 0.420
     - ease / adapted split
   * - EASE_, Table 1
     - netflix
     - 0.362
     - 0.445
     - 0.393
     - ease / adapted split
   * - EASE_, Table 1
     - taste-profile
     - 0.333
     - 0.428
     - 0.389
     - ease / adapted split
   * - MultVAE_, Table 2
     - ml20m
     - 0.395
     - 0.537
     - 0.426
     - multvae / adapted split
   * - MultVAE_, Table 2
     - netflix
     - 0.351
     - 0.444
     - 0.386
     - multvae / adapted split
   * - MultVAE_, Table 2
     - taste-profile
     - 0.266
     - 0.364
     - 0.316
     - multvae / adapted split
   * - MultVAE_ (MultDAE), Table 2
     - ml20m
     - 0.387
     - 0.524
     - 0.419
     - multdae / architecture differs
   * - MultVAE_ (MultDAE), Table 2
     - netflix
     - 0.344
     - 0.438
     - 0.380
     - multdae / architecture differs
   * - MultVAE_ (MultDAE), Table 2
     - taste-profile
     - 0.266
     - 0.363
     - 0.313
     - multdae / architecture differs

.. _EASE: https://arxiv.org/html/1905.03375
.. _MultVAE: https://arxiv.org/html/1802.05814

For original preprocessing and neural training code, see the authors'
`vae_cf repository <https://github.com/dawenl/vae_cf>`_. EASE's paper includes
the closed-form training algorithm. Both papers report standard errors around
0.002 for ML-20M and 0.001 for Netflix/MSD; these are not acceptance tolerances
for a changed protocol.

Checkpoint protocol and limitations
-----------------------------------

.. list-table:: Explicit builder settings
   :header-rows: 1

   * - Dataset
     - Positive interaction
     - Min user / item support
     - Validation / test users
   * - ML-20M
     - Rating >= 4, then binary
     - 5 / 1
     - 10,000 / 10,000
   * - Netflix
     - Rating >= 4, then binary
     - 5 / 1
     - 40,000 / 40,000
   * - MSD Taste Profile
     - Any play count, then binary
     - 20 / 200
     - 50,000 / 50,000

The recipe fixes seed 98765, disjoint held-out users, an 80/20 source/target
split, one evaluation draw, and no minimum metadata-text length. This overrides
the normal ML-20M builder defaults. Netflix requires the source files to be
available locally; consult :doc:`api/datasets` for download availability and terms.

The current builder is paper-inspired, not the authors' preprocessing program:
filter ordering, iterative support pruning, random-number generation, catalog
construction, and holdout rounding can change membership. Compare generated
counts and saved manifests before interpreting score gaps. For strict replication,
import the authors' exact split through :doc:`bring-your-own-dataset` and preserve
their train-derived item vocabulary and evaluation eligibility rules.

MultVAE uses a 600-hidden/200-latent architecture. Our MultDAE is
``items -> 200 -> items``, whereas the main Table 2 reference uses a deeper
network. Its scores are contextual references, not valid pass/fail targets for
our shallower model. Training schedules and validation selection also need
alignment before claiming replication.

EASE forms dense item-by-item matrices. One float64 matrix at 41,140 items is
about 13.5 GB; inversion, retained models during validation selection, and
workspace require substantially more. Do not start the MSD EASE grid on a
memory-constrained machine. Neural recipes stream sparse training minibatches.

Sequential, cold-start, and other models
---------------------------------------------

.. list-table:: Additional coverage and honest comparison boundaries
   :header-rows: 1
   :widths: 18 27 27 28

   * - Model / dataset
     - Paper / upstream code
     - Local builder and training code
     - Published target / status
   * - SASRec / ML-1M
     - `SASRec paper <https://arxiv.org/abs/1808.09781>`_;
       `authors' code <https://github.com/kang205/SASRec>`_
     - :doc:`reproducing-sasrec-results-on-ml1m` contains both
     - HR@10 0.8245, NDCG@10 0.5905 (Table III).
       Sampled-negative protocol only; see notebook caveats.
   * - EASE / Steam
     - EASE_; no Steam target in that paper
     - Same script: build ``--dataset steam``, train ``--model ease``
     - Runnable warm-start baseline; establish a new measured baseline
   * - Steam cold-start / temporal / LLO
     - :ref:`cite-datasets`
     - :doc:`cli-reference` and :doc:`api/datasets` describe checkpoint construction
     - No matched published target established; CF script is not a cold-start trainer
   * - ELSA
     - `Authors' reproduction code <https://github.com/recombee/ELSA/tree/reproduce_movielens>`_;
       paper in :doc:`citing`
     - ``ELSATrainer`` in :doc:`api/models`; dedicated recipe pending
     - Not yet protocol-audited; no acceptance target assigned
   * - TEASER / TEASERGD
     - `Authors' code <https://github.com/JoeyDP/TEASER>`_;
       paper in :doc:`citing`
     - :doc:`api/models`; metadata/protocol recipe pending
     - Not yet protocol-audited; do not substitute CF-only targets
   * - KNN, simple sequence models, compressed ELSA
     - :doc:`citing`
     - :doc:`api/models`
     - No matched validation recipe in this first matrix

Validation checklist
--------------------

1. Record the repository commit and dirty diff with each run, plus data provenance.
2. Build once and reuse the identical checkpoint for competing models.
3. Check users/items/interactions, binary values, source/target separation,
   candidate vocabulary, and full-catalog versus sampled ranking.
4. Tune on validation only. Keep the test set sealed until configuration selection.
5. Compare the JSON test metrics with the matching row, recording absolute gaps.
   Repeat with several training seeds before diagnosing small differences.
6. Label outcomes ``adapted benchmark`` until preprocessing, architecture, training,
   and evaluation all match. Do not treat an arbitrary numerical tolerance as
   evidence of reproduction.
