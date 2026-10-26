Reproducing the ECIR 2027 Benchmark
===================================

The results tables and the warm/cold case study of the compresso-recsys resource
paper (ECIR 2027) come from one runner, ``examples/paper/run.py``, driven by one
configuration file. This page walks through the whole path:

1. build the data checkpoints;
2. train every model;
3. select hyperparameters on validation;
4. rerun the selected configurations with more seeds;
5. compare the models statistically;
6. regenerate the tables.

It also shows how to check the published results deposit without rerunning
anything.

The results were produced with compresso-recsys at commit ``TODO`` (tag ``v0.4.0``).
The results deposit is at ``https://doi.org/10.5281/zenodo.TODO``.

What ``examples/paper/`` contains
---------------------------------

``run.py``
   The runner. Its subcommands:

   - ``list``: print one ``train`` command per fit.
   - ``train``: one fit (block, dataset, model, seed and grid point), scored on
     validation and test.
   - ``select``: choose a grid point on validation.
   - ``compare``: the paper's tables, with per-seed summaries and the manifest.
   - ``encode``: text features for the case study.
   - ``verify-checkpoints``: match results to rebuilt checkpoints.
   - ``relabel-hosts`` and ``bundle``: prepare the deposit.
``ecir2027.toml``
   Every block, dataset, model, grid, epoch rule and seed of the paper. Each
   training rule is either *max epochs + patience* on the block's validation
   metric, or the published fixed budget of SASRec and BERT4Rec on ML-1M.
``smoke.toml``
   A minutes-long ML-1M pass that exercises every command and writes every
   output file. Run it first.
``README.md``
   The results layout and the runner's guarantees: idempotent fits, exit code 2
   on a changed spec or checkpoint, and writes only under ``--out``.

The benchmark has five blocks:

.. list-table::
   :header-rows: 1

   * - Block
     - Data and protocol
     - Models
   * - A
     - ML-20M, Netflix; strong generalisation (held-out users, 80/20 fold-in)
     - Pop, ItemKNN, EASE, ELSA, MultVAE
   * - B
     - ML-1M; leave-last-out, full catalogue
     - the above plus SASRec and BERT4Rec (published budgets)
   * - B′
     - Amazon Reviews 2023 Toys & Games; leave-last-out
     - as B
   * - C
     - ML-1M item split with SWAP features; cold and full catalogues
     - TEASER, Content, MMConcat-TEASER, TEASER-GD, CLCRec, EASE
   * - C′
     - Steam and Amazon Toys; temporal split, warm-only versus mixed catalogues
     - Pop, EASE, Content, TEASER, TEASER-GD on MiniLM text

Step by step
------------

1. **Environment.** Python 3.12 and a CUDA build of PyTorch that matches the GPU
   driver. Then:

   .. code-block:: bash

      pip install "compresso-recsys==0.4.0" sentence-transformers
      git clone https://github.com/zombak79/compresso-recsys && cd compresso-recsys
      git checkout v0.4.0
      export CR_DATA=$PWD/data CR_CKPT=$PWD/checkpoints RESULTS=$PWD/results/ecir2027
      export CR_SOURCE_COMMIT=$(git rev-parse HEAD) CR_HOST_LABEL=my-gpu-host

2. **Data checkpoints** (CPU, about 10 minutes, plus downloads). Each build
   downloads its dataset into ``$CR_DATA`` on first use. The checkpoint records
   the build arguments, the SHA-256 of every raw file it read, and a content
   digest that is identical across rebuilds (:doc:`cli-reference`).

   .. code-block:: python

      import os
      import compresso_recsys as cr

      data, ckpt = os.environ["CR_DATA"], os.environ["CR_CKPT"]
      for name, held_out in (("ml20m", 10_000), ("netflix", 40_000)):
          cr.build_recsys_checkpoint(
              dataset=name, data_dir=data, checkpoint_path=f"{ckpt}/{name}-user_split.zip",
              split_mode="user_split", seed=98765, val_users=held_out, test_users=held_out,
              min_user_support=5, item_min_support=1, min_value_to_keep=4, set_all_values_to=1.0,
              eval_draws=1, min_entity_text_words=0, annotation_source="none", show_progress=False,
          )
      cr.build_recsys_checkpoint(
          dataset="ml1m", data_dir=data, checkpoint_path=f"{ckpt}/ml1m-leave_last_out.zip",
          split_mode="leave_last_out", min_user_support=5, item_min_support=5, min_value_to_keep=1.0,
          min_entity_text_words=0, seed=0, annotation_source="none", show_progress=False,
      )
      cr.build_recsys_checkpoint(
          dataset="ml1m", data_dir=data, checkpoint_path=f"{ckpt}/ml1m-item_split-swap.zip",
          split_mode="item_split", seed=42, annotation_source="none",
          multimodal_features="text/mpnet,text/minilm,image/vit_cls", show_progress=False,
      )
      for mode in ("leave_last_out", "temporal"):
          cr.build_recsys_checkpoint(
              dataset="amazon2023", amazon_category="Toys_and_Games", data_dir=data,
              checkpoint_path=f"{ckpt}/amazon-toys-{mode}.zip", split_mode=mode,
              annotation_source="none", show_progress=False,
          )
      cr.build_recsys_checkpoint(
          dataset="steam", data_dir=data, checkpoint_path=f"{ckpt}/steam-temporal.zip",
          split_mode="temporal", annotation_source="none", show_progress=False,
      )

   Then record the checksums the runner verifies before every fit:

   .. code-block:: bash

      (cd $CR_CKPT && sha256sum *.zip > SHA256SUMS)

3. **Smoke test** (CPU, a few minutes). Every command runs on ML-1M:

   .. code-block:: bash

      python examples/paper/run.py list --config examples/paper/smoke.toml --stage grid --out results/smoke | bash
      for m in Pop ItemKNN EASE MultVAE; do
        python examples/paper/run.py select --config examples/paper/smoke.toml --block B --dataset ml1m-llo --model $m --out results/smoke
      done
      python examples/paper/run.py list --config examples/paper/smoke.toml --stage seeds --out results/smoke | bash
      python examples/paper/run.py compare --config examples/paper/smoke.toml --out results/smoke

4. **Case-study text features** (one-off, under a minute per dataset on a GPU):

   .. code-block:: bash

      python examples/paper/run.py encode --config examples/paper/ecir2027.toml --block Cp --dataset steam-temporal --out $RESULTS
      python examples/paper/run.py encode --config examples/paper/ecir2027.toml --block Cp --dataset amazon-toys --out $RESULTS

5. **Grid stage.** Every grid point on seed 0. ``list`` prints one ``train``
   command per line. Run them in any order, on any number of machines, for
   example as a Slurm array with one line per task:

   .. code-block:: bash

      python examples/paper/run.py list --config examples/paper/ecir2027.toml --stage grid --device gpu > grid-gpu.txt
      python examples/paper/run.py list --config examples/paper/ecir2027.toml --stage grid --device cpu > grid-cpu.txt

   A line whose result already exists with the same spec is skipped. A line
   whose spec or checkpoint differs exits with code 2 instead of overwriting.

6. **Selection** (CPU, seconds). Run one ``select`` per block, dataset and
   model. The highest validation score wins, and ties go to the first grid point
   in the configuration's order:

   .. code-block:: bash

      python examples/paper/run.py select --config examples/paper/ecir2027.toml --block A --dataset ml20m --model EASE --out $RESULTS

7. **Seeds stage.** The selected configuration of every stochastic model, with
   seeds 1 and 2:

   .. code-block:: bash

      python examples/paper/run.py list --config examples/paper/ecir2027.toml --stage seeds --out $RESULTS > seeds.txt

8. **Comparison** (CPU, about 7 minutes). This writes:

   - ``table/<dataset>.json`` and ``casestudy/<dataset>__<catalogue>.json``: paired
     bootstrap CIs and Holm-adjusted randomization tests on the seed-0 per-user
     values, B = 9,999;
   - per-seed means and standard deviations under ``summary/``;
   - the cold-share figures and the full protocol in ``meta.json``;
   - ``MANIFEST.json``.

   .. code-block:: bash

      python examples/paper/run.py compare --config examples/paper/ecir2027.toml --out $RESULTS

9. **Tables.** The paper's table and figure generators read only ``table/``,
   ``casestudy/`` and ``meta.json``: ``make results RESULTS=$RESULTS`` in the
   paper repository. To read the results without the paper sources,
   ``summary/<block>/<dataset>.md`` holds every comparison as Markdown, and
   :meth:`~compresso_recsys.stats.ComparisonReport.to_latex` renders the same
   reports.

Hardware and time
-----------------

The published results ran on a Slurm cluster with NVIDIA H100 80 GB GPUs, at
most six at a time, between 24 and 26 October 2026. GPU-hours are allocated GPUs
times job time, and include grid points that a later grid extension superseded.

.. list-table::
   :header-rows: 1

   * - Stage
     - Resources
   * - Checkpoint builds, all datasets
     - CPU, about 8 minutes (plus first-time downloads)
   * - Blocks A, B, B′, C′: grid, seeds and reruns (incl. BERT4Rec ML-1M, 907 epochs)
     - 7.9 H100-hours
   * - Block C (ML-1M item split): grid, three grid extensions, CLCRec seeds
     - 12.0 H100-hours
   * - CPU stages (EASE, ItemKNN, Pop, TEASER, Content, ``select``, ``compare``)
     - about 4.4 CPU-hours, 8 cores per task
   * - ``compare`` (all blocks, B = 9,999)
     - CPU, about 7 minutes
   * - **Tables and case study in total**
     - **about 19.9 H100-hours**; two nights on six GPUs
   * - BERT4Rec ML-1M reproduction seeds (sampled metrics only, not in the tables)
     - 2.5 H100-hours (22.5 in all)

The longest single fits are the fixed-budget BERT4Rec runs on ML-1M, about
75 minutes each on one H100. A 24 GB consumer GPU runs every GPU model in the
configuration. MultVAE on Netflix needs about 27 GB with ``preload_training_data``,
so set it to ``false`` there or use a larger GPU.

Checking the published results
------------------------------

The results deposit (``results-ecir2027.tar.gz``) is built with
``run.py bundle``. Among the run files it holds every model's per-user test
values. Dataset user IDs are replaced by per-dataset indices, so paired
comparison still works and no dataset identifiers are published. Three checks
need no GPU and no data:

1. **Integrity.** Every file's SHA-256 is listed in ``MANIFEST.json``:

   .. code-block:: bash

      tar -xzf results-ecir2027.tar.gz && cd results-ecir2027
      python - <<'EOF'
      import hashlib, json
      for f in json.load(open("MANIFEST.json"))["files"]:
          if hashlib.sha256(open(f["path"], "rb").read()).hexdigest() != f["sha256"]:
              print("MISMATCH", f["path"])
      EOF

2. **The tables follow from the per-user values.** Recompute every comparison
   from the deposit and diff it against the shipped tables:

   .. code-block:: bash

      cp -r results-ecir2027 recheck
      python examples/paper/run.py compare --config examples/paper/ecir2027.toml --out recheck
      diff <(python -m json.tool results-ecir2027/table/ml20m.json) <(python -m json.tool recheck/table/ml20m.json)

   The statistics are deterministic for a fixed ``random_state``
   (:doc:`statistical-comparison`), so the recomputed tables match bit for bit.

3. **The results belong to these data.** Every ``result.json`` records its
   checkpoint's ``content_sha256``. A checkpoint rebuilt with step 2 above has
   the same digest, which you can read with :func:`~compresso_recsys.load_provenance`.
   If you rebuilt checkpoints after running, ``run.py verify-checkpoints
   --previous <old checkpoint dir>`` checks the match.

See also
--------

- :doc:`statistical-comparison` for what the intervals and p-values mean.
- :doc:`api/training` for the max-epochs-plus-patience rule.
- :doc:`cli-reference` for the checkpoint manifest and provenance.
- ``examples/paper/README.md`` for the results layout.
