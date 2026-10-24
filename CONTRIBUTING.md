# Contributing to compresso-recsys

Thanks for helping. This guide covers setting up a development environment, the checks a change must pass, and where to start when adding a model, a dataset, or a metric.

## Development install

compresso-recsys needs Python ≥ 3.10. On a machine without a GPU, install the CPU build of PyTorch first; it is much smaller.

```bash
git clone https://github.com/zombak79/compresso-recsys.git
cd compresso-recsys
python -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cpu   # skip on a CUDA machine
pip install -e ".[test,knn,dev]"
```

Use `pip install -e ".[docs]"` as well if you will build the documentation.

## Supported versions

- **Python:** 3.10, 3.11 and 3.12 are tested in CI and supported. 3.13 runs in CI
  as an allowed-to-fail job until PyTorch and compresso-pytorch ship stable
  wheels for it.
- **Adding a Python version:** once PyTorch publishes stable wheels for it, the
  version joins the required CI matrix and the classifiers.
- **Dropping a Python version:** about three years after its release, in a minor
  release, announced in the changelog one release ahead.
- **PyTorch:** the same policy applies to the `torch` floor in `pyproject.toml`:
  raised only in a minor release, announced one release ahead.

## Checks before opening a pull request

**Tests.** The whole suite runs offline in about a minute on a laptop:

```bash
pytest -q
```

CI runs the same suite on Python 3.10–3.12 (and 3.13, allowed to fail) with network access disabled (`pytest-socket`). A test must therefore never download anything:
- Build the data it needs in `tmp_path`.
- Monkeypatch the downloader, as `tests/test_grouplens_download.py` and `tests/test_amazon_transport.py` do.

**Lint.** We use [ruff](https://docs.astral.sh/ruff/):

```bash
ruff check --select F src tests      # must be clean: CI enforces pyflakes rules
ruff check src tests                 # the full default rule set: report only for now
ruff format --check src tests        # report only for now
```

Please don't mix repository-wide reformatting into a feature pull request. A dedicated clean-up will turn the full rule set and formatting into enforced checks.

**Docs.** If you changed a public API or a notebook under `docs/source/`, build the docs locally:

```bash
sphinx-build -b html docs/source docs/build/html
```

`tests/test_docs_snippets.py` executes the code in the documentation pages and notebooks, so broken examples fail the test suite.

## Adding things

### A model

Start from the tutorial [Implementing a recommender](https://zombak79.github.io/compresso-recsys/implementing-a-recommender.html) (`docs/source/implementing-a-recommender.ipynb`). It builds a complete model with validation, stable-ID recommendation, persistence, evaluation, and executable contract checks. In short:

1. Subclass the base for your model's input shape:
   - `BaseCollaborativeRecommender` for user–item matrices
   - `BaseSequentialRecommender` for chronological histories
   - `BaseColdStartRecommender` or `BaseMultiModalRecommender` for item features
2. Give it a unique `checkpoint_type`. Implement `fit`, `is_fitted`, `n_items`, `predict_on_batch` and `_from_checkpoint_config`, plus `_save_checkpoint_state` / `_load_checkpoint_state` for state that isn't a torch module.
3. Accept `config=None` and fall back to the default config, so `MyModel()` works.
4. Export it from `compresso_recsys/models/__init__.py`. Add tests: a fit/predict round trip, `save` → `load` equality, and `recommend()` with stable IDs. Add a `citing.rst` entry if it implements a published method.

Every model that implements `predict_on_batch` is evaluated by the same `evaluate_recommender` and compared by the same `compare_models`, so no evaluation code is needed.

### A dataset

Follow [Bring your own dataset](https://zombak79.github.io/compresso-recsys/bring-your-own-dataset.html) (`docs/source/bring-your-own-dataset.ipynb`).

1. Subclass `RecSysDataset` (or `PublicDataset` for a public source). Implement `download()` and `prepare()`, producing the canonical `user_id, item_id, value, timestamp` interactions and an item-metadata frame.
2. Register a `DatasetSpec` in `compresso_recsys/builder` (`DATASETS`). Its defaults should be thought through: `min_value_to_keep`, the support thresholds, `min_entity_text_words`, `temporal_period_hours`.
3. Document it in `docs/source/datasets.rst` using the existing **Loading / Terms / Defaults** layout, and add a row to `docs/source/licenses.rst`. The **licence and terms are mandatory**: we only add loaders for data whose terms we can state.
4. Write tests against a tiny synthetic file in `tmp_path`, never against the real download.

### A metric

Metrics live in `compresso_recsys/metrics.py`.
1. Subclass `_MeanAtCutoffsMetric`. Set a key `prefix`, and return per-row values of shape `(rows, n_cutoffs)` from the hits in a `RankingBatch`; `CalibratedRecall` and `NDCG` are short examples.
2. Per-row values are what `compare_models` resamples. A new metric therefore gets paired significance tests and confidence intervals for free, as long as it is a mean over users.
3. Add the definition, with its formula and any normalisation choice, to the docstring and to `docs/evaluation-statistics-spec.md` §8.

## Pull requests

- **One topic per pull request.** Link the issue it addresses.
- **Changelog:** add an entry under `## [Unreleased]` in `CHANGELOG.md`. Say what changed for users and, for behaviour changes, what they need to do.
- **Tests:** new behaviour comes with tests, and bug fixes come with a regression test.
- **Compatibility:** don't change the on-disk layout of data or model checkpoints without bumping the format version and keeping old files readable. `tests/test_checkpoint_layout.py` pins the layout on purpose.
- **Results:** results quoted in docs or notebooks must state the commit, the seed(s) and the hardware they came from. A number without provenance is not a result.
- **Review:** at least one maintainer approval is required, and CI must be green.

## Reporting bugs and asking questions

Use the GitHub issue templates. For a bug, include:
- the `compresso-recsys`, `torch` and Python versions
- the smallest code that reproduces it
- the full traceback
