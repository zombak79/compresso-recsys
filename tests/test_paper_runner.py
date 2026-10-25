"""The paper benchmark runner (examples/paper/run.py), end to end on a synthetic split."""

from __future__ import annotations

import importlib.util
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

try:
    import tomllib  # noqa: F401
except ModuleNotFoundError:  # Python 3.10: the runner reads its config with tomli
    pytest.importorskip("tomli")

from compresso_recsys.builder import _build_args, _build_leave_last_out_split
from compresso_recsys.checkpoint import save_recsys_split

RUNNER = Path(__file__).parents[1] / "examples" / "paper" / "run.py"
SEQUENCE_KEYS = ("x_train_sequences", "train_source_sequences", "val_source_sequences", "test_source_sequences")
# The table metrics go up to ndcg@100, so the catalogue needs more than 100 items.
N_USERS, N_ITEMS, HISTORY = 300, 120, 8


def _events() -> pd.DataFrame:
    """User ``u`` reads ``HISTORY`` consecutive items of a cycle, starting at ``u % N_ITEMS``."""
    return pd.DataFrame(
        [
            {"user_id": f"u{user:04d}", "item_id": f"i{(user + step) % N_ITEMS:03d}",
             "value": 1.0, "timestamp": 1_000_000 + step * 90_000}
            for user in range(N_USERS)
            for step in range(HISTORY)
        ]
    )

CONFIG = """
[run]
seeds = [0, 1]
eval_batch_size = 64

[paths]
checkpoints = "{checkpoints}"

[statistics]
n_resamples = 99

[blocks.B]
selection = "ndcg@20"
reference = "EASE"
datasets = {{ "toy-llo" = {{ checkpoint = "toy.zip", table = "toy-llo" }} }}

[blocks.B.models.Pop]
kind = "popularity"

[blocks.B.models.EASE]
kind = "ease"
grid = {{ l2 = [1.0, 10.0] }}

[blocks.B.models.MultVAE]
kind = "multvae"
params = {{ latent_dim = 8, hidden_dim = 16, batch_size = 32 }}
training = {{ max_epochs = 3, every = 1, patience = 1 }}

[blocks.B.models.SASRec]
kind = "sasrec"
params = {{ d_model = 16, max_history_length = 10, batch_size = 32 }}
training = {{ epochs = 2, every = 1 }}
"""


@pytest.fixture(scope="module")
def runner():
    spec = importlib.util.spec_from_file_location("paper_run", RUNNER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def setup(tmp_path_factory):
    payload = _build_leave_last_out_split(
        _build_args(dataset="goodbooks", split_mode="leave_last_out"), _events()
    )
    root = tmp_path_factory.mktemp("split")
    save_recsys_split(
        root,
        item_ids=payload["item_ids"],
        x_train=payload["x_train"],
        train_source_matrix=payload["train_source_matrix"],
        train_target_matrix=payload["train_target_matrix"],
        val_source_indices=payload["val_holdout"]["source_indices"],
        val_target_indices=payload["val_holdout"]["target_indices"],
        test_source_indices=payload["test_holdout"]["source_indices"],
        test_target_indices=payload["test_holdout"]["target_indices"],
        **{key: payload[key] for key in SEQUENCE_KEYS},
    )
    checkpoints = tmp_path_factory.mktemp("checkpoints")
    with zipfile.ZipFile(checkpoints / "toy.zip", "w") as archive:
        for path in sorted(root.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(root))
    config = tmp_path_factory.mktemp("config") / "toy.toml"
    config.write_text(CONFIG.format(checkpoints=checkpoints))
    return config, checkpoints, tmp_path_factory.mktemp("results")


def _train_args(config, out, model, seed=0, trial=None):
    args = ["train", "--config", str(config), "--block", "B", "--dataset", "toy-llo",
            "--model", model, "--seed", str(seed), "--out", str(out), "--device", "cpu"]
    return args + ([] if trial is None else ["--grid-trial", str(trial)])


def _commands(runner, capsys, *args):
    assert runner.main(["list", *args]) == 0
    return [line.split(" ", 2)[2].split() for line in capsys.readouterr().out.splitlines()]


def test_the_whole_pipeline(runner, setup, capsys):
    config, _, out = setup
    grid = _commands(runner, capsys, "--config", str(config), "--stage", "grid", "--out", str(out))
    assert len(grid) == 5  # Pop, EASE x2, MultVAE, SASRec
    assert len(_commands(runner, capsys, "--config", str(config), "--stage", "grid", "--device", "gpu")) == 2
    for command in grid:
        assert runner.main([*command, "--device", "cpu"]) == 0
    capsys.readouterr()
    for model in ("Pop", "EASE", "MultVAE", "SASRec"):
        assert runner.main(["select", "--config", str(config), "--block", "B", "--dataset", "toy-llo",
                            "--model", model, "--out", str(out)]) == 0
    capsys.readouterr()
    seeds = _commands(runner, capsys, "--config", str(config), "--stage", "seeds", "--out", str(out))
    assert sorted(command[command.index("--model") + 1] for command in seeds) == ["MultVAE", "SASRec"]
    for command in seeds:
        assert runner.main([*command, "--device", "cpu"]) == 0
    capsys.readouterr()
    assert runner.main(["compare", "--config", str(config), "--out", str(out)]) == 0

    records = json.loads((out / "table" / "toy-llo.json").read_text())
    assert {r["candidate"] for r in records} == {"Pop", "MultVAE", "SASRec"}
    assert {r["baseline"] for r in records} == {"EASE"}
    assert {r["metric"] for r in records} == {"calibrated_recall@20", "ndcg@100"}
    summary = json.loads((out / "summary" / "B" / "toy-llo.json").read_text())
    assert summary["MultVAE"]["n_seeds"] == 2 and summary["EASE"]["n_seeds"] == 1
    multvae = json.loads((out / "runs/B/toy-llo/MultVAE/s0/result.json").read_text())
    assert multvae["training"]["best"]["epoch"] >= 1
    assert all("val_ndcg@20" in record for record in multvae["training"]["history"])
    manifest = {f["path"]: f["tier"] for f in json.loads((out / "MANIFEST.json").read_text())["files"]}
    assert manifest["table/toy-llo.json"] == "git" and manifest["meta.json"] == "git"
    assert manifest["runs/B/toy-llo/Pop/s0/per_user.npz"] == "archive"
    assert manifest["runs/B/toy-llo/Pop/s0/result.json"] == "git"


def test_train_is_idempotent_and_refuses_a_changed_spec(runner, setup, tmp_path, capsys):
    config, checkpoints, _ = setup
    assert runner.main(_train_args(config, tmp_path, "Pop")) == 0
    assert runner.main(_train_args(config, tmp_path, "Pop")) == 0
    assert "skip" in capsys.readouterr().out
    changed = tmp_path / "changed.toml"
    changed.write_text(config.read_text().replace('kind = "popularity"', 'kind = "popularity"\nparams = { use_values = true }'))
    assert runner.main(_train_args(changed, tmp_path, "Pop")) == runner.SPEC_MISMATCH


def test_a_checkpoint_that_differs_from_its_sums_is_a_spec_mismatch(runner, setup, tmp_path):
    config, checkpoints, _ = setup
    (checkpoints / "SHA256SUMS").write_text(f"{'0' * 64}  toy.zip\n")
    try:
        assert runner.main(_train_args(config, tmp_path, "Pop")) == runner.SPEC_MISMATCH
        status = json.loads((tmp_path / "runs/B/toy-llo/Pop/s0/status.json").read_text())
        assert status["phase"] == "spec_mismatch"
    finally:
        (checkpoints / "SHA256SUMS").unlink()


def test_grid_models_need_a_trial_and_deterministic_models_one_seed(runner, setup, tmp_path):
    config, _, _ = setup
    with pytest.raises(SystemExit, match="--grid-trial"):
        runner.main(_train_args(config, tmp_path, "EASE"))
    with pytest.raises(SystemExit, match="runs seeds"):
        runner.main(_train_args(config, tmp_path, "Pop", seed=1))


# --------------------------------------------------------------------------- catalogue blocks

CATALOGUE_CONFIG = """
[run]
seeds = [0, 1]
eval_batch_size = 64

[paths]
checkpoints = "{checkpoints}"

[statistics]
n_resamples = 99

[blocks.Cp]
mode = "catalogue"
catalogues = ["warm", "mixed"]
selection = "ndcg@100"
selection_catalogue = "mixed"
reference = "EASE"
encoder = "stub/encoder"
pairs = [["TEASER", "TEASER-GD"]]

[blocks.Cp.datasets.toy-temporal]
checkpoint = "toy-temporal.zip"
outputs = {{ warm = "casestudy/toy__warm", mixed = "casestudy/toy__mixed" }}

[blocks.Cp.models.EASE]
kind = "ease"

[blocks.Cp.models.TEASER]
kind = "teaser"
params = {{ features = ["text"], l2 = 0.1 }}

[blocks.Cp.models.TEASER-GD]
kind = "teaser_gd"
params = {{ features = ["text"], batch_size = 64, coefficient_regularization_samples = 16 }}
training = {{ max_epochs = 3, every = 1, patience = 1 }}

[blocks.Cp.models.CLCRec]
kind = "clcrec"
params = {{ features = ["text"], embedding_dim = 8, hidden_dim = 16, n_negatives = 4, batch_size = 64 }}
grid = {{ re_weight = [0.3, 0.7] }}
training = {{ max_epochs = 3, every = 1, patience = 1 }}
"""

N_WARM, N_VAL_COLD, N_TEST_COLD = 110, 10, 10


def _rows(rng, n_rows, columns, per_row, width):
    """``n_rows`` rows of ``per_row`` distinct items drawn from ``columns``, ``width`` columns wide."""
    from scipy.sparse import csr_matrix

    indices = [np.sort(rng.choice(columns, size=per_row, replace=False)) for _ in range(n_rows)]
    indptr = np.concatenate([[0], np.cumsum([len(i) for i in indices])])
    matrix = csr_matrix((np.ones(indptr[-1], dtype=np.float32), np.concatenate(indices), indptr),
                        shape=(n_rows, width))
    return matrix, indices


@pytest.fixture(scope="module")
def catalogue_setup(tmp_path_factory):
    """A temporal-style split: training items are a prefix, each later stage appends cold items."""
    rng = np.random.default_rng(0)
    n_items = N_WARM + N_VAL_COLD + N_TEST_COLD
    item_ids = np.array([f"i{i:03d}" for i in range(n_items)])
    warm = np.arange(N_WARM)
    val_catalogue = np.arange(N_WARM + N_VAL_COLD)
    x_train, _ = _rows(rng, 200, warm, 6, N_WARM)
    # Evaluation rows are training users, as in an item split, so CLCRec can
    # look up their embeddings.
    train_user_ids = np.array([f"u{u:03d}" for u in range(200)])
    x_train.data[:3] = 2.0  # repeat events, as a temporal split keeps them
    phases = {}
    for phase, catalogue in (("val", val_catalogue), ("test", np.arange(n_items))):
        source, source_indices = _rows(rng, 60, warm, 5, catalogue.size)
        targets, target_indices = _rows(rng, 60, catalogue, 2, catalogue.size)
        source.data[:2] = 2.0
        phases[phase] = {"source": source, "targets": targets,
                         "source_indices": source_indices, "target_indices": target_indices}
    root = tmp_path_factory.mktemp("split")
    save_recsys_split(
        root, item_ids=item_ids, x_train=x_train,
        train_item_ids=item_ids[:N_WARM], val_item_ids=item_ids[val_catalogue], test_item_ids=item_ids,
        val_source_indices=phases["val"]["source_indices"], val_target_indices=phases["val"]["target_indices"],
        test_source_indices=phases["test"]["source_indices"], test_target_indices=phases["test"]["target_indices"],
        val_source_matrix=phases["val"]["source"], val_target_matrix=phases["val"]["targets"],
        test_source_matrix=phases["test"]["source"], test_target_matrix=phases["test"]["targets"],
        warm_item_indices=warm, val_cold_item_indices=np.arange(N_WARM, N_WARM + N_VAL_COLD),
        test_cold_item_indices=np.arange(N_WARM + N_VAL_COLD, n_items),
        train_user_ids=train_user_ids,
        val_eval_user_ids=train_user_ids[:60], test_eval_user_ids=train_user_ids[60:120],
        entity_metadata=pd.DataFrame({"item_id": item_ids, "entity_text": [f"item {i}" for i in range(n_items)]}),
    )
    checkpoints = tmp_path_factory.mktemp("checkpoints")
    with zipfile.ZipFile(checkpoints / "toy-temporal.zip", "w") as archive:
        for path in sorted(root.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(root))
    config = tmp_path_factory.mktemp("config") / "toy-catalogue.toml"
    config.write_text(CATALOGUE_CONFIG.format(checkpoints=checkpoints))
    return config, tmp_path_factory.mktemp("results")


class _StubEncoder:
    """Stands in for sentence_transformers.SentenceTransformer: no download, fixed vectors."""

    def __init__(self, name, device=None):
        self.name = name

    def encode(self, texts, **kwargs):
        rng = np.random.default_rng(len(texts))
        vectors = rng.standard_normal((len(texts), 8)).astype(np.float32)
        return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


def test_catalogue_block_end_to_end(runner, catalogue_setup, monkeypatch, capsys):
    import types

    config, out = catalogue_setup
    monkeypatch.setitem(sys.modules, "sentence_transformers", types.SimpleNamespace(SentenceTransformer=_StubEncoder))
    base = ["--config", str(config), "--block", "Cp", "--dataset", "toy-temporal", "--out", str(out)]
    assert runner.main(["encode", *base]) == 0
    assert runner.main(["encode", *base]) == 0  # idempotent
    for model in ("EASE", "TEASER", "TEASER-GD"):
        assert runner.main(["train", *base, "--model", model, "--seed", "0", "--device", "cpu"]) == 0
        assert runner.main(["select", *base, "--model", model]) == 0
    assert runner.main(["train", *base, "--model", "TEASER-GD", "--seed", "1", "--device", "cpu"]) == 0
    for trial in (0, 1):
        assert runner.main(["train", *base, "--model", "CLCRec", "--seed", "0", "--grid-trial", str(trial),
                            "--device", "cpu"]) == 0
    assert runner.main(["select", *base, "--model", "CLCRec"]) == 0
    selected = runner.selected_trial(out, runner.load_config(config), "Cp", "toy-temporal", "CLCRec")
    assert runner.main(["train", *base, "--model", "CLCRec", "--seed", "1", "--grid-trial", str(selected),
                        "--device", "cpu"]) == 0
    capsys.readouterr()
    assert runner.main(["compare", "--config", str(config), "--out", str(out)]) == 0

    teaser = json.loads((out / "runs/Cp/toy-temporal/TEASER/s0/result.json").read_text())
    sizes = {key: record["catalogue_items"] for key, record in teaser["evaluation"].items()}
    assert sizes == {"val__warm": N_WARM, "val__mixed": N_WARM + N_VAL_COLD,
                     "test__warm": N_WARM, "test__mixed": N_WARM + N_VAL_COLD + N_TEST_COLD}
    assert teaser["cold_share"]["catalogue_items"] == {"cold": N_VAL_COLD + N_TEST_COLD,
                                                       "total": N_WARM + N_VAL_COLD + N_TEST_COLD}
    ease = json.loads((out / "runs/Cp/toy-temporal/EASE/s0/result.json").read_text())
    for key in ("test__warm", "test__mixed"):  # every model saw the same histories
        assert ease["evaluation"][key]["source_fingerprint"] == teaser["evaluation"][key]["source_fingerprint"]
    # Repeats were binarised for every model; TEASER and TEASER-GD refuse non-binary input.
    assert teaser["binarized_entries"] == {"x_train": 3, "val_source": 2, "test_source": 2}
    gd = json.loads((out / "runs/Cp/toy-temporal/TEASER-GD/s0/result.json").read_text())
    assert gd["training"]["best"] is not None and "val_ndcg@100" in gd["training"]["history"][0]
    for name in ("toy__warm", "toy__mixed"):
        records = json.loads((out / "casestudy" / f"{name}.json").read_text())
        assert {r["candidate"] for r in records} == {"TEASER", "TEASER-GD", "CLCRec"}
    pairs = json.loads((out / "summary/Cp/toy-temporal.pairs.json").read_text())
    assert {(r["catalogue"], r["baseline"], r["candidate"], r["metric"]) for r in pairs} == {
        (c, "TEASER", "TEASER-GD", m) for c in ("warm", "mixed") for m in ("calibrated_recall@20", "ndcg@100")}
    assert all(r["family"] == "TEASER-GD vs TEASER, 2 metrics" for r in pairs)
    # The paper's generator reads these fields by name: keep them stable.
    assert all({"catalogue", "metric", "difference", "adjusted_p_value", "ci_low", "ci_high", "p_value"} <= set(r)
               for r in pairs)
    meta = json.loads((out / "meta.json").read_text())
    assert meta["protocol"]["Cp"]["models"]["TEASER-GD"]["epochs"].startswith("max 3 epochs")
    share = meta["cold_share"]["toy-temporal"]
    n_items = N_WARM + N_VAL_COLD + N_TEST_COLD
    assert share["catalogue_items_vs_training"] == {"cold": N_VAL_COLD + N_TEST_COLD, "total": n_items,
                                                    "share": round((N_VAL_COLD + N_TEST_COLD) / n_items, 4)}
    assert share["catalogue_items_new_in_stage"]["cold"] == N_TEST_COLD
    assert (share["target_interactions_vs_training"]["cold"]
            == teaser["cold_share"]["target_interactions"]["cold"])
    assert share["target_interactions_new_in_stage"]["cold"] <= share["target_interactions_vs_training"]["cold"]
    assert meta["n_resamples"] == 99
    manifest = json.loads((out / "MANIFEST.json").read_text())
    assert manifest["scrub"]["flagged"] == []

    # A result from before binarisation is a spec mismatch for train and blocks compare.
    path = out / "runs/Cp/toy-temporal/EASE/s0/result.json"
    record = json.loads(path.read_text())
    record["spec"].pop("binary_interactions")
    record["spec_sha256"] = runner.spec_digest(record["spec"])
    path.write_text(json.dumps(record))
    assert runner.main(["train", *base, "--model", "EASE", "--seed", "0", "--device", "cpu"]) == runner.SPEC_MISMATCH
    assert runner.main(["compare", "--config", str(config), "--out", str(out)]) == runner.SPEC_MISMATCH


def test_verify_checkpoints_matches_content_not_zip_bytes(runner, setup, tmp_path):
    config, checkpoints, _ = setup
    previous = tmp_path / "previous"
    previous.mkdir()
    # Same content, different ZIP bytes (entries re-packed in reverse order).
    with zipfile.ZipFile(checkpoints / "toy.zip") as source, zipfile.ZipFile(previous / "toy.zip", "w") as copy:
        for name in reversed(source.namelist()):
            copy.writestr(name, source.read(name))
    out = tmp_path / "results"
    assert runner.main(["train", "--config", str(config), "--block", "B", "--dataset", "toy-llo", "--model", "Pop",
                        "--seed", "0", "--out", str(out), "--device", "cpu"]) == 0
    args = ["verify-checkpoints", "--config", str(config), "--previous", str(previous), "--out", str(out)]
    assert runner.main(args) == 0
    row = json.loads((out / "provenance" / "checkpoints.json").read_text())["checkpoints"][0]
    assert row["content_matches_previous"] and row["previous"]["sha256"] != row["sha256"]

    with zipfile.ZipFile(checkpoints / "toy.zip") as source, zipfile.ZipFile(previous / "toy.zip", "w") as copy:
        for name in source.namelist():
            copy.writestr(name, source.read(name))
        copy.writestr("data/extra.txt", b"a file the current build does not have")
    assert runner.main(args) == runner.SPEC_MISMATCH


def test_scrub_check_exempts_host_labels_but_not_the_hostname(runner, tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "_identifying_strings", lambda: ["h100", "someuser", "/home/someuser"])
    monkeypatch.setenv("CR_HOST_LABEL", "ostrava-h100")
    (tmp_path / "env").mkdir()
    (tmp_path / "env" / "ostrava-h100-pip-freeze.txt").write_text("numpy==2.0\n")
    (tmp_path / "labelled.json").write_text('{"host": "ostrava-h100"}')
    (tmp_path / "bare.json").write_text('{"node": "h100"}')
    (tmp_path / "path.json").write_text('{"error": "File /home/someuser/x.py"}')
    files = [{"path": p} for p in ("labelled.json", "bare.json", "path.json", "env/ostrava-h100-pip-freeze.txt")]
    assert sorted(runner.scrub_check(tmp_path, files)) == ["bare.json", "path.json"]
    cleaned = runner.scrubbed('File "/home/someuser/x.py" on h100')
    assert "someuser" not in cleaned and "h100" not in cleaned and "x.py" in cleaned


def test_relabel_hosts_rewrites_every_occurrence(runner, setup, tmp_path, monkeypatch):
    config, _, _ = setup
    monkeypatch.setenv("CR_HOST_LABEL", "site-gpu")
    out = tmp_path / "results"
    assert runner.main(_train_args(config, out, "Pop")) == 0
    assert runner.main(["relabel-hosts", "--config", str(config), "--out", str(out),
                        "--old", "site-gpu", "--new", "gpu-cluster"]) == 0
    result = json.loads((out / "runs/B/toy-llo/Pop/s0/result.json").read_text())
    assert result["env"]["host"] == "gpu-cluster"
    assert json.loads((out / "runs/B/toy-llo/Pop/s0/status.json").read_text())["host"] == "gpu-cluster"
    assert [p.name for p in (out / "env").iterdir()] == ["gpu-cluster-pip-freeze.txt"]


def test_positional_sample_ids_are_recorded_and_refused_for_identity_models(runner, setup, tmp_path, monkeypatch):
    config, _, _ = setup
    import compresso_recsys as cr

    original = runner.load_split

    def without_eval_ids(path, *args, **kwargs):
        split, features = original(path, *args, **kwargs)
        split["val_eval_user_ids"] = split["test_eval_user_ids"] = None
        return split, features

    monkeypatch.setattr(runner, "load_split", without_eval_ids)
    out = tmp_path / "results"
    assert runner.main(_train_args(config, out, "Pop")) == 0
    assert json.loads((out / "runs/B/toy-llo/Pop/s0/result.json").read_text())["sample_ids"] == "positional"

    monkeypatch.setattr(cr.PopularityBaseline, "accepts_user_ids", True, raising=False)
    with pytest.raises(ValueError, match="uses user identity"):
        runner.main(_train_args(config, tmp_path / "other", "Pop"))


def test_bundle_builds_a_verified_pseudonymised_deposit(runner, setup, tmp_path, monkeypatch):
    import tarfile

    config, _, _ = setup
    monkeypatch.setenv("CR_HOST_LABEL", "site-gpu")
    out = tmp_path / "results"
    for model, trial in (("Pop", None), ("EASE", 0), ("EASE", 1)):
        assert runner.main(_train_args(config, out, model, trial=trial)) == 0
    for model in ("Pop", "EASE"):
        assert runner.main(["select", "--config", str(config), "--block", "B", "--dataset", "toy-llo",
                            "--model", model, "--out", str(out)]) == 0
    assert runner.main(["compare", "--config", str(config), "--out", str(out)]) == 0
    table_before = (out / "table" / "toy-llo.json").read_text()

    archive, git_dir = tmp_path / "deposit.tar.gz", tmp_path / "git"
    args = ["bundle", "--config", str(config), "--out", str(out), "--archive", str(archive),
            "--relabel", "site-gpu=gpu-cluster", "--git-dir", str(git_dir), "--name", "dep"]
    assert runner.main(args) == 0
    assert (out / "table" / "toy-llo.json").read_text() == table_before  # --out is never modified

    with tarfile.open(archive) as tar:
        tar.extractall(tmp_path / "x", filter="data")
    deposit = tmp_path / "x" / "dep"
    assert "No raw interaction data" in (deposit / "README.md").read_text()
    assert json.loads((deposit / "meta.json").read_text())["hosts"] == ["gpu-cluster"]
    with np.load(deposit / "runs/B/toy-llo/Pop/s0/per_user.npz") as arrays:
        ids = arrays["test/sample_ids"]
    assert ids.dtype == np.int64 and sorted(ids.tolist()) == list(range(len(ids)))
    assert runner.verify_manifest(deposit) == []
    assert (git_dir / "table" / "toy-llo.json").exists() and not list(git_dir.rglob("per_user.npz"))
    # The deposit's per-user values still reproduce the comparison.
    assert runner.main(["compare", "--config", str(config), "--out", str(deposit)]) == 0
    assert json.loads((deposit / "table" / "toy-llo.json").read_text()) == json.loads(table_before)

    (out / "table" / "toy-llo.json").write_text("[]")  # a change after compare
    assert runner.main(args) == runner.SPEC_MISMATCH


def test_user_embedding_models_refuse_checkpoints_without_user_ids(runner):
    with pytest.raises(ValueError, match="needs user ids"):
        runner.training_user_ids({"train_user_ids": np.array(["u0"]), "val_eval_user_ids": None})
    ids = runner.training_user_ids({"train_user_ids": np.array(["u0", "u1"]),
                                    "val_eval_user_ids": np.array(["u1"]), "test_eval_user_ids": np.array(["u0"])})
    assert ids.tolist() == ["u0", "u1"]
