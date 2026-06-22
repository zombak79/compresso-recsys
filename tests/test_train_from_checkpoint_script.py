"""Regression smoke test for recsys_train_sae_from_checkpoint.py."""

from __future__ import annotations

import subprocess
import sys
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from compresso_recsys.checkpoint import save_recsys_split, update_checkpoint


def test_recsys_train_from_checkpoint_script_smoke(tmp_path: Path):
    n_items = 12
    emb_dim = 8
    rng = np.random.default_rng(0)
    item_ids = np.array([f"i{i}" for i in range(n_items)])
    x_train = csr_matrix(rng.integers(0, 2, size=(6, n_items), dtype=np.int8).astype(np.float32))
    entity_tag_matrix = csr_matrix(
        rng.integers(0, 2, size=(n_items, 3), dtype=np.int8).astype(np.float32)
    )
    tag_names = np.array(["alpha", "beta", "gamma"])
    entity_metadata = pd.DataFrame(
        {
            "item_id": item_ids,
            "title": [f"Item {i}" for i in range(n_items)],
        }
    )

    val_source_indices = [
        np.array([0, 1, 2], dtype=np.int64),
        np.array([3, 4], dtype=np.int64),
        np.array([5, 6, 7], dtype=np.int64),
        np.array([8], dtype=np.int64),
    ]
    val_target_indices = [
        np.array([3, 4], dtype=np.int64),
        np.array([0, 2], dtype=np.int64),
        np.array([1, 9], dtype=np.int64),
        np.array([10, 11], dtype=np.int64),
    ]
    test_source_indices = [
        np.array([1, 2], dtype=np.int64),
        np.array([4, 5], dtype=np.int64),
        np.array([6, 7], dtype=np.int64),
        np.array([9], dtype=np.int64),
    ]
    test_target_indices = [
        np.array([0, 3], dtype=np.int64),
        np.array([1, 8], dtype=np.int64),
        np.array([2, 10], dtype=np.int64),
        np.array([11], dtype=np.int64),
    ]

    ckpt_path = tmp_path / "recsys_checkpoint.zip"
    with update_checkpoint(ckpt_path) as root:
        save_recsys_split(
            root,
            item_ids=item_ids,
            x_train=x_train,
            val_source_indices=val_source_indices,
            val_target_indices=val_target_indices,
            test_source_indices=test_source_indices,
            test_target_indices=test_target_indices,
            entity_tag_matrix=entity_tag_matrix,
            tag_names=tag_names,
            entity_metadata=entity_metadata,
            metadata={"dataset": "synthetic"},
        )
    repo_root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    src_path = str(repo_root / "src")
    env["PYTHONPATH"] = src_path if not env.get("PYTHONPATH") else f"{src_path}{os.pathsep}{env['PYTHONPATH']}"
    elsa_cmd = [
        sys.executable,
        "-m",
        "compresso_recsys.scripts.train_elsa_from_checkpoint",
        "--checkpoint_path",
        str(ckpt_path),
        "--device",
        "cpu",
        "--seed",
        "0",
        "--elsa_dim",
        str(emb_dim),
        "--elsa_epochs",
        "1",
        "--elsa_batch_size",
        "4",
        "--eval_batch_size",
        "4",
    ]
    elsa_proc = subprocess.run(elsa_cmd, cwd=repo_root, env=env, capture_output=True, text=True)
    assert elsa_proc.returncode == 0, elsa_proc.stderr
    assert "ELSA checkpoint metrics:" in elsa_proc.stdout

    cmd = [
        sys.executable,
        "-m",
        "compresso_recsys.scripts.train_sae_from_checkpoint",
        "--checkpoint_path",
        str(ckpt_path),
        "--device",
        "cpu",
        "--seed",
        "0",
        "--sae_hidden_dim",
        "16",
        "--sae_k",
        "4",
        "--sae_epochs",
        "1",
        "--sae_batch_size",
        "4",
        "--eval_batch_size",
        "4",
    ]
    proc = subprocess.run(cmd, cwd=repo_root, env=env, capture_output=True, text=True)

    assert proc.returncode == 0, proc.stderr
    assert "Original embedding metrics:" in proc.stdout
    assert "(from checkpoint)" in proc.stdout
    assert "SAE embedding metrics:" in proc.stdout
    assert "Perf drop vs elsa:" in proc.stdout
    assert "Saved sae stage to checkpoint:" in proc.stdout

    import zipfile

    with zipfile.ZipFile(ckpt_path, "r") as zf:
        names = set(zf.namelist())
    assert "elsa/model.pt" in names
    assert "elsa/item_embeddings.npy" in names
    assert "data/entity_tags.npz" in names
    assert "data/tag_names.npy" in names
    assert "data/entity_metadata.csv" in names
    assert "sae/model.pt" in names
    assert "sae/sparse_embeddings.srp.pt" in names
    assert "sae/metrics.json" in names

    eval_cmd = [
        sys.executable,
        "-m",
        "compresso_recsys.scripts.eval_checkpoint",
        "--checkpoint_path",
        str(ckpt_path),
        "--device",
        "cpu",
        "--eval_batch_size",
        "4",
    ]
    eval_proc = subprocess.run(eval_cmd, cwd=repo_root, env=env, capture_output=True, text=True)
    assert eval_proc.returncode == 0, eval_proc.stderr
    assert "ELSA metrics:" in eval_proc.stdout
    assert "SAE sparse code metrics:" in eval_proc.stdout
    assert "SAE + decoder kernel-trick metrics:" in eval_proc.stdout
