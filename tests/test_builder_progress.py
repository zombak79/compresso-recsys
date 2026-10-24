from __future__ import annotations

import sys
from pathlib import Path

from compresso_recsys import builder
from compresso_recsys.builder.progress import _CheckpointProgress

sys.path.insert(0, str(Path(__file__).parent))
from test_public_datasets import steam_fixture  # noqa: E402


def test_step_messages_go_to_stderr_not_stdout(capsys):
    progress = _CheckpointProgress(enabled=True, total=2)
    progress.bar = None  # the fallback used when tqdm is unavailable
    progress.step("Loading interactions")
    progress.detail("a detail")
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "[compresso-recsys] Loading interactions" in captured.err
    assert "[compresso-recsys] a detail" in captured.err


def test_annotation_step_says_skipping_when_annotations_are_off(tmp_path, monkeypatch):
    steps: list[str] = []

    class Recording(_CheckpointProgress):
        def step(self, message: str) -> None:
            steps.append(message)

    monkeypatch.setattr(builder, "_CheckpointProgress", Recording)
    steam_fixture(tmp_path)
    builder.build_recsys_checkpoint(
        dataset="steam", data_dir=tmp_path, checkpoint_path=str(tmp_path / "s.zip"),
        split_mode="user_split", val_users=2, test_users=2, min_user_support=1,
        item_min_support=1, annotation_source="none", show_progress=False,
    )
    assert "Skipping annotations" in steps
    assert "Building annotations" not in steps
    assert "Loading item metadata" in steps  # metadata is still written to the checkpoint
