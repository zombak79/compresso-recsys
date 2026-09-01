"""Keep documented code honest about the signatures it calls.

Sphinx never executes a ``code-block``, and ``nbsphinx_execute`` is ``"never"``,
so a documented call can go stale in complete silence: building with ``-W``
passes, the HTML renders, and the snippet raises ``TypeError`` for the first
reader who copies it. That already happened once here — ``max_length`` moved from
``SimpleRNNConfig`` to ``SequenceBatcher`` and the example kept passing it.

These tests cannot run the snippets, which mostly need a checkpoint. They check
the two things that rot without anyone touching the docs: that the code still
parses, and that every keyword argument aimed at a public callable still exists
in its signature.
"""

from __future__ import annotations

import ast
import inspect
import json
import re
import textwrap
from pathlib import Path

import numpy as np
import pytest

import compresso_recsys
import compresso_recsys.evaluation
import compresso_recsys.metrics
import compresso_recsys.models

DOCS = Path(__file__).resolve().parent.parent / "docs" / "source"

_BLOCK = re.compile(
    r"\.\. code-block:: python\n(?:\s*:[a-z]+:.*\n)*\n((?:(?:[ \t]+.*)?\n)+)"
)


def _public_callables() -> dict[str, object]:
    """Every class or function a reader could reach by its documented name."""
    found: dict[str, object] = {}
    for module in (
        compresso_recsys,
        compresso_recsys.models,
        compresso_recsys.evaluation,
        compresso_recsys.metrics,
    ):
        for name in dir(module):
            if name.startswith("_"):
                continue
            obj = getattr(module, name)
            if inspect.isclass(obj) or inspect.isfunction(obj):
                found.setdefault(name, obj)
    return found


def _rst_snippets() -> list[tuple[str, int, str]]:
    """``(source, line, code)`` for every python block in the documentation."""
    out = []
    for path in sorted(DOCS.rglob("*.rst")):
        text = path.read_text()
        for match in _BLOCK.finditer(text):
            line = text[: match.start()].count("\n") + 1
            out.append(
                (str(path.relative_to(DOCS)), line, textwrap.dedent(match.group(1)))
            )
    return out


def _notebook_snippets() -> list[tuple[str, int, str]]:
    """Code cells from the shipped tutorials, which are not executed at build."""
    out = []
    for path in sorted(DOCS.rglob("*.ipynb")):
        cells = json.loads(path.read_text())["cells"]
        for index, cell in enumerate(cells):
            if cell["cell_type"] != "code":
                continue
            code = "".join(cell["source"])
            # Notebook magics and shell escapes are not Python.
            if any(l.lstrip().startswith(("!", "%")) for l in code.splitlines()):
                continue
            out.append((str(path.relative_to(DOCS)), index, code))
    return out


SNIPPETS = _rst_snippets() + _notebook_snippets()


def test_the_documentation_actually_contains_snippets():
    """A regex that silently matches nothing would make the rest of this vacuous."""
    assert len(SNIPPETS) > 30, f"only found {len(SNIPPETS)} snippets"
    assert any(s[0].endswith(".ipynb") for s in SNIPPETS)
    assert any(s[0].endswith(".rst") for s in SNIPPETS)


@pytest.mark.parametrize(
    ("source", "line", "code"), SNIPPETS, ids=[f"{s}:{l}" for s, l, _ in SNIPPETS]
)
def test_documented_snippets_parse(source, line, code):
    try:
        ast.parse(code)
    except SyntaxError as error:  # pragma: no cover - only on a docs regression
        pytest.fail(f"{source}:{line} does not parse: {error.msg}")


@pytest.mark.parametrize(
    ("source", "line", "code"), SNIPPETS, ids=[f"{s}:{l}" for s, l, _ in SNIPPETS]
)
def test_documented_keywords_exist_in_their_signatures(source, line, code):
    """The failure mode Sphinx cannot see: a keyword that stopped existing."""
    known = _public_callables()
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return  # the parse test owns this failure

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Name):
            name = func.id
        elif isinstance(func, ast.Attribute):
            name = func.attr
        else:
            continue
        target = known.get(name)
        if target is None:
            continue
        try:
            signature = inspect.signature(target)
        except (TypeError, ValueError):  # pragma: no cover - builtins
            continue
        # **kwargs accepts anything, so there is nothing to check.
        if any(
            p.kind is inspect.Parameter.VAR_KEYWORD
            for p in signature.parameters.values()
        ):
            continue
        for keyword in node.keywords:
            if keyword.arg is None:  # **spread
                continue
            assert keyword.arg in signature.parameters, (
                f"{source}:{line} calls {name}({keyword.arg}=...) but "
                f"{name} takes {sorted(signature.parameters)}"
            )


def test_implementing_a_recommender_notebook_executes():
    """Run the educational path without downloading the benchmark dataset."""
    pytest.importorskip("sklearn")
    path = DOCS / "implementing-a-recommender.ipynb"
    cells = json.loads(path.read_text())["cells"]
    namespace = {"__name__": "__main__"}

    for index, cell in enumerate(cells):
        if cell["cell_type"] != "code":
            continue
        code = "".join(cell["source"])
        if "requires-data" in cell.get("metadata", {}).get("tags", []):
            from scipy.sparse import csr_matrix

            if "x_train" not in namespace:
                n_items = 30
                train = np.zeros((12, n_items), dtype=np.float32)
                source = np.zeros((4, n_items), dtype=np.float32)
                targets = np.zeros((4, n_items), dtype=np.float32)
                for row in range(train.shape[0]):
                    train[
                        row,
                        [row % 10, (row + 3) % 20, (row + 9) % n_items],
                    ] = 1
                for row in range(source.shape[0]):
                    source[row, [row, row + 4, row + 8]] = 1
                    targets[row, row + 20] = 1
                namespace.update(
                    x_train=csr_matrix(train),
                    test_source=csr_matrix(source),
                    test_targets=csr_matrix(targets),
                    item_ids=np.array([f"item-{i}" for i in range(n_items)]),
                )
            continue
        exec(compile(code, f"{path.name}#cell-{index}", "exec"), namespace)

    assert namespace["tutorial_model"].is_fitted
    assert namespace["tutorial_result"].metrics["ndcg@20"] >= 0.0
