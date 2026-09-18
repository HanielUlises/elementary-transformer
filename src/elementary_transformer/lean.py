"""Independent verification of distinguishing formulas by the Lean checker.

The executable ``et-check`` is built from ``lean/Main.lean`` with ``lake
build``. It decides each certificate with an evaluator proved correct with
respect to the Tarskian semantics (``Formula.eval_iff_sat`` and
``Certificate.check_sound``). The checker is located through the environment
variable ``ELEMENTARY_TRANSFORMER_LEAN_CHECKER``, then in the Lake build
directory of the source tree, then on the ``PATH``.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

from . import formulas as fo
from .structures import Structure

if TYPE_CHECKING:
    from .dataset import Example

ENV_VAR = "ELEMENTARY_TRANSFORMER_LEAN_CHECKER"


def checker_path() -> Path | None:
    explicit = os.environ.get(ENV_VAR)
    if explicit:
        return Path(explicit)
    built = Path(__file__).resolve().parents[2] / ".lake" / "build" / "bin" / "et-check"
    if built.is_file():
        return built
    found = shutil.which("et-check")
    return Path(found) if found else None


def available() -> bool:
    return checker_path() is not None


def structure_json(s: Structure) -> dict[str, Any]:
    return {
        "size": s.size,
        "relations": [
            {"name": sym.name, "arity": sym.arity, "tuples": [list(t) for t in s.tuples(sym.name)]} for sym in s.signature
        ],
    }


def certificate(left: Structure, right: Structure, formula: fo.Formula, k: int, q: int) -> dict[str, Any]:
    return {
        "k": k,
        "q": q,
        "left": structure_json(left),
        "right": structure_json(right),
        "formula": fo.to_binary_json(formula),
    }


def check(certificates: Sequence[dict[str, Any]]) -> list[bool]:
    """Run the Lean checker on the certificates and return one verdict per certificate."""
    exe = checker_path()
    if exe is None:
        raise FileNotFoundError("the Lean checker et-check is not built; run `lake build`")
    if not certificates:
        return []
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "certificates.jsonl"
        path.write_text("".join(json.dumps(c) + "\n" for c in certificates))
        proc = subprocess.run([str(exe), str(path)], capture_output=True, text=True)
    if proc.returncode not in (0, 1):
        raise RuntimeError(f"et-check failed: {proc.stderr.strip()}")
    results = [json.loads(line) for line in proc.stdout.splitlines() if line.strip()]
    if len(results) != len(certificates):
        raise RuntimeError("et-check returned an unexpected number of results")
    return [bool(r["valid"]) for r in results]


def check_examples(examples: Sequence[Example]) -> list[bool | None]:
    """Verdicts for the examples that carry a formula, ``None`` for the others."""
    indices = [i for i, ex in enumerate(examples) if ex.formula is not None]
    certs = []
    for i in indices:
        ex = examples[i]
        assert ex.formula is not None and ex.q_star is not None
        certs.append(certificate(ex.pair.left, ex.pair.right, ex.formula, ex.k, ex.q_star))
    verdicts: list[bool | None] = [None] * len(examples)
    for i, ok in zip(indices, check(certs), strict=True):
        verdicts[i] = ok
    return verdicts
