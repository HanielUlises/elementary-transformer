from __future__ import annotations

import itertools
import os

# The data pipeline runs on the CPU; this also keeps the tests from reserving GPU memory.
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np
import pytest

from elementary_transformer.structures import GRAPH, RelationSymbol, Signature, Structure


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(20260918)


def random_graph(rng: np.random.Generator, n: int, p: float = 0.5) -> Structure:
    upper = np.triu(rng.random((n, n)) < p, 1)
    return Structure(GRAPH, n, [upper | upper.T])


def random_structure(rng: np.random.Generator, signature: Signature, n: int, p: float = 0.4) -> Structure:
    arrays = []
    for s in signature:
        a = rng.random((n,) * s.arity) < p
        if s.symmetric:
            a = np.triu(a, 1)
            a = a | a.T
        if s.irreflexive:
            np.fill_diagonal(a, False)
        arrays.append(a)
    return Structure(signature, n, arrays)


MIXED = Signature((RelationSymbol("R", 2), RelationSymbol("P", 1), RelationSymbol("T", 3)))


def naive_atomic_type(s: Structure, t: tuple[int, ...]) -> tuple:
    """Atomic type of a tuple, computed directly from the definition."""
    k = len(t)
    eqs = tuple(t[i] == t[j] for i, j in itertools.combinations(range(k), 2))
    rels = tuple(
        bool(s.relation(sym.name)[tuple(t[c] for c in g)])
        for sym in s.signature
        for g in itertools.product(range(k), repeat=sym.arity)
    )
    return eqs + rels
