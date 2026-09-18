import itertools

import numpy as np
import pytest

from elementary_transformer import formulas as fo
from elementary_transformer.structures import Structure

from .conftest import MIXED, random_structure


def naive_holds(s: Structure, phi: fo.Formula, rho: dict[int, int]) -> bool:
    if isinstance(phi, fo.Rel):
        return bool(s.relation(phi.symbol)[tuple(rho[a] for a in phi.args)])
    if isinstance(phi, fo.Eq):
        return rho[phi.left] == rho[phi.right]
    if isinstance(phi, fo.Not):
        return not naive_holds(s, phi.body, rho)
    if isinstance(phi, fo.And):
        return all(naive_holds(s, p, rho) for p in phi.parts)
    if isinstance(phi, fo.Or):
        return any(naive_holds(s, p, rho) for p in phi.parts)
    values = (naive_holds(s, phi.body, {**rho, phi.var: a}) for a in range(s.size))
    return any(values) if isinstance(phi, fo.Exists) else all(values)


def random_formula(rng: np.random.Generator, k: int, depth: int) -> fo.Formula:
    choice = rng.integers(0, 7 if depth > 0 else 2)
    if choice == 0:
        sym = MIXED.symbols[rng.integers(0, len(MIXED))]
        return fo.Rel(sym.name, tuple(int(x) for x in rng.integers(0, k, sym.arity)))
    if choice == 1:
        return fo.Eq(int(rng.integers(0, k)), int(rng.integers(0, k)))
    if choice == 2:
        return fo.Not(random_formula(rng, k, depth - 1))
    if choice in (3, 4):
        parts = tuple(random_formula(rng, k, depth - 1) for _ in range(int(rng.integers(0, 3))))
        return fo.And(parts) if choice == 3 else fo.Or(parts)
    body = random_formula(rng, k, depth - 1)
    var = int(rng.integers(0, k))
    return fo.Exists(var, body) if choice == 5 else fo.Forall(var, body)


def test_model_checker_agrees_with_definition(rng):
    for _ in range(300):
        k = int(rng.integers(1, 4))
        phi = random_formula(rng, k, 4)
        s = random_structure(rng, MIXED, int(rng.integers(1, 5)))
        free = sorted(fo.free_variables(phi))
        for values in itertools.islice(itertools.product(range(s.size), repeat=len(free)), 20):
            rho = dict(zip(free, values, strict=True))
            assert fo.holds(s, phi, rho) == naive_holds(s, phi, rho)


def test_syntactic_measures():
    phi = fo.Exists(0, fo.And((fo.Rel("R", (0, 1)), fo.Forall(1, fo.Eq(0, 1)))))
    assert fo.rank(phi) == 2
    assert fo.width(phi) == 2
    assert fo.free_variables(phi) == frozenset({1})
    assert not fo.is_sentence(phi)
    assert fo.is_sentence(fo.Exists(1, phi))


def test_serialisations_roundtrip(rng):
    for _ in range(200):
        phi = random_formula(rng, 3, 4)
        assert fo.from_json(fo.to_json(phi)) == phi
        back = fo.from_prefix_tokens(fo.prefix_tokens(phi), MIXED)
        s = random_structure(rng, MIXED, 3)
        free = sorted(fo.free_variables(phi))
        rho = dict.fromkeys(free, 0)
        assert fo.holds(s, back, rho) == fo.holds(s, phi, rho)
        assert fo.rank(back) == fo.rank(phi)
        assert set(fo.prefix_tokens(phi)) <= set(fo.formula_vocabulary(MIXED, 3))


def test_holds_requires_free_variables():
    with pytest.raises(ValueError):
        fo.holds(random_structure(np.random.default_rng(0), MIXED, 2), fo.Rel("R", (0, 1)), {0: 0})
