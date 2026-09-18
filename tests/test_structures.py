import numpy as np
import pytest

from elementary_transformer import generators as gen
from elementary_transformer.structures import GRAPH, ORDER, RelationSymbol, Signature, Structure, edges, graph

from .conftest import MIXED, random_structure


def test_json_roundtrip(rng):
    for n in range(0, 6):
        s = random_structure(rng, MIXED, n)
        assert Structure.from_json(s.to_json()) == s


def test_declared_properties_are_checked():
    with pytest.raises(ValueError):
        Structure(GRAPH, 2, [np.array([[False, True], [False, False]])])
    with pytest.raises(ValueError):
        Structure(ORDER, 1, [np.array([[True]])])
    with pytest.raises(ValueError):
        RelationSymbol("U", 1, symmetric=True)
    with pytest.raises(ValueError):
        Signature((RelationSymbol("E", 2), RelationSymbol("E", 1)))


def test_permute_is_an_isomorphism(rng):
    s = random_structure(rng, MIXED, 6)
    perm = rng.permutation(6)
    t = s.permute(perm)
    for sym in MIXED:
        for tup in s.tuples(sym.name):
            assert t.holds(sym.name, [perm[a] for a in tup])
        assert len(s.tuples(sym.name)) == len(t.tuples(sym.name))


def test_disjoint_union_and_edges():
    g = gen.cycle(3).disjoint_union(gen.path(2))
    assert g.size == 5
    assert edges(g) == [(0, 1), (0, 2), (1, 2), (3, 4)]
    assert graph(3, [(0, 1)]) == Structure.from_tuples(GRAPH, 3, {"E": [(1, 0)]})
