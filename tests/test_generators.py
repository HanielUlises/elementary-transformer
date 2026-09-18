import numpy as np
import pytest

from elementary_transformer import generators as gen
from elementary_transformer import pairs, wl
from elementary_transformer.structures import degrees, edges

from .conftest import random_graph


def test_graph6():
    assert gen.parse_graph6("Bw") == gen.complete(3)
    assert gen.parse_graph6("C~") == gen.complete(4)
    assert gen.parse_graph6(">>graph6<<C?").size == 4
    rng = np.random.default_rng(1)
    for n in (0, 1, 2, 7, 13, 70):
        g = random_graph(rng, n, 0.3)
        assert gen.parse_graph6(gen.to_graph6(g)) == g


def test_random_graphs_are_reproducible():
    assert gen.gnp(7, 12, 0.5) == gen.gnp(7, 12, 0.5)
    assert gen.random_regular(3, 14, 3) == gen.random_regular(3, 14, 3)


@pytest.mark.parametrize("n,d", [(10, 3), (12, 4), (9, 2), (8, 0)])
def test_random_regular_degrees(n, d):
    g = gen.random_regular(n + d, n, d)
    assert set(degrees(g).tolist()) <= {d}


def test_named_graphs():
    assert len(edges(gen.petersen())) == 15
    assert gen.srg_parameters(gen.petersen()) == (10, 3, 0, 1)
    assert gen.srg_parameters(gen.rook_graph(4)) == (16, 6, 2, 2)
    assert gen.srg_parameters(gen.shrikhande()) == (16, 6, 2, 2)
    assert gen.srg_parameters(gen.triangular(8)) == (28, 12, 6, 4)
    for g in gen.chang_graphs():
        assert gen.srg_parameters(g) == (28, 12, 6, 4)
    assert gen.srg_parameters(gen.cycle(6)) is None


def test_chang_graphs_are_pairwise_non_isomorphic():
    graphs = [gen.triangular(8), *gen.chang_graphs()]
    for i in range(len(graphs)):
        for j in range(i + 1, len(graphs)):
            assert not wl.wl_equivalent(graphs[i], graphs[j], 3), (i, j)


def test_cfi_graphs_are_cubic_and_sized():
    for _name, make in gen.CFI_BASES.items():
        base = make()
        x, y = gen.cfi_pair(base)
        assert x.size == y.size == 10 * base.size
        assert set(degrees(x).tolist()) == {3} == set(degrees(y).tolist())


def cfi_flip(base, v, i, j):
    """The permutation of X(G) that swaps a/b at the i-th and j-th ends of v and shifts middles."""
    names = gen.cfi_vertices(base)
    index = {name: pos for pos, name in enumerate(names)}
    perm = list(range(len(names)))
    for pos, (kind, w, x, _) in enumerate(names):
        if w != v:
            continue
        if kind == "m":
            perm[pos] = index[("m", v, x ^ (1 << i) ^ (1 << j), 0)]
        elif x in (i, j):
            perm[pos] = index[("b", v, x, 1) if kind == "a" else ("a", v, x, 0)]
    return perm


def test_cfi_twists_depend_on_parity_only():
    base = gen.complete(4)
    x, x_twisted = gen.cfi_pair(base)
    double = gen.cfi_graph(base, [(0, 1), (0, 2)])
    # Vertex 0 has neighbours 1, 2, 3, so the edges {0,1} and {0,2} are its ends 0 and 1.
    assert x.permute(cfi_flip(base, 0, 0, 1)) == double
    assert x != double
    # Non-isomorphism of the untwisted and twisted graphs, certified by 3-dimensional WL.
    assert wl.wl_equivalent(x, x_twisted, 2)
    assert not wl.wl_equivalent(x, x_twisted, 3)


def test_pair_constructions(rng):
    g = gen.gnp(5, 9, 0.5)
    flip = pairs.edge_flip(1, g)
    assert np.sum(flip.relations[0] != g.relations[0]) == 2
    swap = pairs.edge_swap(2, g)
    assert swap is not None and np.array_equal(degrees(swap), degrees(g)) and swap != g
    assert pairs.edge_swap(0, gen.complete(5)) is None
    copy = pairs.isomorphic_copy(3, g)
    assert sorted(degrees(copy).tolist()) == sorted(degrees(g).tolist())
    p = pairs.regular_pair(4, 10, 3).relabelled(9)
    assert p.left.size == p.right.size == 10


def test_optional_tools_report_availability():
    if gen.geng_executable() is None:
        with pytest.raises(FileNotFoundError):
            next(gen.geng(4))
    else:
        assert len(list(gen.geng(4))) == 11
    if not gen.sage_available():
        with pytest.raises(FileNotFoundError):
            gen.sage_strongly_regular(16, 6, 2, 2)
    else:
        assert gen.srg_parameters(gen.sage_strongly_regular(16, 6, 2, 2)) == (16, 6, 2, 2)
