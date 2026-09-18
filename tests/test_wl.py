from elementary_transformer import games, wl
from elementary_transformer import generators as gen

from .conftest import random_graph


def test_colour_refinement_and_triangles():
    c6, two_triangles = gen.cycle(6), gen.disjoint_cycles([3, 3])
    assert wl.wl_equivalent(c6, two_triangles, 1)
    assert not wl.wl_equivalent(c6, two_triangles, 2)
    assert wl.counting_equivalent(c6, two_triangles, 2)
    assert not wl.counting_equivalent(c6, two_triangles, 3)


def test_strongly_regular_graphs_need_three_dimensions():
    rook, shrikhande = gen.rook_graph(4), gen.shrikhande()
    assert wl.wl_equivalent(rook, shrikhande, 2)
    assert not wl.wl_equivalent(rook, shrikhande, 3)


def test_counting_is_stronger_than_fo_with_the_same_variables():
    star3, star4 = gen.complete_bipartite(1, 3), gen.complete_bipartite(1, 4)
    assert not wl.counting_equivalent(star3, star4, 2)
    assert games.q_star(star3, star4, 2, 8) is None


def test_isomorphic_copies(rng):
    for n in (5, 7):
        g = random_graph(rng, n)
        assert wl.wl_equivalent(g, g.permute(rng.permutation(n)), 2)


def test_consistent_with_the_game_solver(rng):
    # FO^k ⊆ C^k, so C^k-equivalence ((k-1)-dimensional WL) implies FO^k-equivalence at every rank.
    agreements = 0
    for trial in range(80):
        n = int(rng.integers(3, 7))
        a = random_graph(rng, n, 0.5)
        b = a.permute(rng.permutation(n)) if trial % 4 == 0 else random_graph(rng, n, 0.5)
        for k in (2, 3):
            result = games.solve(a, b, k, 6)
            if wl.counting_equivalent(a, b, k):
                assert result.q_star is None
                agreements += 1
            elif result.q_star is not None:
                agreements += 1
    assert agreements > 0


def test_regular_graphs_defeat_colour_refinement():
    for seed in range(3):
        a, b = gen.random_regular(seed, 10, 3), gen.random_regular(seed + 100, 10, 3)
        assert wl.wl_equivalent(a, b, 1)


def test_different_sizes_are_distinguished_immediately():
    result = wl.wl(gen.path(3), gen.path(4), 1)
    assert result.distinguished and result.distinguishing_round == 0
