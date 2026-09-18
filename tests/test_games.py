import itertools

import numpy as np
import pytest

from elementary_transformer import formulas as fo
from elementary_transformer import games
from elementary_transformer import generators as gen
from elementary_transformer.structures import GRAPH, RelationSymbol, Signature, Structure

from .conftest import MIXED, random_graph, random_structure

DIGRAPH = Signature((RelationSymbol("R", 2),))


def expected_order_rank(m: int, n: int) -> int | None:
    """Least q with L_m ≢_q L_n: they agree up to rank q iff m = n or both are ≥ 2^q − 1."""
    if m == n:
        return None
    q = 0
    while min(m, n) >= 2**q - 1:
        q += 1
    return q


@pytest.mark.parametrize("q_max", [1, 2, 3, 4])
def test_linear_orders(q_max):
    # With k = q_max pebbles the q-round game is the ordinary EF game for all q ≤ q_max.
    for m, n in itertools.product(range(1, 18), repeat=2):
        expected = expected_order_rank(m, n)
        if expected is not None and expected > q_max:
            expected = None
        result = games.solve(gen.linear_order(m), gen.linear_order(n), q_max, q_max)
        assert result.q_star == expected, (m, n)
        if result.formula is not None:
            assert fo.distinguishes(result.formula, gen.linear_order(m), gen.linear_order(n))
            assert fo.rank(result.formula) == expected


def test_threshold_orders_equivalent_at_every_rank_up_to_four():
    for q in range(1, 5):
        low = 2**q - 1
        assert games.equivalent(gen.linear_order(low), gen.linear_order(low + 5), q, q)
        assert not games.equivalent(gen.linear_order(low - 1), gen.linear_order(low), q, q)


def test_agrees_with_direct_game_search(rng):
    for trial in range(60):
        signature = [GRAPH, DIGRAPH, MIXED][trial % 3]
        k = int(rng.integers(1, 4))
        q = int(rng.integers(1, 4 if k < 3 else 3))
        a = random_structure(rng, signature, int(rng.integers(1, 5)))
        b = random_structure(rng, signature, int(rng.integers(1, 5)))
        assert games.q_star(a, b, k, q) == games.reference_q_star(a, b, k, q), (trial, k, q)


def test_isomorphic_pairs_are_equivalent(rng):
    for trial in range(40):
        signature = [GRAPH, DIGRAPH, MIXED][trial % 3]
        a = random_structure(rng, signature, int(rng.integers(1, 9)))
        b = a.permute(rng.permutation(a.size))
        result = games.solve(a, b, 3, 6)
        assert result.q_star is None and result.stable


def test_extracted_formulas_are_verified(rng):
    checked = 0
    for trial in range(150):
        signature = [GRAPH, DIGRAPH, MIXED][trial % 3]
        k = int(rng.integers(1, 5))
        a = random_structure(rng, signature, int(rng.integers(1, 8)))
        b = random_structure(rng, signature, int(rng.integers(1, 8)))
        result = games.solve(a, b, k, 4)
        if result.q_star is None:
            continue
        phi = result.formula
        assert phi is not None
        assert fo.is_sentence(phi)
        assert fo.rank(phi) == result.q_star
        assert fo.width(phi) <= k
        assert fo.holds(a, phi) and not fo.holds(b, phi)
        checked += 1
    assert checked > 50


def test_spoiler_strategy_wins_against_every_duplicator(rng):
    for trial in range(30):
        a = random_graph(rng, int(rng.integers(2, 6)))
        b = random_graph(rng, int(rng.integers(2, 6)))
        result = games.solve(a, b, 2 + trial % 2, 3)
        if result.strategy is None:
            continue
        k = result.k
        for play in games.duplicator_answers(result.strategy):
            assert len(play) <= result.q_star
            config: list[tuple[int, int] | None] = [None] * k
            for move, answer in play:
                pair = (move.element, answer) if move.side == games.LEFT else (answer, move.element)
                config[move.pebble] = pair
            placed = [p for p in config if p is not None]
            assert not games.is_partial_isomorphism(a, b, placed)


def test_strategy_tree_and_play():
    c6, two_triangles = gen.cycle(6), gen.disjoint_cycles([3, 3])
    result = games.solve(c6, two_triangles, 3, 5)
    assert result.q_star == 3
    tree = result.strategy.tree()
    assert tree.rounds == 3 and tree.move is not None
    history = result.strategy.play(lambda left, right, move: 0)
    assert 1 <= len(history) <= 3


def test_known_small_cases():
    c6, two_triangles = gen.cycle(6), gen.disjoint_cycles([3, 3])
    assert games.solve(c6, two_triangles, 3, 5).q_star == 3
    two = games.solve(c6, two_triangles, 2, 10)
    assert two.q_star is None and two.stable
    # FO^2 cannot count the leaves of a star.
    star3, star4 = gen.complete_bipartite(1, 3), gen.complete_bipartite(1, 4)
    assert games.solve(star3, star4, 2, 10).stable
    assert games.q_star(star3, star4, 2, 10) is None
    # Four pairwise non-adjacent elements exist only in K_{1,4}; three rounds never suffice.
    assert games.q_star(star3, star4, 4, 5) == 4


def test_signature_mismatch_is_rejected():
    with pytest.raises(ValueError):
        games.solve(gen.linear_order(3), gen.path(3), 2, 2)


def test_empty_structures():
    empty = Structure(GRAPH, 0, [np.zeros((0, 0), dtype=bool)])
    assert games.q_star(empty, empty, 2, 3) is None
    result = games.solve(empty, gen.path(1), 2, 3)
    assert result.q_star == 1
    assert fo.distinguishes(result.formula, gen.path(1), empty) or fo.distinguishes(result.formula, empty, gen.path(1))
