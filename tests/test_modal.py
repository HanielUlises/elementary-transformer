from functools import cache

import numpy as np

from elementary_transformer import games, modal


def reference_bisimilar(m: modal.KripkeModel, n: modal.KripkeModel, depth: int, w: int, v: int) -> bool:
    @cache
    def bis(d: int, x: int, y: int) -> bool:
        if not np.array_equal(m.valuation[x], n.valuation[y]):
            return False
        if d == 0:
            return True
        for a in range(m.num_agents):
            succ_m = np.flatnonzero(m.accessibility[a][x])
            succ_n = np.flatnonzero(n.accessibility[a][y])
            if not all(any(bis(d - 1, x2, y2) for y2 in succ_n) for x2 in succ_m):
                return False
            if not all(any(bis(d - 1, x2, y2) for x2 in succ_m) for y2 in succ_n):
                return False
        return True

    return bis(depth, w, v)


def test_partition_refinement_matches_definition():
    for seed in range(12):
        m = modal.random_kripke(seed, 5, 2, 1, 0.3)
        n = modal.random_kripke(seed + 50, 4, 2, 1, 0.3)
        for w in range(m.num_worlds):
            for v in range(n.num_worlds):
                for depth in range(4):
                    assert modal.n_bisimilar(m, w, n, v, depth) == reference_bisimilar(m, n, depth, w, v)


def test_copies_are_bisimilar():
    m = modal.random_kripke(3, 6, 2, 2, 0.4)
    perm = np.random.default_rng(0).permutation(6)
    inverse = np.argsort(perm)
    copy = modal.KripkeModel(tuple(r[np.ix_(inverse, inverse)] for r in m.accessibility), m.valuation[inverse])
    for w in range(6):
        assert modal.distinguishing_depth(m, w, copy, int(perm[w])) is None


def test_modal_depth_bounds_the_two_variable_rank():
    # The standard translation of a modal formula of depth d uses two variables and rank d, so
    # pointed models separated at modal depth d are separated by an FO^2 sentence of rank ≤ d + 1.
    found = 0
    for seed in range(20):
        m = modal.random_kripke(seed, 4, 1, 1, 0.35)
        n = modal.random_kripke(seed + 100, 4, 1, 1, 0.35)
        d = modal.distinguishing_depth(m, 0, n, 0, 4)
        if d is None:
            continue
        q = games.q_star(m.pointed(0), n.pointed(0), 2, d + 1)
        assert q is not None and q <= d + 1
        found += 1
    assert found > 5
