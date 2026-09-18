import pytest

from elementary_transformer import formulas as fo
from elementary_transformer import games, lean
from elementary_transformer import generators as gen

pytestmark = pytest.mark.skipif(not lean.available(), reason="the Lean checker is not built (run `lake build`)")


def test_lean_accepts_extracted_formulas_and_rejects_wrong_ones():
    cases = [
        (gen.cycle(6), gen.disjoint_cycles([3, 3]), 3),
        (gen.linear_order(6), gen.linear_order(7), 3),
        (gen.rook_graph(4), gen.shrikhande(), 4),
        (gen.path(4), gen.complete_bipartite(1, 3), 2),
    ]
    good, bad = [], []
    for left, right, k in cases:
        result = games.solve(left, right, k, 4)
        assert result.formula is not None
        good.append(lean.certificate(left, right, result.formula, k, result.q_star))
        bad.append(lean.certificate(left, right, fo.Not(result.formula), k, result.q_star))
        bad.append(lean.certificate(left, right, result.formula, k, result.q_star - 1))
        bad.append(lean.certificate(left, right, result.formula, fo.width(result.formula) - 1, result.q_star))
    assert lean.check(good) == [True] * len(good)
    assert lean.check(bad) == [False] * len(bad)
