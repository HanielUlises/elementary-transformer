"""Exact solver for the k-pebble Ehrenfeucht–Fraïssé game with q rounds.

A position of the game on ``(A, B)`` is a pair of pebble configurations: for
every pebble ``i < k`` either both copies are off the board or pebble ``i``
lies on ``a_i`` in ``A`` and on ``b_i`` in ``B``. In each round Spoiler picks a
pebble and places it on an element of one structure; Duplicator answers by
placing the matching pebble in the other structure. Spoiler wins as soon as
the pebbled elements do not induce a partial isomorphism.

The solver never enumerates pairs of configurations. The C++ core computes,
for each structure separately but with a shared naming, the rank-r FO^k type
of every configuration in ``(A ∪ {⊥})^k``: the type at round ``r + 1`` is the
type at round ``r`` together with, for every pebble ``i``, the set of round-r
types reachable by moving pebble ``i``. By induction on ``r``, Duplicator wins
``r`` rounds from ``(ā, b̄)`` iff both configurations have the same round-r
type, so these tables memoise the value of every position. Spoiler's
strategy and a distinguishing formula are read off the tables.
"""

from __future__ import annotations

import itertools
from collections import OrderedDict
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from functools import cache
from typing import Literal

import numpy as np

from . import formulas as fo
from .structures import Structure

try:
    from . import _core
except ImportError as exc:  # pragma: no cover - exercised only without a build
    raise ImportError(
        "the C++ extension elementary_transformer._core is not built; "
        "install the package with `pip install .` (a C++17 compiler and CMake are required)"
    ) from exc

Side = Literal[0, 1]
LEFT: Side = 0
RIGHT: Side = 1
Config = tuple["int | None", ...]


@dataclass(frozen=True)
class Move:
    """Spoiler places pebble ``pebble`` on ``element`` of the left (0) or right (1) structure."""

    pebble: int
    side: Side
    element: int


@dataclass(frozen=True)
class ResponseClass:
    """Duplicator answers that lead to positions of the same type, with Spoiler's continuation."""

    elements: tuple[int, ...]
    node: StrategyNode


@dataclass(frozen=True)
class StrategyNode:
    """Spoiler's strategy from a position that Spoiler wins within ``rounds`` rounds.

    ``move`` is ``None`` when the position is already lost for Duplicator,
    that is, when the pebbled elements do not form a partial isomorphism.
    """

    left: Config
    right: Config
    rounds: int
    move: Move | None
    responses: tuple[ResponseClass, ...]

    def size(self) -> int:
        return 1 + sum(c.node.size() for c in self.responses)


def _relation_inputs(s: Structure) -> list[tuple[int, np.ndarray]]:
    return [(a.ndim, np.ascontiguousarray(a, dtype=np.uint8).ravel()) for a in s.relations]


class TypeTables:
    """Round-by-round FO^k types of all pebble configurations of two structures."""

    def __init__(self, left: Structure, right: Structure, k: int, q_max: int) -> None:
        if left.signature != right.signature:
            raise ValueError("the structures must share a signature")
        raw = _core.refine(left.size, _relation_inputs(left), right.size, _relation_inputs(right), k, q_max)
        self.structures = (left, right)
        self.k = k
        self.q_max = q_max
        self.q_star: int | None = raw["q_star"]
        self.stable: bool = raw["stable"]
        self.num_types: list[int] = list(raw["num_types"])
        self.tables: tuple[list[np.ndarray], list[np.ndarray]] = (list(raw["left"]), list(raw["right"]))
        self._bases = (left.size + 1, right.size + 1)

    @property
    def rounds(self) -> int:
        """Largest round whose tables were computed."""
        return len(self.tables[0]) - 1

    def index(self, side: Side, config: Config) -> int:
        b = self._bases[side]
        return sum((0 if a is None else a + 1) * b**i for i, a in enumerate(config))

    def config(self, side: Side, index: int) -> Config:
        b = self._bases[side]
        out: list[int | None] = []
        for _ in range(self.k):
            d = index % b
            out.append(None if d == 0 else d - 1)
            index //= b
        return tuple(out)

    def type(self, side: Side, r: int, index: int) -> int:
        return int(self.tables[side][r][index])

    def child(self, side: Side, index: int, pebble: int, element: int) -> int:
        b = self._bases[side]
        p = b**pebble
        d = (index // p) % b
        return index + (element + 1 - d) * p

    def children(self, side: Side, index: int, pebble: int) -> np.ndarray:
        """Indices of the configurations obtained by moving ``pebble`` to each element."""
        b = self._bases[side]
        p = b**pebble
        base = index - ((index // p) % b) * p
        return base + (np.arange(b - 1, dtype=np.int64) + 1) * p

    def separation_rank(self, s: int, t: int) -> int | None:
        """Least computed round whose types differ at the left configuration ``s`` and right ``t``."""
        for r in range(self.rounds + 1):
            if self.tables[0][r][s] != self.tables[1][r][t]:
                return r
        return None


class _SatisfactionCache:
    """Satisfaction arrays of subformulas, bounded in total size."""

    def __init__(self, structures: tuple[Structure, Structure], k: int, budget_bytes: int = 1 << 28) -> None:
        self.structures = structures
        self.k = k
        self.budget = budget_bytes
        self.used = 0
        self.entries: OrderedDict[tuple[int, int], tuple[fo.Formula, np.ndarray]] = OrderedDict()

    def holds(self, phi: fo.Formula, side: Side, config: Config) -> bool:
        key = (id(phi), side)
        entry = self.entries.get(key)
        if entry is None or entry[0] is not phi:
            sat = fo.satisfaction(self.structures[side], phi, self.k)
            entry = (phi, sat)
            self.entries[key] = entry
            self.used += sat.nbytes
            while self.used > self.budget and len(self.entries) > 1:
                _, (_, old) = self.entries.popitem(last=False)
                self.used -= old.nbytes
        else:
            self.entries.move_to_end(key)
        sat = entry[1]
        index = tuple(0 if sat.shape[i] == 1 else config[i] for i in range(self.k))
        return bool(sat[index])


class SpoilerStrategy:
    """A winning strategy for Spoiler read off the type tables.

    From a position ``(s, t)`` whose types first differ at round ``r > 0``,
    Spoiler moves a pebble ``i`` to an element whose round-(r-1) child type
    does not occur among the round-(r-1) child types of pebble ``i`` on the
    other side. Every answer of Duplicator then leads to a position whose
    types differ at round ``r - 1``. Among such moves the strategy prefers the
    one with the fewest classes of answers, which keeps the extracted formula
    small.
    """

    def __init__(self, tables: TypeTables) -> None:
        if tables.q_star is None:
            raise ValueError("Duplicator wins every computed round; Spoiler has no winning strategy")
        self.tables = tables
        self.k = tables.k
        self.structures = tables.structures
        self._moves: dict[tuple[int, int], Move | None] = {}
        self._formulas: dict[tuple[int, int], fo.Formula] = {}
        self._sat = _SatisfactionCache(self.structures, self.k)

    @property
    def rounds(self) -> int:
        q = self.tables.q_star
        assert q is not None
        return q

    def root(self) -> tuple[int, int]:
        return 0, 0

    def move_at(self, s: int, t: int) -> Move | None:
        """Spoiler's move at the position given by configuration indices ``(s, t)``."""
        key = (s, t)
        if key in self._moves:
            return self._moves[key]
        tab = self.tables
        r = tab.separation_rank(s, t)
        if r is None:
            raise ValueError("Duplicator wins from this position")
        if r == 0:
            self._moves[key] = None
            return None
        best: tuple[tuple[int, int, int, int], Move] | None = None
        for pebble in range(self.k):
            kids = (tab.children(LEFT, s, pebble), tab.children(RIGHT, t, pebble))
            types = (tab.tables[LEFT][r - 1][kids[0]], tab.tables[RIGHT][r - 1][kids[1]])
            for side in (LEFT, RIGHT):
                other = np.unique(types[1 - side])
                fresh = np.flatnonzero(~np.isin(types[side], other))
                if fresh.size == 0:
                    continue
                cost = (int(other.size), side, pebble, int(fresh[0]))
                if best is None or cost < best[0]:
                    best = (cost, Move(pebble, side, int(fresh[0])))
        assert best is not None, "type tables are inconsistent"
        self._moves[key] = best[1]
        return best[1]

    def response_classes(self, s: int, t: int) -> list[tuple[tuple[int, ...], tuple[int, int]]]:
        """Duplicator's answers to Spoiler's move at ``(s, t)``, grouped by the type they lead to.

        Each class is returned with the position reached by its first element.
        """
        move = self.move_at(s, t)
        if move is None:
            return []
        tab = self.tables
        r = tab.separation_rank(s, t)
        assert r is not None
        answer_side: Side = RIGHT if move.side == LEFT else LEFT
        own = s if answer_side == LEFT else t
        kids = tab.children(answer_side, own, move.pebble)
        types = tab.tables[answer_side][r - 1][kids]
        classes: dict[int, list[int]] = {}
        for element, ty in enumerate(types.tolist()):
            classes.setdefault(ty, []).append(element)
        out = []
        for elements in classes.values():
            s2, t2 = self.next_position(s, t, elements[0])
            out.append((tuple(elements), (s2, t2)))
        return out

    def next_position(self, s: int, t: int, answer: int) -> tuple[int, int]:
        """Position after Spoiler's move at ``(s, t)`` and Duplicator's ``answer``."""
        move = self.move_at(s, t)
        if move is None:
            raise ValueError("the game is already over at this position")
        tab = self.tables
        if move.side == LEFT:
            return tab.child(LEFT, s, move.pebble, move.element), tab.child(RIGHT, t, move.pebble, answer)
        return tab.child(LEFT, s, move.pebble, answer), tab.child(RIGHT, t, move.pebble, move.element)

    def tree(self, max_nodes: int = 100_000) -> StrategyNode:
        """The strategy as an explicit tree with answers grouped into classes."""
        count = 0

        def build(s: int, t: int) -> StrategyNode:
            nonlocal count
            count += 1
            if count > max_nodes:
                raise RuntimeError(f"strategy tree exceeds {max_nodes} nodes")
            r = self.tables.separation_rank(s, t)
            assert r is not None
            responses = tuple(ResponseClass(el, build(*pos)) for el, pos in self.response_classes(s, t))
            return StrategyNode(
                left=self.tables.config(LEFT, s),
                right=self.tables.config(RIGHT, t),
                rounds=r,
                move=self.move_at(s, t),
                responses=responses,
            )

        return build(*self.root())

    def play(self, duplicator: Callable[[Config, Config, Move], int]) -> list[tuple[Move, int]]:
        """Play against ``duplicator`` (a function of the position and Spoiler's move) until Spoiler wins."""
        s, t = self.root()
        history: list[tuple[Move, int]] = []
        while (move := self.move_at(s, t)) is not None:
            answer = duplicator(self.tables.config(LEFT, s), self.tables.config(RIGHT, t), move)
            history.append((move, answer))
            s, t = self.next_position(s, t, answer)
        return history

    # -- formula extraction -------------------------------------------------

    def formula(self) -> fo.Formula:
        """A sentence of FO^k with quantifier rank ``q_star``, true in the left structure and false in the right."""
        return self._formula_at(*self.root())

    def _formula_at(self, s: int, t: int) -> fo.Formula:
        key = (s, t)
        if key in self._formulas:
            return self._formulas[key]
        move = self.move_at(s, t)
        if move is None:
            phi = self._literal(s, t)
        elif move.side == LEFT:
            # A ⊨ ∃x_i ⋀ψ_c witnessed by the chosen element; every answer b in B
            # falls in some class c whose conjunct ψ_c fails at (t, i ↦ b).
            parts: list[fo.Formula] = []
            for _, (s2, t2) in self.response_classes(s, t):
                c2 = self.tables.config(RIGHT, t2)
                if any(not self._sat.holds(p, RIGHT, c2) for p in parts):
                    continue
                parts.append(self._formula_at(s2, t2))
            phi = fo.Exists(move.pebble, fo.conj(parts))
        else:
            # B ⊭ ∀x_i ⋁χ_c at the chosen element; every a in A falls in some
            # class c whose disjunct χ_c holds at (s, i ↦ a).
            parts = []
            for _, (s2, t2) in self.response_classes(s, t):
                c2 = self.tables.config(LEFT, s2)
                if any(self._sat.holds(p, LEFT, c2) for p in parts):
                    continue
                parts.append(self._formula_at(s2, t2))
            phi = fo.Forall(move.pebble, fo.disj(parts))
        self._formulas[key] = phi
        return phi

    def _literal(self, s: int, t: int) -> fo.Formula:
        """An atomic formula or negated atomic formula true at ``s`` and false at ``t``."""
        a = self.tables.config(LEFT, s)
        b = self.tables.config(RIGHT, t)
        placed = [i for i in range(self.k) if a[i] is not None]
        if placed != [i for i in range(self.k) if b[i] is not None]:
            raise AssertionError("positions with different pebbles on the board")
        for i, j in itertools.combinations(placed, 2):
            ea, eb = a[i] == a[j], b[i] == b[j]
            if ea != eb:
                return fo.Eq(i, j) if ea else fo.Not(fo.Eq(i, j))
        left, right = self.structures
        for symbol in left.signature:
            ra, rb = left.relation(symbol.name), right.relation(symbol.name)
            for args in itertools.product(placed, repeat=symbol.arity):
                ha = bool(ra[tuple(a[x] for x in args)])
                hb = bool(rb[tuple(b[x] for x in args)])
                if ha != hb:
                    atom = fo.Rel(symbol.name, tuple(args))
                    return atom if ha else fo.Not(atom)
        raise AssertionError("positions with the same atomic type")


@dataclass
class GameResult:
    """Outcome of the k-pebble game with at most ``q_max`` rounds.

    ``q_star`` is the least number of rounds in which Spoiler wins, which is
    the least quantifier rank of an FO^k sentence separating the structures,
    or ``None`` when Duplicator wins the ``q_max``-round game. When ``stable``
    is true and ``q_star`` is ``None`` the structures are FO^k-equivalent for
    every rank.
    """

    k: int
    q_max: int
    q_star: int | None
    stable: bool
    num_types: list[int]
    strategy: SpoilerStrategy | None
    formula: fo.Formula | None

    @property
    def equivalent(self) -> bool:
        return self.q_star is None


def solve(left: Structure, right: Structure, k: int, q_max: int, *, extract_formula: bool = True) -> GameResult:
    """Solve the k-pebble game with at most ``q_max`` rounds on ``(left, right)``."""
    tables = TypeTables(left, right, k, q_max)
    strategy = SpoilerStrategy(tables) if tables.q_star is not None else None
    formula = strategy.formula() if strategy is not None and extract_formula else None
    return GameResult(
        k=k,
        q_max=q_max,
        q_star=tables.q_star,
        stable=tables.stable,
        num_types=tables.num_types,
        strategy=strategy,
        formula=formula,
    )


def q_star(left: Structure, right: Structure, k: int, q_max: int) -> int | None:
    return solve(left, right, k, q_max, extract_formula=False).q_star


def equivalent(left: Structure, right: Structure, k: int, q: int) -> bool:
    """Whether ``left`` and ``right`` agree on all FO^k sentences of quantifier rank at most ``q``."""
    return q_star(left, right, k, q) is None


# ---------------------------------------------------------------------------
# Direct game search, used to cross-check the solver on small instances


def is_partial_isomorphism(left: Structure, right: Structure, pairs: Sequence[tuple[int, int]]) -> bool:
    """Whether ``a_i -> b_i`` is a well-defined injective map preserving all relations both ways."""
    for (a1, b1), (a2, b2) in itertools.combinations(pairs, 2):
        if (a1 == a2) != (b1 == b2):
            return False
    for symbol in left.signature:
        ra, rb = left.relation(symbol.name), right.relation(symbol.name)
        for idx in itertools.product(range(len(pairs)), repeat=symbol.arity):
            if bool(ra[tuple(pairs[i][0] for i in idx)]) != bool(rb[tuple(pairs[i][1] for i in idx)]):
                return False
    return True


def reference_duplicator_wins(left: Structure, right: Structure, k: int, rounds: int) -> bool:
    """Top-down search over pairs of configurations, memoised on (configuration, rounds left)."""

    @cache
    def wins(config: tuple[tuple[int, int] | None, ...], r: int) -> bool:
        if not is_partial_isomorphism(left, right, [p for p in config if p is not None]):
            return False
        if r == 0:
            return True
        for i in range(k):
            for a in range(left.size):
                if not any(wins(config[:i] + ((a, b),) + config[i + 1 :], r - 1) for b in range(right.size)):
                    return False
            for b in range(right.size):
                if not any(wins(config[:i] + ((a, b),) + config[i + 1 :], r - 1) for a in range(left.size)):
                    return False
        return True

    return wins((None,) * k, rounds)


def reference_q_star(left: Structure, right: Structure, k: int, q_max: int) -> int | None:
    for q in range(q_max + 1):
        if not reference_duplicator_wins(left, right, k, q):
            return q
    return None


def duplicator_answers(strategy: SpoilerStrategy) -> Iterator[list[tuple[Move, int]]]:
    """All plays of ``strategy`` against every possible sequence of Duplicator answers."""
    left, right = strategy.structures

    def walk(s: int, t: int, history: list[tuple[Move, int]]) -> Iterator[list[tuple[Move, int]]]:
        move = strategy.move_at(s, t)
        if move is None:
            yield history
            return
        answers = right.size if move.side == LEFT else left.size
        for answer in range(answers):
            yield from walk(*strategy.next_position(s, t, answer), history + [(move, answer)])

    yield from walk(*strategy.root(), [])
