"""First-order formulas over relational signatures and a model checker.

Variables are natural numbers; variable ``i`` is written ``x{i}`` and, in the
game-theoretic reading, is the variable attached to pebble ``i``. Formulas may
reuse variables, so the ``k``-variable fragment FO^k is the set of formulas
whose variables are among ``0, ..., k - 1``.

The model checker evaluates a formula bottom-up. The value of a subformula is
a boolean array with one axis per variable; an axis has length ``n`` when the
variable occurs free in the subformula and length 1 otherwise, so numpy
broadcasting performs the joins. The cost is ``O(|phi| * n^w)`` where ``w`` is
the number of variables.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from .structures import Signature, Structure


@dataclass(frozen=True)
class Rel:
    symbol: str
    args: tuple[int, ...]


@dataclass(frozen=True)
class Eq:
    left: int
    right: int


@dataclass(frozen=True)
class Not:
    body: Formula


@dataclass(frozen=True)
class And:
    parts: tuple[Formula, ...]


@dataclass(frozen=True)
class Or:
    parts: tuple[Formula, ...]


@dataclass(frozen=True)
class Exists:
    var: int
    body: Formula


@dataclass(frozen=True)
class Forall:
    var: int
    body: Formula


Formula = Rel | Eq | Not | And | Or | Exists | Forall

TRUE: Formula = And(())
FALSE: Formula = Or(())


def conj(parts: Sequence[Formula]) -> Formula:
    """Conjunction with flattening and removal of repeated conjuncts."""
    flat: list[Formula] = []
    for p in parts:
        for q in p.parts if isinstance(p, And) else (p,):
            if q not in flat:
                flat.append(q)
    return flat[0] if len(flat) == 1 else And(tuple(flat))


def disj(parts: Sequence[Formula]) -> Formula:
    """Disjunction with flattening and removal of repeated disjuncts."""
    flat: list[Formula] = []
    for p in parts:
        for q in p.parts if isinstance(p, Or) else (p,):
            if q not in flat:
                flat.append(q)
    return flat[0] if len(flat) == 1 else Or(tuple(flat))


def neg(phi: Formula) -> Formula:
    return phi.body if isinstance(phi, Not) else Not(phi)


def subformulas(phi: Formula) -> Iterator[Formula]:
    yield phi
    if isinstance(phi, Not):
        yield from subformulas(phi.body)
    elif isinstance(phi, (And, Or)):
        for p in phi.parts:
            yield from subformulas(p)
    elif isinstance(phi, (Exists, Forall)):
        yield from subformulas(phi.body)


def rank(phi: Formula) -> int:
    """Quantifier rank (maximal nesting depth of quantifiers)."""
    if isinstance(phi, (Rel, Eq)):
        return 0
    if isinstance(phi, Not):
        return rank(phi.body)
    if isinstance(phi, (And, Or)):
        return max((rank(p) for p in phi.parts), default=0)
    return 1 + rank(phi.body)


def size(phi: Formula) -> int:
    """Number of nodes of the syntax tree."""
    return sum(1 for _ in subformulas(phi))


def variables(phi: Formula) -> frozenset[int]:
    """All variables occurring in ``phi``, free or bound."""
    out: set[int] = set()
    for p in subformulas(phi):
        if isinstance(p, Rel):
            out.update(p.args)
        elif isinstance(p, Eq):
            out.update((p.left, p.right))
        elif isinstance(p, (Exists, Forall)):
            out.add(p.var)
    return frozenset(out)


def free_variables(phi: Formula) -> frozenset[int]:
    if isinstance(phi, Rel):
        return frozenset(phi.args)
    if isinstance(phi, Eq):
        return frozenset((phi.left, phi.right))
    if isinstance(phi, Not):
        return free_variables(phi.body)
    if isinstance(phi, (And, Or)):
        return frozenset().union(*(free_variables(p) for p in phi.parts))
    return free_variables(phi.body) - {phi.var}


def width(phi: Formula) -> int:
    """Least ``k`` such that ``phi`` belongs to FO^k with variables ``0..k-1``."""
    vs = variables(phi)
    return max(vs) + 1 if vs else 0


def is_sentence(phi: Formula) -> bool:
    return not free_variables(phi)


# ---------------------------------------------------------------------------
# Printing


def to_text(phi: Formula) -> str:
    """Unicode rendering, for example ``∃x0 (E(x0,x1) ∧ ¬x0=x1)``."""
    if isinstance(phi, Rel):
        return f"{phi.symbol}({','.join(f'x{a}' for a in phi.args)})"
    if isinstance(phi, Eq):
        return f"x{phi.left}=x{phi.right}"
    if isinstance(phi, Not):
        return f"¬{_wrap(phi.body)}"
    if isinstance(phi, And):
        return "⊤" if not phi.parts else " ∧ ".join(_wrap(p) for p in phi.parts)
    if isinstance(phi, Or):
        return "⊥" if not phi.parts else " ∨ ".join(_wrap(p) for p in phi.parts)
    q = "∃" if isinstance(phi, Exists) else "∀"
    return f"{q}x{phi.var} {_wrap(phi.body)}"


def _wrap(phi: Formula) -> str:
    text = to_text(phi)
    return f"({text})" if isinstance(phi, (And, Or)) and len(phi.parts) > 1 else text


def to_latex(phi: Formula) -> str:
    if isinstance(phi, Rel):
        return f"{phi.symbol}({','.join(f'x_{{{a}}}' for a in phi.args)})"
    if isinstance(phi, Eq):
        return f"x_{{{phi.left}}} = x_{{{phi.right}}}"
    if isinstance(phi, Not):
        return rf"\neg {_wrap_latex(phi.body)}"
    if isinstance(phi, And):
        return r"\top" if not phi.parts else r" \wedge ".join(_wrap_latex(p) for p in phi.parts)
    if isinstance(phi, Or):
        return r"\bot" if not phi.parts else r" \vee ".join(_wrap_latex(p) for p in phi.parts)
    q = r"\exists" if isinstance(phi, Exists) else r"\forall"
    return f"{q} x_{{{phi.var}}}\\, {_wrap_latex(phi.body)}"


def _wrap_latex(phi: Formula) -> str:
    text = to_latex(phi)
    return f"({text})" if isinstance(phi, (And, Or)) and len(phi.parts) > 1 else text


# ---------------------------------------------------------------------------
# Serialisation


def to_json(phi: Formula) -> dict[str, Any]:
    """Nested JSON form; conjunctions and disjunctions keep their arity."""
    if isinstance(phi, Rel):
        return {"op": "rel", "symbol": phi.symbol, "args": list(phi.args)}
    if isinstance(phi, Eq):
        return {"op": "eq", "left": phi.left, "right": phi.right}
    if isinstance(phi, Not):
        return {"op": "not", "body": to_json(phi.body)}
    if isinstance(phi, And):
        return {"op": "and", "parts": [to_json(p) for p in phi.parts]}
    if isinstance(phi, Or):
        return {"op": "or", "parts": [to_json(p) for p in phi.parts]}
    op = "exists" if isinstance(phi, Exists) else "forall"
    return {"op": op, "var": phi.var, "body": to_json(phi.body)}


def from_json(data: Mapping[str, Any]) -> Formula:
    op = data["op"]
    if op == "rel":
        return Rel(str(data["symbol"]), tuple(int(a) for a in data["args"]))
    if op == "eq":
        return Eq(int(data["left"]), int(data["right"]))
    if op == "not":
        return Not(from_json(data["body"]))
    if op == "and":
        return And(tuple(from_json(p) for p in data["parts"]))
    if op == "or":
        return Or(tuple(from_json(p) for p in data["parts"]))
    if op == "exists":
        return Exists(int(data["var"]), from_json(data["body"]))
    if op == "forall":
        return Forall(int(data["var"]), from_json(data["body"]))
    raise ValueError(f"unknown operator {op!r}")


def to_binary_json(phi: Formula) -> dict[str, Any]:
    """JSON form with binary connectives, as read by the Lean checker."""
    if isinstance(phi, (And, Or)):
        unit = {"op": "true"} if isinstance(phi, And) else {"op": "false"}
        op = "and" if isinstance(phi, And) else "or"
        if not phi.parts:
            return unit
        out = to_binary_json(phi.parts[-1])
        for p in reversed(phi.parts[:-1]):
            out = {"op": op, "left": to_binary_json(p), "right": out}
        return out
    if isinstance(phi, Not):
        return {"op": "not", "body": to_binary_json(phi.body)}
    if isinstance(phi, (Exists, Forall)):
        op = "exists" if isinstance(phi, Exists) else "forall"
        return {"op": op, "var": phi.var, "body": to_binary_json(phi.body)}
    return to_json(phi)


def prefix_tokens(phi: Formula) -> list[str]:
    """Prefix (Polish) notation with binary connectives.

    Tokens are ``TRUE``, ``FALSE``, ``NOT``, ``AND``, ``OR``, ``EXISTS``,
    ``FORALL``, ``EQ``, ``R:<symbol>`` and ``x<i>``. The arity of ``R:<symbol>``
    is read from the signature, so the notation is unambiguous.
    """
    if isinstance(phi, Rel):
        return [f"R:{phi.symbol}", *(f"x{a}" for a in phi.args)]
    if isinstance(phi, Eq):
        return ["EQ", f"x{phi.left}", f"x{phi.right}"]
    if isinstance(phi, Not):
        return ["NOT", *prefix_tokens(phi.body)]
    if isinstance(phi, (And, Or)):
        if not phi.parts:
            return ["TRUE" if isinstance(phi, And) else "FALSE"]
        op = "AND" if isinstance(phi, And) else "OR"
        out = prefix_tokens(phi.parts[-1])
        for p in reversed(phi.parts[:-1]):
            out = [op, *prefix_tokens(p), *out]
        return out
    op = "EXISTS" if isinstance(phi, Exists) else "FORALL"
    return [op, f"x{phi.var}", *prefix_tokens(phi.body)]


def from_prefix_tokens(tokens: Sequence[str], signature: Signature) -> Formula:
    pos = 0

    def var() -> int:
        nonlocal pos
        tok = tokens[pos]
        if not tok.startswith("x"):
            raise ValueError(f"expected a variable at position {pos}, got {tok!r}")
        pos += 1
        return int(tok[1:])

    def parse() -> Formula:
        nonlocal pos
        tok = tokens[pos]
        pos += 1
        if tok == "TRUE":
            return TRUE
        if tok == "FALSE":
            return FALSE
        if tok == "NOT":
            return Not(parse())
        if tok in ("AND", "OR"):
            left = parse()
            right = parse()
            return (conj if tok == "AND" else disj)([left, right])
        if tok in ("EXISTS", "FORALL"):
            v = var()
            body = parse()
            return Exists(v, body) if tok == "EXISTS" else Forall(v, body)
        if tok == "EQ":
            return Eq(var(), var())
        if tok.startswith("R:"):
            name = tok[2:]
            return Rel(name, tuple(var() for _ in range(signature.symbol(name).arity)))
        raise ValueError(f"unknown token {tok!r}")

    phi = parse()
    if pos != len(tokens):
        raise ValueError("trailing tokens")
    return phi


def formula_vocabulary(signature: Signature, k: int) -> list[str]:
    """Token vocabulary of :func:`prefix_tokens` for formulas in FO^k over ``signature``."""
    return [
        "TRUE",
        "FALSE",
        "NOT",
        "AND",
        "OR",
        "EXISTS",
        "FORALL",
        "EQ",
        *(f"R:{s.name}" for s in signature),
        *(f"x{i}" for i in range(k)),
    ]


# ---------------------------------------------------------------------------
# Model checking


def satisfaction(structure: Structure, phi: Formula, num_vars: int | None = None) -> np.ndarray:
    """Satisfaction array of ``phi`` in ``structure``.

    The result has ``num_vars`` axes (by default ``width(phi)``); entry
    ``[a_0, ..., a_{w-1}]`` (broadcast along axes of variables that are not
    free) is true iff the assignment ``x_i -> a_i`` satisfies ``phi``.
    """
    w = width(phi) if num_vars is None else num_vars
    if w < width(phi):
        raise ValueError("num_vars is smaller than the width of the formula")
    n = structure.size
    grids = [np.arange(n).reshape((1,) * i + (n,) + (1,) * (w - i - 1)) for i in range(w)]
    ones = (1,) * w
    cache: dict[int, np.ndarray] = {}

    def ev(p: Formula) -> np.ndarray:
        key = id(p)
        if key in cache:
            return cache[key]
        if isinstance(p, Rel):
            symbol = structure.signature.symbol(p.symbol)
            if len(p.args) != symbol.arity:
                raise ValueError(f"{p.symbol} has arity {symbol.arity}")
            table = structure.relation(p.symbol)
            index = tuple(grids[a] for a in p.args)
            out = np.broadcast_to(table[index], _free_shape(p, n, w))
        elif isinstance(p, Eq):
            out = np.ones(ones, dtype=bool) if p.left == p.right else grids[p.left] == grids[p.right]
        elif isinstance(p, Not):
            out = ~ev(p.body)
        elif isinstance(p, And):
            out = np.ones(ones, dtype=bool)
            for q in p.parts:
                out = out & ev(q)
        elif isinstance(p, Or):
            out = np.zeros(ones, dtype=bool)
            for q in p.parts:
                out = out | ev(q)
        elif isinstance(p, Exists):
            out = np.asarray(ev(p.body).any(axis=p.var, keepdims=True)) if n else np.zeros(ones, dtype=bool)
        elif isinstance(p, Forall):
            out = np.asarray(ev(p.body).all(axis=p.var, keepdims=True)) if n else np.ones(ones, dtype=bool)
        else:
            raise TypeError(f"not a formula: {p!r}")
        cache[key] = out
        return out

    return ev(phi)


def _free_shape(p: Rel, n: int, w: int) -> tuple[int, ...]:
    return tuple(n if i in p.args else 1 for i in range(w))


def holds(structure: Structure, phi: Formula, assignment: Mapping[int, int] | None = None) -> bool:
    """Whether ``structure`` satisfies ``phi`` under ``assignment`` (required for the free variables)."""
    assignment = dict(assignment or {})
    missing = free_variables(phi) - set(assignment)
    if missing:
        raise ValueError(f"no value for free variables {sorted(missing)}")
    w = max(width(phi), max(assignment, default=-1) + 1)
    sat = satisfaction(structure, phi, w)
    index = tuple(assignment.get(i, 0) if sat.shape[i] > 1 else 0 for i in range(w))
    return bool(sat[index])


def distinguishes(phi: Formula, left: Structure, right: Structure) -> bool:
    """Whether the sentence ``phi`` holds in ``left`` and fails in ``right``."""
    if not is_sentence(phi):
        raise ValueError("distinguishing formulas must be sentences")
    return holds(left, phi) and not holds(right, phi)
