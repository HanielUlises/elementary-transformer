"""Finite relational structures over arbitrary relational signatures.

A structure with universe ``{0, ..., n - 1}`` stores each relation of arity
``r`` as a dense boolean array of shape ``(n,) * r``. The representation is
meant for the small universes used in the datasets (a few dozen elements) and
low arities; it keeps membership tests, permutations and vectorised model
checking straightforward.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class RelationSymbol:
    """A relation symbol with its arity.

    ``symmetric`` and ``irreflexive`` are declared properties of binary
    symbols. They are checked when a structure is built and let the
    tokenizer use a smaller vocabulary of atomic types.
    """

    name: str
    arity: int
    symmetric: bool = False
    irreflexive: bool = False

    def __post_init__(self) -> None:
        if not self.name or any(c.isspace() for c in self.name):
            raise ValueError(f"invalid relation name {self.name!r}")
        if self.arity < 1:
            raise ValueError("relation symbols must have arity at least 1")
        if (self.symmetric or self.irreflexive) and self.arity != 2:
            raise ValueError("symmetric and irreflexive apply to binary symbols only")

    def to_json(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "arity": self.arity,
            "symmetric": self.symmetric,
            "irreflexive": self.irreflexive,
        }

    @staticmethod
    def from_json(data: Mapping[str, Any]) -> RelationSymbol:
        return RelationSymbol(
            name=str(data["name"]),
            arity=int(data["arity"]),
            symmetric=bool(data.get("symmetric", False)),
            irreflexive=bool(data.get("irreflexive", False)),
        )


@dataclass(frozen=True)
class Signature:
    """A finite relational signature (no constants or function symbols)."""

    symbols: tuple[RelationSymbol, ...]

    def __post_init__(self) -> None:
        names = [s.name for s in self.symbols]
        if len(set(names)) != len(names):
            raise ValueError("relation names must be distinct")

    def __iter__(self):
        return iter(self.symbols)

    def __len__(self) -> int:
        return len(self.symbols)

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(s.name for s in self.symbols)

    @property
    def max_arity(self) -> int:
        return max((s.arity for s in self.symbols), default=0)

    def index(self, name: str) -> int:
        for i, s in enumerate(self.symbols):
            if s.name == name:
                return i
        raise KeyError(name)

    def symbol(self, name: str) -> RelationSymbol:
        return self.symbols[self.index(name)]

    def extend(self, *symbols: RelationSymbol) -> Signature:
        return Signature(self.symbols + tuple(symbols))

    def to_json(self) -> list[dict[str, Any]]:
        return [s.to_json() for s in self.symbols]

    @staticmethod
    def from_json(data: Sequence[Mapping[str, Any]]) -> Signature:
        return Signature(tuple(RelationSymbol.from_json(d) for d in data))


#: Simple undirected graphs: one symmetric irreflexive binary relation.
GRAPH = Signature((RelationSymbol("E", 2, symmetric=True, irreflexive=True),))

#: Strict linear orders.
ORDER = Signature((RelationSymbol("<", 2, irreflexive=True),))


class Structure:
    """A finite structure over ``signature`` with universe ``{0, ..., size - 1}``."""

    __slots__ = ("signature", "size", "_relations")

    def __init__(
        self,
        signature: Signature,
        size: int,
        relations: Sequence[np.ndarray] | Mapping[str, np.ndarray],
    ) -> None:
        if size < 0:
            raise ValueError("size must be non-negative")
        if isinstance(relations, Mapping):
            missing = set(signature.names) - set(relations)
            extra = set(relations) - set(signature.names)
            if missing or extra:
                raise ValueError(f"relations do not match the signature: missing {missing}, extra {extra}")
            arrays = [relations[name] for name in signature.names]
        else:
            arrays = list(relations)
            if len(arrays) != len(signature):
                raise ValueError("one array per relation symbol is required")
        checked = []
        for symbol, array in zip(signature, arrays, strict=True):
            a = np.array(array, dtype=bool, copy=True)
            if a.shape != (size,) * symbol.arity:
                raise ValueError(f"relation {symbol.name} must have shape {(size,) * symbol.arity}, got {a.shape}")
            if symbol.symmetric and not np.array_equal(a, a.T):
                raise ValueError(f"relation {symbol.name} is declared symmetric")
            if symbol.irreflexive and a.diagonal().any():
                raise ValueError(f"relation {symbol.name} is declared irreflexive")
            a.setflags(write=False)
            checked.append(a)
        self.signature = signature
        self.size = size
        self._relations: tuple[np.ndarray, ...] = tuple(checked)

    @classmethod
    def from_tuples(
        cls,
        signature: Signature,
        size: int,
        relations: Mapping[str, Iterable[Sequence[int]]],
    ) -> Structure:
        """Build a structure from lists of tuples; symmetric relations are closed under swapping."""
        arrays = []
        for symbol in signature:
            a = np.zeros((size,) * symbol.arity, dtype=bool)
            for t in relations.get(symbol.name, ()):
                t = tuple(int(x) for x in t)
                if len(t) != symbol.arity or not all(0 <= x < size for x in t):
                    raise ValueError(f"invalid tuple {t} for relation {symbol.name}")
                a[t] = True
                if symbol.symmetric:
                    a[t[::-1]] = True
            arrays.append(a)
        return cls(signature, size, arrays)

    @property
    def relations(self) -> tuple[np.ndarray, ...]:
        return self._relations

    def relation(self, name: str) -> np.ndarray:
        return self._relations[self.signature.index(name)]

    def holds(self, name: str, args: Sequence[int]) -> bool:
        return bool(self.relation(name)[tuple(args)])

    def tuples(self, name: str) -> list[tuple[int, ...]]:
        return [tuple(int(x) for x in t) for t in np.argwhere(self.relation(name))]

    def permute(self, perm: Sequence[int] | np.ndarray) -> Structure:
        """Return the image of this structure under the bijection ``a -> perm[a]``."""
        p = np.asarray(perm, dtype=np.int64)
        if sorted(p.tolist()) != list(range(self.size)):
            raise ValueError("perm must be a permutation of the universe")
        inverse = np.empty_like(p)
        inverse[p] = np.arange(self.size)
        arrays = []
        for a in self._relations:
            arrays.append(a[np.ix_(*([inverse] * a.ndim))] if a.ndim else a)
        return Structure(self.signature, self.size, arrays)

    def disjoint_union(self, other: Structure) -> Structure:
        """Disjoint union; the elements of ``other`` are shifted by ``self.size``."""
        if other.signature != self.signature:
            raise ValueError("signatures differ")
        n = self.size + other.size
        arrays = []
        for a, b in zip(self._relations, other._relations, strict=True):
            c = np.zeros((n,) * a.ndim, dtype=bool)
            c[(slice(0, self.size),) * a.ndim] = a
            c[(slice(self.size, n),) * a.ndim] = b
            arrays.append(c)
        return Structure(self.signature, n, arrays)

    def expand(self, symbol: RelationSymbol, array: np.ndarray) -> Structure:
        """Add a relation for a new symbol."""
        return Structure(self.signature.extend(symbol), self.size, [*self._relations, array])

    def to_json(self) -> dict[str, Any]:
        return {
            "signature": self.signature.to_json(),
            "size": self.size,
            "relations": {s.name: [list(t) for t in self.tuples(s.name)] for s in self.signature},
        }

    @staticmethod
    def from_json(data: Mapping[str, Any]) -> Structure:
        signature = Signature.from_json(data["signature"])
        return Structure.from_tuples(signature, int(data["size"]), data["relations"])

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Structure):
            return NotImplemented
        return (
            self.signature == other.signature
            and self.size == other.size
            and all(np.array_equal(a, b) for a, b in zip(self._relations, other._relations, strict=True))
        )

    def __hash__(self) -> int:
        return hash((self.signature, self.size, tuple(a.tobytes() for a in self._relations)))

    def __repr__(self) -> str:
        counts = ", ".join(f"{s.name}:{int(a.sum())}" for s, a in zip(self.signature, self._relations, strict=True))
        return f"Structure(size={self.size}, {counts})"


def graph(size: int, edges: Iterable[tuple[int, int]]) -> Structure:
    """A simple undirected graph over the signature :data:`GRAPH`."""
    a = np.zeros((size, size), dtype=bool)
    for u, v in edges:
        if u == v:
            raise ValueError("graphs over GRAPH have no loops")
        a[u, v] = a[v, u] = True
    return Structure(GRAPH, size, [a])


def graph_from_adjacency(adjacency: np.ndarray) -> Structure:
    a = np.asarray(adjacency, dtype=bool)
    return Structure(GRAPH, a.shape[0], [a])


def adjacency(g: Structure) -> np.ndarray:
    """The adjacency matrix of a structure over a signature with a single binary relation."""
    if len(g.signature) != 1 or g.signature.symbols[0].arity != 2:
        raise ValueError("expected a signature with a single binary relation")
    return g.relations[0]


def edges(g: Structure) -> list[tuple[int, int]]:
    """Edges ``(u, v)`` with ``u < v`` of an undirected graph."""
    a = adjacency(g)
    return [(int(u), int(v)) for u, v in zip(*np.nonzero(np.triu(a, 1)), strict=True)]


def degrees(g: Structure) -> np.ndarray:
    return adjacency(g).sum(axis=1)
