"""Construction of pairs of structures.

Three kinds of pairs are produced: minimal mutations of a structure (one edge
flipped, or one degree-preserving double swap), isomorphic copies under a
random permutation, and pairs drawn from families that are known to be hard
for bounded-variable logics.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import jax
import numpy as np

from . import generators as gen
from .generators import KeyLike, as_key
from .structures import Structure, adjacency, graph_from_adjacency


@dataclass(frozen=True)
class StructurePair:
    left: Structure
    right: Structure
    family: str
    construction: str
    params: Mapping[str, Any] = field(default_factory=dict)

    def swapped(self) -> StructurePair:
        return StructurePair(self.right, self.left, self.family, self.construction, self.params)

    def relabelled(self, key: KeyLike) -> StructurePair:
        """Apply independent random permutations to both structures."""
        k1, k2 = jax.random.split(as_key(key))
        return StructurePair(
            self.left.permute(gen.random_permutation(k1, self.left.size)),
            self.right.permute(gen.random_permutation(k2, self.right.size)),
            self.family,
            self.construction,
            self.params,
        )


def edge_flip(key: KeyLike, g: Structure) -> Structure:
    """Toggle the adjacency of one uniformly random pair of distinct vertices."""
    n = g.size
    if n < 2:
        raise ValueError("need at least two vertices")
    k1, k2 = jax.random.split(as_key(key))
    u = int(jax.random.randint(k1, (), 0, n))
    v = int(jax.random.randint(k2, (), 0, n - 1))
    v = v + 1 if v >= u else v
    a = adjacency(g).copy()
    a[u, v] = a[v, u] = not a[u, v]
    return graph_from_adjacency(a)


def edge_swap(key: KeyLike, g: Structure, max_tries: int = 1000) -> Structure | None:
    """Replace edges {a,b}, {c,d} by {a,d}, {c,b}; the degree sequence is preserved.

    Returns ``None`` when no valid swap is found, for instance in complete or
    empty graphs.
    """
    a = adjacency(g)
    edges = np.argwhere(np.triu(a, 1))
    if len(edges) < 2:
        return None
    key = as_key(key)
    for _ in range(max_tries):
        key, k1, k2, k3 = jax.random.split(key, 4)
        i, j = (int(x) for x in jax.random.choice(k1, len(edges), (2,), replace=False))
        (p, q), (r, s) = edges[i], edges[j]
        if bool(jax.random.bernoulli(k2)):
            p, q = q, p
        if bool(jax.random.bernoulli(k3)):
            r, s = s, r
        if len({int(p), int(q), int(r), int(s)}) < 4 or a[p, s] or a[r, q]:
            continue
        b = a.copy()
        b[p, q] = b[q, p] = b[r, s] = b[s, r] = False
        b[p, s] = b[s, p] = b[r, q] = b[q, r] = True
        return graph_from_adjacency(b)
    return None


def isomorphic_copy(key: KeyLike, g: Structure) -> Structure:
    return g.permute(gen.random_permutation(key, g.size))


def mutation_pair(key: KeyLike, g: Structure, family: str, construction: str, params: Mapping[str, Any]) -> StructurePair | None:
    """Pair ``g`` with a mutation of it (``edge_flip``, ``edge_swap`` or ``isomorphic``)."""
    if construction == "edge_flip":
        other: Structure | None = edge_flip(key, g)
    elif construction == "edge_swap":
        other = edge_swap(key, g)
    elif construction == "isomorphic":
        other = isomorphic_copy(key, g)
    else:
        raise ValueError(f"unknown construction {construction!r}")
    if other is None:
        return None
    return StructurePair(g, other, family, construction, dict(params))


def linear_order_pair(m: int, n: int) -> StructurePair:
    return StructurePair(gen.linear_order(m), gen.linear_order(n), "linear_order", "hard", {"m": m, "n": n})


def cycle_pair(lengths_left: list[int], lengths_right: list[int]) -> StructurePair:
    """Two disjoint unions of cycles, for example C_n against C_a ⊎ C_{n-a}."""
    return StructurePair(
        gen.disjoint_cycles(lengths_left),
        gen.disjoint_cycles(lengths_right),
        "cycles",
        "hard",
        {"left": list(lengths_left), "right": list(lengths_right)},
    )


def cfi_pair(base_name: str, base: Structure, *, twisted: bool = True) -> StructurePair:
    """X(G) against X̃(G) (non-isomorphic), or against X(G) with two twists (isomorphic to X(G))."""
    x, x_twisted = gen.cfi_pair(base)
    if twisted:
        return StructurePair(x, x_twisted, "cfi", "twist", {"base": base_name})
    edge_list = [(int(u), int(v)) for u, v in np.argwhere(np.triu(adjacency(base), 1))]
    double = gen.cfi_graph(base, edge_list[:2])
    return StructurePair(x, double, "cfi", "double_twist", {"base": base_name})


def regular_pair(key: KeyLike, n: int, d: int) -> StructurePair:
    """Two independent uniform d-regular graphs; colour refinement never separates them."""
    k1, k2 = jax.random.split(as_key(key))
    return StructurePair(gen.random_regular(k1, n, d), gen.random_regular(k2, n, d), "regular", "independent", {"d": d})
