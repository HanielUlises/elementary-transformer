"""Multi-agent Kripke models and bounded bisimilarity.

A Kripke model over agents ``0..m-1`` and propositional letters ``0..p-1``
has a set of worlds, one accessibility relation per agent and a valuation.
Pointed models ``(M, w)`` and ``(N, v)`` are n-bisimilar when Duplicator
survives n rounds of the bisimulation game; for finitely many letters this is
equivalent to agreement on all modal formulas of modal depth at most n.

n-bisimilarity is computed by partition refinement on the disjoint union:
the colour of a world at depth 0 is its valuation, and at depth d + 1 it is
its colour at depth d together with, for every agent, the set of depth-d
colours of its successors. The successor sets are computed with a matrix
product in ``jax.numpy``.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from .generators import KeyLike, as_key
from .structures import RelationSymbol, Signature, Structure


@dataclass(frozen=True)
class KripkeModel:
    accessibility: tuple[np.ndarray, ...]
    valuation: np.ndarray

    def __post_init__(self) -> None:
        n = self.valuation.shape[0]
        for r in self.accessibility:
            if r.shape != (n, n):
                raise ValueError("accessibility relations must be n × n")

    @property
    def num_worlds(self) -> int:
        return int(self.valuation.shape[0])

    @property
    def num_agents(self) -> int:
        return len(self.accessibility)

    @property
    def num_letters(self) -> int:
        return int(self.valuation.shape[1])

    def signature(self) -> Signature:
        return Signature(
            tuple(RelationSymbol(f"R{a}", 2) for a in range(self.num_agents))
            + tuple(RelationSymbol(f"P{i}", 1) for i in range(self.num_letters))
        )

    def to_structure(self) -> Structure:
        """The model as a relational structure with binary R_a and unary P_i."""
        arrays = [np.asarray(r, dtype=bool) for r in self.accessibility]
        arrays += [np.asarray(self.valuation[:, i], dtype=bool) for i in range(self.num_letters)]
        return Structure(self.signature(), self.num_worlds, arrays)

    def pointed(self, world: int) -> Structure:
        """The model with an extra unary predicate ``Pt`` that holds exactly at ``world``."""
        mark = np.zeros(self.num_worlds, dtype=bool)
        mark[world] = True
        return self.to_structure().expand(RelationSymbol("Pt", 1), mark)


def random_kripke(key: KeyLike, n: int, agents: int, letters: int, p_edge: float, p_letter: float = 0.5) -> KripkeModel:
    """Worlds with independent random accessibility edges and valuation bits."""
    k1, k2 = jax.random.split(as_key(key))
    access = np.asarray(jax.random.bernoulli(k1, p_edge, (agents, n, n)))
    valuation = np.asarray(jax.random.bernoulli(k2, p_letter, (n, letters)))
    return KripkeModel(tuple(access[a] for a in range(agents)), valuation)


def _union(m: KripkeModel, n: KripkeModel) -> tuple[jnp.ndarray, jnp.ndarray]:
    if m.num_agents != n.num_agents or m.num_letters != n.num_letters:
        raise ValueError("models over different agents or letters")
    size = m.num_worlds + n.num_worlds
    access = np.zeros((m.num_agents, size, size), dtype=np.float32)
    for a in range(m.num_agents):
        access[a, : m.num_worlds, : m.num_worlds] = m.accessibility[a]
        access[a, m.num_worlds :, m.num_worlds :] = n.accessibility[a]
    valuation = np.concatenate([m.valuation, n.valuation]).astype(np.int32)
    return jnp.asarray(access), jnp.asarray(valuation)


def bisimulation_colours(m: KripkeModel, n: KripkeModel, max_depth: int | None = None) -> list[np.ndarray]:
    """Colours of the worlds of ``m`` followed by those of ``n``, for depths 0, 1, ... until stable.

    Worlds ``w`` of ``m`` and ``v`` of ``n`` are d-bisimilar iff their colours
    at depth d coincide.
    """
    access, valuation = _union(m, n)
    size = valuation.shape[0]
    if valuation.shape[1] == 0:
        colours = jnp.zeros((size,), dtype=jnp.int32)
    else:
        _, colours = jnp.unique(valuation, axis=0, return_inverse=True)
    colours = colours.reshape(-1)
    count = int(colours.max()) + 1 if size else 0
    history = [np.asarray(colours)]
    depth = 0
    while max_depth is None or depth < max_depth:
        onehot = jax.nn.one_hot(colours, count, dtype=jnp.float32)
        successors = (jnp.einsum("aij,jc->aic", access, onehot) > 0).astype(jnp.int32)
        rows = jnp.concatenate([colours[:, None], jnp.transpose(successors, (1, 0, 2)).reshape(size, -1)], axis=1)
        _, new = jnp.unique(rows, axis=0, return_inverse=True)
        new = new.reshape(-1)
        new_count = int(new.max()) + 1 if size else 0
        depth += 1
        if new_count == count:
            break
        colours, count = new, new_count
        history.append(np.asarray(colours))
    return history


def distinguishing_depth(m: KripkeModel, w: int, n: KripkeModel, v: int, max_depth: int | None = None) -> int | None:
    """Least d such that (m, w) and (n, v) are not d-bisimilar, or ``None`` if they are bisimilar
    (within ``max_depth`` when it is given)."""
    history = bisimulation_colours(m, n, max_depth)
    offset = m.num_worlds
    for d, colours in enumerate(history):
        if colours[w] != colours[offset + v]:
            return d
    return None


def n_bisimilar(m: KripkeModel, w: int, n: KripkeModel, v: int, depth: int) -> bool:
    d = distinguishing_depth(m, w, n, v, depth)
    return d is None or d > depth
