"""The k-dimensional Weisfeiler–Leman algorithm and counting-logic labels.

``wl(A, B, d)`` runs the d-dimensional (folklore) refinement of Cai, Fürer
and Immerman on d-tuples of both structures with a joint, collision-free
naming of colours:

    c_0(t)     = atomic type of t
    c_{r+1}(t) = (c_r(t), {{ (atp(t w), c_r(t[1/w]), ..., c_r(t[d/w])) : w ∈ V }})

where ``t[i/w]`` replaces the i-th entry of t by w and ``atp(t w)`` is the
atomic type of the (d+1)-tuple. For graphs the extra component is implied by
the others when d ≥ 2 and it makes d = 1 coincide with colour refinement;
for arbitrary signatures of arity at most d + 1 it makes one refinement step
match one round of the bijective (d+1)-pebble game. Two structures are
distinguished iff the multisets of stable colours differ, which happens iff
they are separated by a sentence of the counting logic C^{d+1}.

The refinement is written with ``jax.numpy``: colours of all tuples are
gathered in one array and renamed with ``jnp.unique``, which is exact.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp

from .atomic import AtomicTypes
from .structures import Structure


@dataclass(frozen=True)
class WLResult:
    dim: int
    distinguished: bool
    #: First round whose colour histograms differ (0 is the atomic colouring).
    distinguishing_round: int | None
    #: Number of refinement rounds performed.
    rounds: int
    #: True if the joint colouring reached a fixed point.
    stable: bool
    num_colours: int


def _replacement_indices(n: int, d: int) -> list[jnp.ndarray]:
    """``idx[i][t, w]`` is the index of t with entry i replaced by w (lexicographic order)."""
    total = n**d
    t = jnp.arange(total)
    coords = jnp.stack(jnp.unravel_index(t, (n,) * d), axis=1) if total else jnp.zeros((0, d), jnp.int32)
    w = jnp.arange(n)
    return [t[:, None] + (w[None, :] - coords[:, i : i + 1]) * n ** (d - 1 - i) for i in range(d)]


def _histograms_differ(colours: jnp.ndarray) -> bool:
    return bool(jnp.any(jnp.sort(colours[0]) != jnp.sort(colours[1])))


def wl(left: Structure, right: Structure, dim: int, max_rounds: int | None = None) -> WLResult:
    """Run d-dimensional Weisfeiler–Leman on both structures jointly."""
    if dim < 1:
        raise ValueError("dim must be at least 1")
    if left.signature != right.signature:
        raise ValueError("the structures must share a signature")
    if left.size != right.size:
        # C^2 already counts the elements: ∃^{≥m} x (x = x).
        return WLResult(dim, True, 0, 0, False, 0)
    n = left.size
    total = n**dim
    tuples_d = AtomicTypes(left.signature, dim)
    tuples_d1 = AtomicTypes(left.signature, dim + 1)
    colours = jnp.stack([tuples_d.codes(left), tuples_d.codes(right)])
    extended = jnp.stack([tuples_d1.codes(left), tuples_d1.codes(right)]).reshape(2, total, n)
    _, colours = jnp.unique(colours.reshape(-1), return_inverse=True)
    colours = colours.reshape(2, total)
    count = int(colours.max()) + 1 if total else 0
    if _histograms_differ(colours):
        return WLResult(dim, True, 0, 0, False, count)
    idx = _replacement_indices(n, dim)
    limit = max_rounds if max_rounds is not None else 2 * total + 1
    for r in range(1, limit + 1):
        rows = jnp.stack([extended, *(colours[:, i] for i in idx)], axis=-1)
        _, ids = jnp.unique(rows.reshape(-1, dim + 1), axis=0, return_inverse=True)
        ids = jnp.sort(ids.reshape(2, total, n), axis=-1)
        signature = jnp.concatenate([colours[..., None], ids], axis=-1)
        _, new = jnp.unique(signature.reshape(2 * total, n + 1), axis=0, return_inverse=True)
        new = new.reshape(2, total)
        new_count = int(new.max()) + 1
        if _histograms_differ(new):
            return WLResult(dim, True, r, r, False, new_count)
        if new_count == count:
            return WLResult(dim, False, None, r, True, new_count)
        colours, count = new, new_count
    return WLResult(dim, False, None, limit, False, count)


def wl_equivalent(left: Structure, right: Structure, dim: int) -> bool:
    """Whether d-dimensional WL fails to distinguish the structures (C^{d+1}-equivalence)."""
    return not wl(left, right, dim).distinguished


def counting_equivalent(left: Structure, right: Structure, k: int) -> bool:
    """Whether the structures agree on all sentences of the counting logic C^k (k ≥ 2)."""
    if k < 2:
        raise ValueError("C^k-equivalence is computed through (k-1)-dimensional WL, so k >= 2")
    return wl_equivalent(left, right, k - 1)
