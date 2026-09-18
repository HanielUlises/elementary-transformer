"""Canonical codes of atomic types of k-tuples.

The atomic type of a tuple ``(a_0, ..., a_{k-1})`` is the set of atomic
formulas ``x_i = x_j`` and ``R(x_{i_1}, ..., x_{i_r})`` that it satisfies. It
is determined by the equality pattern, a set partition of ``{0..k-1}``
written as a restricted growth string, together with the truth values of the
relational atoms on one representative per block (the least coordinate of
the block). Declared symmetry or irreflexivity of binary symbols removes
atoms whose value is forced.

Codes enumerate exactly the consistent atomic types: partitions in
lexicographic order of their restricted growth strings and, inside a
partition, the bit vector of block-level atoms read as a binary number. The
same codes serve as tokens of the dataset and as initial colours of the
Weisfeiler–Leman refinement.
"""

from __future__ import annotations

import bisect
import itertools
from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np

from .structures import Signature, Structure


def set_partitions(k: int) -> list[tuple[int, ...]]:
    """Restricted growth strings of length ``k`` in lexicographic order."""
    if k == 0:
        return [()]
    out: list[tuple[int, ...]] = []

    def extend(prefix: list[int], blocks: int) -> None:
        if len(prefix) == k:
            out.append(tuple(prefix))
            return
        for b in range(blocks + 1):
            extend(prefix + [b], max(blocks, b + 1))

    extend([0], 1)
    return out


@dataclass(frozen=True)
class _Atom:
    symbol: int
    coords: tuple[int, ...]


class AtomicTypes:
    """Vocabulary of atomic k-types over a signature and a vectorised encoder."""

    def __init__(self, signature: Signature, k: int) -> None:
        if k < 1:
            raise ValueError("k must be positive")
        self.signature = signature
        self.k = k
        self.partitions = set_partitions(k)
        self._partition_index = {self._rgs_key(p): i for i, p in enumerate(self.partitions)}
        # Column layout of the coordinate-level atom matrix: for each symbol,
        # all maps {0..r-1} -> {0..k-1} in lexicographic order.
        self._column_offset = []
        col = 0
        for s in signature:
            self._column_offset.append(col)
            col += k**s.arity
        self._num_columns = col
        self.atoms: list[list[_Atom]] = []
        offsets = [0]
        for rgs in self.partitions:
            reps = [rgs.index(b) for b in range(max(rgs) + 1)]
            atoms = []
            for si, s in enumerate(signature):
                for g in itertools.product(range(len(reps)), repeat=s.arity):
                    if s.irreflexive and g[0] == g[1]:
                        continue
                    if s.symmetric and g[0] > g[1]:
                        continue
                    atoms.append(_Atom(si, tuple(reps[b] for b in g)))
            if len(atoms) > 30:
                raise ValueError("atomic type vocabulary too large; reduce k or the signature")
            self.atoms.append(atoms)
            offsets.append(offsets[-1] + (1 << len(atoms)))
        self.offsets = offsets
        self.size = offsets[-1]
        if self.size >= 1 << 31:
            raise ValueError("atomic type vocabulary exceeds 2^31 codes")
        width = max(len(a) for a in self.atoms)
        gather = np.full((len(self.partitions), max(width, 1)), self._num_columns, dtype=np.int32)
        for pi, atoms in enumerate(self.atoms):
            for j, atom in enumerate(atoms):
                gather[pi, j] = self._column(atom)
        self._gather = gather
        lookup = np.full(k**k, -1, dtype=np.int32)
        for key, i in self._partition_index.items():
            lookup[key] = i
        self._lookup = lookup

    def _rgs_key(self, rgs: tuple[int, ...]) -> int:
        return sum(b * self.k**j for j, b in enumerate(rgs))

    def _column(self, atom: _Atom) -> int:
        offset = 0
        for c in atom.coords:
            offset = offset * self.k + c
        return self._column_offset[atom.symbol] + offset

    @property
    def dtype(self) -> np.dtype:
        return np.dtype(np.uint16) if self.size <= 1 << 16 else np.dtype(np.uint32)

    def codes(self, structure: Structure) -> jnp.ndarray:
        """Codes of all k-tuples of ``structure`` in lexicographic order (first coordinate most significant)."""
        if structure.signature != self.signature:
            raise ValueError("signature mismatch")
        n, k = structure.size, self.k
        total = n**k
        coords = jnp.stack(jnp.unravel_index(jnp.arange(total), (n,) * k), axis=1) if total else jnp.zeros((0, k), jnp.int32)
        rgs = jnp.zeros((total, k), dtype=jnp.int32)
        blocks = jnp.ones((total,), dtype=jnp.int32)
        for j in range(1, k):
            eq = coords[:, :j] == coords[:, j : j + 1]
            has = eq.any(axis=1)
            first = jnp.argmax(eq, axis=1)
            value = jnp.where(has, jnp.take_along_axis(rgs, first[:, None], axis=1)[:, 0], blocks)
            rgs = rgs.at[:, j].set(value)
            blocks = jnp.where(has, blocks, blocks + 1)
        key = (rgs * (k ** jnp.arange(k, dtype=jnp.int32))).sum(axis=1)
        pid = jnp.asarray(self._lookup)[key]
        columns = []
        for s, table in zip(self.signature, structure.relations, strict=True):
            t = jnp.asarray(table)
            for g in itertools.product(range(k), repeat=s.arity):
                columns.append(t[tuple(coords[:, c] for c in g)] if total else jnp.zeros((0,), bool))
        columns.append(jnp.zeros((total,), dtype=bool))
        matrix = jnp.stack(columns, axis=1)
        gather = jnp.asarray(self._gather)[pid]
        bits = jnp.take_along_axis(matrix, gather, axis=1).astype(jnp.int32)
        within = (bits << jnp.arange(bits.shape[1], dtype=jnp.int32)).sum(axis=1)
        return jnp.asarray(self.offsets[:-1], dtype=jnp.int32)[pid] + within

    def describe(self, code: int) -> str:
        """Human-readable form, for example ``{x0,x1}{x2} E(x0,x2)``."""
        if not 0 <= code < self.size:
            raise ValueError("code out of range")
        pi = bisect.bisect_right(self.offsets, code) - 1
        rgs = self.partitions[pi]
        within = code - self.offsets[pi]
        blocks = "".join("{" + ",".join(f"x{j}" for j in range(self.k) if rgs[j] == b) + "}" for b in range(max(rgs) + 1))
        names = self.signature.names
        true_atoms = [
            f"{names[a.symbol]}({','.join(f'x{c}' for c in a.coords)})" for j, a in enumerate(self.atoms[pi]) if within >> j & 1
        ]
        return " ".join([blocks, *true_atoms])
