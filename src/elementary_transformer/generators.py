"""Families of finite structures.

Random generators take a JAX PRNG key (or an integer seed), so every sample of
a dataset is reproducible from the global seed and its index through
``jax.random.fold_in``.
"""

from __future__ import annotations

import itertools
import shutil
import subprocess
from collections.abc import Iterable, Iterator, Sequence

import jax
import numpy as np

from .structures import ORDER, Structure, adjacency, graph, graph_from_adjacency

KeyLike = jax.Array | int


def as_key(key: KeyLike) -> jax.Array:
    return jax.random.PRNGKey(key) if isinstance(key, int) else key


# ---------------------------------------------------------------------------
# Deterministic families


def linear_order(n: int) -> Structure:
    """The strict linear order L_n = ({0, ..., n-1}, <)."""
    idx = np.arange(n)
    return Structure(ORDER, n, [idx[:, None] < idx[None, :]])


def path(n: int) -> Structure:
    return graph(n, [(i, i + 1) for i in range(n - 1)])


def cycle(n: int) -> Structure:
    if n < 3:
        raise ValueError("cycles have at least three vertices")
    return graph(n, [(i, (i + 1) % n) for i in range(n)])


def disjoint_cycles(lengths: Sequence[int]) -> Structure:
    """Disjoint union of cycles of the given lengths."""
    out = graph(0, [])
    for length in lengths:
        out = out.disjoint_union(cycle(length))
    return out


def complete(n: int) -> Structure:
    return graph(n, itertools.combinations(range(n), 2))


def complete_bipartite(m: int, n: int) -> Structure:
    return graph(m + n, [(i, m + j) for i in range(m) for j in range(n)])


def cartesian_product(g: Structure, h: Structure) -> Structure:
    a, b = adjacency(g), adjacency(h)
    eye_a, eye_b = np.eye(g.size, dtype=bool), np.eye(h.size, dtype=bool)
    return graph_from_adjacency(np.kron(a, eye_b) | np.kron(eye_a, b))


def prism(n: int) -> Structure:
    """The prism C_n □ K_2."""
    return cartesian_product(cycle(n), complete(2))


def hypercube(d: int) -> Structure:
    n = 1 << d
    return graph(n, [(u, u ^ (1 << i)) for u in range(n) for i in range(d) if u < u ^ (1 << i)])


def petersen() -> Structure:
    """The Petersen graph as the Kneser graph K(5, 2)."""
    pairs = list(itertools.combinations(range(5), 2))
    return graph(10, [(i, j) for i, j in itertools.combinations(range(10), 2) if not set(pairs[i]) & set(pairs[j])])


def rook_graph(m: int) -> Structure:
    """The m × m rook's graph K_m □ K_m."""
    return cartesian_product(complete(m), complete(m))


def shrikhande() -> Structure:
    """The Shrikhande graph: Cayley graph of Z_4 × Z_4 with connection set ±(1,0), ±(0,1), ±(1,1)."""
    steps = {(1, 0), (3, 0), (0, 1), (0, 3), (1, 1), (3, 3)}
    cells = [(i, j) for i in range(4) for j in range(4)]
    return graph(
        16,
        [
            (u, v)
            for u, v in itertools.combinations(range(16), 2)
            if ((cells[v][0] - cells[u][0]) % 4, (cells[v][1] - cells[u][1]) % 4) in steps
        ],
    )


def triangular(m: int) -> Structure:
    """The triangular graph T(m) = L(K_m): 2-subsets of [m], adjacent iff they meet."""
    pairs = list(itertools.combinations(range(m), 2))
    n = len(pairs)
    return graph(n, [(i, j) for i, j in itertools.combinations(range(n), 2) if len(set(pairs[i]) & set(pairs[j])) == 1])


def seidel_switch(g: Structure, subset: Iterable[int]) -> Structure:
    """Seidel switching: complement the adjacency between ``subset`` and its complement."""
    a = adjacency(g).copy()
    mask = np.zeros(g.size, dtype=bool)
    mask[list(subset)] = True
    cross = mask[:, None] ^ mask[None, :]
    a[cross] = ~a[cross]
    return graph_from_adjacency(a)


def chang_graphs() -> list[Structure]:
    """The three Chang graphs, obtained from T(8) by Seidel switching.

    The switching sets are the vertices of T(8) = L(K_8) that correspond to
    the edges of a perfect matching 4K_2, of a cycle C_8 and of C_3 ∪ C_5 in
    K_8. Together with T(8) these are the strongly regular graphs with
    parameters (28, 12, 6, 4).
    """
    pairs = list(itertools.combinations(range(8), 2))
    position = {p: i for i, p in enumerate(pairs)}

    def vertices(edge_list: Iterable[tuple[int, int]]) -> list[int]:
        return [position[(min(e), max(e))] for e in edge_list]

    matching = [(0, 1), (2, 3), (4, 5), (6, 7)]
    octagon = [(i, (i + 1) % 8) for i in range(8)]
    triangle_pentagon = [(0, 1), (1, 2), (2, 0), (3, 4), (4, 5), (5, 6), (6, 7), (7, 3)]
    t8 = triangular(8)
    return [seidel_switch(t8, vertices(s)) for s in (matching, octagon, triangle_pentagon)]


def srg_parameters(g: Structure) -> tuple[int, int, int, int] | None:
    """``(v, k, λ, μ)`` if ``g`` is strongly regular (and neither complete nor empty), else ``None``."""
    a = adjacency(g).astype(np.int64)
    n = g.size
    deg = a.sum(axis=1)
    if n == 0 or not np.all(deg == deg[0]) or deg[0] in (0, n - 1):
        return None
    common = a @ a
    off = ~np.eye(n, dtype=bool)
    adj = a.astype(bool) & off
    non = ~a.astype(bool) & off
    lam, mu = np.unique(common[adj]), np.unique(common[non])
    if lam.size != 1 or mu.size != 1:
        return None
    return n, int(deg[0]), int(lam[0]), int(mu[0])


def strongly_regular_pairs() -> list[tuple[str, Structure, str, Structure]]:
    """Pairs of non-isomorphic strongly regular graphs with equal parameters.

    Built without external software: the 4 × 4 rook's graph and the
    Shrikhande graph, srg(16, 6, 2, 2), and T(8) with the three Chang graphs,
    srg(28, 12, 6, 4).
    """
    out = [("rook4", rook_graph(4), "shrikhande", shrikhande())]
    named = [("T8", triangular(8))] + [(f"chang{i + 1}", g) for i, g in enumerate(chang_graphs())]
    for (na, a), (nb, b) in itertools.combinations(named, 2):
        out.append((na, a, nb, b))
    return out


# ---------------------------------------------------------------------------
# Random families


def gnp(key: KeyLike, n: int, p: float) -> Structure:
    """Erdős–Rényi–Gilbert random graph G(n, p)."""
    coins = np.asarray(jax.random.bernoulli(as_key(key), p, (n, n)))
    upper = np.triu(coins, 1)
    return graph_from_adjacency(upper | upper.T)


def sparse_gnp(key: KeyLike, n: int, c: float) -> Structure:
    """G(n, c/n)."""
    return gnp(key, n, min(1.0, c / n) if n else 0.0)


def random_regular(key: KeyLike, n: int, d: int, max_tries: int = 100_000) -> Structure:
    """Uniform random simple d-regular graph via the pairing model with rejection.

    Conditioned on producing a simple graph, the pairing model is uniform on
    labelled simple d-regular graphs. The expected number of trials grows
    like exp((d^2 - 1) / 4), so the method is intended for small d.
    """
    if not 0 <= d < n or (n * d) % 2:
        raise ValueError("need 0 <= d < n and n * d even")
    key = as_key(key)
    points = np.repeat(np.arange(n), d)
    for _ in range(max_tries):
        key, sub = jax.random.split(key)
        perm = np.asarray(jax.random.permutation(sub, n * d))
        u, v = points[perm[0::2]], points[perm[1::2]]
        if np.any(u == v):
            continue
        lo, hi = np.minimum(u, v), np.maximum(u, v)
        if np.unique(lo * n + hi).size != lo.size:
            continue
        a = np.zeros((n, n), dtype=bool)
        a[lo, hi] = a[hi, lo] = True
        return graph_from_adjacency(a)
    raise RuntimeError("no simple pairing found; d is too large for rejection sampling")


def is_connected(g: Structure) -> bool:
    a = adjacency(g)
    if g.size == 0:
        return True
    seen = np.zeros(g.size, dtype=bool)
    seen[0] = True
    frontier = seen.copy()
    while frontier.any():
        frontier = a[frontier].any(axis=0) & ~seen
        seen |= frontier
    return bool(seen.all())


def random_connected_cubic(key: KeyLike, m: int, max_tries: int = 1000) -> Structure:
    key = as_key(key)
    for _ in range(max_tries):
        key, sub = jax.random.split(key)
        g = random_regular(sub, m, 3)
        if is_connected(g):
            return g
    raise RuntimeError("no connected cubic graph found")


def random_permutation(key: KeyLike, n: int) -> np.ndarray:
    return np.asarray(jax.random.permutation(as_key(key), n))


# ---------------------------------------------------------------------------
# Cai–Fürer–Immerman graphs


CfiVertex = tuple[str, int, int, int]


def cfi_vertices(base: Structure) -> list[CfiVertex]:
    """Names of the vertices of X(G) in index order.

    End vertices are ``("a", v, j, 0)`` and ``("b", v, j, 1)`` for the j-th
    neighbour of v; middle vertices are ``("m", v, S, 0)`` with S a bit mask.
    """
    a = adjacency(base)
    names: list[CfiVertex] = []
    for v in range(base.size):
        d = int(a[v].sum())
        for j in range(d):
            names.append(("a", v, j, 0))
            names.append(("b", v, j, 1))
        names.extend(("m", v, mask, 0) for mask in range(1 << d) if bin(mask).count("1") % 2 == 0)
    return names


def cfi_graph(base: Structure, twisted: Iterable[tuple[int, int]] = ()) -> Structure:
    """The CFI graph X(G) of a base graph G, with the given base edges twisted.

    Every vertex v of degree d with neighbours w_0 < ... < w_{d-1} becomes a
    gadget with middle vertices m(v, S) for the subsets S ⊆ {0..d-1} of even
    size and end vertices a(v, j), b(v, j); m(v, S) is adjacent to a(v, j) if
    j ∈ S and to b(v, j) otherwise. For a base edge {v, w} with w = w_j at v
    and v = w_l at w, the untwisted connection joins a(v, j)–a(w, l) and
    b(v, j)–b(w, l), while a twisted one joins a(v, j)–b(w, l) and
    b(v, j)–a(w, l). For connected G the isomorphism type depends only on the
    parity of the number of twisted edges; X(G) with one twist is the graph
    denoted X̃(G) by Cai, Fürer and Immerman. Vertex names are given by
    :func:`cfi_vertices`.
    """
    a = adjacency(base)
    neighbours = [np.flatnonzero(a[v]).tolist() for v in range(base.size)]
    twist = {tuple(sorted(e)) for e in twisted}
    for u, v in twist:
        if not a[u, v]:
            raise ValueError(f"({u}, {v}) is not an edge of the base graph")
    names = cfi_vertices(base)
    end = {(v, j, bit): i for i, (kind, v, j, bit) in enumerate(names) if kind != "m"}
    edge_list: list[tuple[int, int]] = []
    for i, (kind, v, mask, _) in enumerate(names):
        if kind == "m":
            edge_list.extend((i, end[(v, j, mask >> j & 1)]) for j in range(len(neighbours[v])))
    for v in range(base.size):
        for j, w in enumerate(neighbours[v]):
            if v > w:
                continue
            l_ = neighbours[w].index(v)
            flip = 1 if (v, w) in twist else 0
            for bit in (0, 1):
                edge_list.append((end[(v, j, bit)], end[(w, l_, bit ^ flip)]))
    return graph(len(names), edge_list)


def cfi_pair(base: Structure) -> tuple[Structure, Structure]:
    """X(G) and X̃(G), the latter with the lexicographically first edge twisted."""
    a = adjacency(base)
    first = next(((int(u), int(v)) for u, v in zip(*np.nonzero(np.triu(a, 1)), strict=True)), None)
    if first is None:
        raise ValueError("the base graph has no edges")
    return cfi_graph(base), cfi_graph(base, [first])


CFI_BASES = {
    "K4": lambda: complete(4),
    "K33": lambda: complete_bipartite(3, 3),
    "prism3": lambda: prism(3),
    "cube": lambda: hypercube(3),
    "petersen": petersen,
}


# ---------------------------------------------------------------------------
# graph6 and geng (nauty)


def parse_graph6(line: str) -> Structure:
    """Decode a graph in graph6 format."""
    s = line.strip()
    if s.startswith(">>graph6<<"):
        s = s[len(">>graph6<<") :]
    data = [ord(c) - 63 for c in s]
    if any(not 0 <= x < 64 for x in data):
        raise ValueError("invalid graph6 string")
    if data[0] < 63:
        n, pos = data[0], 1
    elif data[1] < 63:
        n, pos = (data[1] << 12) | (data[2] << 6) | data[3], 4
    else:
        n = 0
        for x in data[2:8]:
            n = (n << 6) | x
        pos = 8
    bits = [(x >> (5 - i)) & 1 for x in data[pos:] for i in range(6)]
    needed = n * (n - 1) // 2
    if len(bits) < needed:
        raise ValueError("graph6 string too short")
    a = np.zeros((n, n), dtype=bool)
    k = 0
    for j in range(1, n):
        for i in range(j):
            if bits[k]:
                a[i, j] = a[j, i] = True
            k += 1
    return graph_from_adjacency(a)


def to_graph6(g: Structure) -> str:
    a = adjacency(g)
    n = g.size
    if n < 63:
        head = [n]
    elif n < 258048:
        head = [63, (n >> 12) & 63, (n >> 6) & 63, n & 63]
    else:
        head = [63, 63] + [(n >> (6 * i)) & 63 for i in range(5, -1, -1)]
    bits = [int(a[i, j]) for j in range(1, n) for i in range(j)]
    bits += [0] * (-len(bits) % 6)
    body = [int("".join(map(str, bits[i : i + 6])), 2) for i in range(0, len(bits), 6)]
    return "".join(chr(x + 63) for x in head + body)


def geng_executable() -> str | None:
    """Path of nauty's ``geng`` (Debian installs it as ``nauty-geng``), if available."""
    return shutil.which("geng") or shutil.which("nauty-geng")


def geng(
    n: int,
    *,
    connected: bool = False,
    min_degree: int | None = None,
    max_degree: int | None = None,
    min_edges: int | None = None,
    max_edges: int | None = None,
) -> Iterator[Structure]:
    """All graphs on ``n`` vertices up to isomorphism, one per class, generated by ``geng``."""
    exe = geng_executable()
    if exe is None:
        raise FileNotFoundError("nauty's geng is not installed")
    args = [exe, "-q"]
    if connected:
        args.append("-c")
    if min_degree is not None:
        args.append(f"-d{min_degree}")
    if max_degree is not None:
        args.append(f"-D{max_degree}")
    args.append(str(n))
    if min_edges is not None or max_edges is not None:
        low = 0 if min_edges is None else min_edges
        high = "" if max_edges is None else str(max_edges)
        args.append(f"{low}:{high}")
    with subprocess.Popen(args, stdout=subprocess.PIPE, text=True) as proc:
        assert proc.stdout is not None
        for line in proc.stdout:
            if line.strip():
                yield parse_graph6(line)
    if proc.returncode:
        raise RuntimeError(f"geng exited with status {proc.returncode}")


# ---------------------------------------------------------------------------
# SageMath (optional)


def sage_available() -> bool:
    try:
        import sage.all  # noqa: F401
    except ImportError:
        return shutil.which("sage") is not None
    return True


def sage_strongly_regular(v: int, k: int, lam: int, mu: int) -> Structure:
    """A strongly regular graph with parameters (v, k, λ, μ) from SageMath's constructions."""
    expr = f"graphs.strongly_regular_graph({v}, {k}, {lam}, {mu}).graph6_string()"
    try:
        from sage.all import graphs  # type: ignore[import-not-found]
    except ImportError:
        exe = shutil.which("sage")
        if exe is None:
            raise FileNotFoundError("SageMath is not installed") from None
        out = subprocess.run([exe, "-c", f"print({expr})"], capture_output=True, text=True, check=True)
        return parse_graph6(out.stdout.strip().splitlines()[-1])
    return parse_graph6(graphs.strongly_regular_graph(v, k, lam, mu).graph6_string())
