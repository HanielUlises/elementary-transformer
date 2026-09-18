"""Labelled datasets of structure pairs.

Every example is a pair (A, B) of structures over a common signature with

* ``q_star``: the least quantifier rank of an FO^k sentence true in A and
  false in B (``None`` when A ≡^k_{q_max} B), computed by the game solver;
* a distinguishing sentence of FO^k of rank ``q_star`` extracted from
  Spoiler's strategy and checked by the model checker (and optionally by the
  Lean checker);
* the C^k label computed with (k-1)-dimensional Weisfeiler–Leman;
* both structures tokenised as the atomic types of all their k-tuples.

Splits follow a fixed protocol. ``train`` and ``val`` contain small
structures (``n ≤ 12`` by default) from the in-distribution families with
``q_star < q_max`` or equivalent. ``test_size`` keeps the families and
changes the size range (``20 ≤ n ≤ 50``), ``test_family`` uses the held-out
families (CFI and strongly regular graphs), and ``test_rank`` keeps families
and sizes but contains only pairs with ``q_star = q_max``. Within each split
the examples are stratified by ``q_star``.
"""

from __future__ import annotations

import json
import time
from collections.abc import Iterator, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import jax
import numpy as np

from . import formulas as fo
from . import games, pairs, wl
from . import generators as gen
from .atomic import AtomicTypes
from .structures import GRAPH, ORDER, RelationSymbol, Signature, Structure

IN_DISTRIBUTION_FAMILIES = ("linear_order", "cycles", "gnp", "sparse", "regular", "geng")
HELD_OUT_FAMILIES = ("cfi", "srg")
ALL_FAMILIES = IN_DISTRIBUTION_FAMILIES + HELD_OUT_FAMILIES
SIGNATURES: dict[str, Signature] = {"graph": GRAPH, "order": ORDER}
SPLITS = ("train", "val", "test_size", "test_family", "test_rank")


def signature_name(signature: Signature) -> str:
    for name, s in SIGNATURES.items():
        if s == signature:
            return name
    raise ValueError("unsupported signature for datasets")


# ---------------------------------------------------------------------------
# Configuration and split protocol


@dataclass
class DatasetConfig:
    k: int = 3
    q_max: int = 4
    families: tuple[str, ...] = ("linear_order", "cycles", "gnp", "sparse", "regular", "cfi", "srg")
    train_sizes: tuple[int, int] = (3, 12)
    test_sizes: tuple[int, int] = (20, 50)
    family_sizes: tuple[int, int] = (16, 60)
    num_train: int = 1000
    num_val: int = 200
    num_test: int = 200
    seed: int = 0
    relabel: bool = True
    shard_size: int = 1000
    max_attempts_factor: int = 50
    wl_labels: bool = True
    lean_verify: bool = False
    max_formula_size: int = 5000

    def validate(self) -> None:
        if self.k < 1 or self.q_max < 1:
            raise ValueError("k and q_max must be positive")
        unknown = set(self.families) - set(ALL_FAMILIES)
        if unknown:
            raise ValueError(f"unknown families {sorted(unknown)}")
        if "geng" in self.families and gen.geng_executable() is None:
            raise ValueError("the geng family requires nauty's geng on the PATH")
        for lo, hi in (self.train_sizes, self.test_sizes, self.family_sizes):
            if not 1 <= lo <= hi:
                raise ValueError("size ranges must satisfy 1 <= low <= high")


@dataclass(frozen=True)
class SplitSpec:
    name: str
    families: tuple[str, ...]
    sizes: tuple[int, int]
    strata: tuple[int | None, ...]
    target: int


def minimum_separation_rank(signatures: Sequence[Signature]) -> int:
    """Least quantifier rank that can separate two non-empty structures over these signatures.

    A sentence of rank 1 is a Boolean combination of sentences ∃x α(x) with α
    quantifier-free in the single variable x. If every atom in one variable is
    trivial, as for irreflexive binary relations, such sentences only express
    that the universe is non-empty, so rank 2 is needed.
    """
    for signature in signatures:
        for symbol in signature:
            if symbol.arity == 1 or not symbol.irreflexive:
                return 1
    return 2


def split_specs(config: DatasetConfig) -> list[SplitSpec]:
    in_dist = tuple(f for f in config.families if f in IN_DISTRIBUTION_FAMILIES)
    held_out = tuple(f for f in config.families if f in HELD_OUT_FAMILIES)
    low = minimum_separation_rank(list(SIGNATURES.values()))
    below: tuple[int | None, ...] = (*range(low, config.q_max), None)
    every: tuple[int | None, ...] = (*range(low, config.q_max + 1), None)
    specs = [
        SplitSpec("train", in_dist, config.train_sizes, below, config.num_train),
        SplitSpec("val", in_dist, config.train_sizes, below, config.num_val),
        SplitSpec("test_size", in_dist, config.test_sizes, every, config.num_test),
        SplitSpec("test_family", held_out, config.family_sizes, every, config.num_test),
        SplitSpec("test_rank", in_dist, config.train_sizes, (config.q_max,), config.num_test),
    ]
    return [s for s in specs if s.families and s.target > 0]


# ---------------------------------------------------------------------------
# Candidate pairs


def _randint(key: jax.Array, low: int, high: int) -> int:
    """Uniform integer in [low, high]."""
    return int(jax.random.randint(key, (), low, high + 1))


def _choice(key: jax.Array, options: Sequence[Any]) -> Any:
    return options[_randint(key, 0, len(options) - 1)]


def _cycle_lengths(key: jax.Array, n: int) -> list[int]:
    """A random partition of n into parts of size at least 3 (at most three parts)."""
    k1, k2, k3 = jax.random.split(key, 3)
    parts = _randint(k1, 1, max(1, min(3, n // 3)))
    if parts == 1:
        return [n]
    if parts == 2:
        a = _randint(k2, 3, n - 3)
        return sorted([a, n - a])
    a = _randint(k2, 3, n - 6)
    b = _randint(k3, 3, n - a - 3)
    return sorted([a, b, n - a - b])


_GENG_CACHE: dict[int, list[Structure]] = {}


def _geng_graphs(n: int) -> list[Structure]:
    if n not in _GENG_CACHE:
        _GENG_CACHE[n] = list(gen.geng(n))
    return _GENG_CACHE[n]


def sample_pair(family: str, key: jax.Array, sizes: tuple[int, int]) -> pairs.StructurePair | None:
    """One candidate pair from ``family`` with universes in ``sizes`` (``None`` if the draw does not apply)."""
    keys = jax.random.split(key, 6)
    lo, hi = sizes

    if family == "linear_order":
        m = _randint(keys[0], lo, hi)
        if bool(jax.random.bernoulli(keys[1])):
            n = min(hi, m + _randint(keys[2], 0, 3))
        else:
            n = _randint(keys[2], lo, hi)
        return pairs.linear_order_pair(m, n)

    if family == "cycles":
        lo = max(lo, 3)
        if lo > hi:
            return None
        n = _randint(keys[0], lo, hi)
        return pairs.cycle_pair(_cycle_lengths(keys[1], n), _cycle_lengths(keys[2], n))

    if family in ("gnp", "sparse"):
        lo = max(lo, 4)
        if lo > hi:
            return None
        n = _randint(keys[0], lo, hi)
        if family == "gnp":
            params: dict[str, Any] = {"p": 0.5}
            g = gen.gnp(keys[1], n, 0.5)
        else:
            c = _choice(keys[2], [1.0, 2.0, 3.0])
            params = {"c": c}
            g = gen.sparse_gnp(keys[1], n, c)
        construction = _choice(keys[3], ["edge_flip", "edge_swap", "isomorphic"])
        return pairs.mutation_pair(keys[4], g, family, construction, params)

    if family == "regular":
        lo = max(lo, 4)
        if lo > hi:
            return None
        n = _randint(keys[0], lo, hi)
        d = _choice(keys[1], [2, 3, 4])
        if d >= n or (n * d) % 2:
            return None
        construction = _choice(keys[2], ["independent", "edge_swap", "isomorphic"])
        if construction == "independent":
            return pairs.regular_pair(keys[3], n, d)
        return pairs.mutation_pair(keys[4], gen.random_regular(keys[3], n, d), "regular", construction, {"d": d})

    if family == "geng":
        lo, hi = max(lo, 2), min(hi, 9)
        if lo > hi:
            return None
        n = _randint(keys[0], lo, hi)
        graphs = _geng_graphs(n)
        i, j = _randint(keys[1], 0, len(graphs) - 1), _randint(keys[2], 0, len(graphs) - 1)
        return pairs.StructurePair(graphs[i], graphs[j], "geng", "enumerated", {"n": n, "i": i, "j": j})

    if family == "cfi":
        bases = [(name, make()) for name, make in gen.CFI_BASES.items()]
        bases = [(name, b) for name, b in bases if lo <= 10 * b.size <= hi]
        if not bases:
            return None
        name, base = _choice(keys[0], bases)
        construction = _choice(keys[1], ["twist", "double_twist", "edge_flip", "edge_swap"])
        if construction in ("twist", "double_twist"):
            return pairs.cfi_pair(name, base, twisted=construction == "twist")
        return pairs.mutation_pair(keys[2], gen.cfi_graph(base), "cfi", construction, {"base": name})

    if family == "srg":
        candidates = [(na, a, nb, b) for na, a, nb, b in gen.strongly_regular_pairs() if lo <= a.size <= hi]
        if not candidates:
            return None
        construction = _choice(keys[0], ["cospectral_pair", "edge_flip", "edge_swap", "isomorphic"])
        na, a, nb, b = _choice(keys[1], candidates)
        if construction == "cospectral_pair":
            return pairs.StructurePair(a, b, "srg", "same_parameters", {"left": na, "right": nb})
        return pairs.mutation_pair(keys[2], a, "srg", construction, {"graph": na})

    raise ValueError(f"unknown family {family!r}")


# ---------------------------------------------------------------------------
# Examples


@dataclass
class Example:
    pair: pairs.StructurePair
    k: int
    q_max: int
    q_star: int | None
    stable: bool
    formula: fo.Formula | None
    counting_equivalent: bool | None = None
    wl_round: int | None = None
    root_move: games.Move | None = None
    lean_verified: bool | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def equivalent(self) -> bool:
        return self.q_star is None

    def metadata(self, index: str) -> dict[str, Any]:
        return {
            "id": index,
            "family": self.pair.family,
            "construction": self.pair.construction,
            "params": dict(self.pair.params),
            "signature": signature_name(self.pair.left.signature),
            "k": self.k,
            "q_max": self.q_max,
            "q_star": self.q_star,
            "equivalent": self.equivalent,
            "stable": self.stable,
            "counting_equivalent": self.counting_equivalent,
            "wl_round": self.wl_round,
            "formula": None if self.formula is None else fo.to_json(self.formula),
            "formula_text": None if self.formula is None else fo.to_text(self.formula),
            "formula_size": None if self.formula is None else fo.size(self.formula),
            "root_move": None if self.root_move is None else asdict(self.root_move),
            "lean_verified": self.lean_verified,
            "left": self.pair.left.to_json(),
            "right": self.pair.right.to_json(),
            **self.extra,
        }


def label_pair(
    pair: pairs.StructurePair,
    k: int,
    q_max: int,
    *,
    wl_labels: bool = True,
    max_formula_size: int = 5000,
    result: games.GameResult | None = None,
) -> Example:
    """Solve the game, extract and verify the distinguishing sentence and compute the C^k label."""
    if result is None:
        result = games.solve(pair.left, pair.right, k, q_max, extract_formula=False)
    formula = None
    root_move = None
    extra: dict[str, Any] = {}
    if result.strategy is not None:
        formula = result.strategy.formula()
        root_move = result.strategy.move_at(*result.strategy.root())
        if not fo.distinguishes(formula, pair.left, pair.right):
            raise AssertionError("extracted formula does not distinguish the pair")
        if fo.rank(formula) != result.q_star or fo.width(formula) > k:
            raise AssertionError("extracted formula has the wrong rank or width")
        if fo.size(formula) > max_formula_size:
            extra["formula_dropped_size"] = fo.size(formula)
            formula = None
    example = Example(pair, k, q_max, result.q_star, result.stable, formula, root_move=root_move, extra=extra)
    if wl_labels and k >= 2:
        w = wl.wl(pair.left, pair.right, k - 1)
        example.counting_equivalent = not w.distinguished
        example.wl_round = w.distinguishing_round
        if example.counting_equivalent and result.q_star is not None:
            raise AssertionError("C^k-equivalent structures cannot be separated in FO^k")
    return example


def _even_quotas(total: int, strata: Sequence[int | None]) -> dict[int | None, int]:
    return {s: total // len(strata) + (1 if i < total % len(strata) else 0) for i, s in enumerate(strata)}


def generate_split(spec: SplitSpec, config: DatasetConfig, key: jax.Array) -> tuple[list[Example], dict[str, Any]]:
    """Rejection sampling of labelled pairs with one quota per value of ``q_star``.

    Quotas start equal. When half of the sampling budget is spent, strata
    that have not received a single example are dropped and their quota is
    shared among the others, so that rare or unreachable ranks do not
    exhaust the budget. The manifest records the dropped strata.
    """
    strata = list(spec.strata)
    quotas = _even_quotas(spec.target, strata)
    counts: dict[int | None, int] = {s: 0 for s in strata}
    dropped: list[int | None] = []
    attempts = 0
    max_attempts = config.max_attempts_factor * spec.target
    examples: list[Example] = []
    families_seen: dict[str, int] = {}
    while any(counts[s] < quotas[s] for s in strata) and attempts < max_attempts:
        if attempts == max_attempts // 2:
            empty = [s for s in strata if counts[s] == 0]
            live = [s for s in strata if counts[s] > 0]
            if empty and live:
                extra = _even_quotas(sum(quotas[s] for s in empty), live)
                for s in empty:
                    quotas[s] = 0
                for s in live:
                    quotas[s] += extra[s]
                dropped.extend(empty)
        sub = jax.random.fold_in(key, attempts)
        attempts += 1
        k_family, k_pair, k_relabel, k_swap = jax.random.split(sub, 4)
        family = _choice(k_family, spec.families)
        pair = sample_pair(family, k_pair, spec.sizes)
        if pair is None:
            continue
        if config.relabel:
            pair = pair.relabelled(k_relabel)
        if bool(jax.random.bernoulli(k_swap)):
            pair = pair.swapped()
        result = games.solve(pair.left, pair.right, config.k, config.q_max, extract_formula=False)
        q = result.q_star
        if q not in quotas or counts[q] >= quotas[q]:
            continue
        example = label_pair(
            pair,
            config.k,
            config.q_max,
            wl_labels=config.wl_labels,
            max_formula_size=config.max_formula_size,
            result=result,
        )
        examples.append(example)
        counts[q] += 1
        families_seen[family] = families_seen.get(family, 0) + 1
    stats = {
        "target": spec.target,
        "generated": len(examples),
        "attempts": attempts,
        "quotas": {_stratum_name(s): quotas[s] for s in strata},
        "counts": {_stratum_name(s): counts[s] for s in strata},
        "dropped_strata": [_stratum_name(s) for s in dropped],
        "families": families_seen,
        "sizes": list(spec.sizes),
    }
    return examples, stats


def _stratum_name(s: int | None) -> str:
    return "equivalent" if s is None else str(s)


# ---------------------------------------------------------------------------
# Tokenisation and serialisation


class Vocabulary:
    """Token vocabularies: one block of atomic k-types per signature, and formula tokens."""

    def __init__(self, k: int, signatures: Sequence[str] = ("graph", "order")) -> None:
        self.k = k
        self.names = list(signatures)
        self.types = {name: AtomicTypes(SIGNATURES[name], k) for name in self.names}
        self.offsets: dict[str, int] = {}
        total = 0
        for name in self.names:
            self.offsets[name] = total
            total += self.types[name].size
        self.size = total
        symbols: dict[str, RelationSymbol] = {}
        for name in self.names:
            for s in SIGNATURES[name]:
                symbols.setdefault(s.name, s)
        union = Signature(tuple(symbols.values()))
        self.formula_tokens = fo.formula_vocabulary(union, k)
        self.formula_index = {t: i for i, t in enumerate(self.formula_tokens)}

    @property
    def token_dtype(self) -> np.dtype:
        return np.dtype(np.uint16) if self.size < 1 << 16 else np.dtype(np.uint32)

    def tokenize(self, structure: Structure) -> np.ndarray:
        name = signature_name(structure.signature)
        codes = np.asarray(self.types[name].codes(structure))
        return (codes + self.offsets[name]).astype(self.token_dtype)

    def encode_formula(self, phi: fo.Formula) -> np.ndarray:
        return np.array([self.formula_index[t] for t in fo.prefix_tokens(phi)], dtype=np.uint16)

    def to_json(self) -> dict[str, Any]:
        return {
            "k": self.k,
            "size": self.size,
            "blocks": [
                {
                    "signature": name,
                    "symbols": SIGNATURES[name].to_json(),
                    "offset": self.offsets[name],
                    "size": self.types[name].size,
                }
                for name in self.names
            ],
            "formula_tokens": self.formula_tokens,
        }

    def describe(self, token: int) -> str:
        for name in reversed(self.names):
            if token >= self.offsets[name]:
                return f"{name}: {self.types[name].describe(token - self.offsets[name])}"
        raise ValueError("token out of range")


def write_shard(
    path: Path, examples: Sequence[Example], vocab: Vocabulary, families: Sequence[str], start: int, split: str
) -> None:
    tokens: list[np.ndarray] = []
    formula_tokens: list[np.ndarray] = []
    for ex in examples:
        tokens.append(vocab.tokenize(ex.pair.left))
        tokens.append(vocab.tokenize(ex.pair.right))
        formula_tokens.append(vocab.encode_formula(ex.formula) if ex.formula is not None else np.zeros(0, np.uint16))
    token_offsets = np.concatenate([[0], np.cumsum([t.size for t in tokens])]).astype(np.int64)
    formula_offsets = np.concatenate([[0], np.cumsum([t.size for t in formula_tokens])]).astype(np.int64)
    np.savez_compressed(
        path.with_suffix(".npz"),
        tokens=np.concatenate(tokens) if tokens else np.zeros(0, vocab.token_dtype),
        token_offsets=token_offsets,
        sizes=np.array([[ex.pair.left.size, ex.pair.right.size] for ex in examples], dtype=np.int32).reshape(-1, 2),
        q_star=np.array([-1 if ex.q_star is None else ex.q_star for ex in examples], dtype=np.int8),
        stable=np.array([ex.stable for ex in examples], dtype=bool),
        counting_equivalent=np.array([ex.counting_equivalent is True for ex in examples], dtype=bool),
        family=np.array([families.index(ex.pair.family) for ex in examples], dtype=np.int16),
        formula_tokens=np.concatenate(formula_tokens) if formula_tokens else np.zeros(0, np.uint16),
        formula_offsets=formula_offsets,
    )
    with open(path.with_suffix(".jsonl"), "w") as f:
        for i, ex in enumerate(examples):
            f.write(json.dumps(ex.metadata(f"{split}-{start + i:07d}")) + "\n")


def build_dataset(config: DatasetConfig, out_dir: str | Path, *, log: Any = None) -> dict[str, Any]:
    """Generate every split of the protocol and write it under ``out_dir``."""
    config.validate()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    vocab = Vocabulary(config.k)
    families = list(ALL_FAMILIES)
    root = jax.random.PRNGKey(config.seed)
    manifest: dict[str, Any] = {
        "format": "elementary-transformer/1",
        "config": asdict(config),
        "vocabulary": vocab.to_json(),
        "families": families,
        "splits": {},
    }
    started = time.time()
    for index, spec in enumerate(split_specs(config)):
        t0 = time.time()
        examples, stats = generate_split(spec, config, jax.random.fold_in(root, index))
        if config.lean_verify:
            from . import lean

            verdicts = lean.check_examples(examples)
            for ex, ok in zip(examples, verdicts, strict=True):
                ex.lean_verified = ok
            if not all(v is not False for v in verdicts):
                raise AssertionError(f"the Lean checker rejected a certificate in split {spec.name}")
        split_dir = out / spec.name
        split_dir.mkdir(exist_ok=True)
        shards = []
        for s, start in enumerate(range(0, max(len(examples), 1), config.shard_size)):
            chunk = examples[start : start + config.shard_size]
            name = f"shard-{s:05d}"
            write_shard(split_dir / name, chunk, vocab, families, start, spec.name)
            shards.append(name)
        stats["shards"] = shards
        stats["seconds"] = round(time.time() - t0, 2)
        manifest["splits"][spec.name] = stats
        if log is not None:
            log(f"{spec.name}: {stats['generated']}/{spec.target} examples, counts {stats['counts']}, {stats['seconds']} s")
    manifest["seconds"] = round(time.time() - started, 2)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


# ---------------------------------------------------------------------------
# Loading


def load_manifest(dataset_dir: str | Path) -> dict[str, Any]:
    return json.loads((Path(dataset_dir) / "manifest.json").read_text())


def iter_examples(dataset_dir: str | Path, split: str) -> Iterator[dict[str, Any]]:
    """Examples of a split as dictionaries with token arrays and metadata."""
    manifest = load_manifest(dataset_dir)
    for shard in manifest["splits"][split]["shards"]:
        base = Path(dataset_dir) / split / shard
        arrays = np.load(base.with_suffix(".npz"))
        with open(base.with_suffix(".jsonl")) as f:
            metadata = [json.loads(line) for line in f]
        off, foff = arrays["token_offsets"], arrays["formula_offsets"]
        for i, meta in enumerate(metadata):
            yield {
                "left_tokens": arrays["tokens"][off[2 * i] : off[2 * i + 1]],
                "right_tokens": arrays["tokens"][off[2 * i + 1] : off[2 * i + 2]],
                "formula_tokens": arrays["formula_tokens"][foff[i] : foff[i + 1]],
                **meta,
            }


def pad_batch(examples: Sequence[dict[str, Any]], pad_id: int, formula_pad_id: int) -> dict[str, jax.Array]:
    """Stack a batch into padded ``jax.numpy`` arrays with boolean masks.

    ``pad_id`` fills the structure token sequences and ``formula_pad_id`` the
    formula sequences; natural choices are the sizes of the two vocabularies.
    """
    import jax.numpy as jnp

    def stack(key: str, pad: int) -> tuple[jax.Array, jax.Array]:
        length = max((len(e[key]) for e in examples), default=0)
        out = np.full((len(examples), length), pad, dtype=np.int32)
        mask = np.zeros((len(examples), length), dtype=bool)
        for i, e in enumerate(examples):
            out[i, : len(e[key])] = e[key]
            mask[i, : len(e[key])] = True
        return jnp.asarray(out), jnp.asarray(mask)

    left, left_mask = stack("left_tokens", pad_id)
    right, right_mask = stack("right_tokens", pad_id)
    formula, formula_mask = stack("formula_tokens", formula_pad_id)
    q = np.array([-1 if e["q_star"] is None else e["q_star"] for e in examples], dtype=np.int32)
    return {
        "left": left,
        "left_mask": left_mask,
        "right": right,
        "right_mask": right_mask,
        "formula": formula,
        "formula_mask": formula_mask,
        "q_star": jnp.asarray(q),
        "equivalent": jnp.asarray(q < 0),
    }
