"""Training and evaluation of the axial tuple transformer on a generated dataset.

The model predicts, for every rank q ≤ q_max, whether the two structures are
FO^k_q-equivalent, and decodes a distinguishing sentence token by token. The
decoder is constrained by the grammar of Definition 3 of the README, so every
output is a well-formed formula; whether it is a sentence that separates the
two structures is then decided by the model checker, which makes synthesis
accuracy an exact metric.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Iterator, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from . import formulas as fo
from . import model as mdl
from .dataset import SIGNATURES, iter_examples, load_manifest
from .structures import Structure


@dataclass
class TrainConfig:
    data: str
    out: str
    d_model: int = 128
    heads: int = 4
    layers: int = 6
    d_ff: int = 256
    decoder_layers: int = 3
    max_formula_len: int = 256
    batch_size: int = 32
    epochs: int = 10
    lr: float = 5e-4
    warmup: int = 500
    formula_weight: float = 1.0
    seed: int = 0
    eval_batch_size: int = 16
    synthesis_limit: int = 500
    log_every: int = 100


# ---------------------------------------------------------------------------
# Data


def _round_up(n: int, multiple: int) -> int:
    return -(-n // multiple) * multiple


def load_examples(data_dir: str | Path, split: str) -> list[dict[str, Any]]:
    return list(iter_examples(data_dir, split))


def _pad_structure(tokens: np.ndarray, n: int, size: int, k: int, pad: int) -> np.ndarray:
    out = np.full((size,) * k, pad, dtype=np.int32)
    if n:
        out[(slice(0, n),) * k] = tokens.reshape((n,) * k)
    return out.reshape(-1)


def make_batch(examples: Sequence[dict[str, Any]], cfg: mdl.ModelConfig, size: int) -> dict[str, np.ndarray]:
    """Arrays for a batch with every structure padded to ``size`` elements."""
    b, k, length = len(examples), cfg.k, cfg.max_formula_len
    left = np.stack([_pad_structure(e["left_tokens"], e["left"]["size"], size, k, cfg.pad_token) for e in examples])
    right = np.stack([_pad_structure(e["right_tokens"], e["right"]["size"], size, k, cfg.pad_token) for e in examples])
    left_mask = np.arange(size)[None, :] < np.array([e["left"]["size"] for e in examples])[:, None]
    right_mask = np.arange(size)[None, :] < np.array([e["right"]["size"] for e in examples])[:, None]
    q = np.array([cfg.q_max + 1 if e["q_star"] is None else e["q_star"] for e in examples])
    equivalence = (q[:, None] > np.arange(1, cfg.q_max + 1)[None, :]).astype(np.float32)
    formula_in = np.full((b, length), cfg.formula_pad, dtype=np.int32)
    formula_out = np.full((b, length), cfg.formula_pad, dtype=np.int32)
    formula_mask = np.zeros((b, length), dtype=np.float32)
    for i, e in enumerate(examples):
        t = np.asarray(e["formula_tokens"], dtype=np.int32)
        if e["q_star"] is None or t.size == 0 or t.size > length:
            continue
        formula_in[i, 0] = cfg.formula_bos
        formula_in[i, 1 : t.size] = t[:-1]
        formula_out[i, : t.size] = t
        formula_mask[i, : t.size] = 1.0
    return {
        "left": left,
        "left_mask": left_mask,
        "right": right,
        "right_mask": right_mask,
        "equivalence": equivalence,
        "formula_in": formula_in,
        "formula_out": formula_out,
        "formula_mask": formula_mask,
    }


def batches_by_size(examples: Sequence[dict[str, Any]], batch_size: int, multiple: int = 4) -> Iterator[tuple[list[int], int]]:
    """Indices grouped by padded universe size, so that each group compiles once."""
    groups: dict[int, list[int]] = {}
    for i, e in enumerate(examples):
        size = _round_up(max(e["left"]["size"], e["right"]["size"], 1), multiple)
        groups.setdefault(size, []).append(i)
    for size in sorted(groups):
        idx = groups[size]
        for start in range(0, len(idx), batch_size):
            yield idx[start : start + batch_size], size


# ---------------------------------------------------------------------------
# Grammar-constrained decoding


class FormulaGrammar:
    """Allowed next tokens of the prefix notation, tracked with a stack of open slots."""

    def __init__(self, tokens: Sequence[str], k: int) -> None:
        self.tokens = list(tokens)
        self.k = k
        self.children: list[list[str]] = []
        for t in self.tokens:
            if t in ("TRUE", "FALSE") or t.startswith("x"):
                self.children.append([])
            elif t == "NOT":
                self.children.append(["F"])
            elif t in ("AND", "OR"):
                self.children.append(["F", "F"])
            elif t in ("EXISTS", "FORALL"):
                self.children.append(["V", "F"])
            elif t == "EQ":
                self.children.append(["V", "V"])
            elif t.startswith("R:"):
                arity = next(s.arity for sig in SIGNATURES.values() for s in sig if s.name == t[2:])
                self.children.append(["V"] * arity)
            else:
                raise ValueError(f"unknown formula token {t}")
        self.is_var = np.array([t.startswith("x") for t in self.tokens])
        self.growth = np.array([len(c) - 1 for c in self.children])

    def allowed(self, stack: list[str], remaining: int, symbols: set[str]) -> np.ndarray:
        slot = stack[-1]
        ok = self.is_var.copy() if slot == "V" else ~self.is_var
        if slot == "F":
            ok &= np.array([not t.startswith("R:") or t[2:] in symbols for t in self.tokens])
        ok &= len(stack) + self.growth <= remaining
        return ok

    def push(self, stack: list[str], token: int) -> None:
        stack.pop()
        stack.extend(reversed(self.children[token]))


def greedy_decode(
    decode_fn: Callable,
    params: mdl.Params,
    cfg: mdl.ModelConfig,
    mem: jax.Array,
    mem_mask: jax.Array,
    grammar: FormulaGrammar,
    symbols: Sequence[set[str]],
) -> list[list[int]]:
    b, length = mem.shape[0], cfg.max_formula_len
    buffer = np.full((b, length), cfg.formula_pad, dtype=np.int32)
    buffer[:, 0] = cfg.formula_bos
    stacks: list[list[str]] = [["F"] for _ in range(b)]
    outputs: list[list[int]] = [[] for _ in range(b)]
    for t in range(length):
        live = [i for i in range(b) if stacks[i]]
        if not live:
            break
        logits = np.asarray(decode_fn(params, mem, mem_mask, jnp.asarray(buffer))[:, t, : len(grammar.tokens)])
        for i in live:
            ok = grammar.allowed(stacks[i], length - t - 1, symbols[i])
            token = int(np.argmax(np.where(ok, logits[i], -np.inf)))
            outputs[i].append(token)
            grammar.push(stacks[i], token)
            if t + 1 < length:
                buffer[i, t + 1] = token
    return outputs


# ---------------------------------------------------------------------------
# Evaluation


def predicted_q_star(probabilities: np.ndarray) -> np.ndarray:
    """Least q whose predicted probability of equivalence is below 1/2, or q_max + 1 (for ⊥)."""
    below = probabilities < 0.5
    first = np.argmax(below, axis=1) + 1
    return np.where(below.any(axis=1), first, probabilities.shape[1] + 1)


def classification_metrics(q_true: np.ndarray, q_pred: np.ndarray, q_max: int) -> dict[str, Any]:
    """Exact q* accuracy, per-rank accuracy of the equivalence decision and per-stratum accuracy."""
    out: dict[str, Any] = {"n": int(q_true.size)}
    if q_true.size == 0:
        return out
    out["q_star_accuracy"] = float(np.mean(q_true == q_pred))
    out["rank_accuracy"] = {str(q): float(np.mean((q_true > q) == (q_pred > q))) for q in range(1, q_max + 1)}
    equiv_true, equiv_pred = q_true > q_max, q_pred > q_max
    out["decision_accuracy"] = float(np.mean(equiv_true == equiv_pred))
    recalls = [float(np.mean(equiv_pred[equiv_true == c] == c)) for c in (True, False) if np.any(equiv_true == c)]
    out["decision_balanced_accuracy"] = float(np.mean(recalls))
    out["per_stratum"] = {
        ("equivalent" if s > q_max else str(s)): {
            "n": int(np.sum(q_true == s)),
            "q_star_accuracy": float(np.mean(q_pred[q_true == s] == s)),
            "decision_accuracy": float(np.mean(equiv_pred[q_true == s] == (s > q_max))),
        }
        for s in sorted(set(q_true.tolist()))
    }
    return out


def evaluate(
    params: mdl.Params,
    cfg: mdl.ModelConfig,
    examples: Sequence[dict[str, Any]],
    formula_tokens: Sequence[str],
    *,
    batch_size: int = 16,
    synthesis_limit: int = 0,
    forward_fn: Callable | None = None,
    decode_fn: Callable | None = None,
) -> dict[str, Any]:
    forward_fn = forward_fn or jax.jit(lambda p, b: mdl.forward(p, cfg, b))
    q_true = np.array([cfg.q_max + 1 if e["q_star"] is None else e["q_star"] for e in examples])
    probabilities = np.zeros((len(examples), cfg.q_max))
    memories: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    synth_candidates = set(
        [i for i, e in enumerate(examples) if e["q_star"] is not None][:synthesis_limit] if synthesis_limit else []
    )
    for idx, size in batches_by_size(examples, batch_size):
        batch = make_batch([examples[i] for i in idx], cfg, size)
        inputs = {key: jnp.asarray(batch[key]) for key in ("left", "left_mask", "right", "right_mask")}
        out = forward_fn(params, inputs)
        probabilities[idx] = np.asarray(jax.nn.sigmoid(out["equivalence_logits"]))
        for j, i in enumerate(idx):
            if i in synth_candidates:
                memories[i] = (np.asarray(out["memory"][j]), np.asarray(out["memory_mask"][j]))
    q_pred = predicted_q_star(probabilities)
    result: dict[str, Any] = {"classification": classification_metrics(q_true, q_pred, cfg.q_max)}
    ck = np.array([bool(e.get("counting_equivalent")) for e in examples])
    result["baseline_counting"] = {
        "decision_accuracy": float(np.mean(ck == (q_true > cfg.q_max))),
        "note": "predicts FO^k-equivalence iff the pair is C^k-equivalent",
    }
    if memories:
        result["synthesis"] = synthesis(params, cfg, examples, memories, formula_tokens, decode_fn)
    return result


def synthesis(
    params: mdl.Params,
    cfg: mdl.ModelConfig,
    examples: Sequence[dict[str, Any]],
    memories: dict[int, tuple[np.ndarray, np.ndarray]],
    formula_tokens: Sequence[str],
    decode_fn: Callable | None,
    batch_size: int = 16,
) -> dict[str, Any]:
    """Decode a formula for each selected example and check it with the model checker."""
    decode_fn = decode_fn or jax.jit(lambda p, m, mm, t: mdl.decode(p, cfg, m, mm, t))
    grammar = FormulaGrammar(formula_tokens, cfg.k)
    valid = optimal = sentences = 0
    sizes: list[int] = []
    order = sorted(memories, key=lambda i: memories[i][0].shape[0])
    groups: dict[int, list[int]] = {}
    for i in order:
        groups.setdefault(int(memories[i][0].shape[0]), []).append(i)
    for idx_all in groups.values():
        for start in range(0, len(idx_all), batch_size):
            idx = idx_all[start : start + batch_size]
            mem = jnp.asarray(np.stack([memories[i][0] for i in idx]))
            mem_mask = jnp.asarray(np.stack([memories[i][1] for i in idx]))
            symbols = [{s.name for s in SIGNATURES[examples[i]["signature"]]} for i in idx]
            decoded = greedy_decode(decode_fn, params, cfg, mem, mem_mask, grammar, symbols)
            for i, tokens in zip(idx, decoded, strict=True):
                e = examples[i]
                left, right = Structure.from_json(e["left"]), Structure.from_json(e["right"])
                try:
                    phi = fo.from_prefix_tokens([formula_tokens[t] for t in tokens], left.signature)
                except (ValueError, IndexError, KeyError):
                    continue
                if not fo.is_sentence(phi):
                    continue
                sentences += 1
                if fo.distinguishes(phi, left, right):
                    valid += 1
                    sizes.append(fo.size(phi))
                    if fo.rank(phi) <= e["q_star"]:
                        optimal += 1
    n = len(memories)
    return {
        "n": n,
        "sentence_rate": sentences / n,
        "verified_rate": valid / n,
        "rank_optimal_rate": optimal / n,
        "mean_size_of_verified": float(np.mean(sizes)) if sizes else None,
    }


# ---------------------------------------------------------------------------
# Training


def model_config(manifest: dict[str, Any], tc: TrainConfig) -> mdl.ModelConfig:
    config = manifest["config"]
    return mdl.ModelConfig(
        vocab_size=manifest["vocabulary"]["size"],
        formula_vocab_size=len(manifest["vocabulary"]["formula_tokens"]),
        k=config["k"],
        q_max=config["q_max"],
        d_model=tc.d_model,
        heads=tc.heads,
        layers=tc.layers,
        d_ff=tc.d_ff,
        decoder_layers=tc.decoder_layers,
        max_formula_len=tc.max_formula_len,
    )


def train(tc: TrainConfig, log: Callable[[str], None] = print) -> dict[str, Any]:
    manifest = load_manifest(tc.data)
    cfg = model_config(manifest, tc)
    out = Path(tc.out)
    out.mkdir(parents=True, exist_ok=True)
    formula_tokens = manifest["vocabulary"]["formula_tokens"]
    train_set = load_examples(tc.data, "train")
    val_set = load_examples(tc.data, "val")
    size = max(max(e["left"]["size"], e["right"]["size"]) for e in train_set)
    steps_per_epoch = len(train_set) // tc.batch_size
    total = steps_per_epoch * tc.epochs
    rng = np.random.default_rng(tc.seed)
    params = mdl.init_params(jax.random.PRNGKey(tc.seed), cfg)
    state = mdl.adamw_init(params)
    step_fn = mdl.make_train_step(cfg, tc.lr, tc.warmup, total, tc.formula_weight)
    forward_fn = jax.jit(lambda p, b: mdl.forward(p, cfg, b))
    log(f"parameters: {mdl.count_params(params)}, steps: {total}, padded universe: {size}")
    best: dict[str, Any] = {"score": -1.0}
    history = []
    started = time.time()
    for epoch in range(tc.epochs):
        order = rng.permutation(len(train_set))
        running: dict[str, float] = {}
        for s in range(steps_per_epoch):
            idx = order[s * tc.batch_size : (s + 1) * tc.batch_size]
            batch = make_batch([train_set[i] for i in idx], cfg, size)
            params, state, metrics = step_fn(params, state, {k: jnp.asarray(v) for k, v in batch.items()})
            for key, value in metrics.items():
                running[key] = running.get(key, 0.0) + float(value)
        val = evaluate(params, cfg, val_set, formula_tokens, batch_size=tc.eval_batch_size, forward_fn=forward_fn)
        score = val["classification"]["q_star_accuracy"]
        entry = {
            "epoch": epoch + 1,
            "seconds": round(time.time() - started, 1),
            **{k: v / steps_per_epoch for k, v in running.items()},
            "val_q_star_accuracy": score,
            "val_decision_accuracy": val["classification"]["decision_accuracy"],
        }
        history.append(entry)
        log(json.dumps(entry))
        if score >= best["score"]:  # ties go to the later epoch, whose decoder has trained longer
            best = {"score": score, "epoch": epoch + 1}
            np.savez(out / "params.npz", **mdl.flatten(params))  # type: ignore[arg-type]
    template = mdl.init_params(jax.random.PRNGKey(0), cfg)
    params = mdl.unflatten(dict(np.load(out / "params.npz")), template)
    results = {
        "train_config": asdict(tc),
        "model_config": asdict(cfg),
        "parameters": mdl.count_params(params),
        "best_epoch": best["epoch"],
        "history": history,
        "splits": {},
    }
    decode_fn = jax.jit(lambda p, m, mm, t: mdl.decode(p, cfg, m, mm, t))
    for split in manifest["splits"]:
        examples = val_set if split == "val" else load_examples(tc.data, split)
        t0 = time.time()
        results["splits"][split] = evaluate(
            params,
            cfg,
            examples,
            formula_tokens,
            batch_size=tc.eval_batch_size if split != "test_size" else 2,
            synthesis_limit=tc.synthesis_limit,
            forward_fn=forward_fn,
            decode_fn=decode_fn,
        )
        log(f"{split}: {json.dumps(results['splits'][split])} ({time.time() - t0:.0f} s)")
    (out / "results.json").write_text(json.dumps(results, indent=2))
    return results


def report(runs: Sequence[str | Path]) -> dict[str, Any]:
    """Mean and standard deviation over runs of the main metrics of every split."""
    results = [json.loads((Path(r) / "results.json").read_text()) for r in runs]

    def collect(path: Sequence[str]) -> list[float]:
        values = []
        for res in results:
            node: Any = res
            for key in path:
                node = node.get(key) if isinstance(node, dict) else None
                if node is None:
                    break
            if isinstance(node, (int, float)):
                values.append(float(node))
        return values

    def stat(values: list[float]) -> dict[str, float] | None:
        if not values:
            return None
        return {"mean": float(np.mean(values)), "std": float(np.std(values)), "runs": len(values)}

    out: dict[str, Any] = {"runs": [str(r) for r in runs], "splits": {}}
    for split in results[0]["splits"]:
        base = ["splits", split]
        entry: dict[str, Any] = {
            "q_star_accuracy": stat(collect([*base, "classification", "q_star_accuracy"])),
            "decision_accuracy": stat(collect([*base, "classification", "decision_accuracy"])),
            "decision_balanced_accuracy": stat(collect([*base, "classification", "decision_balanced_accuracy"])),
            "counting_baseline_decision_accuracy": stat(collect([*base, "baseline_counting", "decision_accuracy"])),
            "synthesis_verified_rate": stat(collect([*base, "synthesis", "verified_rate"])),
            "synthesis_rank_optimal_rate": stat(collect([*base, "synthesis", "rank_optimal_rate"])),
        }
        strata = results[0]["splits"][split]["classification"].get("per_stratum", {})
        entry["per_stratum_q_star_accuracy"] = {
            s: stat(collect([*base, "classification", "per_stratum", s, "q_star_accuracy"])) for s in strata
        }
        entry["per_stratum_decision_accuracy"] = {
            s: stat(collect([*base, "classification", "per_stratum", s, "decision_accuracy"])) for s in strata
        }
        out["splits"][split] = entry
    return out


def baselines(data_dir: str | Path) -> dict[str, Any]:
    """Reference decision rules evaluated on every split.

    ``majority`` predicts the most frequent value of q* (and the most frequent
    decision) of the training split; ``wl1`` and ``counting`` predict
    FO^k-equivalence at rank q_max iff colour refinement, respectively the
    (k-1)-dimensional Weisfeiler–Leman algorithm, does not distinguish the pair.
    """
    from . import wl

    manifest = load_manifest(data_dir)
    q_max = manifest["config"]["q_max"]
    train_q = np.array([q_max + 1 if e["q_star"] is None else e["q_star"] for e in iter_examples(data_dir, "train")])
    values, counts = np.unique(train_q, return_counts=True)
    majority_q = int(values[np.argmax(counts)])
    majority_equivalent = bool(np.mean(train_q > q_max) >= 0.5)
    out: dict[str, Any] = {"majority_q_star": majority_q, "majority_equivalent": majority_equivalent, "splits": {}}
    for split in manifest["splits"]:
        examples = list(iter_examples(data_dir, split))
        q = np.array([q_max + 1 if e["q_star"] is None else e["q_star"] for e in examples])
        equivalent = q > q_max
        ck = np.array([bool(e.get("counting_equivalent")) for e in examples])
        wl1 = np.array(
            [not wl.wl(Structure.from_json(e["left"]), Structure.from_json(e["right"]), 1).distinguished for e in examples]
        )
        out["splits"][split] = {
            "n": len(examples),
            "majority_q_star_accuracy": float(np.mean(q == majority_q)),
            "majority_decision_accuracy": float(np.mean(equivalent == majority_equivalent)),
            "wl1_decision_accuracy": float(np.mean(wl1 == equivalent)),
            "counting_decision_accuracy": float(np.mean(ck == equivalent)),
            "fraction_equivalent": float(np.mean(equivalent)),
        }
    return out
