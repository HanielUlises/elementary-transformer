"""An axial tuple transformer for FO^k equivalence and formula synthesis.

A structure with n elements enters as the tensor of atomic types of its
k-tuples, of shape ``(n,) * k``. Each encoder layer lets every tuple attend,
separately along each axis i, to the n tuples that differ from it in the i-th
coordinate, and then applies a position-wise MLP. This is the continuous
analogue of one step of the type refinement of the game solver, in which
moving pebble i ranges over the same n tuples; the layer is equivariant under
permutations of the universe, and the pooled readout is invariant.

The model is written directly in JAX. Parameters are nested dictionaries of
arrays, and the optimiser is a plain implementation of AdamW.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

Params = dict[str, Any]
NEG_INF = -1e9


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int
    formula_vocab_size: int
    k: int
    q_max: int
    d_model: int = 128
    heads: int = 4
    layers: int = 6
    d_ff: int = 256
    decoder_layers: int = 3
    max_formula_len: int = 256

    @property
    def pad_token(self) -> int:
        return self.vocab_size

    @property
    def formula_bos(self) -> int:
        return self.formula_vocab_size

    @property
    def formula_pad(self) -> int:
        return self.formula_vocab_size + 1


# Building blocks


def _dense(key: jax.Array, fan_in: int, fan_out: int) -> Params:
    return {"w": jax.random.normal(key, (fan_in, fan_out)) / math.sqrt(fan_in), "b": jnp.zeros((fan_out,))}


def dense(p: Params, x: jax.Array) -> jax.Array:
    return x @ p["w"] + p["b"]


def _norm(d: int) -> Params:
    return {"g": jnp.ones((d,)), "b": jnp.zeros((d,))}


def layer_norm(p: Params, x: jax.Array) -> jax.Array:
    mu = x.mean(-1, keepdims=True)
    var = x.var(-1, keepdims=True)
    return (x - mu) * jax.lax.rsqrt(var + 1e-5) * p["g"] + p["b"]


def _mlp(key: jax.Array, sizes: list[int]) -> Params:
    keys = jax.random.split(key, len(sizes) - 1)
    return {"layers": [_dense(kk, a, b) for kk, a, b in zip(keys, sizes[:-1], sizes[1:], strict=True)]}


def mlp(p: Params, x: jax.Array) -> jax.Array:
    layers = p["layers"]
    for i, layer in enumerate(layers):
        x = dense(layer, x)
        if i < len(layers) - 1:
            x = jax.nn.gelu(x)
    return x


def _attention(key: jax.Array, d: int) -> Params:
    kq, kk, kv, ko = jax.random.split(key, 4)
    return {"q": _dense(kq, d, d), "k": _dense(kk, d, d), "v": _dense(kv, d, d), "o": _dense(ko, d, d)}


def attention(p: Params, xq: jax.Array, xkv: jax.Array, mask: jax.Array, heads: int) -> jax.Array:
    """Multi-head attention over the second-to-last axis; ``mask`` broadcasts to (..., heads, Lq, Lk)."""
    d = xq.shape[-1]
    dh = d // heads
    q = dense(p["q"], xq).reshape(*xq.shape[:-1], heads, dh)
    k = dense(p["k"], xkv).reshape(*xkv.shape[:-1], heads, dh)
    v = dense(p["v"], xkv).reshape(*xkv.shape[:-1], heads, dh)
    scores = jnp.einsum("...qhd,...khd->...hqk", q, k) / math.sqrt(dh)
    scores = jnp.where(mask, scores, NEG_INF)
    weights = jax.nn.softmax(scores, axis=-1)
    out = jnp.einsum("...hqk,...khd->...qhd", weights, v).reshape(*xq.shape[:-1], d)
    return dense(p["o"], out)


# Model


def init_params(key: jax.Array, cfg: ModelConfig) -> Params:
    d = cfg.d_model
    keys = iter(jax.random.split(key, 16 + cfg.layers * (cfg.k + 1) + cfg.decoder_layers * 3))
    params: Params = {
        "embed": jax.random.normal(next(keys), (cfg.vocab_size + 1, d)) * 0.02,
        "encoder": [
            {
                "ln1": _norm(d),
                "attn": [_attention(next(keys), d) for _ in range(cfg.k)],
                "ln2": _norm(d),
                "mlp": _mlp(next(keys), [d, cfg.d_ff, d]),
            }
            for _ in range(cfg.layers)
        ],
        "enc_ln": _norm(d),
        "readout": _mlp(next(keys), [2 * d, d, d]),
        "head": _mlp(next(keys), [3 * d, d, cfg.q_max]),
        "memory": _dense(next(keys), 2 * d, d),
        "side": jax.random.normal(next(keys), (2, d)) * 0.02,
        "fembed": jax.random.normal(next(keys), (cfg.formula_vocab_size + 2, d)) * 0.02,
        "fpos": jax.random.normal(next(keys), (cfg.max_formula_len, d)) * 0.02,
        "decoder": [
            {
                "ln1": _norm(d),
                "self": _attention(next(keys), d),
                "ln2": _norm(d),
                "cross": _attention(next(keys), d),
                "ln3": _norm(d),
                "mlp": _mlp(next(keys), [d, cfg.d_ff, d]),
            }
            for _ in range(cfg.decoder_layers)
        ],
        "dec_ln": _norm(d),
        "fout": _dense(next(keys), d, cfg.formula_vocab_size + 2),
    }
    return params


def count_params(params: Params) -> int:
    return sum(int(np.prod(x.shape)) for x in jax.tree_util.tree_leaves(params))


def _tuple_mask(mask: jax.Array, k: int) -> jax.Array:
    """``mask`` (B, N) over elements to (B, N, ..., N) over k-tuples."""
    b, n = mask.shape
    out = jnp.ones((b,) + (n,) * k, dtype=bool)
    for i in range(k):
        shape = [b] + [1] * k
        shape[i + 1] = n
        out = out & mask.reshape(shape)
    return out


def encoder_layer(layer: Params, cfg: ModelConfig, x: jax.Array, key_mask: jax.Array) -> jax.Array:
    """Axial attention along each of the k tuple axes, then a position-wise MLP (pre-norm residual)."""
    y = layer_norm(layer["ln1"], x)
    update = jnp.zeros_like(x)
    for axis in range(cfg.k):
        yt = jnp.moveaxis(y, axis + 1, -2)
        out = attention(layer["attn"][axis], yt, yt, key_mask, cfg.heads)
        update = update + jnp.moveaxis(out, -2, axis + 1)
    x = x + update
    return x + mlp(layer["mlp"], layer_norm(layer["ln2"], x))


def encode(params: Params, cfg: ModelConfig, tokens: jax.Array, mask: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Tuple embeddings (B, N, ..., N, d) and the tuple mask, from tokens (B, N^k) and element mask (B, N)."""
    b, n = mask.shape
    k = cfg.k
    x = params["embed"][tokens].reshape((b,) + (n,) * k + (cfg.d_model,))
    key_mask = mask.reshape((b,) + (1,) * (k - 1) + (1, 1, n))
    # Recompute each layer in the backward pass instead of storing its activations.
    layer_fn = jax.checkpoint(lambda layer, x, km: encoder_layer(layer, cfg, x, km))
    for layer in params["encoder"]:
        x = layer_fn(layer, x, key_mask)
    return layer_norm(params["enc_ln"], x), _tuple_mask(mask, k)


def readout(params: Params, x: jax.Array, tuple_mask: jax.Array) -> jax.Array:
    """Mean and max pooling over the valid tuples, followed by an MLP."""
    axes = tuple(range(1, x.ndim - 1))
    m = tuple_mask[..., None]
    mean = (x * m).sum(axes) / jnp.maximum(m.sum(axes), 1)
    top = jnp.where(m, x, NEG_INF).max(axes)
    return mlp(params["readout"], jnp.concatenate([mean, top], axis=-1))


def classify(params: Params, left: jax.Array, right: jax.Array) -> jax.Array:
    """Logits of ``left ≡^k_q right`` for q = 1..q_max; symmetric in the two structures."""
    features = jnp.concatenate([left + right, jnp.abs(left - right), left * right], axis=-1)
    return mlp(params["head"], features)


def memory(params: Params, cfg: ModelConfig, x: jax.Array, tuple_mask: jax.Array, side: int) -> tuple[jax.Array, jax.Array]:
    """Decoder memory: tuple embeddings pooled over the last coordinate, (B, N^(k-1), d)."""
    m = tuple_mask[..., None]
    mean = (x * m).sum(-2) / jnp.maximum(m.sum(-2), 1)
    top = jnp.where(m, x, NEG_INF).max(-2)
    mem = dense(params["memory"], jnp.concatenate([mean, top], axis=-1)) + params["side"][side]
    b = x.shape[0]
    return mem.reshape(b, -1, cfg.d_model), tuple_mask.any(-1).reshape(b, -1)


def decode(params: Params, cfg: ModelConfig, mem: jax.Array, mem_mask: jax.Array, tokens: jax.Array) -> jax.Array:
    """Next-token logits (B, L, V) for the formula prefix ``tokens`` (B, L)."""
    length = tokens.shape[1]
    y = params["fembed"][tokens] + params["fpos"][:length]
    causal = jnp.tril(jnp.ones((length, length), dtype=bool))[None, None]
    cross_mask = mem_mask[:, None, None, :]
    for layer in params["decoder"]:
        h = layer_norm(layer["ln1"], y)
        y = y + attention(layer["self"], h, h, causal, cfg.heads)
        y = y + attention(layer["cross"], layer_norm(layer["ln2"], y), mem, cross_mask, cfg.heads)
        y = y + mlp(layer["mlp"], layer_norm(layer["ln3"], y))
    return dense(params["fout"], layer_norm(params["dec_ln"], y))


def forward(params: Params, cfg: ModelConfig, batch: dict[str, jax.Array]) -> dict[str, jax.Array]:
    xa, ma = encode(params, cfg, batch["left"], batch["left_mask"])
    xb, mb = encode(params, cfg, batch["right"], batch["right_mask"])
    logits = classify(params, readout(params, xa, ma), readout(params, xb, mb))
    mem_a, mask_a = memory(params, cfg, xa, ma, 0)
    mem_b, mask_b = memory(params, cfg, xb, mb, 1)
    mem = jnp.concatenate([mem_a, mem_b], axis=1)
    mem_mask = jnp.concatenate([mask_a, mask_b], axis=1)
    out = {"equivalence_logits": logits, "memory": mem, "memory_mask": mem_mask}
    if "formula_in" in batch:
        out["formula_logits"] = decode(params, cfg, mem, mem_mask, batch["formula_in"])
    return out


def loss_fn(params: Params, cfg: ModelConfig, batch: dict[str, jax.Array], formula_weight: float) -> tuple[jax.Array, dict]:
    out = forward(params, cfg, batch)
    logits = out["equivalence_logits"]
    labels = batch["equivalence"]
    bce = jnp.mean(jnp.maximum(logits, 0) - logits * labels + jnp.log1p(jnp.exp(-jnp.abs(logits))))
    target = batch["formula_out"]
    tmask = batch["formula_mask"]
    logp = jax.nn.log_softmax(out["formula_logits"], axis=-1)
    nll = -jnp.take_along_axis(logp, target[..., None], axis=-1)[..., 0]
    ce = (nll * tmask).sum() / jnp.maximum(tmask.sum(), 1)
    return bce + formula_weight * ce, {"bce": bce, "formula_ce": ce}


# Optimisation


def adamw_init(params: Params) -> dict[str, Any]:
    zeros = jax.tree_util.tree_map(jnp.zeros_like, params)
    return {"step": jnp.zeros((), jnp.int32), "m": zeros, "v": jax.tree_util.tree_map(jnp.zeros_like, params)}


def adamw_update(
    params: Params,
    grads: Params,
    state: dict[str, Any],
    lr: jax.Array,
    *,
    b1: float = 0.9,
    b2: float = 0.98,
    eps: float = 1e-8,
    weight_decay: float = 0.01,
    clip: float = 1.0,
) -> tuple[Params, dict[str, Any]]:
    norm = jnp.sqrt(sum(jnp.sum(g**2) for g in jax.tree_util.tree_leaves(grads)))
    scale = jnp.minimum(1.0, clip / (norm + 1e-12))
    grads = jax.tree_util.tree_map(lambda g: g * scale, grads)
    step = state["step"] + 1
    m = jax.tree_util.tree_map(lambda m, g: b1 * m + (1 - b1) * g, state["m"], grads)
    v = jax.tree_util.tree_map(lambda v, g: b2 * v + (1 - b2) * g * g, state["v"], grads)
    c1 = 1 - b1 ** step.astype(jnp.float32)
    c2 = 1 - b2 ** step.astype(jnp.float32)

    def update(p: jax.Array, m_: jax.Array, v_: jax.Array) -> jax.Array:
        decay = weight_decay * p if p.ndim >= 2 else 0.0
        return p - lr * ((m_ / c1) / (jnp.sqrt(v_ / c2) + eps) + decay)

    return jax.tree_util.tree_map(update, params, m, v), {"step": step, "m": m, "v": v}


def schedule(step: jax.Array, peak: float, warmup: int, total: int) -> jax.Array:
    """Linear warm-up followed by cosine decay to a tenth of the peak."""
    s = step.astype(jnp.float32)
    warm = peak * s / max(warmup, 1)
    progress = jnp.clip((s - warmup) / max(total - warmup, 1), 0.0, 1.0)
    cosine = peak * (0.1 + 0.9 * 0.5 * (1 + jnp.cos(math.pi * progress)))
    return jnp.where(s < warmup, warm, cosine)


def make_train_step(cfg: ModelConfig, peak_lr: float, warmup: int, total: int, formula_weight: float) -> Callable:
    def step(params: Params, state: dict[str, Any], batch: dict[str, jax.Array]):
        (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(params, cfg, batch, formula_weight)
        lr = schedule(state["step"], peak_lr, warmup, total)
        params, state = adamw_update(params, grads, state, lr)
        return params, state, {"loss": loss, "lr": lr, **aux}

    return jax.jit(step, donate_argnums=(0, 1))


# Checkpoints


def flatten(params: Params, prefix: str = "") -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    if isinstance(params, dict):
        for key, value in params.items():
            out.update(flatten(value, f"{prefix}{key}/"))
    elif isinstance(params, list):
        for i, value in enumerate(params):
            out.update(flatten(value, f"{prefix}{i}/"))
    else:
        out[prefix.rstrip("/")] = np.asarray(params)
    return out


def unflatten(flat: dict[str, np.ndarray], template: Params) -> Params:
    def build(node: Any, prefix: str) -> Any:
        if isinstance(node, dict):
            return {key: build(value, f"{prefix}{key}/") for key, value in node.items()}
        if isinstance(node, list):
            return [build(value, f"{prefix}{i}/") for i, value in enumerate(node)]
        return jnp.asarray(flat[prefix.rstrip("/")])

    return build(template, "")
