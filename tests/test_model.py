import jax
import jax.numpy as jnp
import numpy as np

from elementary_transformer import dataset
from elementary_transformer import formulas as fo
from elementary_transformer import generators as gen
from elementary_transformer import model as mdl
from elementary_transformer import train as tr

from .conftest import random_graph

VOCAB = dataset.Vocabulary(3)
CFG = mdl.ModelConfig(
    vocab_size=VOCAB.size,
    formula_vocab_size=len(VOCAB.formula_tokens),
    k=3,
    q_max=3,
    d_model=32,
    heads=2,
    layers=2,
    d_ff=64,
    decoder_layers=1,
    max_formula_len=24,
)


def example(left, right, q_star, formula=None):
    return {
        "left": {"size": left.size},
        "right": {"size": right.size},
        "left_tokens": VOCAB.tokenize(left),
        "right_tokens": VOCAB.tokenize(right),
        "q_star": q_star,
        "formula_tokens": VOCAB.encode_formula(formula) if formula is not None else np.zeros(0, np.uint16),
        "signature": "graph",
    }


def test_readout_is_invariant_under_permutations(rng):
    params = mdl.init_params(jax.random.PRNGKey(0), CFG)
    g = random_graph(rng, 7)
    h = g.permute(rng.permutation(7))
    batch = tr.make_batch([example(g, h, None)], CFG, 8)
    x, m = mdl.encode(params, CFG, jnp.asarray(batch["left"]), jnp.asarray(batch["left_mask"]))
    y, n = mdl.encode(params, CFG, jnp.asarray(batch["right"]), jnp.asarray(batch["right_mask"]))
    np.testing.assert_allclose(mdl.readout(params, x, m), mdl.readout(params, y, n), rtol=1e-4, atol=1e-4)


def test_loss_decreases_on_a_fixed_batch():
    pairs_ = [
        (gen.cycle(6), gen.disjoint_cycles([3, 3])),
        (gen.path(5), gen.path(5)),
        (gen.complete(4), gen.cycle(4)),
        (gen.linear_order(3), gen.linear_order(3)),
    ]
    examples = []
    for a, b in pairs_:
        if a.signature != b.signature:
            continue
        result = dataset.label_pair(dataset.pairs.StructurePair(a, b, "t", "t"), 3, 3, wl_labels=False)
        examples.append(example(a, b, result.q_star, result.formula) | {"signature": dataset.signature_name(a.signature)})
    batch = {k: jnp.asarray(v) for k, v in tr.make_batch(examples, CFG, 6).items()}
    params = mdl.init_params(jax.random.PRNGKey(1), CFG)
    state = mdl.adamw_init(params)
    step = mdl.make_train_step(CFG, 3e-3, 1, 60, 1.0)
    first = None
    for _ in range(60):
        params, state, metrics = step(params, state, batch)
        first = first if first is not None else float(metrics["loss"])
    assert float(metrics["loss"]) < 0.5 * first


def test_grammar_constrained_decoding_is_well_formed():
    params = mdl.init_params(jax.random.PRNGKey(2), CFG)
    grammar = tr.FormulaGrammar(VOCAB.formula_tokens, 3)
    mem = jnp.asarray(np.random.default_rng(0).normal(size=(3, 10, CFG.d_model)), dtype=jnp.float32)
    mask = jnp.ones((3, 10), dtype=bool)
    decode_fn = jax.jit(lambda p, m, mm, t: mdl.decode(p, CFG, m, mm, t))
    outputs = tr.greedy_decode(decode_fn, params, CFG, mem, mask, grammar, [{"E"}] * 3)
    for tokens in outputs:
        assert len(tokens) <= CFG.max_formula_len
        phi = fo.from_prefix_tokens([VOCAB.formula_tokens[t] for t in tokens], gen.path(2).signature)
        assert fo.width(phi) <= 3


def test_predicted_q_star_and_metrics():
    probs = np.array([[0.9, 0.2, 0.1], [0.9, 0.8, 0.7], [0.9, 0.9, 0.3]])
    assert tr.predicted_q_star(probs).tolist() == [2, 4, 3]
    metrics = tr.classification_metrics(np.array([2, 4, 2]), np.array([2, 4, 3]), 3)
    assert metrics["q_star_accuracy"] == 2 / 3
    assert metrics["decision_accuracy"] == 1.0
