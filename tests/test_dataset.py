import itertools

import numpy as np
import pytest

from elementary_transformer import dataset, lean
from elementary_transformer import formulas as fo
from elementary_transformer.atomic import AtomicTypes, set_partitions
from elementary_transformer.structures import GRAPH, ORDER

from .conftest import MIXED, naive_atomic_type, random_graph, random_structure


def test_set_partitions_are_counted_by_bell_numbers():
    assert [len(set_partitions(k)) for k in range(1, 6)] == [1, 2, 5, 15, 52]


def test_vocabulary_sizes():
    # Simple graphs: sum over partitions of [k] with b blocks of 2^(b choose 2).
    assert AtomicTypes(GRAPH, 2).size == 1 + 2
    assert AtomicTypes(GRAPH, 3).size == 1 + 3 * 2 + 8
    assert AtomicTypes(GRAPH, 4).size == 1 + 7 * 2 + 6 * 8 + 64
    # Strict orders without declared asymmetry: 2^(b(b-1)) per partition.
    assert AtomicTypes(ORDER, 2).size == 1 + 4


@pytest.mark.parametrize("signature,k", [(GRAPH, 3), (ORDER, 3), (MIXED, 2), (GRAPH, 4)])
def test_codes_are_exactly_the_atomic_types(rng, signature, k):
    types = AtomicTypes(signature, k)
    s = random_structure(rng, signature, 5)
    codes = np.asarray(types.codes(s))
    tuples = list(itertools.product(range(5), repeat=k))
    assert len(codes) == len(tuples)
    seen: dict[int, tuple] = {}
    for code, t in zip(codes.tolist(), tuples, strict=True):
        atp = naive_atomic_type(s, t)
        assert seen.setdefault(code, atp) == atp
    by_type: dict[tuple, int] = {}
    for code, t in zip(codes.tolist(), tuples, strict=True):
        assert by_type.setdefault(naive_atomic_type(s, t), code) == code
    assert 0 <= codes.min() and codes.max() < types.size


def test_tokens_are_equivariant(rng):
    vocab = dataset.Vocabulary(3)
    g = random_graph(rng, 6)
    perm = rng.permutation(6)
    tokens = vocab.tokenize(g).reshape(6, 6, 6)
    permuted = vocab.tokenize(g.permute(perm)).reshape(6, 6, 6)
    assert np.array_equal(permuted[np.ix_(perm, perm, perm)], tokens)


def test_label_pair():
    from elementary_transformer import generators as gen
    from elementary_transformer import pairs

    example = dataset.label_pair(pairs.cycle_pair([6], [3, 3]), 3, 4)
    assert example.q_star == 3 and example.formula is not None
    assert example.counting_equivalent is False
    example = dataset.label_pair(pairs.StructurePair(gen.rook_graph(4), gen.shrikhande(), "srg", "same_parameters"), 3, 4)
    assert example.q_star is None and example.counting_equivalent is True


@pytest.fixture(scope="module")
def small_dataset(tmp_path_factory):
    out = tmp_path_factory.mktemp("data")
    config = dataset.DatasetConfig(
        k=3,
        q_max=3,
        train_sizes=(3, 8),
        test_sizes=(14, 16),
        family_sizes=(16, 40),
        num_train=16,
        num_val=4,
        num_test=6,
        shard_size=10,
        max_attempts_factor=20,
        seed=1,
        lean_verify=lean.available(),
    )
    manifest = dataset.build_dataset(config, out)
    return out, manifest


def test_split_protocol(small_dataset):
    out, manifest = small_dataset
    assert set(manifest["splits"]) == {"train", "val", "test_size", "test_family", "test_rank"}
    train = manifest["splits"]["train"]
    assert train["generated"] == 16 and len(train["shards"]) == 2
    assert set(train["counts"]) == {"2", "equivalent"}
    for split in manifest["splits"]:
        examples = list(dataset.iter_examples(out, split))
        assert len(examples) == manifest["splits"][split]["generated"]
        for ex in examples:
            sizes = (ex["left"]["size"], ex["right"]["size"])
            if split == "train" or split == "val":
                assert ex["q_star"] is None or ex["q_star"] < 3
                assert ex["family"] not in dataset.HELD_OUT_FAMILIES
                assert all(3 <= n <= 8 for n in sizes)
            elif split == "test_size":
                assert ex["family"] not in dataset.HELD_OUT_FAMILIES
                assert all(14 <= n <= 16 for n in sizes)
            elif split == "test_family":
                assert ex["family"] in dataset.HELD_OUT_FAMILIES
            elif split == "test_rank":
                assert ex["q_star"] == 3
                assert all(3 <= n <= 8 for n in sizes)


def test_stored_examples_are_consistent(small_dataset):
    out, manifest = small_dataset
    vocab = dataset.Vocabulary(3)
    from elementary_transformer.structures import Structure

    for split in manifest["splits"]:
        for ex in dataset.iter_examples(out, split):
            left, right = Structure.from_json(ex["left"]), Structure.from_json(ex["right"])
            assert np.array_equal(ex["left_tokens"], vocab.tokenize(left))
            assert np.array_equal(ex["right_tokens"], vocab.tokenize(right))
            if ex["q_star"] is None:
                assert ex["formula"] is None and len(ex["formula_tokens"]) == 0
                continue
            phi = fo.from_json(ex["formula"])
            assert fo.distinguishes(phi, left, right) and fo.rank(phi) == ex["q_star"]
            tokens = [vocab.formula_tokens[i] for i in ex["formula_tokens"]]
            decoded = fo.from_prefix_tokens(tokens, left.signature)
            assert fo.distinguishes(decoded, left, right)
            if manifest["config"]["lean_verify"]:
                assert ex["lean_verified"] is True
            if ex["counting_equivalent"]:
                raise AssertionError("an FO^k-separated pair cannot be C^k-equivalent")


def test_padded_batches(small_dataset):
    out, manifest = small_dataset
    examples = list(dataset.iter_examples(out, "train"))[:4]
    vocab_size = manifest["vocabulary"]["size"]
    batch = dataset.pad_batch(examples, vocab_size, len(manifest["vocabulary"]["formula_tokens"]))
    assert batch["left"].shape[0] == 4
    assert int(batch["left_mask"].sum()) == sum(len(e["left_tokens"]) for e in examples)
    assert bool((batch["left"][~batch["left_mask"]] == vocab_size).all())
