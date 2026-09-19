"""Command-line interface: ``elementary-transformer {generate,solve,wl,inspect,train,baselines,report}``."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from . import __version__

STRUCTURE_HELP = (
    "a structure: order:N, path:N, cycle:N, cycles:A,B,..., complete:N, petersen, rook:M, shrikhande, "
    "triangular:M, chang:I, cfi:BASE, cfi-twisted:BASE (BASE in K4, K33, prism3, cube, petersen), "
    "gnp:N:P:SEED, regular:N:D:SEED, graph6:STRING, or a path to a structure JSON file or a graph6 file"
)


def parse_structure(spec: str) -> Any:
    from . import generators as gen
    from .structures import Structure

    path = Path(spec)
    if path.is_file():
        text = path.read_text()
        if path.suffix == ".json":
            return Structure.from_json(json.loads(text))
        return gen.parse_graph6(text.splitlines()[0])
    name, _, rest = spec.partition(":")
    args = rest.split(":") if rest else []
    if name == "order":
        return gen.linear_order(int(args[0]))
    if name == "path":
        return gen.path(int(args[0]))
    if name == "cycle":
        return gen.cycle(int(args[0]))
    if name == "cycles":
        return gen.disjoint_cycles([int(x) for x in args[0].split(",")])
    if name == "complete":
        return gen.complete(int(args[0]))
    if name == "petersen":
        return gen.petersen()
    if name == "rook":
        return gen.rook_graph(int(args[0]))
    if name == "shrikhande":
        return gen.shrikhande()
    if name == "triangular":
        return gen.triangular(int(args[0]))
    if name == "chang":
        return gen.chang_graphs()[int(args[0]) - 1]
    if name in ("cfi", "cfi-twisted"):
        pair = gen.cfi_pair(gen.CFI_BASES[args[0]]())
        return pair[1] if name == "cfi-twisted" else pair[0]
    if name == "gnp":
        return gen.gnp(int(args[2]), int(args[0]), float(args[1]))
    if name == "regular":
        return gen.random_regular(int(args[2]), int(args[0]), int(args[1]))
    if name == "graph6":
        return gen.parse_graph6(rest)
    raise ValueError(f"cannot parse structure {spec!r}; expected {STRUCTURE_HELP}")


def _size_range(text: str) -> tuple[int, int]:
    lo, _, hi = text.partition(":")
    return int(lo), int(hi or lo)


def cmd_generate(args: argparse.Namespace) -> int:
    from .dataset import DatasetConfig, build_dataset

    config = DatasetConfig(
        k=args.k,
        q_max=args.q,
        families=tuple(args.families.split(",")),
        train_sizes=args.train_sizes,
        test_sizes=args.test_sizes,
        family_sizes=args.family_sizes,
        num_train=args.train,
        num_val=args.val,
        num_test=args.test,
        seed=args.seed,
        relabel=not args.no_relabel,
        shard_size=args.shard_size,
        max_attempts_factor=args.max_attempts_factor,
        wl_labels=not args.no_wl,
        lean_verify=args.lean_verify,
    )
    manifest = build_dataset(config, args.out, workers=args.workers, log=lambda m: print(m, file=sys.stderr))
    print(json.dumps({name: s["counts"] for name, s in manifest["splits"].items()}, indent=2))
    return 0


def cmd_solve(args: argparse.Namespace) -> int:
    from . import formulas as fo
    from . import games

    left, right = parse_structure(args.left), parse_structure(args.right)
    result = games.solve(left, right, args.k, args.q)
    out: dict[str, Any] = {
        "k": args.k,
        "q_max": args.q,
        "q_star": result.q_star,
        "equivalent": result.equivalent,
        "stable": result.stable,
        "types_per_round": result.num_types,
    }
    if result.strategy is not None and result.formula is not None:
        phi = result.formula
        move = result.strategy.move_at(*result.strategy.root())
        out["formula"] = fo.to_text(phi)
        out["formula_latex"] = fo.to_latex(phi)
        out["formula_size"] = fo.size(phi)
        out["model_checker"] = fo.distinguishes(phi, left, right)
        out["spoiler_first_move"] = (
            None
            if move is None
            else {
                "pebble": move.pebble,
                "structure": "left" if move.side == 0 else "right",
                "element": move.element,
            }
        )
        if args.lean:
            from . import lean

            out["lean"] = lean.check([lean.certificate(left, right, phi, args.k, result.q_star or 0)])[0]
    if args.json:
        print(json.dumps(out, indent=2, ensure_ascii=False))
    else:
        for key, value in out.items():
            print(f"{key}: {value}")
    return 0


def cmd_wl(args: argparse.Namespace) -> int:
    from . import wl

    left, right = parse_structure(args.left), parse_structure(args.right)
    result = wl.wl(left, right, args.dim)
    print(json.dumps(result.__dict__, indent=2))
    return 0


def cmd_inspect(args: argparse.Namespace) -> int:
    from .dataset import load_manifest

    manifest = load_manifest(args.dir)
    config = manifest["config"]
    print(f"k = {config['k']}, q_max = {config['q_max']}, seed = {config['seed']}")
    vocab = manifest["vocabulary"]
    print(f"vocabulary: {vocab['size']} structure tokens, {len(vocab['formula_tokens'])} formula tokens")
    for name, stats in manifest["splits"].items():
        print(f"{name}: {stats['generated']}/{stats['target']} examples, sizes {stats['sizes']}, q* counts {stats['counts']}")
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    from .train import TrainConfig, train

    config = TrainConfig(
        data=args.data,
        out=args.out,
        d_model=args.d_model,
        heads=args.heads,
        layers=args.layers,
        d_ff=args.d_ff,
        decoder_layers=args.decoder_layers,
        max_formula_len=args.max_formula_len,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        warmup=args.warmup,
        formula_weight=args.formula_weight,
        seed=args.seed,
        synthesis_limit=args.synthesis_limit,
    )
    train(config, log=lambda m: print(m, flush=True))
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    from .train import report

    print(json.dumps(report(args.runs), indent=2))
    return 0


def cmd_baselines(args: argparse.Namespace) -> int:
    from .train import baselines

    result = baselines(args.data)
    text = json.dumps(result, indent=2)
    if args.out:
        Path(args.out).write_text(text)
    print(text)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="elementary-transformer",
        description="Pairs of finite structures labelled by FO^k equivalence of bounded quantifier rank.",
    )
    parser.add_argument("--version", action="version", version=__version__)
    sub = parser.add_subparsers(dest="command", required=True)

    g = sub.add_parser("generate", help="generate a dataset with the split protocol")
    g.add_argument("--out", required=True, help="output directory")
    g.add_argument("--k", type=int, default=3, help="number of pebbles (variables)")
    g.add_argument("--q", type=int, default=4, help="maximal quantifier rank q_max")
    g.add_argument(
        "--families",
        default="linear_order,cycles,gnp,sparse,regular,cfi,srg",
        help="comma-separated families among linear_order, cycles, gnp, sparse, regular, geng, cfi, srg",
    )
    g.add_argument(
        "--train-sizes", type=_size_range, default=(3, 12), metavar="LO:HI", help="universe sizes of train, val and test_rank"
    )
    g.add_argument("--test-sizes", type=_size_range, default=(20, 50), metavar="LO:HI", help="universe sizes of test_size")
    g.add_argument("--family-sizes", type=_size_range, default=(16, 60), metavar="LO:HI", help="universe sizes of test_family")
    g.add_argument("--train", type=int, default=1000, help="number of training examples")
    g.add_argument("--val", type=int, default=200, help="number of validation examples")
    g.add_argument("--test", type=int, default=200, help="number of examples of each test split")
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--shard-size", type=int, default=1000)
    g.add_argument("--workers", type=int, default=1, help="number of worker processes (does not change the result)")
    g.add_argument("--max-attempts-factor", type=int, default=50, help="sampling budget per requested example")
    g.add_argument("--no-relabel", action="store_true", help="keep the vertex labelling produced by the generators")
    g.add_argument("--no-wl", action="store_true", help="skip the C^k labels")
    g.add_argument("--lean-verify", action="store_true", help="re-check every formula with the Lean checker")
    g.set_defaults(func=cmd_generate)

    s = sub.add_parser("solve", help="solve the k-pebble game on two structures", description=STRUCTURE_HELP)
    s.add_argument("left")
    s.add_argument("right")
    s.add_argument("--k", type=int, default=3)
    s.add_argument("--q", type=int, default=4)
    s.add_argument("--lean", action="store_true", help="also check the formula with the Lean checker")
    s.add_argument("--json", action="store_true")
    s.set_defaults(func=cmd_solve)

    w = sub.add_parser("wl", help="run d-dimensional Weisfeiler-Leman on two structures", description=STRUCTURE_HELP)
    w.add_argument("left")
    w.add_argument("right")
    w.add_argument("--dim", type=int, default=1)
    w.set_defaults(func=cmd_wl)

    i = sub.add_parser("inspect", help="summarise a generated dataset")
    i.add_argument("dir")
    i.set_defaults(func=cmd_inspect)

    t = sub.add_parser("train", help="train the axial tuple transformer on a dataset")
    t.add_argument("--data", required=True)
    t.add_argument("--out", required=True)
    t.add_argument("--d-model", type=int, default=128)
    t.add_argument("--heads", type=int, default=4)
    t.add_argument("--layers", type=int, default=6)
    t.add_argument("--d-ff", type=int, default=256)
    t.add_argument("--decoder-layers", type=int, default=3)
    t.add_argument("--max-formula-len", type=int, default=256)
    t.add_argument("--batch-size", type=int, default=32)
    t.add_argument("--epochs", type=int, default=10)
    t.add_argument("--lr", type=float, default=5e-4)
    t.add_argument("--warmup", type=int, default=500)
    t.add_argument("--formula-weight", type=float, default=1.0)
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--synthesis-limit", type=int, default=500, help="examples per split used to evaluate synthesis")
    t.set_defaults(func=cmd_train)

    b = sub.add_parser("baselines", help="evaluate majority and Weisfeiler-Leman decision rules on a dataset")
    b.add_argument("--data", required=True)
    b.add_argument("--out", help="write the result to this JSON file")
    b.set_defaults(func=cmd_baselines)

    r = sub.add_parser("report", help="aggregate the results of several training runs")
    r.add_argument("runs", nargs="+")
    r.set_defaults(func=cmd_report)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command != "train":
        # Labelling works on many small arrays, for which the CPU backend is faster than a GPU.
        os.environ.setdefault("JAX_PLATFORMS", "cpu")
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
