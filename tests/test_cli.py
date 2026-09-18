import json

from elementary_transformer import cli


def test_solve(capsys):
    assert cli.main(["solve", "cycle:6", "cycles:3,3", "--k", "3", "--q", "4", "--json"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["q_star"] == 3 and out["model_checker"] is True


def test_solve_structure_files(tmp_path, capsys):
    from elementary_transformer import generators as gen

    (tmp_path / "a.json").write_text(json.dumps(gen.linear_order(5).to_json()))
    (tmp_path / "b.json").write_text(json.dumps(gen.linear_order(6).to_json()))
    cli.main(["solve", str(tmp_path / "a.json"), str(tmp_path / "b.json"), "--k", "3", "--q", "3", "--json"])
    assert json.loads(capsys.readouterr().out)["q_star"] == 3


def test_wl(capsys):
    cli.main(["wl", "rook:4", "shrikhande", "--dim", "2"])
    assert json.loads(capsys.readouterr().out)["distinguished"] is False


def test_generate_and_inspect(tmp_path, capsys):
    out = tmp_path / "ds"
    code = cli.main(
        [
            "generate",
            "--out",
            str(out),
            "--k",
            "2",
            "--q",
            "2",
            "--families",
            "linear_order,cycles,srg",
            "--train-sizes",
            "3:6",
            "--test-sizes",
            "9:10",
            "--train",
            "6",
            "--val",
            "2",
            "--test",
            "3",
        ]
    )
    assert code == 0
    capsys.readouterr()
    cli.main(["inspect", str(out)])
    assert "train" in capsys.readouterr().out
