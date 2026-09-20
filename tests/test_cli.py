import pytest

import src.cli as cli


def test_build_parser_parses_known_commands():
    parser = cli.build_parser()
    for command in ["pipeline", "analysis", "train"]:
        assert parser.parse_args([command]).command == command


def test_build_parser_requires_a_command():
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args([])


def test_build_parser_rejects_unknown_command():
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(["nope"])


def test_main_dispatches_to_analysis(monkeypatch):
    called = {}

    def fake_run():
        called["analysis"] = True

    monkeypatch.setattr("src.run_analysis.run", fake_run)
    assert cli.main(["analysis"]) == 0
    assert called == {"analysis": True}
