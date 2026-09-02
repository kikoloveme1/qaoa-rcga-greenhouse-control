from dataclasses import asdict

from paper_protocol import cli


def test_cli_does_not_create_results_without_explicit_out(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        cli,
        "build_landscape",
        lambda protocol, out: {
            "metadata": {"metrics": {}, "landscape_hash": "test"},
        },
    )
    monkeypatch.setattr(
        cli,
        "run_one",
        lambda protocol, method, scenario, seed, out, landscape, data_path=None: {
            "profit": 1.0,
            "penalty": 0.0,
            "total_seconds": 0.1,
            "protocol": asdict(protocol),
        },
    )

    cli.main(
        [
            "principal",
            "--smoke",
            "--methods",
            "es_policy_search",
            "--scenarios",
            "baseline",
            "--seeds",
            "42",
            "--backend",
            "numpy",
        ]
    )

    assert not (tmp_path / "results").exists()
