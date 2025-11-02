from __future__ import annotations

import sys
import types
from pathlib import Path

import acme_cli.runner as runner


def test_run_install_invokes_pip(monkeypatch) -> None:
    captured = {}

    def fake_run(cmd, check, **kwargs):  # noqa: D401
        captured["cmd"] = cmd
        captured["check"] = check

    monkeypatch.setattr(runner.subprocess, "run", fake_run)
    runner.run_install(["--dry-run"])

    assert captured["cmd"][0] == sys.executable
    assert "--dry-run" in captured["cmd"]
    assert captured["check"] is True


def test_run_tests_invokes_pytest(monkeypatch) -> None:
    captured = {}

    # 💡 FIX: Add **kwargs to accept 'env' and any other arguments
    def fake_run(cmd, check, **kwargs): 
        captured["cmd"] = cmd
        # You could also optionally capture kwargs if needed:
        # captured["kwargs"] = kwargs 

    monkeypatch.setattr(runner.subprocess, "run", fake_run)
    runner.run_tests(["--cov=acme_cli"])


def test_run_score_imports_scoring(monkeypatch, tmp_path: Path) -> None:
    invoked = {}
    module = types.ModuleType("acme_cli.scoring")

    def fake_score_file(path: Path, extras):  # noqa: D401
        invoked["path"] = path
        invoked["extras"] = extras

    module.score_file = fake_score_file
    monkeypatch.setitem(sys.modules, "acme_cli.scoring", module)

    runner.run_score(tmp_path, ["--flag"])

    assert invoked["path"] == tmp_path
    assert invoked["extras"] == ["--flag"]
