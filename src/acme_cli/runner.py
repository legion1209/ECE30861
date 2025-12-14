"""Top-level command handlers for the ACME CLI."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
USER_BIN = Path.home() / '.local' / 'bin'

class CommandError(RuntimeError):
    """Raised when a subprocess command fails."""


def run_install(extra_args: Sequence[str] | None = None) -> int:
    """Install project dependencies into the user's site-packages."""
    args: list[str] = [sys.executable, "-m", "pip", "install", "--user", ".[dev]"]
    if extra_args:
        args.extend(extra_args)
    return _checked_run(args)


def run_tests(pytest_args: Sequence[str] | None = None) -> int:
    """Execute the test suite with coverage enabled (when flags are provided)."""
    args: list[str] = [sys.executable, "-m", "pytest"]
    # The user/CLI must now provide the coverage flags via pytest_args
    if pytest_args:
        args.extend(pytest_args)
    return _checked_run(args)


def run_score(url_file: Path | None = None) -> int:
    """Score the models referenced in *url_file*.

    The heavy lifting lives in :mod:`acme_cli.scoring`. This thin wrapper merely
    ensures the package is importable without installation by augmenting
    ``sys.path`` to include ``src/`` when running from the repo root.
    """
    sys.path.insert(0, str(SRC_ROOT))
    from acme_cli.scoring import score_file  # pylint: disable=import-error

    score_file(url_file or [])
    return 0

def _checked_run(cmd: Sequence[str]) -> int:
    try:
        # Create a copy of the current environment
        env = os.environ.copy()
        
        # Explicitly add the user's bin directory to the PATH for the subprocess
        if USER_BIN.exists():
            env['PATH'] = str(USER_BIN) + os.pathsep + env.get('PATH', '')

        subprocess.run(cmd, check=True, env=env)
        print(f"Command {' '.join(cmd)} executed successfully.")
    except subprocess.CalledProcessError as exc:  # pragma: no cover - defensive
        print(f"Command {' '.join(cmd)} failed with exit code {exc.returncode}.")
        raise CommandError(str(exc)) from exc
    return 0

def score_artifact_for_worker(url: str):
    sys.path.insert(0, str(SRC_ROOT))

    from acme_cli.scoring import score_file
    scores = score_file(url)

    return scores

__all__ = ["CommandError", "run_install", "run_tests", "run_score"]
