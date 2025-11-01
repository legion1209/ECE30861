"""Load and organize scoring targets from the user-supplied URL file."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Iterator

from acme_cli.types import ScoreTarget
from acme_cli.urls import is_code_url, is_dataset_url, is_model_url


def parse_url_file(path: Path) -> list[ScoreTarget]:
    """Parse *path* into a list of :class:`ScoreTarget` objects."""
    lines = [line.strip() for line in _iter_lines(path) if line.strip()]
    dataset_buffer: list[str] = []
    code_buffer: list[str] = []
    targets: list[ScoreTarget] = []

    for line in lines:
        # Split the line by commas to separate model, dataset, and code links
        url_parts = [part.strip() for part in line.split(',')]

        if len(url_parts) > 3 or len(url_parts) < 1:
            raise ValueError(f"Invalid line format: {line}. Expected 'model_url, dataset_url, code_url'")

        # Extract the model, dataset, and code URLs
        code_url, dataset_url, model_url = url_parts
        dataset_buffer.append(dataset_url)
        code_buffer.append(code_url)
        
        # Create a ScoreTarget object for each line
        targets.append(ScoreTarget(model_url=model_url, dataset_urls=list(dataset_buffer), code_urls=list(code_buffer)))

    return targets



def _iter_lines(path: Path) -> Iterator[str]:
    with path.open("r", encoding="utf-8") as handle:
        yield from handle


__all__ = ["parse_url_file"]
