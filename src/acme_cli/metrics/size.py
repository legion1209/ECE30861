"""Model size compatibility metric."""
from __future__ import annotations

from pathlib import Path

from acme_cli.metrics.base import Metric
from acme_cli.types import ModelContext, RepoFile


_THRESHOLDS = {
    "raspberry_pi": (200 * 1024 * 1024, 600 * 1024 * 1024),
    "jetson_nano": (600 * 1024 * 1024, 2 * 1024 * 1024 * 1024),
    "desktop_pc": (2 * 1024 * 1024 * 1024, 6 * 1024 * 1024 * 1024),
    "aws_server": (6 * 1024 * 1024 * 1024, 20 * 1024 * 1024 * 1024),
}

_WEIGHT_EXTENSIONS = (".bin", ".safetensors", ".pt", ".onnx", ".gguf", ".ggml")


class SizeMetric(Metric):
    name = "size_score"

    def compute(self, context: ModelContext) -> dict[str, float]:
        local_repo = context.local_repo
        total_bytes = 0
        if local_repo and local_repo.path:
            total_bytes = SizeMetric._collect_weight_bytes(local_repo.path)
        if total_bytes == 0 and context.model_metadata:
            total_bytes = self._collect_metadata_bytes(context.model_metadata.files)
        if total_bytes == 0:
            return {key: 0.0 for key in _THRESHOLDS}
        return {hardware: self._hardware_score(total_bytes, limits) for hardware, limits in _THRESHOLDS.items()}

    def _collect_weight_bytes(path: Path) -> int:
        total = 0                             
        for file in path.rglob("*"):
            try:
                file_size = file.stat().st_size
                total += file_size
            except OSError as e:
                continue
        return total

    @staticmethod
    def _hardware_score(total_bytes: int, limits: tuple[int, int]) -> float:
        sweet_spot, upper_bound = limits
        if total_bytes <= sweet_spot:
            return 1.0
        if total_bytes <= upper_bound:
            return 0.7
        if total_bytes <= upper_bound * 1.5:
            return 0.4
        return 0.1

    def _collect_metadata_bytes(self, files: list[RepoFile]) -> int:
        total = 0
        for file in files:
            # 1. Filter only for files that are model weights
            if any(file.path.endswith(ext) for ext in _WEIGHT_EXTENSIONS):
                file_size = file.size_bytes
                
                # 2. Check if size is missing (None or 0)
                if not file_size:
                    # 3. Fallback: Attempt to fetch size remotely
                    # NOTE: You need the full URL for this file to make the request.
                    # Assuming 'file.path' is enough to construct the URL or that 
                    # 'RepoFile' has a 'url' attribute.
                    file_size = self._fetch_remote_size(file.path) 

                # 4. If size is found (either via metadata or remote fetch), add it to total
                if file_size and isinstance(file_size, int) and file_size > 0:
                    total += file_size
                    
        return total


__all__ = ["SizeMetric"]
