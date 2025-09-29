"""Metric registry and execution engine."""
from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Iterable

from acme_cli.llm import LlmEvaluator
from acme_cli.metrics.base import Metric
from acme_cli.metrics.bus_factor import BusFactorMetric
from acme_cli.metrics.code_quality import CodeQualityMetric
from acme_cli.metrics.dataset_code import DatasetAndCodeMetric
from acme_cli.metrics.dataset_quality import DatasetQualityMetric
from acme_cli.metrics.license import LicenseMetric
from acme_cli.metrics.performance import PerformanceClaimsMetric
from acme_cli.metrics.ramp_up import RampUpMetric
from acme_cli.metrics.size import SizeMetric
from acme_cli.types import EvaluationOutcome, MetricFailure, MetricResult, ModelContext


def build_metrics(llm: LlmEvaluator | None = None) -> list[Metric]:
    shared_llm = llm or LlmEvaluator()
    return [
        RampUpMetric(shared_llm),
        BusFactorMetric(),
        PerformanceClaimsMetric(shared_llm),
        LicenseMetric(),
        SizeMetric(),
        DatasetAndCodeMetric(),
        DatasetQualityMetric(),
        CodeQualityMetric(),
    ]


def evaluate_metrics(context: ModelContext, metrics: Iterable[Metric]) -> EvaluationOutcome:
    metrics = list(metrics)
    max_workers = min(len(metrics), int(os.getenv("ACME_MAX_WORKERS", os.cpu_count() or 4))) or 1
    results: dict[str, MetricResult] = {}
    failures: list[MetricFailure] = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_map = {executor.submit(metric.evaluate, context): metric for metric in metrics}
        for future in as_completed(future_map):
            metric = future_map[future]
            try:
                result = future.result()
                results[result.name] = result
            except Exception as exc:  # noqa: BLE001
                failures.append(MetricFailure(name=metric.name, message=str(exc)))
    return EvaluationOutcome(metrics=results, failures=failures)


__all__ = ["build_metrics", "evaluate_metrics"]
