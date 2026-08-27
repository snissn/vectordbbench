import time
from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from vectordb_bench.backend.clients import api
from vectordb_bench.backend.runner.read_write_runner import ReadWriteRunner
from vectordb_bench.backend.task_runner import CaseRunner
from vectordb_bench.models import PerformanceTimeoutError, TaskStage


class ImmediateExecutor:
    result = None
    seen_timeout = None

    def __init__(self, *args, **kwargs):
        self._processes = {}

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return None

    def submit(self, _function: object):
        executor = self

        class Future:
            def result(self, timeout: float | None = None):
                ImmediateExecutor.seen_timeout = timeout
                return executor.result

        return Future()


def case_runner(optimize_timeout: float, allowance: float) -> CaseRunner:
    runner = object.__new__(CaseRunner)
    object.__setattr__(runner, "db", SimpleNamespace(optimize_timeout_allowance=allowance))
    object.__setattr__(runner, "ca", SimpleNamespace(optimize_timeout=optimize_timeout))
    return runner


def test_case_runner_reports_database_duration_and_preserves_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "vectordb_bench.backend.task_runner.concurrent.futures.ProcessPoolExecutor",
        ImmediateExecutor,
    )
    runner = case_runner(optimize_timeout=0.05, allowance=90.0)
    ImmediateExecutor.result = (api.OptimizeResult(0.02), 0.11)

    assert runner._optimize() == 0.02
    assert ImmediateExecutor.seen_timeout == 90.05

    ImmediateExecutor.result = (api.OptimizeResult(0.051), 0.06)
    with pytest.raises(PerformanceTimeoutError):
        runner._optimize()


def test_performance_load_duration_uses_database_optimize_duration() -> None:
    runner = object.__new__(CaseRunner)
    object.__setattr__(
        runner,
        "config",
        SimpleNamespace(stages={TaskStage.LOAD}, load_concurrency=None, case_config=SimpleNamespace(k=10)),
    )
    object.__setattr__(
        runner,
        "ca",
        SimpleNamespace(
            payload_profile=SimpleNamespace(value="ids_only"),
            estimated_payload_bytes_per_query=lambda _k: 0,
        ),
    )
    object.__setattr__(runner, "_last_load_concurrency", 4)
    object.__setattr__(runner, "_load_train_data", lambda: (None, 1.25))
    object.__setattr__(runner, "_optimize", lambda: 0.02)

    metric = runner._run_perf_case()

    assert metric.insert_duration == 1.25
    assert metric.optimize_duration == 0.02
    assert metric.load_duration == 1.27


def test_read_write_runner_preserves_explicit_database_duration() -> None:
    @contextmanager
    def init():
        yield

    def optimize(*, data_size: int | None):
        assert data_size == 50_000
        time.sleep(0.06)
        return api.OptimizeResult(0.01)

    runner = object.__new__(ReadWriteRunner)
    runner.db = SimpleNamespace(init=init, optimize=optimize)
    runner.data_volume = 50_000

    result, wall_duration = runner.run_optimize()

    assert result == api.OptimizeResult(0.01)
    assert wall_duration >= 0.05
    assert api.reported_optimize_duration(result, wall_duration) == 0.01


def test_existing_optimize_return_values_keep_wall_duration() -> None:
    assert api.reported_optimize_duration({"duration_seconds": 0.01}, 2.5) == 2.5
