from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from vectordb_bench.backend.runner.rate_runner import RatedMultiThreadingInsertRunner
from vectordb_bench.backend.runner.read_write_runner import ReadWriteRunner
from vectordb_bench.backend.task_runner import CaseRunner
from vectordb_bench.metric import Metric


def test_fixed_rate_runner_rejects_backlog_instead_of_skipping_data() -> None:
    @contextmanager
    def init():
        yield

    runner = RatedMultiThreadingInsertRunner(
        rate=100,
        db=SimpleNamespace(name="Test", init=init),
        dataset_iter=iter(()),
    )
    runner.executing_futures = [object()] * 201

    with pytest.raises(RuntimeError, match="backlog exceeded 200"):
        runner.run_with_rate(None)


def test_streaming_search_error_is_not_reported_as_zero_metrics() -> None:
    runner = object.__new__(ReadWriteRunner)
    runner.data_volume = 1
    runner.insert_rate = 1
    runner.search_stages = [0.0]
    runner.serial_search_runner = SimpleNamespace(run=lambda: (_ for _ in ()).throw(RuntimeError("search failed")))

    with pytest.raises(RuntimeError, match="search failed"):
        runner.run_search_by_sig(None)


@pytest.mark.parametrize("requires_preflight", [False, True])
def test_treedb_streaming_only_runs_preflight_for_vector_index_cases(requires_preflight: bool) -> None:
    @contextmanager
    def init():
        yield

    calls = []
    db = SimpleNamespace(
        name="TreeDB",
        init=init,
        requires_live_ann_preflight=requires_preflight,
        preflight_live_ann=lambda: calls.append("preflight"),
    )
    runner = object.__new__(CaseRunner)
    object.__setattr__(runner, "db", db)
    object.__setattr__(runner, "read_write_runner", SimpleNamespace(run_read_write=lambda: Metric()))
    object.__setattr__(runner, "_init_read_write_runner", lambda: calls.append("runner"))

    assert runner._run_streaming_case() == Metric()
    assert calls == (["preflight", "runner"] if requires_preflight else ["runner"])
