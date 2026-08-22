import concurrent.futures
import multiprocessing as mp
import queue
from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any

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
    runner.executing_futures = [concurrent.futures.Future() for _ in range(201)]

    with pytest.raises(RuntimeError, match="backlog exceeded 200"):
        runner.run_with_rate(None)


def test_insert_backlog_wakes_waiting_streaming_search() -> None:
    @contextmanager
    def init():
        yield

    manager = mp.Manager()
    try:
        q, stop = manager.Queue(), manager.Event()
        insert = RatedMultiThreadingInsertRunner(
            rate=100, db=SimpleNamespace(name="Test", init=init), dataset_iter=iter(())
        )
        insert.executing_futures = [concurrent.futures.Future() for _ in range(201)]
        search = object.__new__(ReadWriteRunner)
        search.data_volume, search.insert_rate, search.search_stages = 200, 100, [0.5]

        with pytest.raises(RuntimeError, match="backlog exceeded 200"):
            insert.run_with_rate(q, stop)
        assert stop.is_set()
        assert search.run_search_by_sig(q, stop) is None
    finally:
        manager.shutdown()


def test_fixed_rate_runner_does_not_count_completed_tasks_as_backlog() -> None:
    @contextmanager
    def init():
        yield

    runner = RatedMultiThreadingInsertRunner(
        rate=100,
        db=SimpleNamespace(name="Test", init=init),
        dataset_iter=iter(()),
    )
    runner.executing_futures = [concurrent.futures.Future() for _ in range(201)]
    for future in runner.executing_futures:
        future.set_result(None)

    runner.run_with_rate(queue.Queue())


def test_streaming_search_failure_stops_insert_and_parent_promptly(monkeypatch: pytest.MonkeyPatch) -> None:
    class ThreadExecutor:
        def __init__(self, *args, **kwargs):
            self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.shutdown(wait=True)

        def submit(self, *args, **kwargs):
            return self.executor.submit(*args, **kwargs)

        def shutdown(self, **kwargs):
            self.executor.shutdown(**kwargs)

    runner = object.__new__(ReadWriteRunner)
    stopped = []

    def insert(_q: Any, stop: Any) -> tuple[None, int]:
        stop.wait(1)
        stopped.append(stop.is_set())
        return None, 0

    def search(_q: Any, _stop: Any) -> None:
        raise RuntimeError("search failed")

    runner.run_with_rate = insert
    runner.run_search_by_sig = search
    monkeypatch.setattr(
        "vectordb_bench.backend.runner.read_write_runner.concurrent.futures.ProcessPoolExecutor", ThreadExecutor
    )

    with pytest.raises(RuntimeError, match="search failed"):
        runner.run_read_write()
    assert stopped == [True]


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
