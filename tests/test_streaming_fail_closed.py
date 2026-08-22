from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from vectordb_bench.backend.runner.rate_runner import RatedMultiThreadingInsertRunner
from vectordb_bench.backend.runner.read_write_runner import ReadWriteRunner


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
