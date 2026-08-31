import http.client
import threading
from contextlib import contextmanager
from email.message import Message
from types import SimpleNamespace

import pytest

from vectordb_bench import config as bench_config
from vectordb_bench.backend.clients.treedb.config import TreeDBConfig, TreeDBHNSWConfig
from vectordb_bench.backend.clients.treedb.treedb import PartitionSearchError, TreeDB
from vectordb_bench.backend.runner.cold_warm_runner import ColdWarmSearchRunner
from vectordb_bench.backend.runner.mp_runner import MultiProcessingSearchRunner
from vectordb_bench.backend.runner.serial_runner import SerialSearchRunner


def bridge() -> TreeDB:
    db = object.__new__(TreeDB)
    db.index_name, db.partition_generation = "idx", 7
    db.partition_node_config_sha256 = "a" * 64
    db.partition_count = 3
    db.document_embedding_encoding = db.query_embedding_encoding = "json"
    db.stats_mode, db.response_format = "full_diagnostics", "full"
    db.live_ann_visibility_timeout, db.live_ann_visibility_poll_interval = 5.0, 0.05
    db.db_case_config = SimpleNamespace(strategy="column_graph")
    db._search_param = {
        "partition_probes": 2,
        "ef_search": 64,
        "use_vector_index": False,
        "query_mode": "exact",
        "quantized_index_name": "",
        "quantized_rerank_candidates": 0,
        "experimental": False,
        "require_vector_index_guards": True,
    }
    return db


def response() -> dict:
    return {
        "version": 1,
        "route": "treedb.nativewire.vector_search_v1",
        "node_config_sha256": "a" * 64,
        "index": "idx",
        "generation": 7,
        "ids": [1],
        "scores": [0.5],
        "counters": {"SelectedPartitions": 2, "HNSWServedPartitions": 2, "ExactScanPartitions": 0},
    }


def test_partition_search_accepts_hnsw_only_complete_route(monkeypatch) -> None:
    db = bridge()
    monkeypatch.setattr(db, "_partition_request", lambda *_, **__: response())
    assert db._partition_search([1.0], 1) == [1]


@pytest.mark.parametrize(
    "field,value",
    [
        ("version", 2), ("version", True), ("route", "wrong"),
        ("node_config_sha256", "wrong"), ("index", "wrong"),
        ("generation", 8), ("generation", True), ("ids", []),
        ("ids", ["1"]), ("scores", []),
        ("counters", {"SelectedPartitions": 1, "HNSWServedPartitions": 1, "ExactScanPartitions": 0}),
        ("counters", {"SelectedPartitions": 2, "HNSWServedPartitions": 1, "ExactScanPartitions": 1}),
    ],
)
def test_partition_search_fails_closed(monkeypatch, field, value) -> None:
    db, result = bridge(), response()
    result[field] = value
    monkeypatch.setattr(db, "_partition_request", lambda *_, **__: result)
    with pytest.raises(RuntimeError):
        db._partition_search([1.0], 1)


@pytest.mark.parametrize("field,value", [("ready", False), ("state", "staging"), ("version", 2), ("generation", 8)])
def test_partition_preflight_fails_closed(monkeypatch, field, value) -> None:
    db, result = bridge(), response()
    result.update({"ready": True, "state": "active"})
    result[field] = value
    db._emit_lifecycle = lambda *_args, **_kwargs: None
    monkeypatch.setattr(db, "_partition_request", lambda *_, **__: result)
    with pytest.raises(RuntimeError):
        db._partition_preflight()


def test_partition_preflight_records_external_count_provenance(monkeypatch) -> None:
    db, captured = bridge(), {}
    monkeypatch.setattr(db, "_partition_request", lambda *_, **__: {**response(), "ready": True, "state": "active"})
    db._emit_lifecycle = lambda event, **values: captured.update(event=event, **values)
    db._partition_preflight()
    assert captured == {
        "event": "partition_preflight",
        "mapping_model": "one_to_one_partition_pack_v1",
        "configured_partition_count": 3,
        "observed_route": "treedb.nativewire.vector_search_v1",
        "observed_node_config_sha256": "a" * 64,
        "observed_index": "idx",
        "observed_generation": 7,
        "partition_count_provenance": "external_manifest_join_required",
    }


def test_partition_search_has_no_status_request(monkeypatch) -> None:
    db, calls = bridge(), []
    def request(path, payload=None, **kwargs):
        calls.append(path)
        return response()
    monkeypatch.setattr(db, "_partition_request", request)
    db._partition_search([1.0], 1)
    assert calls == ["/v1/search"]


def test_serial_runner_does_not_replay_public_partition_search(monkeypatch) -> None:
    db = object.__new__(TreeDB)
    db.name, db.transport = "TreeDB", "partition_bridge_v1"
    calls = []

    def failed_partition_search(*_args):
        calls.append(1)
        raise RuntimeError("failed")

    monkeypatch.setattr(db, "_partition_search", failed_partition_search)
    monkeypatch.setattr(bench_config, "MAX_SEARCH_RETRY", 3)
    runner = SerialSearchRunner(db=db, test_data=[[1.0]], ground_truth=[[1]], k=1)
    with pytest.raises(PartitionSearchError):
        runner._get_db_search_res([1.0])
    assert calls == [1]


def test_serial_runner_keeps_retrying_ordinary_errors(monkeypatch) -> None:
    class RetryableDB:
        name = "retryable"

        @staticmethod
        def supports_payload_profile(_payload):
            return True

        def __init__(self):
            self.calls = 0

        def search_embedding(self, *_args):
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("retry")
            return [1]

    db = RetryableDB()
    monkeypatch.setattr(bench_config, "MAX_SEARCH_RETRY", 1)
    runner = SerialSearchRunner(db=db, test_data=[[1.0]], ground_truth=[[1]], k=1)
    assert runner._get_db_search_res([1.0]) == [1]
    assert db.calls == 2


@pytest.mark.parametrize(
    ("config", "kwargs"),
    [
        (TreeDBConfig, {"partition_generation": True}),
        (TreeDBConfig, {"partition_count": True}),
        (TreeDBConfig, {"timeout": True}),
        (TreeDBHNSWConfig, {"partition_probes": True}),
        (TreeDBHNSWConfig, {"ef_search": True}),
    ],
)
def test_treedb_config_rejects_raw_boolean_numbers(config, kwargs) -> None:
    base = {"base_url": "http://127.0.0.1:7120"} if config is TreeDBConfig else {}
    with pytest.raises((TypeError, ValueError), match="cannot be boolean"):
        config(**base, **kwargs)


def test_cold_warm_runner_does_not_retry_non_retryable_partition_error(monkeypatch) -> None:
    class DB:
        name = "TreeDB"

        @staticmethod
        def supports_payload_profile(_payload):
            return True

        def search_embedding(self, *_args):
            raise PartitionSearchError("partition failed")

    runner = ColdWarmSearchRunner(DB(), [[1.0]], k=1, query_count=1)
    monkeypatch.setattr(bench_config, "MAX_SEARCH_RETRY", 3)
    with pytest.raises(PartitionSearchError):
        runner._get_db_search_res([1.0])


@pytest.mark.parametrize("method", ["search", "search_by_dur"])
def test_multiprocessing_runner_propagates_non_retryable_partition_error(monkeypatch, method) -> None:
    class DB:
        name = "TreeDB"

        @staticmethod
        def supports_payload_profile(_payload):
            return True

        @contextmanager
        def init(self):
            yield

        def prepare_filter(self, _filters):
            pass

        def search_embedding(self, *_args):
            raise PartitionSearchError("partition failed")

    class Queue:
        @staticmethod
        def put(_value):
            pass

    class Condition:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        @staticmethod
        def wait():
            pass

    runner = MultiProcessingSearchRunner(DB(), [[1.0]], k=1, duration=1)
    monkeypatch.setattr("vectordb_bench.backend.runner.mp_runner.time.perf_counter", lambda: 0.0)
    with pytest.raises(PartitionSearchError):
        if method == "search":
            runner.search([[1.0]], Queue(), Condition())
        else:
            runner.search_by_dur(1, [[1.0]], Queue(), Condition())


@pytest.mark.parametrize("method", ["_run_all_concurrencies_mem_efficient", "_run_by_dur"])
@pytest.mark.parametrize("non_retryable", [True, False])
def test_multiprocessing_sweep_later_failure_respects_retry_marker(monkeypatch, method, non_retryable) -> None:
    class DB:
        name = "TreeDB"

        @staticmethod
        def supports_payload_profile(_payload):
            return True

    class Queue:
        @staticmethod
        def qsize():
            return 99

    class Condition:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        @staticmethod
        def notify_all():
            pass

    class Manager:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        @staticmethod
        def Queue():
            return Queue()

        @staticmethod
        def Condition():
            return Condition()

    state = {"concurrency": 0}

    class Future:
        def result(self):
            if state["concurrency"] == 2:
                error = PartitionSearchError("failed") if non_retryable else RuntimeError("failed")
                raise error
            return (1, 0, {"p99": 0.01, "p95": 0.01, "avg": 0.01, "count": 1}) if method == "_run_by_dur" else (1, 1, [0.01])

    class Executor:
        def __init__(self, *args, max_workers, **kwargs):
            state["concurrency"] = max_workers

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        @staticmethod
        def submit(*_args):
            return Future()

    runner = MultiProcessingSearchRunner(DB(), [[1.0]], concurrencies=[1, 2], duration=1)
    monkeypatch.setattr("vectordb_bench.backend.runner.mp_runner.mp.Manager", Manager)
    monkeypatch.setattr("vectordb_bench.backend.runner.mp_runner.concurrent.futures.ProcessPoolExecutor", Executor)
    monkeypatch.setattr(runner, "get_mp_context", lambda: None)
    monkeypatch.setattr(runner, "_wait_for_queue_fill", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(runner, "stop", lambda: None)
    ticks = iter(range(1, 20))
    monkeypatch.setattr("vectordb_bench.backend.runner.mp_runner.time.perf_counter", lambda: next(ticks))
    if non_retryable:
        with pytest.raises(PartitionSearchError):
            getattr(runner, method)(1) if method == "_run_by_dur" else getattr(runner, method)()
    else:
        result = getattr(runner, method)(1) if method == "_run_by_dur" else getattr(runner, method)()
        assert result[0] > 0


def test_partition_init_reuses_and_closes_only_bridge_connection(monkeypatch) -> None:
    db = bridge()
    db.transport = "partition_bridge_v1"
    db._clients = threading.local()
    connection = type("Connection", (), {"closed": False, "close": lambda self: setattr(self, "closed", True)})()
    monkeypatch.setattr(db, "_new_partition_connection", lambda: connection)
    monkeypatch.setattr(db, "_new_client", lambda: pytest.fail("partition mode must not create document clients"))
    with db.init():
        assert db._thread_clients().partition_connection is connection
    assert connection.closed


def test_partition_constructor_never_creates_document_client(monkeypatch) -> None:
    monkeypatch.setattr(TreeDB, "_new_client", lambda *_: pytest.fail("partition mode must not create document clients"))
    monkeypatch.setattr(TreeDB, "_partition_preflight", lambda *_: None)
    db = TreeDB(
        dim=1,
        db_config={
            "base_url": "http://127.0.0.1:7120",
            "transport": "partition_bridge_v1",
            "partition_generation": 7,
            "partition_node_config_sha256": "a" * 64,
            "partition_count": 3,
        },
        db_case_config=TreeDBHNSWConfig(partition_probes=2),
    )
    assert db.requires_live_ann_preflight is False


def test_partition_constructor_rejects_drop_old(monkeypatch) -> None:
    monkeypatch.setattr(TreeDB, "_partition_preflight", lambda *_: pytest.fail("must not preflight"))
    with pytest.raises(ValueError, match="rejects drop_old"):
        TreeDB(
            dim=1,
            db_config={
                "base_url": "http://127.0.0.1:7120",
                "transport": "partition_bridge_v1",
                "partition_generation": 7,
                "partition_node_config_sha256": "a" * 64,
                "partition_count": 3,
            },
            db_case_config=TreeDBHNSWConfig(partition_probes=2),
            drop_old=True,
        )


def test_partition_mode_rejects_mutation_paths() -> None:
    db = bridge()
    db.transport = "partition_bridge_v1"
    with pytest.raises(RuntimeError, match="rejects insert"):
        db.insert_embeddings([], [])
    with pytest.raises(RuntimeError, match="rejects optimize"):
        db.optimize()


def test_partition_config_rejects_probe_count_at_or_above_count() -> None:
    db = bridge()
    db.base_url, db.timeout, db._metric = "http://127.1.2.3:7120", 30, "cosine"
    db._search_param["partition_probes"] = db.partition_count
    with pytest.raises(ValueError, match="partition_count"):
        db._validate_partition_config_shape()


@pytest.mark.parametrize(
    ("db_config", "case_config"),
    [
        ({"query_embedding_encoding": "f32_le"}, {}),
        ({"document_embedding_encoding": "f32_le_b64"}, {}),
        ({"stats_mode": "production"}, {}),
        ({"response_format": "ids"}, {}),
        ({"live_ann_visibility_timeout": 4.0}, {}),
        ({"live_ann_visibility_poll_interval": 0.1}, {}),
        ({}, {"use_vector_index": True}),
        ({}, {"query_mode": "quantized_only"}),
        ({}, {"quantized_index_name": "quantized"}),
        ({}, {"quantized_rerank_candidates": 1}),
        ({}, {"quantized_codec": "scalar_u8"}),
        ({}, {"experimental": True}),
        ({}, {"require_vector_index_guards": False}),
        ({}, {"strategy": "native_runtime"}),
    ],
)
def test_partition_constructor_rejects_noncanonical_document_knobs(monkeypatch, db_config, case_config) -> None:
    monkeypatch.setattr(TreeDB, "_partition_preflight", lambda *_: pytest.fail("must not preflight"))
    config = {
        "base_url": "http://127.0.0.1:7120",
        "transport": "partition_bridge_v1",
        "partition_generation": 7,
        "partition_node_config_sha256": "a" * 64,
        "partition_count": 3,
        **db_config,
    }
    with pytest.raises(ValueError, match="rejects"):
        TreeDB(dim=1, db_config=config, db_case_config=TreeDBHNSWConfig(partition_probes=2, **case_config))


def test_partition_search_reuses_one_worker_connection(monkeypatch) -> None:
    db = bridge()
    db._clients = threading.local()
    connection = object()
    db._clients.partition_connection = connection
    calls = []
    monkeypatch.setattr(db, "_partition_response", lambda actual, path, payload: calls.append((actual, path)) or response())
    assert db._partition_request("/v1/search", {}) == response()
    assert db._partition_request("/v1/search", {}) == response()
    assert calls == [(connection, "/v1/search"), (connection, "/v1/search")]


def test_partition_failed_row_replaces_connection_without_resubmitting(monkeypatch) -> None:
    db = bridge()
    db.transport, db._clients = "partition_bridge_v1", threading.local()
    original = type("Connection", (), {"closed": False, "close": lambda self: setattr(self, "closed", True)})()
    replacement = type("Connection", (), {"closed": False, "close": lambda self: setattr(self, "closed", True)})()
    connections, calls = iter((original, replacement)), []
    monkeypatch.setattr(db, "_new_partition_connection", lambda: next(connections))

    def partition_response(connection, path, payload):
        calls.append((connection, path))
        if len(calls) == 1:
            raise http.client.RemoteDisconnected("stale")
        return response()

    monkeypatch.setattr(db, "_partition_response", partition_response)
    with db.init():
        with pytest.raises(http.client.RemoteDisconnected):
            db._partition_search([1.0], 1)
        assert db._partition_search([1.0], 1) == [1]
    assert original.closed and replacement.closed
    assert not hasattr(db._clients, "partition_connection")
    assert [path for _, path in calls] == ["/v1/search", "/v1/search"]
    assert calls[0][0] is original and calls[1][0] is replacement


def test_partition_oversized_response_discards_worker_connection(monkeypatch) -> None:
    db = bridge()
    db._clients = threading.local()
    original = type("Connection", (), {"closed": False, "close": lambda self: setattr(self, "closed", True)})()
    replacement = type("Connection", (), {"closed": False, "close": lambda self: setattr(self, "closed", True)})()
    db._clients.partition_connection = original
    monkeypatch.setattr(db, "_new_partition_connection", lambda: replacement)
    calls = []

    def partition_response(connection, path, payload):
        calls.append(connection)
        if connection is original:
            raise RuntimeError("TreeDB partition bridge response exceeds limit")
        return response()

    monkeypatch.setattr(db, "_partition_response", partition_response)
    with pytest.raises(RuntimeError, match="exceeds limit"):
        db._partition_request("/v1/search", {})
    assert original.closed
    assert db._partition_request("/v1/search", {}) == response()
    assert calls == [original, replacement]


@pytest.mark.parametrize("ids", [[1, 1], [1 << 64]])
def test_partition_search_rejects_duplicate_or_out_of_range_ids(monkeypatch, ids) -> None:
    db, result = bridge(), response()
    result["ids"] = ids
    result["scores"] = [0.5] * len(ids)
    monkeypatch.setattr(db, "_partition_request", lambda *_, **__: result)
    with pytest.raises(RuntimeError):
        db._partition_search([1.0], len(ids))


@pytest.mark.parametrize("score", [True, "1", float("nan"), float("inf")])
def test_partition_search_rejects_invalid_scores(monkeypatch, score) -> None:
    db, result = bridge(), response()
    result["scores"] = [score]
    monkeypatch.setattr(db, "_partition_request", lambda *_, **__: result)
    with pytest.raises(RuntimeError):
        db._partition_search([1.0], 1)


@pytest.mark.parametrize("query", [[], [True], [float("nan")], [float("inf")], ["1"]])
def test_partition_search_rejects_invalid_query(monkeypatch, query) -> None:
    db = bridge()
    monkeypatch.setattr(db, "_partition_request", lambda *_, **__: pytest.fail("must not request"))
    with pytest.raises(ValueError):
        db._partition_search(query, 1)


class _Response:
    def __init__(self, status, content_type, body):
        self.status, self.body = status, body
        self.headers = Message()
        self.headers["Content-Type"] = content_type
        self.read_limit = None

    def read(self, limit=None):
        self.read_limit = limit
        return self.body[:limit]


class _Connection:
    def __init__(self, response):
        self.response = response
        self.requests = []

    def request(self, *args, **kwargs):
        self.requests.append((args, kwargs))

    def getresponse(self):
        return self.response


@pytest.mark.parametrize(
    "status,content_type,body",
    [
        (500, "application/json", b"{}"),
        (200, "text/plain", b"{}"),
        (200, "application/json", b"{"),
        (200, "application/json", b"{} trailing"),
        (200, "application/json", b"x" * (1024 * 1024 + 1)),
        (500, "application/json", b"x" * (1024 * 1024 + 1)),
    ],
)
def test_partition_response_fails_closed_with_bounded_read(status, content_type, body) -> None:
    db, response_object = bridge(), _Response(status, content_type, body)
    with pytest.raises(RuntimeError):
        db._partition_response(_Connection(response_object), "/v1/search", {})
    assert response_object.read_limit == 1024 * 1024 + 1


def test_partition_response_rejects_oversized_request_before_send() -> None:
    db = bridge()
    connection = _Connection(_Response(200, "application/json", b"{}"))
    with pytest.raises(ValueError, match="request exceeds"):
        db._partition_response(connection, "/v1/search", {"query": "x" * (1024 * 1024)})
    assert not connection.requests


@pytest.mark.parametrize("base_url,timeout", [("https://127.0.0.1:7120", 30), ("http://localhost:7120", 30), ("http://127.0.0.1:7120/path", 30), ("http://127.0.0.1:7120", 0), ("http://127.0.0.1:7120", 61)])
def test_partition_config_rejects_invalid_url_or_timeout(base_url, timeout) -> None:
    db = bridge()
    db.base_url, db.timeout, db._metric = base_url, timeout, "cosine"
    with pytest.raises(ValueError):
        db._validate_partition_config_shape()
