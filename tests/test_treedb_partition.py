import pytest

from vectordb_bench.backend.clients.treedb.config import TreeDBHNSWConfig
from vectordb_bench.backend.clients.treedb.treedb import TreeDB


def bridge() -> TreeDB:
    db = object.__new__(TreeDB)
    db.index_name, db.partition_generation, db.partition_node_config_sha256 = "idx", 7, "node"
    db._search_param = {"partition_probes": 2, "ef_search": 64}
    return db


def response() -> dict:
    return {"version": 1, "route": "treedb.nativewire.vector_search_v1", "node_config_sha256": "node", "index": "idx", "generation": 7, "ids": [1], "scores": [0.5], "counters": {"SelectedPartitions": 2, "HNSWServedPartitions": 2, "ExactScanPartitions": 0}}


def test_partition_search_accepts_hnsw_only_complete_route(monkeypatch) -> None:
    db = bridge()
    monkeypatch.setattr(db, "_partition_request", lambda *_, **__: response())
    assert db._partition_search([1.0], 1) == [1]


@pytest.mark.parametrize("field,value", [("version", 2), ("version", True), ("route", "wrong"), ("node_config_sha256", "wrong"), ("index", "wrong"), ("generation", 8), ("generation", True), ("ids", []), ("ids", ["1"]), ("scores", []), ("counters", {"SelectedPartitions": 1, "HNSWServedPartitions": 1, "ExactScanPartitions": 0}), ("counters", {"SelectedPartitions": 2, "HNSWServedPartitions": 1, "ExactScanPartitions": 1})])
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


def test_partition_search_has_no_status_request(monkeypatch) -> None:
    db, calls = bridge(), []
    def request(path, payload=None, **kwargs):
        calls.append(path)
        return response()
    monkeypatch.setattr(db, "_partition_request", request)
    db._partition_search([1.0], 1)
    assert calls == ["/v1/search"]


def test_partition_init_reuses_and_closes_only_bridge_connection(monkeypatch) -> None:
    db = bridge()
    db.transport = "partition_bridge_v1"
    db._clients = __import__("threading").local()
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
        db_config={"base_url": "http://127.0.0.1:7120", "transport": "partition_bridge_v1", "partition_generation": 7, "partition_node_config_sha256": "node", "partition_count": 3},
        db_case_config=TreeDBHNSWConfig(partition_probes=2),
    )
    assert db.requires_live_ann_preflight is False


def test_partition_search_reuses_one_worker_connection(monkeypatch) -> None:
    db = bridge()
    db._clients = __import__("threading").local()
    connection = object()
    db._clients.partition_connection = connection
    calls = []
    monkeypatch.setattr(db, "_partition_response", lambda actual, path, payload: calls.append((actual, path)) or response())
    assert db._partition_request("/v1/search", {}) == response()
    assert db._partition_request("/v1/search", {}) == response()
    assert calls == [(connection, "/v1/search"), (connection, "/v1/search")]
