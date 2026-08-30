import pytest

from vectordb_bench.backend.clients.treedb.treedb import TreeDB


def bridge() -> TreeDB:
    db = object.__new__(TreeDB)
    db.index_name, db.partition_generation, db.partition_node_config_sha256 = "idx", 7, "node"
    db._search_param = {"partition_probes": 2, "ef_search": 64}
    return db


def response() -> dict:
    return {"version": 1, "route": "/v1/vector-partition/public", "node_config_sha256": "node", "index": "idx", "generation": 7, "ids": [1], "scores": [0.5], "counters": {"selected_partitions": 2, "hnsw_served_partitions": 2, "exact_scan_partitions": 0}}


def test_partition_search_accepts_hnsw_only_complete_route(monkeypatch) -> None:
    db = bridge()
    monkeypatch.setattr(db, "_partition_request", lambda *_: response())
    assert db._partition_search([1.0], 1) == [1]


@pytest.mark.parametrize("field,value", [("version", 2), ("route", "wrong"), ("node_config_sha256", "wrong"), ("index", "wrong"), ("generation", 8), ("ids", []), ("ids", ["1"]), ("scores", []), ("counters", {"selected_partitions": 1, "hnsw_served_partitions": 1, "exact_scan_partitions": 0}), ("counters", {"selected_partitions": 2, "hnsw_served_partitions": 1, "exact_scan_partitions": 1})])
def test_partition_search_fails_closed(monkeypatch, field, value) -> None:
    db, result = bridge(), response()
    result[field] = value
    monkeypatch.setattr(db, "_partition_request", lambda *_: result)
    with pytest.raises(RuntimeError):
        db._partition_search([1.0], 1)


@pytest.mark.parametrize("field,value", [("ready", False), ("state", "staging"), ("version", 2), ("generation", 8)])
def test_partition_preflight_fails_closed(monkeypatch, field, value) -> None:
    db, result = bridge(), response()
    result.update({"ready": True, "state": "active"})
    result[field] = value
    db._emit_lifecycle = lambda *_args, **_kwargs: None
    monkeypatch.setattr(db, "_partition_request", lambda *_: result)
    with pytest.raises(RuntimeError):
        db._partition_preflight()


def test_partition_search_has_no_status_request(monkeypatch) -> None:
    db, calls = bridge(), []
    def request(path, payload=None):
        calls.append(path)
        return response()
    monkeypatch.setattr(db, "_partition_request", request)
    db._partition_search([1.0], 1)
    assert calls == ["/v1/search"]
