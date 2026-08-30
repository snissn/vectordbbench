import base64
import json
import pickle
import sys
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from types import ModuleType, SimpleNamespace
from typing import TYPE_CHECKING, Any

import pytest
from click.testing import CliRunner
from pytest import MonkeyPatch

from vectordb_bench.backend.clients import DB, IndexType
from vectordb_bench.backend.clients.treedb import cli as treedb_cli
from vectordb_bench.backend.clients.treedb.config import (
    TreeDBColumnGraphExactConfig,
    TreeDBConfig,
    TreeDBHNSWConfig,
    TreeDBScalarU8RerankConfig,
)
from vectordb_bench.backend.runner.concurrent_runner import ConcurrentInsertRunner

if TYPE_CHECKING:
    from vectordb_bench.backend.clients.treedb.treedb import TreeDB


def test_treedb_concurrent_insert_uses_and_closes_distinct_worker_clients(monkeypatch: MonkeyPatch) -> None:
    class Data:
        train_id_field = "id"
        train_vector_field = "vector"

    class Dataset:
        data = Data()

        def iter_batches(self, batch_size):
            import numpy as np
            import pandas as pd

            return iter(
                [pd.DataFrame({"id": [row_id], "vector": [np.array([float(row_id), 1.0])]}) for row_id in range(2)]
            )

    clients = []
    insert_gate = threading.Barrier(2)

    class FakeDocument:
        def __init__(self, id, embedding):
            self.id = id
            self.embedding = embedding

    class FakeClient:
        def __init__(self, base_url, timeout=30.0):
            self.closed = False
            self.inserting = False
            clients.append(self)

        def create_index(self, *args, **kwargs):
            pass

        def close(self):
            self.closed = True

        def upsert_documents(self, index_name, documents, *, defer_vector_index_rebuild=False):
            assert not self.closed
            assert not self.inserting
            self.inserting = True
            try:
                insert_gate.wait(timeout=2)
                return SimpleNamespace(upserted=len(documents))
            finally:
                self.inserting = False

    fake_module = ModuleType("treedb_client")
    fake_module.Document = FakeDocument
    fake_module.TreeDBClient = FakeClient
    monkeypatch.setitem(sys.modules, "treedb_client", fake_module)

    from vectordb_bench.backend.clients.treedb.treedb import TreeDB

    db = TreeDB(
        dim=2,
        db_config={"base_url": "http://127.0.0.1:7120", "index_name": "bench"},
        db_case_config=TreeDBColumnGraphExactConfig(),
    )
    assert clients[0].closed

    runner = ConcurrentInsertRunner(db, Dataset(), normalize=False, max_workers=2, batch_size=1)
    assert runner.load_concurrency == {"requested": 2, "effective": 2}
    with db.init():
        assert pickle.loads(pickle.dumps(db))._client is None  # noqa: S301
    assert runner.task() == 2
    worker_clients = clients[2:]
    assert len(worker_clients) == 2
    assert len({id(client) for client in worker_clients}) == 2
    assert all(client.closed for client in worker_clients)


def test_treedb_async_insert_clamps_to_one_thread_local_worker() -> None:
    from vectordb_bench.backend.clients.treedb.treedb import TreeDB
    from vectordb_bench.backend.runner.concurrent_runner import ExecutorBackend

    db = object.__new__(TreeDB)
    db.name = "TreeDB"
    runner = ConcurrentInsertRunner(
        db, SimpleNamespace(), normalize=False, max_workers=4, backend=ExecutorBackend.ASYNC
    )

    assert runner.load_concurrency == {"requested": 4, "effective": 1}


@pytest.mark.parametrize("accepts_numpy_embeddings", [False, True])
def test_concurrent_insert_runner_preserves_numpy_only_for_opted_in_db(accepts_numpy_embeddings) -> None:
    import numpy as np
    import pandas as pd

    class Data:
        train_id_field = "id"
        train_vector_field = "vector"

    class Dataset:
        data = Data()

        def iter_batches(self, batch_size):
            return iter([pd.DataFrame({"id": [1], "vector": [np.array([0.1, 0.2], dtype=np.float32)]})])

    class DB:
        thread_safe = True
        name = "fake"

        def __init__(self):
            self.accepts_numpy_embeddings = accepts_numpy_embeddings
            self.received = None

        @contextmanager
        def init(self):
            yield

        def insert_embeddings(self, embeddings, metadata, labels_data=None):
            self.received = embeddings
            return len(metadata), None

    db = DB()

    assert ConcurrentInsertRunner(db, Dataset(), normalize=False, max_workers=1).task() == 1
    assert isinstance(db.received, np.ndarray) is accepts_numpy_embeddings


def test_treedb_concurrent_insert_closes_worker_clients_after_failures(monkeypatch: MonkeyPatch) -> None:
    from vectordb_bench import config
    from vectordb_bench.backend.clients.treedb.treedb import TreeDB

    class Data:
        train_id_field = "id"
        train_vector_field = "vector"

    class Dataset:
        data = Data()

        def iter_batches(self, batch_size):
            import numpy as np
            import pandas as pd

            return iter([pd.DataFrame({"id": [0], "vector": [np.array([0.0])]})])

    class RetryableFailure(RuntimeError):
        pass

    clients = []
    db = object.__new__(TreeDB)
    db.name = "TreeDB"
    db._client = None
    db._clients = threading.local()
    db._new_client = lambda: SimpleNamespace(close=lambda: None, closed=False)

    def new_client():
        client = SimpleNamespace(closed=False)
        client.close = lambda: setattr(client, "closed", True)
        clients.append(client)
        return client

    db._new_client = new_client
    db.insert_embeddings = lambda **kwargs: (0, RetryableFailure("retryable failure"))
    monkeypatch.setattr(config, "MAX_INSERT_RETRY", 0)

    with pytest.raises(RuntimeError, match="retried more than 0 times"):
        ConcurrentInsertRunner(db, Dataset(), normalize=False, max_workers=2, batch_size=1).task()

    assert len(clients) == 2
    assert all(client.closed for client in clients)


def test_treedb_concurrent_insert_closes_worker_clients_after_deadline() -> None:
    from vectordb_bench.backend.clients.treedb.treedb import TreeDB

    class Data:
        train_id_field = "id"
        train_vector_field = "vector"

    class Dataset:
        data = Data()

        def iter_batches(self, batch_size):
            return iter(())

    clients = []
    db = object.__new__(TreeDB)
    db.name = "TreeDB"
    db._client = None
    db._clients = threading.local()

    def new_client():
        client = SimpleNamespace(closed=False)
        client.close = lambda: setattr(client, "closed", True)
        clients.append(client)
        return client

    db._new_client = new_client

    runner = ConcurrentInsertRunner(db, Dataset(), normalize=False, max_workers=2, duration=0)
    assert runner.task() == 0
    assert len(clients) == 2
    assert all(client.closed for client in clients)


def test_treedb_config_to_dict_and_case_config_scalar_u8_rerank() -> None:
    config = TreeDBConfig(
        db_label="local",
        base_url="http://127.0.0.1:7120",
        index_name="bench",
        timeout=5,
        document_embedding_encoding="f32_le_b64",
        query_embedding_encoding="f32_le_b64",
    )
    assert config.to_dict() == {
        "base_url": "http://127.0.0.1:7120",
        "index_name": "bench",
        "timeout": 5,
        "document_embedding_encoding": "f32_le_b64",
        "query_embedding_encoding": "f32_le_b64",
        "stats_mode": "full_diagnostics",
        "response_format": "full",
        "live_ann_visibility_timeout": 5.0,
            "live_ann_visibility_poll_interval": 0.05,
            "transport": "document_service",
            "partition_generation": 0,
            "partition_node_config_sha256": "",
            "partition_count": 0,
    }

    case = TreeDBHNSWConfig(
        index=IndexType.HNSW,
        strategy="column_graph",
        m=16,
        ef_construction=128,
        ef_search=64,
        use_vector_index=True,
        query_mode="quantized_rerank",
        quantized_codec="scalar_u8",
        quantized_index_name="embedding.scalar_u8.fast",
        quantized_rerank_candidates=32,
    )

    assert case.index_param()["quantized_indexes"] == [
        {"name": "embedding.scalar_u8.fast", "codec": "scalar_u8", "version": 1}
    ]
    assert case.search_param()["query_mode"] == "quantized_rerank"
    assert case.search_param()["quantized_rerank_candidates"] == 32


def test_treedb_cli_dry_run_captures_scalar_u8_rerank(monkeypatch: MonkeyPatch) -> None:
    captured = {}

    def fake_run(*args, **kwargs):
        captured["args"] = args
        captured.update(kwargs)

    monkeypatch.setattr(treedb_cli, "run", fake_run)

    result = CliRunner().invoke(
        treedb_cli.TreeDBHNSW,
        [
            "--base-url",
            "http://127.0.0.1:7120",
            "--index-name",
            "bench",
            "--m",
            "16",
            "--ef-construction",
            "128",
            "--ef-search",
            "64",
            "--use-vector-index",
            "--query-mode",
            "quantized_rerank",
            "--quantized-codec",
            "scalar_u8",
            "--quantized-index-name",
            "embedding.scalar_u8.fast",
            "--quantized-rerank-candidates",
            "32",
            "--query-embedding-encoding",
            "f32_le_b64",
            "--document-embedding-encoding",
            "f32_le_b64",
            "--stats-mode",
            "production",
            "--skip-load",
            "--skip-search-serial",
            "--skip-search-concurrent",
            "--dry-run",
        ],
    )

    assert result.exit_code == 0, result.output
    assert captured["args"][0] == DB.TreeDB
    assert captured["args"][1].base_url == "http://127.0.0.1:7120"
    assert captured["args"][1].document_embedding_encoding == "f32_le_b64"
    assert captured["args"][1].query_embedding_encoding == "f32_le_b64"
    assert captured["args"][1].stats_mode == "production"
    assert captured["args"][2].use_vector_index is True
    assert captured["args"][2].query_mode == "quantized_rerank"
    assert captured["args"][2].quantized_rerank_candidates == 32


def test_treedb_dense_init_omits_vector_index_options(monkeypatch: MonkeyPatch) -> None:
    calls = []

    class FakeClient:
        def __init__(self, base_url, timeout=30.0):
            self.base_url = base_url
            self.timeout = timeout

        def create_index(self, index_name, dimension, metric, vector_index_options=None):
            calls.append((index_name, dimension, metric, vector_index_options))

    fake_module = ModuleType("treedb_client")
    fake_module.TreeDBClient = FakeClient
    monkeypatch.setitem(sys.modules, "treedb_client", fake_module)

    from vectordb_bench.backend.clients.treedb.treedb import TreeDB

    TreeDB(
        dim=3,
        db_config={"base_url": "http://127.0.0.1:7120", "index_name": "bench", "timeout": 5},
        db_case_config=TreeDBHNSWConfig(use_vector_index=False),
    )

    assert calls == [("bench", 3, "cosine", None)]


def test_treedb_vector_index_init_passes_options(monkeypatch: MonkeyPatch) -> None:
    calls = []

    class FakeClient:
        def __init__(self, base_url, timeout=30.0):
            self.base_url = base_url
            self.timeout = timeout

        def create_index(self, index_name, dimension, metric, vector_index_options=None):
            calls.append((index_name, dimension, metric, vector_index_options))

    fake_module = ModuleType("treedb_client")
    fake_module.TreeDBClient = FakeClient
    monkeypatch.setitem(sys.modules, "treedb_client", fake_module)

    from vectordb_bench.backend.clients.treedb.treedb import TreeDB

    TreeDB(
        dim=3,
        db_config={"base_url": "http://127.0.0.1:7120", "index_name": "bench", "timeout": 5},
        db_case_config=TreeDBHNSWConfig(
            use_vector_index=True,
            query_mode="quantized_rerank",
            quantized_codec="scalar_u8",
            quantized_index_name="embedding.scalar_u8.fast",
            quantized_rerank_candidates=32,
        ),
    )

    assert calls[0][0:3] == ("bench", 3, "cosine")
    assert calls[0][3]["strategy"] == "column_graph"
    assert calls[0][3]["quantized_indexes"] == [
        {"name": "embedding.scalar_u8.fast", "codec": "scalar_u8", "version": 1}
    ]


def test_treedb_vector_index_inserts_defer_rebuild_until_optimize(monkeypatch: MonkeyPatch) -> None:
    calls = []

    class FakeDocument:
        def __init__(self, id, embedding):
            self.id = id
            self.embedding = embedding

    class FakeClient:
        def __init__(self, base_url, timeout=30.0):
            self.base_url = base_url
            self.timeout = timeout

        def create_index(self, *args, **kwargs):
            pass

        def upsert_documents(self, index_name, documents, *, defer_vector_index_rebuild=False):
            calls.append((index_name, documents, defer_vector_index_rebuild))
            return SimpleNamespace(upserted=len(documents))

    fake_module = ModuleType("treedb_client")
    fake_module.Document = FakeDocument
    fake_module.TreeDBClient = FakeClient
    monkeypatch.setitem(sys.modules, "treedb_client", fake_module)

    from vectordb_bench.backend.clients.treedb.treedb import TreeDB

    db = TreeDB(
        dim=2,
        db_config={"base_url": "http://127.0.0.1:7120", "index_name": "bench", "timeout": 5},
        db_case_config=TreeDBColumnGraphExactConfig(),
    )

    count, err = db.insert_embeddings([[1.0, 0.0]], [7])

    assert err is None
    assert count == 1
    assert calls[0][0] == "bench"
    assert calls[0][2] is True


def test_treedb_lifecycle_sidecar_is_absent_by_default(monkeypatch: MonkeyPatch, tmp_path) -> None:
    sidecar = tmp_path / "lifecycle.jsonl"
    monkeypatch.delenv("TREEDB_LIFECYCLE_SIDECAR", raising=False)

    class FakeDocument:
        def __init__(self, id, embedding):
            self.id = id
            self.embedding = embedding

    class FakeClient:
        def __init__(self, base_url, timeout=30.0):
            pass

        def reset_index(self, *args, **kwargs):
            return SimpleNamespace(index_name="bench", generation=1)

        def upsert_documents(self, index_name, documents, *, defer_vector_index_rebuild=False):
            return SimpleNamespace(upserted=len(documents))

        def optimize_index(self, index_name):
            return SimpleNamespace(root_id=9)

    fake_module = ModuleType("treedb_client")
    fake_module.Document = FakeDocument
    fake_module.TreeDBClient = FakeClient
    monkeypatch.setitem(sys.modules, "treedb_client", fake_module)

    from vectordb_bench.backend.clients.treedb.treedb import TreeDB

    db = TreeDB(
        dim=2,
        db_config={"base_url": "http://127.0.0.1:7120", "index_name": "bench"},
        db_case_config=TreeDBColumnGraphExactConfig(),
        drop_old=True,
    )
    assert db.insert_embeddings([[1.0, 0.0]], [7]) == (1, None)
    db.optimize()

    assert not sidecar.exists()


def test_treedb_lifecycle_sidecar_preserves_load_and_optimize_boundaries(monkeypatch: MonkeyPatch, tmp_path) -> None:
    sidecar = tmp_path / "lifecycle.jsonl"
    monkeypatch.setenv("TREEDB_LIFECYCLE_SIDECAR", str(sidecar))

    class FakeDocument:
        def __init__(self, id, embedding):
            self.id = id
            self.embedding = embedding

    class FakeResponse:
        def __init__(self, **values):
            self.values = values

        def model_dump(self, *, mode):
            assert mode == "json"
            return self.values

    class FakeClient:
        def __init__(self, base_url, timeout=30.0):
            pass

        def reset_index(self, *args, **kwargs):
            return FakeResponse(index_name="bench", generation=1)

        def upsert_documents(self, index_name, documents, *, defer_vector_index_rebuild=False):
            return SimpleNamespace(upserted=len(documents))

        def optimize_index(self, index_name):
            return FakeResponse(
                index_name=index_name,
                generation=2,
                maintenance={"root_id": 9},
                timing={"cache_prime_seconds": 0.5, "cache_warm_seconds": 0.25},
            )

    fake_module = ModuleType("treedb_client")
    fake_module.Document = FakeDocument
    fake_module.TreeDBClient = FakeClient
    monkeypatch.setitem(sys.modules, "treedb_client", fake_module)

    from vectordb_bench.backend.clients.treedb.treedb import TreeDB

    db = TreeDB(
        dim=2,
        db_config={"base_url": "http://127.0.0.1:7120", "index_name": "bench"},
        db_case_config=TreeDBColumnGraphExactConfig(),
        drop_old=True,
    )
    start = threading.Barrier(2)
    results = []

    def insert(embeddings, metadata):
        start.wait(timeout=2)
        results.append(db.insert_embeddings(embeddings, metadata))

    workers = [
        threading.Thread(target=insert, args=([[1.0, 0.0], [0.0, 1.0]], [7, 8])),
        threading.Thread(target=insert, args=([[0.5, 0.5]], [9])),
    ]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join(timeout=2)
    assert sorted(results) == [(1, None), (2, None)]

    restored = pickle.loads(pickle.dumps(db))  # noqa: S301
    restored.optimize()

    records = [json.loads(line) for line in sidecar.read_text().splitlines()]
    assert [record["event"] for record in records] == [
        "reset",
        "load_start",
        "batch_accepted",
        "batch_accepted",
        "load_end",
        "optimize_start",
        "optimize_end",
    ]
    assert sum(record.get("client_sent", 0) for record in records) == 3
    assert sum(record.get("server_accepted", 0) for record in records) == 3
    assert records[-1]["response"]["maintenance"]["root_id"] == 9
    assert all(isinstance(record["timestamp_ns"], int) for record in records)


def test_treedb_lifecycle_waits_for_exact_diagnostics_at_each_optimize_boundary(
    monkeypatch: MonkeyPatch, tmp_path
) -> None:
    sidecar = tmp_path / "lifecycle.jsonl"
    acknowledgement = tmp_path / "lifecycle-boundary-diagnostics.json"
    monkeypatch.setenv("TREEDB_LIFECYCLE_SIDECAR", str(sidecar))
    monkeypatch.setenv("TREEDB_LIFECYCLE_BOUNDARY_ACK", str(acknowledgement))
    optimize_started = threading.Event()
    optimize_results = []

    class FakeClient:
        def __init__(self, base_url, timeout=30.0):
            pass

        def reset_index(self, *args, **kwargs):
            return SimpleNamespace(index_name="bench", generation=1)

        def optimize_index(self, index_name):
            optimize_started.set()
            time.sleep(0.02)
            return SimpleNamespace(root_id=9)

    fake_module = ModuleType("treedb_client")
    fake_module.Document = object
    fake_module.TreeDBClient = FakeClient
    monkeypatch.setitem(sys.modules, "treedb_client", fake_module)

    from vectordb_bench.backend.clients.treedb.treedb import TreeDB

    db = TreeDB(
        dim=2,
        db_config={"base_url": "http://127.0.0.1:7120", "index_name": "bench"},
        db_case_config=TreeDBColumnGraphExactConfig(),
        drop_old=True,
    )
    started = time.perf_counter()
    worker = threading.Thread(target=lambda: optimize_results.append(db.optimize()))
    worker.start()

    def acknowledge(event: str) -> None:
        deadline = time.monotonic() + 2
        record = None
        while time.monotonic() < deadline:
            if sidecar.exists():
                records = [json.loads(line) for line in sidecar.read_text().splitlines()]
                record = next((item for item in records if item["event"] == event), None)
                if record is not None:
                    break
            time.sleep(0.01)
        assert record is not None
        time.sleep(0.03)
        acknowledgement.write_text(
            json.dumps(
                {
                    "boundary": event,
                    "boundary_timestamp_ns": record["timestamp_ns"],
                    "sample_timestamp_ns": record["timestamp_ns"] + 1,
                }
            )
        )

    acknowledge("load_end")
    assert not optimize_started.is_set()
    acknowledge("optimize_start")
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline and not optimize_started.is_set():
        time.sleep(0.01)
    assert optimize_started.is_set()
    acknowledge("optimize_end")
    worker.join(timeout=2)
    wall_duration = time.perf_counter() - started

    assert not worker.is_alive()
    assert len(optimize_results) == 1
    assert optimize_results[0].duration_seconds == pytest.approx(0.02, abs=0.02)
    assert wall_duration - optimize_results[0].duration_seconds >= 0.06
    assert db.optimize_timeout_allowance == 90.0


def test_treedb_lifecycle_waits_at_real_search_boundaries(monkeypatch: MonkeyPatch, tmp_path) -> None:
    sidecar = tmp_path / "lifecycle.jsonl"
    acknowledgement = tmp_path / "lifecycle-boundary-diagnostics.json"
    monkeypatch.setenv("TREEDB_LIFECYCLE_SIDECAR", str(sidecar))
    monkeypatch.setenv("TREEDB_LIFECYCLE_BOUNDARY_ACK", str(acknowledgement))

    fake_module = ModuleType("treedb_client")
    fake_module.Document = object
    fake_module.TreeDBClient = type(
        "FakeClient",
        (),
        {
            "__init__": lambda self, *args, **kwargs: None,
            "create_index": lambda self, *args, **kwargs: None,
            "close": lambda self: None,
        },
    )
    monkeypatch.setitem(sys.modules, "treedb_client", fake_module)

    from vectordb_bench.backend.clients.treedb.treedb import TreeDB

    db = TreeDB(
        dim=2,
        db_config={"base_url": "http://127.0.0.1:7120", "index_name": "bench"},
        db_case_config=TreeDBColumnGraphExactConfig(),
    )

    for boundary in ("cache_prime", "cache_warm"):
        worker = threading.Thread(target=db.complete_lifecycle_search_phase, args=(boundary,))
        worker.start()
        deadline = time.monotonic() + 2
        record = None
        while time.monotonic() < deadline:
            if not sidecar.exists():
                time.sleep(0.01)
                continue
            records = [json.loads(line) for line in sidecar.read_text().splitlines()]
            record = next((item for item in records if item["event"] == boundary), None)
            if record is not None:
                break
            time.sleep(0.01)
        assert record is not None
        assert worker.is_alive()
        acknowledgement.write_text(
            json.dumps(
                {
                    "boundary": boundary,
                    "boundary_timestamp_ns": record["timestamp_ns"],
                    "sample_timestamp_ns": record["timestamp_ns"] + 1,
                }
            )
        )
        worker.join(timeout=2)
        assert not worker.is_alive()


def test_treedb_lifecycle_ignores_stale_ack_until_current_run_replaces_it(tmp_path) -> None:
    from vectordb_bench.backend.clients.treedb.treedb import _wait_for_boundary_ack

    acknowledgement = tmp_path / "lifecycle-boundary-diagnostics.json"
    acknowledgement.write_text(
        json.dumps(
            {
                "boundary": "load_end",
                "boundary_timestamp_ns": 10,
                "sample_timestamp_ns": 11,
            }
        )
    )
    completed = threading.Event()

    def wait() -> None:
        _wait_for_boundary_ack(str(acknowledgement), "load_end", 20)
        completed.set()

    worker = threading.Thread(target=wait)
    worker.start()
    time.sleep(0.03)
    assert not completed.is_set()

    acknowledgement.write_text(
        json.dumps(
            {
                "boundary": "load_end",
                "boundary_timestamp_ns": 20,
                "sample_timestamp_ns": 21,
            }
        )
    )
    worker.join(timeout=2)

    assert not worker.is_alive()
    assert completed.is_set()


def test_treedb_lifecycle_waits_through_partial_ack_but_persistent_malformed_fails_closed(tmp_path) -> None:
    from vectordb_bench.backend.clients.treedb.treedb import _wait_for_boundary_ack

    acknowledgement = tmp_path / "lifecycle-boundary-diagnostics.json"
    acknowledgement.write_text('{"boundary":')
    completed = threading.Event()

    worker = threading.Thread(
        target=lambda: (
            _wait_for_boundary_ack(str(acknowledgement), "load_end", 20, timeout=1),
            completed.set(),
        )
    )
    worker.start()
    time.sleep(0.03)
    assert not completed.is_set()
    acknowledgement.write_text(
        json.dumps(
            {
                "boundary": "load_end",
                "boundary_timestamp_ns": 20,
                "sample_timestamp_ns": 21,
            }
        )
    )
    worker.join(timeout=2)

    assert not worker.is_alive()
    assert completed.is_set()

    acknowledgement.write_text("{")
    with pytest.raises(TimeoutError, match="remained malformed"):
        _wait_for_boundary_ack(str(acknowledgement), "optimize_start", 30, timeout=0.03)


def test_treedb_lifecycle_failure_after_acceptance_is_not_retried(monkeypatch: MonkeyPatch, tmp_path) -> None:
    monkeypatch.setenv("TREEDB_LIFECYCLE_SIDECAR", str(tmp_path / "lifecycle.jsonl"))
    calls = []

    class FakeDocument:
        def __init__(self, id, embedding):
            self.id = id
            self.embedding = embedding

    class FakeClient:
        def __init__(self, base_url, timeout=30.0):
            pass

        def create_index(self, *args, **kwargs):
            pass

        def upsert_documents(self, index_name, documents, *, defer_vector_index_rebuild=False):
            calls.append([document.id for document in documents])
            return SimpleNamespace(upserted=len(documents))

    fake_module = ModuleType("treedb_client")
    fake_module.Document = FakeDocument
    fake_module.TreeDBClient = FakeClient
    monkeypatch.setitem(sys.modules, "treedb_client", fake_module)

    from vectordb_bench.backend.clients.treedb import treedb as treedb_module

    real_append = treedb_module._append_lifecycle_record

    def fail_after_acceptance(path, event, **values):
        if event == "batch_accepted":
            raise OSError("sidecar full")
        return real_append(path, event, **values)

    monkeypatch.setattr(treedb_module, "_append_lifecycle_record", fail_after_acceptance)
    db = treedb_module.TreeDB(
        dim=2,
        db_config={"base_url": "http://127.0.0.1:7120", "index_name": "bench"},
        db_case_config=TreeDBColumnGraphExactConfig(),
    )
    runner = ConcurrentInsertRunner.__new__(ConcurrentInsertRunner)

    with pytest.raises(RuntimeError, match="after 2 inserted rows"):
        runner._insert_batch_with_retry(db, [[1.0, 0.0], [0.0, 1.0]], [7, 8])

    assert calls == [["7", "8"]]


def test_treedb_lifecycle_sidecar_serializes_nested_client_dataclasses(monkeypatch: MonkeyPatch, tmp_path) -> None:
    from vectordb_bench.backend.clients.treedb.treedb import _append_lifecycle_record, _jsonable_response

    @dataclass
    class Status:
        root_id: int

    @dataclass
    class Response:
        status: Status

    path = tmp_path / "lifecycle.jsonl"
    _append_lifecycle_record(str(path), "optimize_end", response=_jsonable_response(Response(Status(9))))

    assert json.loads(path.read_text())["response"] == {"status": {"root_id": 9}}


def test_treedb_compact_document_insert_sends_exact_f32le_bytes(monkeypatch: MonkeyPatch) -> None:
    import numpy as np

    calls = []

    class NumericDocument:
        def __init__(self, *args, **kwargs):
            raise AssertionError("compact mode used the numeric Document path")

    class FakeClient:
        def __init__(self, base_url, timeout=30.0):
            pass

        def create_index(self, *args, **kwargs):
            pass

        def close(self):
            pass

        def upsert_documents(self, index_name, documents, *, defer_vector_index_rebuild=False):
            calls.append((index_name, documents, defer_vector_index_rebuild))
            return SimpleNamespace(upserted=len(documents))

    fake_module = ModuleType("treedb_client")
    fake_module.Document = NumericDocument
    fake_module.TreeDBClient = FakeClient
    monkeypatch.setitem(sys.modules, "treedb_client", fake_module)

    from vectordb_bench.backend.clients.treedb.treedb import TreeDB

    db = TreeDB(
        dim=3,
        db_config={
            "base_url": "http://127.0.0.1:7120",
            "index_name": "bench",
            "document_embedding_encoding": "f32_le_b64",
        },
        db_case_config=TreeDBColumnGraphExactConfig(),
    )
    assert pickle.loads(pickle.dumps(db)).accepts_numpy_embeddings is True  # noqa: S301
    embeddings = [np.arange(6, dtype=np.float64)[::2], np.array([1, -2, 3], dtype=np.int16)]

    count, err = db.insert_embeddings(embeddings, [7, 9])

    assert err is None
    assert count == 2
    assert calls[0][0] == "bench"
    assert calls[0][2] is True
    assert [document["id"] for document in calls[0][1]] == ["7", "9"]
    assert all("embedding" not in document for document in calls[0][1])
    for source, document in zip(embeddings, calls[0][1], strict=True):
        assert base64.b64decode(document["embedding_f32_le_b64"]) == np.asarray(source, dtype="<f4").tobytes()


@pytest.mark.parametrize("encoding", ["json", "f32_le_b64"])
@pytest.mark.parametrize("embeddings,metadata", [([[0.1], [0.2]], [1]), ([[0.1]], [1, 2])])
def test_treedb_insert_rejects_mismatched_embedding_metadata_counts(
    monkeypatch: MonkeyPatch, encoding, embeddings, metadata
) -> None:
    class FakeClient:
        def __init__(self, base_url, timeout=30.0):
            pass

        def create_index(self, *args, **kwargs):
            pass

        def close(self):
            pass

        def upsert_documents(self, *args, **kwargs):
            raise AssertionError("mismatched batch reached the client")

    fake_module = ModuleType("treedb_client")
    fake_module.Document = lambda **kwargs: kwargs
    fake_module.TreeDBClient = FakeClient
    monkeypatch.setitem(sys.modules, "treedb_client", fake_module)

    from vectordb_bench.backend.clients.treedb.treedb import TreeDB

    db = TreeDB(
        dim=1,
        db_config={
            "base_url": "http://127.0.0.1:7120",
            "index_name": "bench",
            "document_embedding_encoding": encoding,
        },
        db_case_config=TreeDBColumnGraphExactConfig(),
    )

    count, err = db.insert_embeddings(embeddings, metadata)

    assert count == 0
    assert isinstance(err, ValueError)


def test_treedb_rejects_unknown_document_embedding_encoding() -> None:
    from vectordb_bench.backend.clients.treedb.treedb import TreeDB

    with pytest.raises(ValueError, match="document_embedding_encoding"):
        TreeDB(
            dim=2,
            db_config={"base_url": "http://127.0.0.1:7120", "document_embedding_encoding": "raw"},
            db_case_config=TreeDBColumnGraphExactConfig(),
        )


def test_treedb_named_exact_cli_uses_vector_index_guards(monkeypatch: MonkeyPatch) -> None:
    captured = {}

    def fake_run(*args, **kwargs):
        captured["args"] = args
        captured.update(kwargs)

    monkeypatch.setattr(treedb_cli, "run", fake_run)

    result = CliRunner().invoke(
        treedb_cli.TreeDBColumnGraphExact,
        [
            "--base-url",
            "http://127.0.0.1:7120",
            "--index-name",
            "bench_exact",
            "--m",
            "16",
            "--ef-construction",
            "128",
            "--ef-search",
            "64",
            "--query-embedding-encoding",
            "f32_le",
            "--skip-load",
            "--skip-search-serial",
            "--skip-search-concurrent",
            "--dry-run",
        ],
    )

    assert result.exit_code == 0, result.output
    assert captured["args"][1].query_embedding_encoding == "f32_le"
    case = captured["args"][2]
    assert case.use_vector_index is True
    assert case.query_mode == "exact"
    assert case.require_vector_index_guards is True
    assert case.quantized_index_name == ""


def test_treedb_named_scalar_u8_cli_defaults_to_rerank32(monkeypatch: MonkeyPatch) -> None:
    captured = {}

    def fake_run(*args, **kwargs):
        captured["args"] = args
        captured.update(kwargs)

    monkeypatch.setattr(treedb_cli, "run", fake_run)

    result = CliRunner().invoke(
        treedb_cli.TreeDBScalarU8Rerank,
        [
            "--base-url",
            "http://127.0.0.1:7120",
            "--index-name",
            "bench_scalar",
            "--m",
            "16",
            "--ef-construction",
            "128",
            "--ef-search",
            "64",
            "--skip-load",
            "--skip-search-serial",
            "--skip-search-concurrent",
            "--dry-run",
        ],
    )

    assert result.exit_code == 0, result.output
    case = captured["args"][2]
    assert case.use_vector_index is True
    assert case.query_mode == "quantized_rerank"
    assert case.quantized_codec == "scalar_u8"
    assert case.quantized_index_name == "embedding.scalar_u8.fast"
    assert case.quantized_rerank_candidates == 32


def test_treedb_named_rabitq_cli_is_experimental(monkeypatch: MonkeyPatch) -> None:
    captured = {}

    def fake_run(*args, **kwargs):
        captured["args"] = args
        captured.update(kwargs)

    monkeypatch.setattr(treedb_cli, "run", fake_run)

    result = CliRunner().invoke(
        treedb_cli.TreeDBRaBitQ1BitExperimental,
        [
            "--base-url",
            "http://127.0.0.1:7120",
            "--m",
            "16",
            "--ef-construction",
            "128",
            "--ef-search",
            "64",
            "--skip-load",
            "--skip-search-serial",
            "--skip-search-concurrent",
            "--dry-run",
        ],
    )

    assert result.exit_code == 0, result.output
    case = captured["args"][2]
    assert case.experimental is True
    assert case.quantized_codec == "rabitq_1bit"
    assert case.query_mode == "quantized_only"


class _FakeTreeDBClient:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def search_vector_index(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return self.response


def test_treedb_worker_lifecycle_closes_client() -> None:
    from vectordb_bench.backend.clients.treedb.treedb import TreeDB

    client = SimpleNamespace(close=lambda: setattr(client, "closed", True), closed=False)
    db = object.__new__(TreeDB)
    db._client = None
    db._new_client = lambda: client

    with db.init():
        assert db.client is client

    assert client.closed is True
    assert db._client is None


def test_treedb_cleanup_preserves_primary_errors(monkeypatch: MonkeyPatch) -> None:
    class FailingClient:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def create_index(self, *args, **kwargs) -> None:
            msg = "create failed"
            raise RuntimeError(msg)

        def close(self) -> None:
            msg = "close failed"
            raise RuntimeError(msg)

    fake_module = ModuleType("treedb_client")
    fake_module.TreeDBClient = FailingClient
    monkeypatch.setitem(sys.modules, "treedb_client", fake_module)

    from vectordb_bench.backend.clients.treedb.treedb import TreeDB

    with pytest.raises(RuntimeError, match="create failed"):
        TreeDB(
            dim=2,
            db_config={"base_url": "http://127.0.0.1:7120"},
            db_case_config=TreeDBHNSWConfig(),
        )

    db = object.__new__(TreeDB)
    db._client = None
    db._new_client = FailingClient
    with pytest.raises(RuntimeError, match="worker failed"):
        with db.init():
            msg = "worker failed"
            raise RuntimeError(msg)
    assert db._client is None


def _tree_db_for_response(search_param: dict, response) -> "TreeDB":
    from vectordb_bench.backend.clients.treedb.treedb import TreeDB

    db = object.__new__(TreeDB)
    db.index_name = "bench"
    db._client = _FakeTreeDBClient(response)
    db._new_client = lambda _timeout=None: db._client
    db.query_embedding_encoding = "json"
    db.stats_mode = "full_diagnostics"
    db.response_format = "full"
    db._search_param = search_param
    return db


def _result_response(**overrides):
    data = {
        "results": [SimpleNamespace(id="7")],
        "query_mode": "exact",
        "quantized_index_name": "",
        "no_documents": True,
        "stats": {"documents_fetched": 0, "search_route_hnsw_search_pack": 1},
        "diagnostics": {"route": "exact_hnsw_search_pack_v1", "fallback_reason": "none"},
    }
    data.update(overrides)
    return SimpleNamespace(**data)


def test_treedb_exact_vector_index_response_guard_allows_exact_route() -> None:
    db = _tree_db_for_response(
        {"use_vector_index": True, "query_mode": "exact", "ef_search": 64, "require_vector_index_guards": True},
        _result_response(),
    )

    assert db.search_embedding([1.0, 0.0], 1) == [7]


def test_treedb_live_ann_preflight_requires_live_mutation_counters() -> None:
    db = _tree_db_for_response(
        {"use_vector_index": True, "query_mode": "exact", "ef_search": 64, "require_vector_index_guards": True},
        _result_response(),
    )
    db.dim = 2
    db.document_embedding_encoding = "f32_le_b64"
    db.live_ann_visibility_timeout = 0.01
    db.live_ann_visibility_poll_interval = 0
    db._client.upsert_documents = lambda *args, **kwargs: SimpleNamespace(upserted=1)
    db._client.delete_documents = lambda *args, **kwargs: SimpleNamespace(deleted=1)

    with pytest.raises(RuntimeError, match="live mutation counters"):
        db.preflight_live_ann()


def test_treedb_live_ann_visibility_sleep_does_not_overshoot_deadline(monkeypatch: MonkeyPatch) -> None:
    db = _tree_db_for_response(
        {"use_vector_index": True, "query_mode": "exact", "ef_search": 64, "require_vector_index_guards": True},
        _result_response(),
    )
    db.live_ann_visibility_timeout = 1
    db.live_ann_visibility_poll_interval = 10
    db._client.search_vector_index = lambda *_args, **_kwargs: _result_response(
        results=[],
        diagnostics={
            "route": "exact_hnsw_search_pack_v1",
            "fallback_reason": "none",
            "live_ann": {"enabled": True, "exact_fallbacks": 0, "full_rebuilds": 0},
        },
    )
    clock = iter([0.0, 0.1, 0.2, 1.0])
    sleeps = []
    monkeypatch.setattr("vectordb_bench.backend.clients.treedb.treedb.time.perf_counter", lambda: next(clock))
    monkeypatch.setattr("vectordb_bench.backend.clients.treedb.treedb.time.sleep", sleeps.append)

    with pytest.raises(RuntimeError, match="visibility deadline"):
        db._wait_for_live_ann("probe", [1.0, 0.0], present=True, phase="insert")
    assert sleeps == [0.8]


@pytest.mark.parametrize(
    ("strategy", "configured_stats_mode", "expected_stats_mode"),
    [("column_graph", "production", "full_diagnostics"), ("native_runtime", "production", "production")],
)
def test_treedb_live_ann_probe_uses_supported_diagnostics_transport(
    strategy: str, configured_stats_mode: str, expected_stats_mode: str
) -> None:
    probe_id = "__vectordbbench_live_ann_probe__"
    response = _result_response(
        results=[SimpleNamespace(id=probe_id)],
        diagnostics={
            "route": "native_runtime" if strategy == "native_runtime" else "exact_hnsw_search_pack_v1",
            "fallback_reason": "none",
            "live_ann": {"enabled": True, "exact_fallbacks": 0, "full_rebuilds": 0},
        },
    )
    db = _tree_db_for_response(
        {"use_vector_index": True, "query_mode": "exact", "ef_search": 64},
        response,
    )
    db.db_case_config = SimpleNamespace(strategy=strategy)
    db.stats_mode = configured_stats_mode
    db.response_format = "ids"
    db.live_ann_visibility_timeout = 1
    db.live_ann_visibility_poll_interval = 0

    db._wait_for_live_ann(probe_id, [1.0, 0.0], present=True, phase="insert")

    options = db._client.calls[0][1]
    assert options["stats_mode"] == expected_stats_mode
    assert "response_format" not in options


def test_treedb_live_ann_probe_rejects_unselected_native_route() -> None:
    db = _tree_db_for_response(
        {"use_vector_index": True, "query_mode": "exact", "ef_search": 64},
        _result_response(),
    )
    db.db_case_config = SimpleNamespace(strategy="native_runtime")
    response = _result_response(
        diagnostics={
            "route": "exact_hnsw_search_pack_v1",
            "fallback_reason": "none",
            "live_ann": {"enabled": True, "exact_fallbacks": 0, "full_rebuilds": 0},
        },
    )

    with pytest.raises(RuntimeError, match="native_runtime ANN route"):
        db._validate_live_ann_response(response)


def test_treedb_live_ann_probe_rejects_missing_route_identity() -> None:
    db = _tree_db_for_response(
        {"use_vector_index": True, "query_mode": "exact", "ef_search": 64},
        _result_response(),
    )
    response = _result_response(
        diagnostics={
            "fallback_reason": "none",
            "live_ann": {"enabled": True, "exact_fallbacks": 0, "full_rebuilds": 0},
        },
    )

    with pytest.raises(RuntimeError, match="selected-route identity"):
        db._validate_live_ann_response(response)


def test_treedb_live_ann_probe_rejects_response_after_deadline(monkeypatch: MonkeyPatch) -> None:
    probe_id = "__vectordbbench_live_ann_probe__"
    response = _result_response(
        results=[SimpleNamespace(id=probe_id)],
        diagnostics={
            "route": "exact_hnsw_search_pack_v1",
            "fallback_reason": "none",
            "live_ann": {"enabled": True, "exact_fallbacks": 0, "full_rebuilds": 0},
        },
    )
    db = _tree_db_for_response({"use_vector_index": True, "query_mode": "exact"}, response)
    db.live_ann_visibility_timeout = 1
    db.live_ann_visibility_poll_interval = 0
    request_timeouts = []
    db._new_client = lambda timeout=None: request_timeouts.append(timeout) or db._client
    clock = iter([0.0, 0.1, 1.1])
    monkeypatch.setattr("vectordb_bench.backend.clients.treedb.treedb.time.perf_counter", lambda: next(clock))

    with pytest.raises(RuntimeError, match="visibility deadline"):
        db._wait_for_live_ann(probe_id, [1.0, 0.0], present=True, phase="insert")
    assert request_timeouts == [pytest.approx(0.9)]


@pytest.mark.parametrize(
    ("search_param", "overrides", "match"),
    [
        ({"query_mode": "exact"}, {"query_mode": "quantized_only"}, "query_mode mismatch"),
        ({"query_mode": "exact"}, {"no_documents": False}, "no-document route"),
        ({"query_mode": "exact"}, {"stats": {"documents_fetched": 1}}, "fetched/materialized documents"),
        (
            {"query_mode": "quantized_only", "quantized_index_name": "expected"},
            {"query_mode": "quantized_only", "quantized_index_name": "other"},
            "quantized_index_name mismatch",
        ),
    ],
)
def test_treedb_live_ann_preflight_rejects_non_live_transport_proof(
    search_param: dict[str, str], overrides: dict[str, Any], match: str
) -> None:
    db = _tree_db_for_response({"use_vector_index": True, **search_param}, _result_response())
    response = _result_response(
        diagnostics={
            "route": "exact_hnsw_search_pack_v1",
            "fallback_reason": "none",
            "live_ann": {"enabled": True, "exact_fallbacks": 0, "full_rebuilds": 0},
        },
        **overrides,
    )

    with pytest.raises(RuntimeError, match=match):
        db._validate_live_ann_response(response)


def test_treedb_live_ann_preflight_proves_insert_update_and_delete() -> None:
    probe = SimpleNamespace(id="__vectordbbench_live_ann_probe__")
    anchor = SimpleNamespace(id="__vectordbbench_live_ann_anchor__")
    live = {"enabled": True, "exact_fallbacks": 0, "full_rebuilds": 0}
    responses = [
        _result_response(
            results=[anchor],
            diagnostics={
                "route": "exact_hnsw_search_pack_v1",
                "fallback_reason": "none",
                "live_ann": live,
            },
        ),
        _result_response(
            results=[probe],
            diagnostics={
                "route": "exact_hnsw_search_pack_v1",
                "fallback_reason": "none",
                "live_ann": live,
            },
        ),
        _result_response(
            results=[probe],
            diagnostics={
                "route": "exact_hnsw_search_pack_v1",
                "fallback_reason": "none",
                "live_ann": live,
            },
        ),
        _result_response(
            results=[anchor],
            diagnostics={"route": "exact_hnsw_search_pack_v1", "fallback_reason": "none", "live_ann": live},
        ),
        _result_response(
            results=[],
            diagnostics={"route": "exact_hnsw_search_pack_v1", "fallback_reason": "none", "live_ann": live},
        ),
        _result_response(
            results=[],
            diagnostics={"route": "exact_hnsw_search_pack_v1", "fallback_reason": "none", "live_ann": live},
        ),
    ]
    db = _tree_db_for_response(
        {"use_vector_index": True, "query_mode": "exact", "ef_search": 64, "require_vector_index_guards": True},
        responses[0],
    )
    db.dim = 2
    db.document_embedding_encoding = "f32_le_b64"
    db.live_ann_visibility_timeout = 0.01
    db.live_ann_visibility_poll_interval = 0
    db._client.upsert_documents = lambda *args, **kwargs: SimpleNamespace(upserted=1)
    db._client.delete_documents = lambda *args, **kwargs: SimpleNamespace(deleted=1)
    db._client.search_vector_index = lambda *args, **kwargs: responses.pop(0)

    db.preflight_live_ann()


def test_treedb_live_ann_preflight_fails_when_anchor_cleanup_is_not_visible() -> None:
    anchor = SimpleNamespace(id="__vectordbbench_live_ann_anchor__")
    probe = SimpleNamespace(id="__vectordbbench_live_ann_probe__")
    live = {"enabled": True, "exact_fallbacks": 0, "full_rebuilds": 0}
    responses = [
        _result_response(results=[item], diagnostics={"route": "exact_hnsw_search_pack_v1", "live_ann": live})
        for item in (anchor, probe, probe, anchor)
    ]
    responses.extend(
        [
            _result_response(results=[], diagnostics={"route": "exact_hnsw_search_pack_v1", "live_ann": live}),
            _result_response(results=[anchor], diagnostics={"route": "exact_hnsw_search_pack_v1", "live_ann": live}),
        ]
    )
    db = _tree_db_for_response({"use_vector_index": True, "query_mode": "exact"}, responses[0])
    db.dim = 2
    db.document_embedding_encoding = "f32_le_b64"
    db.live_ann_visibility_timeout = 0.01
    db.live_ann_visibility_poll_interval = 0
    db._client.upsert_documents = lambda *args, **kwargs: SimpleNamespace(upserted=1)
    db._client.delete_documents = lambda *args, **kwargs: SimpleNamespace(deleted=1)
    db._client.search_vector_index = lambda *args, **kwargs: responses.pop(0) if len(responses) > 1 else responses[0]

    with pytest.raises(RuntimeError, match="cleanup.*not absent"):
        db.preflight_live_ann()


def test_treedb_live_ann_preflight_propagates_cleanup_delete_failure() -> None:
    anchor = SimpleNamespace(id="__vectordbbench_live_ann_anchor__")
    probe = SimpleNamespace(id="__vectordbbench_live_ann_probe__")
    live = {"enabled": True, "exact_fallbacks": 0, "full_rebuilds": 0}
    responses = [
        _result_response(results=results, diagnostics={"route": "exact_hnsw_search_pack_v1", "live_ann": live})
        for results in ([anchor], [probe], [probe], [anchor], [])
    ]
    db = _tree_db_for_response({"use_vector_index": True, "query_mode": "exact"}, responses[0])
    db.dim = 2
    db.document_embedding_encoding = "f32_le_b64"
    db.live_ann_visibility_timeout = 1
    db.live_ann_visibility_poll_interval = 0
    db._client.upsert_documents = lambda *args, **kwargs: SimpleNamespace(upserted=1)
    deletes = []

    def delete(*args, **kwargs):
        deletes.append(args[1])
        if len(deletes) == 2:
            raise RuntimeError("cleanup delete failed")

    db._client.delete_documents = delete
    db._client.search_vector_index = lambda *args, **kwargs: responses.pop(0)

    with pytest.raises(RuntimeError, match="cleanup.*cleanup delete failed"):
        db.preflight_live_ann()


def test_treedb_live_ann_preflight_preserves_primary_failure_over_cleanup_failure() -> None:
    db = _tree_db_for_response({"use_vector_index": True, "query_mode": "exact"}, _result_response())
    db.dim = 2
    db.document_embedding_encoding = "f32_le_b64"
    db.live_ann_visibility_timeout = 1
    db.live_ann_visibility_poll_interval = 0
    db._client.upsert_documents = lambda *args, **kwargs: SimpleNamespace(upserted=1)
    db._client.search_vector_index = lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("search failed"))
    db._client.delete_documents = lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("cleanup delete failed"))

    with pytest.raises(RuntimeError, match="search failed"):
        db.preflight_live_ann()


def test_treedb_live_ann_preflight_rejects_update_visible_at_old_vector() -> None:
    probe = SimpleNamespace(id="__vectordbbench_live_ann_probe__")
    anchor = SimpleNamespace(id="__vectordbbench_live_ann_anchor__")
    live = {"enabled": True, "exact_fallbacks": 0, "full_rebuilds": 0}
    responses = [
        _result_response(
            results=[anchor],
            diagnostics={"route": "exact_hnsw_search_pack_v1", "fallback_reason": "none", "live_ann": live},
        ),
        _result_response(
            results=[probe],
            diagnostics={"route": "exact_hnsw_search_pack_v1", "fallback_reason": "none", "live_ann": live},
        ),
        _result_response(
            results=[probe],
            diagnostics={"route": "exact_hnsw_search_pack_v1", "fallback_reason": "none", "live_ann": live},
        ),
        _result_response(
            results=[probe],
            diagnostics={"route": "exact_hnsw_search_pack_v1", "fallback_reason": "none", "live_ann": live},
        ),
    ]
    db = _tree_db_for_response(
        {"use_vector_index": True, "query_mode": "exact", "ef_search": 64, "require_vector_index_guards": True},
        responses[0],
    )
    db.dim = 2
    db.document_embedding_encoding = "f32_le_b64"
    db.live_ann_visibility_timeout = 0.01
    db.live_ann_visibility_poll_interval = 0
    db._client.upsert_documents = lambda *args, **kwargs: SimpleNamespace(upserted=1)
    deleted = []
    db._client.delete_documents = lambda *args, **kwargs: deleted.append(args[1])
    db._client.search_vector_index = lambda *args, **kwargs: responses.pop(0) if len(responses) > 1 else responses[0]

    with pytest.raises(RuntimeError, match="update replacement.*not visible"):
        db.preflight_live_ann()
    assert deleted == [["__vectordbbench_live_ann_probe__", "__vectordbbench_live_ann_anchor__"]]


def test_treedb_live_ann_preflight_rejects_canonical_fallback_reason() -> None:
    db = _tree_db_for_response(
        {"use_vector_index": True, "query_mode": "exact", "ef_search": 64, "require_vector_index_guards": True},
        _result_response(),
    )
    db.dim = 2
    db.document_embedding_encoding = "f32_le_b64"
    db.live_ann_visibility_timeout = 1
    db.live_ann_visibility_poll_interval = 0
    db._client.upsert_documents = lambda *args, **kwargs: SimpleNamespace(upserted=1)
    db._client.delete_documents = lambda *args, **kwargs: SimpleNamespace(deleted=1)
    db._client.search_vector_index = lambda *args, **kwargs: _result_response(
        results=[SimpleNamespace(id="__vectordbbench_live_ann_anchor__")],
        diagnostics={
            "route": "exact_hnsw_search_pack_v1",
            "fallback_reason": "exact_fallback",
            "live_ann": {"enabled": True, "exact_fallbacks": 0, "full_rebuilds": 0},
        },
    )

    with pytest.raises(RuntimeError, match="fallback_reason='exact_fallback'"):
        db.preflight_live_ann()


def test_treedb_live_ann_preflight_cleans_up_after_search_failure() -> None:
    db = _tree_db_for_response(
        {"use_vector_index": True, "query_mode": "exact", "ef_search": 64, "require_vector_index_guards": True},
        _result_response(),
    )
    db.dim = 2
    db.document_embedding_encoding = "f32_le_b64"
    db.live_ann_visibility_timeout = 0.01
    db.live_ann_visibility_poll_interval = 0
    db._client.upsert_documents = lambda *args, **kwargs: SimpleNamespace(upserted=1)
    deleted = []
    db._client.delete_documents = lambda *args, **kwargs: deleted.append(args[1])
    db._client.search_vector_index = lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("search failed"))

    with pytest.raises(RuntimeError, match="search failed"):
        db.preflight_live_ann()
    assert deleted == [["__vectordbbench_live_ann_probe__", "__vectordbbench_live_ann_anchor__"]]


def test_treedb_live_ann_preflight_requires_two_dimensions() -> None:
    db = _tree_db_for_response(
        {"use_vector_index": True, "query_mode": "exact", "ef_search": 64, "require_vector_index_guards": True},
        _result_response(),
    )
    db.dim = 1

    with pytest.raises(RuntimeError, match="dim >= 2"):
        db.preflight_live_ann()


def test_treedb_live_ann_preflight_names_unsupported_selected_route() -> None:
    db = _tree_db_for_response(
        {"use_vector_index": True, "query_mode": "exact", "ef_search": 64, "require_vector_index_guards": True},
        _result_response(),
    )
    db.dim = 2
    db.document_embedding_encoding = "f32_le_b64"
    db.live_ann_visibility_timeout = 0.01
    db.live_ann_visibility_poll_interval = 0
    db.db_case_config = SimpleNamespace(strategy="native_runtime")
    db._client.upsert_documents = lambda *args, **kwargs: SimpleNamespace(upserted=1)
    db._client.delete_documents = lambda *args, **kwargs: SimpleNamespace(deleted=1)
    db._client.search_vector_index = lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("unsupported"))

    with pytest.raises(RuntimeError, match="strategy=native_runtime.*unsupported"):
        db.preflight_live_ann()


@pytest.mark.parametrize("encoding", ["f32_le_b64", "f32_le"])
def test_treedb_vector_index_search_passes_query_embedding_encoding(encoding: str) -> None:
    db = _tree_db_for_response(
        {"use_vector_index": True, "query_mode": "exact", "ef_search": 64, "require_vector_index_guards": True},
        _result_response(),
    )
    db.query_embedding_encoding = encoding

    assert db.search_embedding([1.0, 0.0], 1) == [7]
    assert db._client.calls[0][1]["query_embedding_encoding"] == encoding


def test_treedb_compact_production_search_returns_ordered_ids() -> None:
    db = _tree_db_for_response(
        {
            "use_vector_index": True,
            "query_mode": "quantized_rerank",
            "ef_search": 64,
            "require_vector_index_guards": False,
        },
        SimpleNamespace(ids=["7", "3"]),
    )
    db.query_embedding_encoding = "f32_le"
    db.stats_mode = "production"
    db.response_format = "ids"

    assert db.search_embedding([1.0, 0.0], 2) == [7, 3]
    assert db._client.calls[0][1]["stats_mode"] == "production"
    assert db._client.calls[0][1]["response_format"] == "ids"


def test_treedb_exact_vector_index_response_guard_rejects_quantized_activity() -> None:
    db = _tree_db_for_response(
        {"use_vector_index": True, "query_mode": "exact", "ef_search": 64, "require_vector_index_guards": True},
        _result_response(
            stats={"documents_fetched": 0, "search_route_hnsw_search_pack": 1, "quantized_score_calls": 1}
        ),
    )

    with pytest.raises(RuntimeError, match="unexpectedly used a quantized score plane"):
        db.search_embedding([1.0, 0.0], 1)


def test_treedb_scalar_u8_rerank_response_guard_requires_bounded_exact_reads() -> None:
    db = _tree_db_for_response(
        {
            "use_vector_index": True,
            "query_mode": "quantized_rerank",
            "ef_search": 64,
            "quantized_index_name": "embedding.scalar_u8.fast",
            "quantized_rerank_candidates": 32,
            "require_vector_index_guards": True,
        },
        _result_response(
            query_mode="quantized_rerank",
            quantized_index_name="embedding.scalar_u8.fast",
            stats={
                "documents_fetched": 0,
                "search_route_quantized_rerank": 1,
                "quantized_scorer_active": 1,
                "quantized_score_calls": 12,
                "quantized_rerank_candidates": 32,
                "quantized_rerank_exact_score_calls": 32,
            },
            diagnostics={"route": "quantized_rerank", "fallback_reason": "none"},
        ),
    )

    assert db.search_embedding([1.0, 0.0], 1) == [7]


def test_treedb_scalar_u8_rerank_response_guard_rejects_excess_exact_reads() -> None:
    db = _tree_db_for_response(
        {
            "use_vector_index": True,
            "query_mode": "quantized_rerank",
            "ef_search": 64,
            "quantized_index_name": "embedding.scalar_u8.fast",
            "quantized_rerank_candidates": 32,
            "require_vector_index_guards": True,
        },
        _result_response(
            query_mode="quantized_rerank",
            quantized_index_name="embedding.scalar_u8.fast",
            stats={
                "documents_fetched": 0,
                "search_route_quantized_rerank": 1,
                "quantized_scorer_active": 1,
                "quantized_score_calls": 12,
                "quantized_rerank_candidates": 64,
                "quantized_rerank_exact_score_calls": 64,
            },
            diagnostics={"route": "quantized_rerank", "fallback_reason": "none"},
        ),
    )

    with pytest.raises(RuntimeError, match="exceeded request"):
        db.search_embedding([1.0, 0.0], 1)


def test_treedb_quantized_only_response_guard_rejects_exact_rerank() -> None:
    db = _tree_db_for_response(
        {
            "use_vector_index": True,
            "query_mode": "quantized_only",
            "ef_search": 64,
            "quantized_index_name": "embedding.rabitq_1bit.experimental",
            "quantized_rerank_candidates": 0,
            "require_vector_index_guards": True,
        },
        _result_response(
            query_mode="quantized_only",
            quantized_index_name="embedding.rabitq_1bit.experimental",
            stats={
                "documents_fetched": 0,
                "search_route_quantized_only": 1,
                "quantized_scorer_active": 1,
                "quantized_score_calls": 12,
                "quantized_rerank_candidates": 4,
                "quantized_rerank_exact_score_calls": 4,
            },
            diagnostics={"route": "quantized_only", "fallback_reason": "none"},
        ),
    )

    with pytest.raises(RuntimeError, match="unexpectedly performed exact rerank reads"):
        db.search_embedding([1.0, 0.0], 1)


def test_treedb_config_shape_rejects_quantized_rerank_without_index() -> None:
    from vectordb_bench.backend.clients.treedb.treedb import TreeDB

    db = object.__new__(TreeDB)
    db.query_embedding_encoding = "json"
    db.stats_mode = "full_diagnostics"
    db.response_format = "full"
    db._metric = "cosine"
    db._search_param = {
        "use_vector_index": True,
        "query_mode": "quantized_rerank",
        "quantized_index_name": "",
        "quantized_rerank_candidates": 32,
    }

    with pytest.raises(ValueError, match="requires quantized_index_name"):
        db._validate_config_shape()

    db._search_param = {"use_vector_index": False}
    for encoding in ("f32_le_b64", "f32_le"):
        db.query_embedding_encoding = encoding
        with pytest.raises(ValueError, match="supported only for the vector-index route"):
            db._validate_config_shape()

    db.query_embedding_encoding = "f32_le"
    db._search_param = {
        "use_vector_index": True,
        "query_mode": "quantized_rerank",
        "quantized_index_name": "embedding.scalar_u8.fast",
        "quantized_rerank_candidates": 32,
    }
    db._validate_config_shape()

    db.query_embedding_encoding = "json"
    db.response_format = "ids"
    db._search_param = {"use_vector_index": False}
    with pytest.raises(ValueError, match="compact IDs responses are supported only for the vector-index route"):
        db._validate_config_shape()

    db.response_format = "full"
    db.stats_mode = "production"
    with pytest.raises(ValueError, match="production stats mode is supported only for the vector-index route"):
        db._validate_config_shape()

    db.stats_mode = "full_diagnostics"
    db.response_format = "ids"
    db._search_param = {
        "use_vector_index": True,
        "query_mode": "quantized_rerank",
        "quantized_index_name": "embedding.scalar_u8.fast",
        "quantized_rerank_candidates": 32,
    }
    db._search_param["require_vector_index_guards"] = True
    with pytest.raises(ValueError, match="separate full-response preflight"):
        db._validate_config_shape()

    db.db_case_config = SimpleNamespace(strategy="native_runtime")
    db.response_format = "full"
    db.stats_mode = "full_diagnostics"
    db._search_param["require_vector_index_guards"] = False
    with pytest.raises(ValueError, match="native_runtime.*stats_mode=production"):
        db._validate_config_shape()

    db.response_format = "full"
    db.stats_mode = "production"
    db._search_param["require_vector_index_guards"] = True
    with pytest.raises(ValueError, match="separate full-response preflight"):
        db._validate_config_shape()


def test_treedb_named_configs_have_expected_modes() -> None:
    assert TreeDBColumnGraphExactConfig().search_param()["query_mode"] == "exact"
    scalar = TreeDBScalarU8RerankConfig()
    assert scalar.search_param()["query_mode"] == "quantized_rerank"
    assert scalar.search_param()["quantized_rerank_candidates"] == 32


@pytest.mark.parametrize(
    ("timeout", "interval", "message"),
    [(0, 0, "visibility_timeout"), (1, -0.1, "poll_interval")],
)
def test_treedb_rejects_invalid_live_ann_probe_timing(timeout: float, interval: float, message: str) -> None:
    from vectordb_bench.backend.clients.treedb.treedb import TreeDB

    db = object.__new__(TreeDB)
    db.query_embedding_encoding = "json"
    db.stats_mode = "full_diagnostics"
    db.response_format = "full"
    db.live_ann_visibility_timeout = timeout
    db.live_ann_visibility_poll_interval = interval
    db._search_param = {"use_vector_index": False}

    with pytest.raises(ValueError, match=message):
        db._validate_config_shape()
