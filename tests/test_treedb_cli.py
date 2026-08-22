import pickle
import sys
import threading
from types import ModuleType, SimpleNamespace
from typing import TYPE_CHECKING

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
                [
                    pd.DataFrame({"id": [row_id], "vector": [np.array([float(row_id), 1.0])]})
                    for row_id in range(2)
                ]
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
    runner = ConcurrentInsertRunner(db, SimpleNamespace(), normalize=False, max_workers=4, backend=ExecutorBackend.ASYNC)

    assert runner.load_concurrency == {"requested": 4, "effective": 1}


def test_treedb_config_to_dict_and_case_config_scalar_u8_rerank() -> None:
    config = TreeDBConfig(
        db_label="local",
        base_url="http://127.0.0.1:7120",
        index_name="bench",
        timeout=5,
        query_embedding_encoding="f32_le_b64",
    )
    assert config.to_dict() == {
        "base_url": "http://127.0.0.1:7120",
        "index_name": "bench",
        "timeout": 5,
        "query_embedding_encoding": "f32_le_b64",
        "stats_mode": "full_diagnostics",
        "response_format": "full",
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

    db.response_format = "full"
    db.stats_mode = "production"
    with pytest.raises(ValueError, match="separate full-response preflight"):
        db._validate_config_shape()


def test_treedb_named_configs_have_expected_modes() -> None:
    assert TreeDBColumnGraphExactConfig().search_param()["query_mode"] == "exact"
    scalar = TreeDBScalarU8RerankConfig()
    assert scalar.search_param()["query_mode"] == "quantized_rerank"
    assert scalar.search_param()["quantized_rerank_candidates"] == 32
