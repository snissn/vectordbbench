import sys
from types import ModuleType, SimpleNamespace

from click.testing import CliRunner
import pytest
from pytest import MonkeyPatch

from vectordb_bench.backend.clients import DB, IndexType
from vectordb_bench.backend.clients.treedb import cli as treedb_cli
from vectordb_bench.backend.clients.treedb.config import (
    TreeDBColumnGraphExactConfig,
    TreeDBConfig,
    TreeDBHNSWConfig,
    TreeDBScalarU8RerankConfig,
)


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


def _tree_db_for_response(search_param: dict, response):
    from vectordb_bench.backend.clients.treedb.treedb import TreeDB

    db = object.__new__(TreeDB)
    db.index_name = "bench"
    db._client = _FakeTreeDBClient(response)
    db.query_embedding_encoding = "json"
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
    with pytest.raises(ValueError, match="exact vector-index search"):
        db._validate_config_shape()


def test_treedb_named_configs_have_expected_modes() -> None:
    assert TreeDBColumnGraphExactConfig().search_param()["query_mode"] == "exact"
    scalar = TreeDBScalarU8RerankConfig()
    assert scalar.search_param()["query_mode"] == "quantized_rerank"
    assert scalar.search_param()["quantized_rerank_candidates"] == 32
