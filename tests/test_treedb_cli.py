import sys
from types import ModuleType

from click.testing import CliRunner
from pytest import MonkeyPatch

from vectordb_bench.backend.clients import DB, IndexType
from vectordb_bench.backend.clients.treedb import cli as treedb_cli
from vectordb_bench.backend.clients.treedb.config import TreeDBConfig, TreeDBHNSWConfig


def test_treedb_config_to_dict_and_case_config_scalar_u8_rerank() -> None:
    config = TreeDBConfig(db_label="local", base_url="http://127.0.0.1:7120", index_name="bench", timeout=5)
    assert config.to_dict() == {"base_url": "http://127.0.0.1:7120", "index_name": "bench", "timeout": 5}

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
            "--skip-load",
            "--skip-search-serial",
            "--skip-search-concurrent",
            "--dry-run",
        ],
    )

    assert result.exit_code == 0, result.output
    assert captured["args"][0] == DB.TreeDB
    assert captured["args"][1].base_url == "http://127.0.0.1:7120"
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
