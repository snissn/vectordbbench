import logging
from contextlib import contextmanager
from typing import Any

from vectordb_bench.backend.filter import Filter, FilterOp

from ..api import MetricType, VectorDB
from .config import TreeDBHNSWConfig

log = logging.getLogger(__name__)

_QUANTIZED_ASSET_FAILURE_STATS = (
    "quantized_asset_unavailable",
    "quantized_asset_missing",
    "quantized_asset_invalid",
    "quantized_asset_stale",
    "quantized_asset_closed",
)


class TreeDB(VectorDB):
    supported_filter_types: list[FilterOp] = [FilterOp.NonFilter]
    thread_safe: bool = False

    def __init__(
        self,
        dim: int,
        db_config: dict,
        db_case_config: TreeDBHNSWConfig,
        collection_name: str = "vector_bench_test",
        drop_old: bool = False,
        **kwargs: Any,
    ):
        self.name = "TreeDB"
        self.dim = dim
        self.db_config = db_config
        self.db_case_config = db_case_config
        self.index_name = db_config.get("index_name") or collection_name
        self.base_url = db_config["base_url"]
        self.timeout = db_config.get("timeout", 30.0)
        self.query_embedding_encoding = db_config.get("query_embedding_encoding", "json")
        self._client = None
        self._search_param = db_case_config.search_param()
        self._metric = self._parse_metric(db_case_config.metric_type)
        self._vector_index_options = (
            db_case_config.index_param() if self._search_param.get("use_vector_index") else None
        )
        self._validate_config_shape()

        # Do setup in __init__ with a short-lived client so the object remains
        # pickle-safe for VectorDBBench subprocess runners.
        client = self._new_client()
        if drop_old:
            client.reset_index(
                self.index_name,
                dimension=self.dim,
                metric=self._metric,
                drop_old=True,
                vector_index_options=self._vector_index_options,
            )
        else:
            client.create_index(
                self.index_name,
                self.dim,
                self._metric,
                vector_index_options=self._vector_index_options,
            )

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_client"] = None
        return state

    def _new_client(self):
        from treedb_client import TreeDBClient

        return TreeDBClient(self.base_url, timeout=self.timeout)

    @contextmanager
    def init(self):
        self._client = self._new_client()
        try:
            yield
        finally:
            self._client = None

    @property
    def client(self):
        if self._client is None:
            self._client = self._new_client()
        return self._client

    def insert_embeddings(
        self,
        embeddings: list[list[float]],
        metadata: list[int],
        labels_data: list[str] | None = None,
        tenant_labels_data: list[str] | None = None,
        **kwargs: Any,
    ) -> tuple[int, Exception | None]:
        try:
            from treedb_client import Document

            documents = [
                Document(id=str(meta), embedding=[float(value) for value in embedding])
                for meta, embedding in zip(metadata, embeddings, strict=False)
            ]
            response = self.client.upsert_documents(
                self.index_name,
                documents,
                defer_vector_index_rebuild=bool(self._search_param.get("use_vector_index")),
            )
            return response.upserted, None
        except Exception as exc:  # noqa: BLE001
            log.warning("Failed to insert embeddings into TreeDB index %s: %s", self.index_name, exc)
            return 0, exc

    def optimize(self, data_size: int | None = None):
        self.client.optimize_index(self.index_name)

    def search_embedding(self, query: list[float], k: int = 100, **kwargs: Any) -> list[int]:
        if self._search_param.get("use_vector_index"):
            result = self.client.search_vector_index(
                self.index_name,
                query,
                k,
                ef_search=self._search_param.get("ef_search") or None,
                query_mode=self._search_param.get("query_mode") or None,
                quantized_index_name=self._search_param.get("quantized_index_name") or None,
                quantized_rerank_candidates=self._search_param.get("quantized_rerank_candidates") or None,
                query_embedding_encoding=self.query_embedding_encoding,
            )
            if self._search_param.get("require_vector_index_guards", True):
                self._validate_vector_index_response(result)
            return [int(item.id) for item in result.results]
        result = self.client.query_by_embedding(self.index_name, query, k)
        return [int(doc.id) for doc in result.documents]

    def prepare_filter(self, filters: Filter):
        if filters.type != FilterOp.NonFilter:
            msg = f"Unsupported TreeDB filter for VectorDBBench: {filters}"
            raise ValueError(msg)

    def _parse_metric(self, metric: MetricType) -> str:
        if metric == MetricType.COSINE:
            return "cosine"
        if metric == MetricType.L2:
            return "l2"
        if metric in (MetricType.IP, MetricType.DP):
            return "inner_product"
        msg = f"Metric type {metric} is not supported by TreeDB document service"
        raise ValueError(msg)

    def _validate_config_shape(self) -> None:
        if self.query_embedding_encoding not in ("json", "f32_le_b64"):
            msg = f"TreeDB query_embedding_encoding={self.query_embedding_encoding!r} is not supported"
            raise ValueError(msg)
        if not self._search_param.get("use_vector_index"):
            if self.query_embedding_encoding != "json":
                msg = "TreeDB f32_le_b64 query embedding encoding is supported only for the vector-index route"
                raise ValueError(msg)
            return
        if self._metric != "cosine":
            msg = "TreeDB vector-index benchmark route currently supports only cosine metric"
            raise ValueError(msg)
        mode = self._search_param.get("query_mode") or "exact"
        quantized_name = self._search_param.get("quantized_index_name") or ""
        rerank_candidates = int(self._search_param.get("quantized_rerank_candidates") or 0)
        if mode == "exact":
            if quantized_name or rerank_candidates:
                msg = "TreeDB exact column_graph row must not set quantized_index_name or quantized_rerank_candidates"
                raise ValueError(msg)
            return
        if mode == "quantized_only":
            if not quantized_name:
                msg = "TreeDB quantized_only row requires quantized_index_name"
                raise ValueError(msg)
            if rerank_candidates:
                msg = "TreeDB quantized_only row must not set quantized_rerank_candidates"
                raise ValueError(msg)
            return
        if mode == "quantized_rerank":
            if not quantized_name:
                msg = "TreeDB quantized_rerank row requires quantized_index_name"
                raise ValueError(msg)
            if rerank_candidates <= 0:
                msg = "TreeDB quantized_rerank row requires quantized_rerank_candidates > 0"
                raise ValueError(msg)
            return
        msg = f"TreeDB vector-index benchmark route does not support query_mode={mode!r}"
        raise ValueError(msg)

    def _validate_vector_index_response(self, response: Any) -> None:
        mode = self._search_param.get("query_mode") or "exact"
        stats = getattr(response, "stats", {}) or {}
        diagnostics = getattr(response, "diagnostics", {}) or {}
        if getattr(response, "no_documents", False) is not True:
            msg = "TreeDB vector-index benchmark response did not use the no-document route"
            raise RuntimeError(msg)
        if getattr(response, "query_mode", "") != mode:
            got = getattr(response, "query_mode", "")
            msg = f"TreeDB vector-index response query_mode mismatch: got {got!r}, want {mode!r}"
            raise RuntimeError(msg)
        if _int_stat(stats, "documents_fetched") != 0 or _int_stat(stats, "document_bytes") != 0:
            msg = "TreeDB vector-index benchmark route fetched/materialized documents"
            raise RuntimeError(msg)
        fallback_reason = str(diagnostics.get("fallback_reason") or "none")
        if fallback_reason not in ("none", ""):
            msg = f"TreeDB vector-index benchmark route reported fallback_reason={fallback_reason!r}"
            raise RuntimeError(msg)

        if mode == "exact":
            self._validate_exact_vector_index_response(stats, diagnostics)
            return
        self._validate_quantized_vector_index_response(response, stats, diagnostics, mode)

    def _validate_exact_vector_index_response(self, stats: dict, diagnostics: dict) -> None:
        route = str(diagnostics.get("route") or "")
        if route != "exact_hnsw_search_pack_v1" and _int_stat(stats, "search_route_hnsw_search_pack") != 1:
            msg = "TreeDB exact column_graph row did not use the exact FP32 hnsw_search_pack route"
            raise RuntimeError(msg)
        if _int_stat(stats, "quantized_score_calls") != 0 or _int_stat(stats, "quantized_scorer_active") != 0:
            msg = "TreeDB exact column_graph row unexpectedly used a quantized score plane"
            raise RuntimeError(msg)

    def _validate_quantized_vector_index_response(
        self,
        response: Any,
        stats: dict,
        diagnostics: dict,
        mode: str,
    ) -> None:
        expected_quantized_name = self._search_param.get("quantized_index_name") or ""
        if getattr(response, "quantized_index_name", "") != expected_quantized_name:
            msg = (
                "TreeDB quantized response quantized_index_name mismatch: "
                f"got {getattr(response, 'quantized_index_name', '')!r}, want {expected_quantized_name!r}"
            )
            raise RuntimeError(msg)
        failed_assets = {
            name: _int_stat(stats, name)
            for name in _QUANTIZED_ASSET_FAILURE_STATS
            if _int_stat(stats, name) != 0
        }
        if failed_assets:
            msg = f"TreeDB quantized score-plane reported unavailable assets: {failed_assets}"
            raise RuntimeError(msg)
        if _int_stat(stats, "quantized_scorer_active") != 1:
            msg = "TreeDB quantized row did not report an active quantized scorer"
            raise RuntimeError(msg)
        if _int_stat(stats, "quantized_score_calls") <= 0:
            msg = "TreeDB quantized row did not report quantized score calls"
            raise RuntimeError(msg)

        route = str(diagnostics.get("route") or "")
        if mode == "quantized_only":
            if route != "quantized_only" and _int_stat(stats, "search_route_quantized_only") != 1:
                msg = "TreeDB quantized_only row did not report the quantized_only route"
                raise RuntimeError(msg)
            if (
                _int_stat(stats, "quantized_rerank_candidates") != 0
                or _int_stat(stats, "quantized_rerank_exact_score_calls") != 0
            ):
                msg = "TreeDB quantized_only row unexpectedly performed exact rerank reads"
                raise RuntimeError(msg)
            if _int_stat(stats, "prepared_score_calls") != 0:
                msg = "TreeDB quantized_only row unexpectedly used exact prepared scoring"
                raise RuntimeError(msg)
            return

        if mode == "quantized_rerank":
            if route != "quantized_rerank" and _int_stat(stats, "search_route_quantized_rerank") != 1:
                msg = "TreeDB quantized_rerank row did not report the quantized_rerank route"
                raise RuntimeError(msg)
            requested = int(self._search_param.get("quantized_rerank_candidates") or 0)
            actual = _int_stat(stats, "quantized_rerank_candidates")
            exact_calls = _int_stat(stats, "quantized_rerank_exact_score_calls")
            if actual <= 0 or exact_calls != actual:
                msg = "TreeDB quantized_rerank row did not report shortlist-bounded exact rerank calls"
                raise RuntimeError(msg)
            if requested > 0 and actual > requested:
                msg = f"TreeDB quantized_rerank exact reads exceeded request: got {actual}, want <= {requested}"
                raise RuntimeError(msg)
            return

        msg = f"unsupported TreeDB vector-index response mode {mode!r}"
        raise RuntimeError(msg)


def _int_stat(stats: dict, name: str) -> int:
    value = stats.get(name, 0)
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return int(value)
    return 0
