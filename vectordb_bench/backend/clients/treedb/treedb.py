import base64
import logging
import math
import threading
import time
from contextlib import contextmanager
from typing import Any

import numpy as np

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
_QUERY_EMBEDDING_ENCODINGS = ("json", "f32_le_b64", "f32_le")
_DOCUMENT_EMBEDDING_ENCODINGS = ("json", "f32_le_b64")


class TreeDB(VectorDB):
    supported_filter_types: list[FilterOp] = [FilterOp.NonFilter]
    # ConcurrentInsertRunner gives every worker its own init() scope and this
    # adapter keeps the resulting client in thread-local state.
    thread_safe: bool = True
    worker_owned_clients: bool = True

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
        self.document_embedding_encoding = db_config.get("document_embedding_encoding", "json")
        self.accepts_numpy_embeddings = self.document_embedding_encoding == "f32_le_b64"
        self.query_embedding_encoding = db_config.get("query_embedding_encoding", "json")
        self.stats_mode = db_config.get("stats_mode", "full_diagnostics")
        self.response_format = db_config.get("response_format", "full")
        self.live_ann_visibility_timeout = db_config.get("live_ann_visibility_timeout", 5.0)
        self.live_ann_visibility_poll_interval = db_config.get("live_ann_visibility_poll_interval", 0.05)
        self._client = None
        self._clients = threading.local()
        self._search_param = db_case_config.search_param()
        self._metric = self._parse_metric(db_case_config.metric_type)
        self._vector_index_options = (
            db_case_config.index_param() if self._search_param.get("use_vector_index") else None
        )
        if self.document_embedding_encoding not in _DOCUMENT_EMBEDDING_ENCODINGS:
            msg = f"TreeDB document_embedding_encoding={self.document_embedding_encoding!r} is not supported"
            raise ValueError(msg)
        self._validate_config_shape()

        # Do setup in __init__ with a short-lived client so the object remains
        # pickle-safe for VectorDBBench subprocess runners.
        client = self._new_client()
        try:
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
        finally:
            _close_client(client)

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_client"] = None
        state.pop("_clients", None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._clients = threading.local()

    def _new_client(self):
        from treedb_client import TreeDBClient

        return TreeDBClient(self.base_url, timeout=self.timeout)

    def _thread_clients(self):
        clients = getattr(self, "_clients", None)
        if clients is None:
            clients = threading.local()
            self._clients = clients
        return clients

    @contextmanager
    def init(self):
        client = self._new_client()
        clients = self._thread_clients()
        clients.client = client
        try:
            yield
        finally:
            if getattr(clients, "client", None) is client:
                del clients.client
            _close_client(client)

    @property
    def client(self):
        clients = self._thread_clients()
        client = getattr(clients, "client", None)
        if client is None:
            client = getattr(self, "_client", None)
            if client is None:
                client = self._new_client()
                clients.client = client
        return client

    def insert_embeddings(
        self,
        embeddings: list[list[float]] | np.ndarray,
        metadata: list[int],
        labels_data: list[str] | None = None,
        tenant_labels_data: list[str] | None = None,
        **kwargs: Any,
    ) -> tuple[int, Exception | None]:
        try:
            if self.document_embedding_encoding == "f32_le_b64":
                documents = [
                    {
                        "id": str(meta),
                        "embedding_f32_le_b64": base64.b64encode(np.ascontiguousarray(embedding, dtype="<f4")).decode(
                            "ascii"
                        ),
                    }
                    for meta, embedding in zip(metadata, embeddings, strict=True)
                ]
            else:
                from treedb_client import Document

                documents = [
                    Document(id=str(meta), embedding=[float(value) for value in embedding])
                    for meta, embedding in zip(metadata, embeddings, strict=True)
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

    @property
    def requires_live_ann_preflight(self) -> bool:
        return bool(self._search_param.get("use_vector_index"))

    def preflight_live_ann(self) -> None:
        """Prove one insert, update, and delete reaches the selected ANN route.

        This deliberately does not call optimize: a rebuild makes this a bulk
        benchmark, not a live-ANN one.
        """
        if not self._search_param.get("use_vector_index"):
            raise RuntimeError("TreeDB live-ANN preflight requires the vector-index route")
        if self.dim < 2:
            raise RuntimeError("TreeDB live-ANN preflight requires dim >= 2 for replacement visibility")
        if not callable(getattr(self.client, "delete_documents", None)):
            raise RuntimeError("TreeDB live-ANN preflight requires delete_documents support")

        probe_id = "__vectordbbench_live_ann_probe__"
        anchor_id = "__vectordbbench_live_ann_anchor__"
        positive = [1.0, *([0.0] * (self.dim - 1))]
        negative = [-1.0, *([0.0] * (self.dim - 1))]
        anchor = [0.0, 1.0, *([0.0] * (self.dim - 2))]
        try:
            self._upsert_live_ann_probe(anchor_id, anchor)
            self._wait_for_live_ann(anchor_id, anchor, present=True, phase="anchor insert")
            self._upsert_live_ann_probe(probe_id, positive)
            self._wait_for_live_ann(probe_id, positive, present=True, phase="insert")
            self._upsert_live_ann_probe(probe_id, negative)
            self._wait_for_live_ann(probe_id, negative, present=True, phase="update")
            self._wait_for_live_ann(anchor_id, positive, present=True, phase="update replacement")
            self.client.delete_documents(self.index_name, [probe_id])
            self._wait_for_live_ann(probe_id, negative, present=False, phase="delete")
        except Exception as exc:  # noqa: BLE001
            msg = f"TreeDB live-ANN preflight failed on selected route {self._selected_ann_route()}: {exc}"
            raise RuntimeError(msg) from exc
        finally:
            try:
                self.client.delete_documents(self.index_name, [probe_id, anchor_id])
            except Exception:  # noqa: BLE001
                log.warning("TreeDB live-ANN preflight probe cleanup failed", exc_info=True)

    def _selected_ann_route(self) -> str:
        strategy = getattr(getattr(self, "db_case_config", None), "strategy", "unknown")
        return f"strategy={strategy}, mode={self._search_param.get('query_mode') or 'exact'}"

    def _upsert_live_ann_probe(self, probe_id: str, embedding: list[float]) -> None:
        if self.document_embedding_encoding == "f32_le_b64":
            document = {
                "id": probe_id,
                "embedding_f32_le_b64": base64.b64encode(np.asarray(embedding, dtype="<f4")).decode("ascii"),
            }
        else:
            from treedb_client import Document

            document = Document(id=probe_id, embedding=embedding)
        self.client.upsert_documents(self.index_name, [document], defer_vector_index_rebuild=True)

    def _wait_for_live_ann(self, probe_id: str, query: list[float], *, present: bool, phase: str) -> None:
        deadline = time.perf_counter() + self.live_ann_visibility_timeout
        while True:
            response = self._search_vector_index(query, 1)
            self._validate_live_ann_response(response)
            ids = _response_ids(response)
            if (probe_id in ids) is present:
                return
            if time.perf_counter() >= deadline:
                expectation = "visible" if present else "absent"
                msg = f"TreeDB live-ANN {phase} probe was not {expectation} before the visibility deadline"
                raise RuntimeError(msg)
            time.sleep(self.live_ann_visibility_poll_interval)

    def _validate_live_ann_response(self, response: Any) -> None:
        diagnostics = getattr(response, "diagnostics", {}) or {}
        live = diagnostics.get("live_ann")
        if not isinstance(live, dict) or live.get("enabled") is not True:
            raise RuntimeError("TreeDB live-ANN preflight response is missing live mutation counters")
        if not isinstance(diagnostics.get("route"), str) or not diagnostics["route"]:
            raise RuntimeError("TreeDB live-ANN preflight response is missing selected-route proof")
        fallback_reason = str(diagnostics.get("fallback_reason") or "none")
        if fallback_reason not in ("none", ""):
            raise RuntimeError(f"TreeDB live-ANN preflight reported fallback_reason={fallback_reason!r}")
        if int(live.get("exact_fallbacks", -1)) != 0:
            raise RuntimeError("TreeDB live-ANN preflight used an exact fallback")
        if int(live.get("full_rebuilds", -1)) != 0:
            raise RuntimeError("TreeDB live-ANN preflight performed a full rebuild")

    def search_embedding(self, query: list[float], k: int = 100, **kwargs: Any) -> list[int]:
        if self._search_param.get("use_vector_index"):
            result = self._search_vector_index(query, k)
            if self._search_param.get("require_vector_index_guards", True):
                self._validate_vector_index_response(result)
            if self.response_format == "ids":
                return [int(id_value) for id_value in result.ids]
            return [int(item.id) for item in result.results]
        result = self.client.query_by_embedding(self.index_name, query, k)
        return [int(doc.id) for doc in result.documents]

    def _search_vector_index(self, query: list[float], k: int):
        search_options = {
            "ef_search": self._search_param.get("ef_search") or None,
            "query_mode": self._search_param.get("query_mode") or None,
            "quantized_index_name": self._search_param.get("quantized_index_name") or None,
            "quantized_rerank_candidates": self._search_param.get("quantized_rerank_candidates") or None,
            "query_embedding_encoding": self.query_embedding_encoding,
            "stats_mode": self.stats_mode,
        }
        if self.response_format == "ids":
            search_options["response_format"] = "ids"
        return self.client.search_vector_index(self.index_name, query, k, **search_options)

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
        if not math.isfinite(getattr(self, "live_ann_visibility_timeout", 5.0)) or getattr(
            self, "live_ann_visibility_timeout", 5.0
        ) <= 0:
            raise ValueError("TreeDB live_ann_visibility_timeout must be > 0")
        if not math.isfinite(getattr(self, "live_ann_visibility_poll_interval", 0.05)) or getattr(
            self, "live_ann_visibility_poll_interval", 0.05
        ) < 0:
            raise ValueError("TreeDB live_ann_visibility_poll_interval must be >= 0")
        if self.query_embedding_encoding not in _QUERY_EMBEDDING_ENCODINGS:
            msg = f"TreeDB query_embedding_encoding={self.query_embedding_encoding!r} is not supported"
            raise ValueError(msg)
        if self.stats_mode not in ("full_diagnostics", "production"):
            msg = f"TreeDB stats_mode={self.stats_mode!r} is not supported"
            raise ValueError(msg)
        if self.response_format not in ("full", "ids"):
            msg = f"TreeDB response_format={self.response_format!r} is not supported"
            raise ValueError(msg)
        if not self._search_param.get("use_vector_index"):
            if self.query_embedding_encoding != "json":
                msg = "TreeDB typed/binary query embedding encodings are supported only for the vector-index route"
                raise ValueError(msg)
            if self.stats_mode != "full_diagnostics":
                msg = "TreeDB production stats mode is supported only for the vector-index route"
                raise ValueError(msg)
            if self.response_format == "ids":
                msg = "TreeDB compact IDs responses are supported only for the vector-index route"
                raise ValueError(msg)
            return
        if self._metric != "cosine":
            msg = "TreeDB vector-index benchmark route currently supports only cosine metric"
            raise ValueError(msg)
        mode = self._search_param.get("query_mode") or "exact"
        quantized_name = self._search_param.get("quantized_index_name") or ""
        rerank_candidates = int(self._search_param.get("quantized_rerank_candidates") or 0)
        if (self.stats_mode == "production" or self.response_format == "ids") and self._search_param.get(
            "require_vector_index_guards", True
        ):
            msg = "TreeDB production transport requires --skip-vector-index-guards after a separate full-response preflight"
            raise ValueError(msg)
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
            name: _int_stat(stats, name) for name in _QUANTIZED_ASSET_FAILURE_STATS if _int_stat(stats, name) != 0
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


def _response_ids(response: Any) -> list[str]:
    if hasattr(response, "ids"):
        return [str(value) for value in response.ids]
    return [str(item.id) for item in getattr(response, "results", [])]


def _close_client(client: Any) -> None:
    close = getattr(client, "close", None)
    if close is None:
        return
    try:
        close()
    except Exception as exc:  # noqa: BLE001
        log.warning("Failed to close TreeDB client: %s", exc)
