import logging
from contextlib import contextmanager
from typing import Any

from vectordb_bench.backend.filter import Filter, FilterOp

from ..api import MetricType, VectorDB
from .config import TreeDBHNSWConfig

log = logging.getLogger(__name__)


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
        self._client = None
        self._search_param = db_case_config.search_param()
        self._metric = self._parse_metric(db_case_config.metric_type)
        self._vector_index_options = db_case_config.index_param()

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
            response = self.client.upsert_documents(self.index_name, documents)
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
            )
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
