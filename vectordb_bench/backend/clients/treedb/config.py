from pydantic import BaseModel

from ..api import DBCaseConfig, DBConfig, IndexType, MetricType


class TreeDBConfig(DBConfig):
    """TreeDB document-service connection configuration."""

    base_url: str
    index_name: str = "vector_bench_test"
    timeout: float = 30.0

    def to_dict(self) -> dict:
        return {
            "base_url": self.base_url,
            "index_name": self.index_name,
            "timeout": self.timeout,
        }


class TreeDBHNSWConfig(BaseModel, DBCaseConfig):
    """TreeDB vector benchmark case configuration.

    The default route is the exact dense document-service path. `use_vector_index`
    opts into TreeDB's no-document vector-index benchmark route, which requires
    a `column_graph` cosine index and fails closed on unsupported/missing assets.
    """

    index: IndexType = IndexType.HNSW
    metric_type: MetricType = MetricType.COSINE
    m: int = 16
    ef_construction: int = 128
    ef_search: int = 64
    strategy: str = "column_graph"
    use_vector_index: bool = False
    query_mode: str = "exact"
    quantized_codec: str = ""
    quantized_index_name: str = ""
    quantized_rerank_candidates: int = 0
    require_vector_index_guards: bool = True
    experimental: bool = False

    def index_param(self) -> dict:
        params: dict = {
            "strategy": self.strategy,
            "m": self.m,
            "ef_construction": self.ef_construction,
            "ef_search": self.ef_search,
        }
        if self.quantized_index_name:
            params["quantized_indexes"] = [
                {
                    "name": self.quantized_index_name,
                    "codec": self.quantized_codec or "scalar_u8",
                    "version": 1,
                }
            ]
        return params

    def search_param(self) -> dict:
        return {
            "use_vector_index": self.use_vector_index,
            "query_mode": self.query_mode,
            "ef_search": self.ef_search,
            "quantized_index_name": self.quantized_index_name,
            "quantized_rerank_candidates": self.quantized_rerank_candidates,
            "require_vector_index_guards": self.require_vector_index_guards,
            "experimental": self.experimental,
        }


class TreeDBColumnGraphExactConfig(TreeDBHNSWConfig):
    """Named TreeDB no-document exact FP32 column_graph row."""

    use_vector_index: bool = True
    query_mode: str = "exact"
    quantized_codec: str = ""
    quantized_index_name: str = ""
    quantized_rerank_candidates: int = 0


class TreeDBScalarU8RerankConfig(TreeDBHNSWConfig):
    """Named TreeDB scalar_u8 + exact-rerank column_graph row.

    The default rerank shortlist is 32 to match TreeDB's current rerank32
    evidence lane unless a benchmark explicitly overrides it.
    """

    use_vector_index: bool = True
    query_mode: str = "quantized_rerank"
    quantized_codec: str = "scalar_u8"
    quantized_index_name: str = "embedding.scalar_u8.fast"
    quantized_rerank_candidates: int = 32


class TreeDBRaBitQ1BitExperimentalConfig(TreeDBHNSWConfig):
    """Optional experimental RaBitQ v1 row.

    This preserves the existing `rabitq_1bit` codec identity and keeps the row
    explicitly labeled experimental so it cannot be confused with scalar_u8's
    exact-like rerank evidence.
    """

    use_vector_index: bool = True
    query_mode: str = "quantized_only"
    quantized_codec: str = "rabitq_1bit"
    quantized_index_name: str = "embedding.rabitq_1bit.experimental"
    quantized_rerank_candidates: int = 0
    experimental: bool = True


_treedb_case_config = {
    IndexType.HNSW: TreeDBHNSWConfig,
}
