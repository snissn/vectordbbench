from typing import Annotated, Unpack

import click

from vectordb_bench.backend.clients import DB
from vectordb_bench.cli.cli import (
    CommonTypedDict,
    HNSWFlavor3,
    cli,
    click_parameter_decorators_from_typed_dict,
    run,
)


class TreeDBTypedDict(CommonTypedDict):
    base_url: Annotated[str, click.option("--base-url", type=str, help="TreeDB document service base URL", required=True)]
    index_name: Annotated[str, click.option("--index-name", type=str, help="TreeDB service index name", default="vector_bench_test", show_default=True)]
    timeout: Annotated[float, click.option("--timeout", type=float, help="HTTP timeout in seconds", default=30.0, show_default=True)]
    use_vector_index: Annotated[bool, click.option("--use-vector-index/--use-dense-vector", type=bool, help="Use TreeDB no-document vector-index route instead of exact dense service route", default=False, show_default=True)]
    strategy: Annotated[str, click.option("--strategy", type=click.Choice(["native_runtime", "column_graph"]), default="column_graph", show_default=True)]
    query_mode: Annotated[str, click.option("--query-mode", type=click.Choice(["exact", "quantized_only", "quantized_rerank"]), default="exact", show_default=True)]
    quantized_codec: Annotated[str, click.option("--quantized-codec", type=click.Choice(["", "scalar_u8", "rabitq_1bit"]), default="", show_default=True)]
    quantized_index_name: Annotated[str, click.option("--quantized-index-name", type=str, default="", help="TreeDB quantized score-plane name")]
    quantized_rerank_candidates: Annotated[int, click.option("--quantized-rerank-candidates", type=int, default=0, help="Exact rerank shortlist for quantized_rerank; use 32 for scalar_u8 rerank32 evidence")]


class TreeDBHNSWTypedDict(TreeDBTypedDict, HNSWFlavor3): ...


@cli.command()
@click_parameter_decorators_from_typed_dict(TreeDBHNSWTypedDict)
def TreeDBHNSW(**parameters: Unpack[TreeDBHNSWTypedDict]):
    """Run VectorDBBench against TreeDB's document service."""
    from ..api import IndexType
    from .config import TreeDBConfig, TreeDBHNSWConfig

    run(
        DB.TreeDB,
        TreeDBConfig(
            db_label=parameters["db_label"],
            base_url=parameters["base_url"],
            index_name=parameters["index_name"],
            timeout=parameters["timeout"],
        ),
        TreeDBHNSWConfig(
            m=parameters["m"],
            ef_construction=parameters["ef_construction"],
            ef_search=parameters["ef_search"],
            index=IndexType.HNSW,
            strategy=parameters["strategy"],
            use_vector_index=parameters["use_vector_index"],
            query_mode=parameters["query_mode"],
            quantized_codec=parameters["quantized_codec"],
            quantized_index_name=parameters["quantized_index_name"],
            quantized_rerank_candidates=parameters["quantized_rerank_candidates"],
        ),
        **parameters,
    )
