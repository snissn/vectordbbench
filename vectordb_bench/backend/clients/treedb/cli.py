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


class TreeDBBaseTypedDict(CommonTypedDict):
    base_url: Annotated[
        str,
        click.option("--base-url", type=str, help="TreeDB document service base URL", required=True),
    ]
    index_name: Annotated[
        str,
        click.option(
            "--index-name",
            type=str,
            help="TreeDB service index name",
            default="vector_bench_test",
            show_default=True,
        ),
    ]
    timeout: Annotated[
        float,
        click.option("--timeout", type=float, help="HTTP timeout in seconds", default=30.0, show_default=True),
    ]
    strategy: Annotated[
        str,
        click.option(
            "--strategy",
            type=click.Choice(["native_runtime", "column_graph"]),
            default="column_graph",
            show_default=True,
        ),
    ]
    require_vector_index_guards: Annotated[
        bool,
        click.option(
            "--require-vector-index-guards/--skip-vector-index-guards",
            type=bool,
            help="Verify TreeDB no-document/vector-index route counters in search responses",
            default=True,
            show_default=True,
        ),
    ]
    query_embedding_encoding: Annotated[
        str,
        click.option(
            "--query-embedding-encoding",
            type=click.Choice(["json", "f32_le_b64"]),
            default="json",
            show_default=True,
            help="Encode TreeDB vector-index query embeddings as JSON floats or base64 little-endian float32 bytes",
        ),
    ]


class TreeDBTypedDict(TreeDBBaseTypedDict):
    use_vector_index: Annotated[
        bool,
        click.option(
            "--use-vector-index/--use-dense-vector",
            type=bool,
            help="Use TreeDB no-document vector-index route instead of exact dense service route",
            default=False,
            show_default=True,
        ),
    ]
    query_mode: Annotated[
        str,
        click.option(
            "--query-mode",
            type=click.Choice(["exact", "quantized_only", "quantized_rerank"]),
            default="exact",
            show_default=True,
        ),
    ]
    quantized_codec: Annotated[
        str,
        click.option(
            "--quantized-codec",
            type=click.Choice(["", "scalar_u8", "rabitq_1bit"]),
            default="",
            show_default=True,
        ),
    ]
    quantized_index_name: Annotated[
        str,
        click.option("--quantized-index-name", type=str, default="", help="TreeDB quantized score-plane name"),
    ]
    quantized_rerank_candidates: Annotated[
        int,
        click.option(
            "--quantized-rerank-candidates",
            type=int,
            default=0,
            help="Exact rerank shortlist for quantized_rerank; use 32 for scalar_u8 rerank32 evidence",
        ),
    ]


class TreeDBScalarU8RerankTypedDict(TreeDBBaseTypedDict):
    quantized_index_name: Annotated[
        str,
        click.option(
            "--quantized-index-name",
            type=str,
            default="embedding.scalar_u8.fast",
            show_default=True,
            help="TreeDB scalar_u8 score-plane name",
        ),
    ]
    quantized_rerank_candidates: Annotated[
        int,
        click.option(
            "--quantized-rerank-candidates",
            type=int,
            default=32,
            show_default=True,
            help="Exact rerank shortlist; 32 is the baseline scalar_u8 rerank32 evidence target",
        ),
    ]


class TreeDBRaBitQ1BitExperimentalTypedDict(TreeDBBaseTypedDict):
    query_mode: Annotated[
        str,
        click.option(
            "--query-mode",
            type=click.Choice(["quantized_only", "quantized_rerank"]),
            default="quantized_only",
            show_default=True,
            help="Experimental RaBitQ v1 benchmark mode",
        ),
    ]
    quantized_index_name: Annotated[
        str,
        click.option(
            "--quantized-index-name",
            type=str,
            default="embedding.rabitq_1bit.experimental",
            show_default=True,
            help="TreeDB RaBitQ v1 score-plane name",
        ),
    ]
    quantized_rerank_candidates: Annotated[
        int,
        click.option(
            "--quantized-rerank-candidates",
            type=int,
            default=0,
            show_default=True,
            help="Optional exact rerank shortlist for experimental RaBitQ v1 rows",
        ),
    ]


class TreeDBHNSWTypedDict(TreeDBTypedDict, HNSWFlavor3): ...


class TreeDBColumnGraphExactTypedDict(TreeDBBaseTypedDict, HNSWFlavor3): ...


class TreeDBScalarU8RerankHNSWTypedDict(TreeDBScalarU8RerankTypedDict, HNSWFlavor3): ...


class TreeDBRaBitQ1BitExperimentalHNSWTypedDict(TreeDBRaBitQ1BitExperimentalTypedDict, HNSWFlavor3): ...


def _treedb_config(parameters):
    from .config import TreeDBConfig

    return TreeDBConfig(
        db_label=parameters["db_label"],
        base_url=parameters["base_url"],
        index_name=parameters["index_name"],
        timeout=parameters["timeout"],
        query_embedding_encoding=parameters["query_embedding_encoding"],
    )


def _run_treedb(parameters, db_case_config):
    run(DB.TreeDB, _treedb_config(parameters), db_case_config, **parameters)


@cli.command()
@click_parameter_decorators_from_typed_dict(TreeDBHNSWTypedDict)
def TreeDBHNSW(**parameters: Unpack[TreeDBHNSWTypedDict]):
    """Run VectorDBBench against TreeDB's document service."""
    from ..api import IndexType
    from .config import TreeDBHNSWConfig

    _run_treedb(
        parameters,
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
            require_vector_index_guards=parameters["require_vector_index_guards"],
        ),
    )


@cli.command()
@click_parameter_decorators_from_typed_dict(TreeDBColumnGraphExactTypedDict)
def TreeDBColumnGraphExact(**parameters: Unpack[TreeDBColumnGraphExactTypedDict]):
    """Run TreeDB column_graph exact FP32 no-document search."""
    from ..api import IndexType
    from .config import TreeDBColumnGraphExactConfig

    _run_treedb(
        parameters,
        TreeDBColumnGraphExactConfig(
            m=parameters["m"],
            ef_construction=parameters["ef_construction"],
            ef_search=parameters["ef_search"],
            index=IndexType.HNSW,
            strategy=parameters["strategy"],
            require_vector_index_guards=parameters["require_vector_index_guards"],
        ),
    )


@cli.command()
@click_parameter_decorators_from_typed_dict(TreeDBScalarU8RerankHNSWTypedDict)
def TreeDBScalarU8Rerank(**parameters: Unpack[TreeDBScalarU8RerankHNSWTypedDict]):
    """Run TreeDB scalar_u8 + exact-rerank no-document search (rerank32 by default)."""
    from ..api import IndexType
    from .config import TreeDBScalarU8RerankConfig

    _run_treedb(
        parameters,
        TreeDBScalarU8RerankConfig(
            m=parameters["m"],
            ef_construction=parameters["ef_construction"],
            ef_search=parameters["ef_search"],
            index=IndexType.HNSW,
            strategy=parameters["strategy"],
            quantized_index_name=parameters["quantized_index_name"],
            quantized_rerank_candidates=parameters["quantized_rerank_candidates"],
            require_vector_index_guards=parameters["require_vector_index_guards"],
        ),
    )


@cli.command()
@click_parameter_decorators_from_typed_dict(TreeDBRaBitQ1BitExperimentalHNSWTypedDict)
def TreeDBRaBitQ1BitExperimental(**parameters: Unpack[TreeDBRaBitQ1BitExperimentalHNSWTypedDict]):
    """Run an explicitly experimental TreeDB RaBitQ v1 no-document search row."""
    from ..api import IndexType
    from .config import TreeDBRaBitQ1BitExperimentalConfig

    _run_treedb(
        parameters,
        TreeDBRaBitQ1BitExperimentalConfig(
            m=parameters["m"],
            ef_construction=parameters["ef_construction"],
            ef_search=parameters["ef_search"],
            index=IndexType.HNSW,
            strategy=parameters["strategy"],
            query_mode=parameters["query_mode"],
            quantized_index_name=parameters["quantized_index_name"],
            quantized_rerank_candidates=parameters["quantized_rerank_candidates"],
            require_vector_index_guards=parameters["require_vector_index_guards"],
        ),
    )
