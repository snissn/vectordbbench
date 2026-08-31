import pytest
import logging
from vectordb_bench.models import (
    TaskConfig, CaseConfig,
    CaseResult, TestResult,
    Metric, CaseType
)
from vectordb_bench.backend.clients import (
    DB,
    IndexType
)

from vectordb_bench import config


log = logging.getLogger("vectordb_bench")


@pytest.mark.parametrize("value", [True, False])
def test_case_config_rejects_boolean_k(value):
    with pytest.raises(TypeError, match="cannot be boolean"):
        CaseConfig(case_id=CaseType.Performance1024D1M, k=value)


@pytest.mark.parametrize("value", [None, 10, "10"])
def test_case_config_accepts_nonboolean_k(value):
    assert CaseConfig(case_id=CaseType.Performance1024D1M, k=value).k == (None if value is None else 10)


class TestModels:
    @pytest.mark.skip("runs locally")
    def test_test_result(self):
        result = CaseResult(
            task_config=TaskConfig(
                db=DB.Milvus,
                db_config=DB.Milvus.config(),
                db_case_config=DB.Milvus.case_config_cls(index=IndexType.Flat)(),
                case_config=CaseConfig(case_id=CaseType.Performance10M),
            ),
            metrics=Metric(),
        )

        test_result = TestResult(run_id=10000, results=[result])
        test_result.flush()

        with pytest.raises(ValueError):
            result = TestResult.read_file('nosuchfile.json')

    def test_test_result_read_write(self):
        result_dir = config.RESULTS_LOCAL_DIR
        for json_file in result_dir.rglob("result*.json"):
            res = TestResult.read_file(json_file)
            res.flush()

    def test_test_result_merge(self):
        result_dir = config.RESULTS_LOCAL_DIR
        all_results = []

        first_result = None
        for json_file in result_dir.glob("*.json"):
            res = TestResult.read_file(json_file)

            for cr in res.results:
                all_results.append(cr)

            if not first_result:
                first_result = res

        tr = TestResult(
            run_id=first_result.run_id,
            task_label="standard",
            results=all_results,
        )
        tr.flush()

    def test_test_result_display(self):
        result_dir = config.RESULTS_LOCAL_DIR
        for json_file in result_dir.rglob("result*.json"):
            log.info(json_file)
            res = TestResult.read_file(json_file)
            res.display()
