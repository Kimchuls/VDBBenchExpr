import time
import itertools
import logging
from vectordb_bench import config
from vectordb_bench.interface import BenchMarkRunner
from vectordb_bench.models import (
    DB, IndexType, CaseType, TaskConfig, CaseConfig,
)
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--db", type=str, required=True)
parser.add_argument("--algo", type=str, required=True)
parser.add_argument("--dataset", type=str, required=True)
args = parser.parse_args()


log = logging.getLogger(__name__)

# nprobes = [10, 20, 30, 40, 50, 70, 100, 150, 200, 300, 400]
nprobes = [50,70,100,130]
# ks = [100, 200, 300, 500, 1000, 2000,3000]
ks = [100, 200, 300, 500]
ivf_configs = itertools.product(nprobes, ks)

# efss = [100, 300, 400, 500, 600, 700, 800]
efss = [200]
hnsw_config = efss


def test_s2_ivf(dataset, nlist, nprobe, k, segments, rows):
    runner = BenchMarkRunner()

    task_config = TaskConfig(
        db=DB.SingleStore,
        db_config=DB.SingleStore.config_cls(segments=segments, rows=rows),
        db_case_config=DB.SingleStore.case_config_cls(index_type=IndexType.IVFPQFS)(nlist=nlist,
                                                                                    nprobe=nprobe,
                                                                                    reorder_k=k,
                                                                                    quantizationRatio=4,
                                                                                    segments=segments, rows=rows),
        case_config=CaseConfig(case_id=dataset),
    )

    runner.run([task_config])
    runner._sync_running_task()
    result = runner.get_results()
    log.info(f"test result: {result}")


def test_milvus_ivf(dataset, nlist, nprobe, k):
    runner = BenchMarkRunner()

    task_config = TaskConfig(
        db=DB.Milvus,
        db_config=DB.Milvus.config_cls(uri="http://localhost:19530"),
        db_case_config=DB.Milvus.case_config_cls(index_type=IndexType.IVFPQFS)(nlist=nlist,
                                                                               nprobe=nprobe,
                                                                               reorder_k=k,
                                                                               quantizationRatio=4),
        case_config=CaseConfig(case_id=dataset),
    )

    runner.run([task_config])
    runner._sync_running_task()
    result = runner.get_results()
    log.info(f"test result: {result}")


def test_pgvector_ivf(dataset, nlist, nprobe):
    runner = BenchMarkRunner()
    task_config = TaskConfig(
        db=DB.PgVector,
        db_config=DB.PgVector.config_cls(
            user_name="zhan4404", password="zhan4404", url="localhost", db_name="pgvtest"),
        db_case_config=DB.PgVector.case_config_cls(index_type=IndexType.IVFPQFS)(lists=nlist,
                                                   probe=nprobe),
        case_config=CaseConfig(case_id=dataset),
    )

    runner.run([task_config])
    runner._sync_running_task()
    result = runner.get_results()
    log.info(f"test result: {result}")


def test_s2_hnsw(dataset, efs, segments, rows):
    runner = BenchMarkRunner()

    task_config = TaskConfig(
        db=DB.SingleStore,
        db_config=DB.SingleStore.config_cls(segments=segments, rows=rows),
        db_case_config=DB.SingleStore.case_config_cls(index_type=IndexType.HNSW)(M=16,
                                                                                 efConstruction=128,
                                                                                 ef=efs,
                                                                                 segments=segments, rows=rows),
        case_config=CaseConfig(case_id=dataset),
    )

    runner.run([task_config])
    runner._sync_running_task()
    result = runner.get_results()
    log.info(f"test result: {result}")


def test_milvus_hnsw(dataset, efs):
    runner = BenchMarkRunner()

    task_config = TaskConfig(
        db=DB.Milvus,
        db_config=DB.Milvus.config_cls(uri="http://localhost:19530"),
        db_case_config=DB.Milvus.case_config_cls(index_type=IndexType.HNSW)(M=16,
                                                                            efConstruction=128,
                                                                            ef=efs),
        case_config=CaseConfig(case_id=dataset),
    )

    runner.run([task_config])
    runner._sync_running_task()
    result = runner.get_results()
    log.info(f"test result: {result}")


def test_pgvector_hnsw(dataset, efs):
    runner = BenchMarkRunner()
    task_config = TaskConfig(
        db=DB.PgVector2,
        db_config=DB.PgVector2.config_cls(
            user_name="zhan4404", password="zhan4404", url="localhost", db_name="pgvtest"),
        db_case_config=DB.PgVector2.case_config_cls(index_type=IndexType.IVFPQFS)(M=16,
                                                    efConstruction=128,
                                                    ef=efs),
        case_config=CaseConfig(case_id=dataset),
    )

    runner.run([task_config])
    runner._sync_running_task()
    result = runner.get_results()
    log.info(f"test result: {result}")


def main():
    if args.dataset == "cohere10m":
        dataset = CaseType.Performance768D10M
        nlist = 3162
        segments = 74
        rows = 10_000_000
    elif args.dataset == "gist1m":
        dataset = CaseType.Performance960D1M
        nlist = 1000
        segments = 9
        rows = 1_000_000
    elif args.dataset == "sift10m":
        dataset = CaseType.Performance128D10M
        nlist = 3162
        segments = 12
        rows = 10_000_000
    elif args.dataset == "sift1m":
        dataset = CaseType.Performance128D1M
        nlist = 1000
        segments = 12
        rows = 1_000_000
    else:
        assert False

    if args.algo == "ivf":
        if args.db == "pgvector":
            for nprobe in nprobes:
                test_pgvector_ivf(dataset, nlist, nprobe)
        else:
            for nprobe, k in ivf_configs:
                if args.db == "singlestore":
                    test_s2_ivf(dataset, nlist, nprobe, k, segments, rows)
                if args.db == "milvus":
                    test_milvus_ivf(dataset, nlist, nprobe, k)
                
                if config.DROP_OLD == True:
                    print("Exiting after first iteration due to drop_old")
                    return
    elif args.algo == "hnsw":
        for efs in hnsw_config:
            print(efs)
            if args.db == "singlestore":
                test_s2_hnsw(dataset, efs, segments, rows)
            if args.db == "milvus":
                test_milvus_hnsw(dataset, efs)
            if args.db == "pgvector":
                test_pgvector_hnsw(dataset, efs)
            if config.DROP_OLD == True:
                print("Exiting after first iteration due to drop_old")
                return


if __name__ == '__main__':
    main()
