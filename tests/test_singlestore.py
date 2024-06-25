import logging
from vectordb_bench.models import (
    DB,
)
from vectordb_bench.backend.clients import (
    MetricType,
)
from vectordb_bench.backend.clients.singlestore.config import (
    SingleStoreIndexConfig, SingleStoreConfig, HNSWConfig
)

from vectordb_bench.backend.clients.chroma.config import ChromaConfig
import numpy as np
import chromadb


log = logging.getLogger(__name__)

""" Tests for Chroma, assumes Chroma is running on localhost:8000, 
    Chroma docs: https://docs.trychroma.com/usage-guide
    To configure Chroma to run in a docker container client/server
    
    To get running: clone chroma repo and run docker-compose up in chroma directory:
    1. git clone chroma repo https://github.com/chroma-core/chroma
    2. cd chroma, docker-compose up -d --build  # start chroma server
    3. default port is 8000, default host is localhost"""



s2Config = SingleStoreConfig(db_name = "db")
s2Config.user = "root"
s2Config.password = ""
s2Config.db_name = "db"

# caseConfig = SingleStoreIndexConfig()
caseConfig = HNSWConfig(M=12, efConstruction=10, ef=10)
caseConfig.metric_type = MetricType.L2
caseConfig.lists = 1000
caseConfig.probes = 10

class TestSingleStore:
    def test_insert_and_search(self):
        assert DB.SingleStore.value == "SingleStore"

        dbcls = DB.SingleStore.init_cls
        dbConfig = DB.SingleStore.config_cls

        dim = 16
        s2 = dbcls(
            dim=dim,
            db_config=s2Config.to_dict(),
            db_case_config=caseConfig,
            indice="example",
            drop_old=True,
        )

        count = 10_000
        filter_value = 0.9
        embeddings = [[np.random.random() for _ in range(dim)] for _ in range(count)]


        # insert
        with s2.init():
            res = s2.insert_embeddings(embeddings=embeddings, metadata=range(count))
            # bulk_insert return
            # assert (
            #     res[0] == count
            # ), f"the return count of bulk insert ({res}) is not equal to count ({count})"

            # count entries in chroma database
            # countRes = s2.collection.count()

            # assert (
            #     countRes == count
            # ), f"the return count of redis client ({countRes}) is not equal to count ({count})"

            s2.optimize()

        # search
        with s2.init():
            test_id = np.random.randint(count)
            #log.info(f"test_id: {test_id}")
            q = embeddings[test_id]

            res = s2.search_embedding(query=q, k=100)
            print(res)
            assert (
                res[0] == int(test_id)
            ), f"the most nearest neighbor ({res[0]}) id is not test_id ({int(test_id)}"
            

        # search with filters, assumes filter format {id: int, metadata: >=int}
        # with s2.init():
        #     filter_value = int(count * filter_value)
        #     test_id = np.random.randint(filter_value, count)
        #     q = embeddings[test_id]


        #     res = s2.search_embedding(
        #         query=q, k=100, filters={"metadata": filter_value}
        #     )
        #     assert (
        #         res[0] == int(test_id)
        #     ), f"the most nearest neighbor ({res[0]}) id is not test_id ({test_id})"
        #     isFilter = True
        #     id_list = []
        #     for id in res:
        #         id_list.append(id)
        #         if int(id) < filter_value:
        #             isFilter = False
        #             break
        #     assert isFilter, f"Filter not working, id_list: {id_list}"

        #     #Test id filter
        #     res = s2.search_embedding(
        #         query=q, k=100, filters={"id": 9999}
        #     )
        #     assert (
        #         res[0] == 9999
        #     )

        #     print("tron")

        #     #Test two filters, id and metadata
        #     res = s2.search_embedding(
        #         query=q, k=100, filters={"metadata": filter_value, "id": 9999}
        #     )
        #     assert (
        #         res[0] == 9999 and len(res) == 1
        #     ), f"filters failed, got: ({res[0]}), expected ({9999})"


TestSingleStore().test_insert_and_search()
