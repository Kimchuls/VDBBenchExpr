from pydantic import BaseModel, SecretStr
from ..api import DBConfig, DBCaseConfig, MetricType, IndexType

POSTGRE_URL_PLACEHOLDER = "postgresql://%s:%s@%s/%s"


class SingleStoreConfig(DBConfig):
    # WIP - make it stop asking
    #
    # user: str = "root"
    # password: str = ""
    # db_name: str

    def to_dict(self) -> dict:
        return {
            # "user" : self.user.get_secret_value(),
            # "password" : self.password.get_secret_value(),
            "user": "root",
            "password": "",
            # "host" : "svc-5ee568b2-281c-4097-ba5b-18dfd1fb0680-dml.aws-ohio-1.svc.singlestore.com",
            # "port" : 3306,
            # "user" : "admin",
            # "password" : "Purdueb3nchmark?",
            "local_infile": True
        }


class SingleStoreIndexConfig(BaseModel, DBCaseConfig):
    metric_type: MetricType | None = None
    segments: int | None = 12
    rows: int | None = 10_000_000

    def parse_metric(self) -> str:
        if self.effective_metric() == MetricType.L2:
            return "EUCLIDEAN_DISTANCE"
        if self.effective_metric() == MetricType.IP:
            return "DOT_PRODUCT"

    def effective_metric(self) -> MetricType:
        # WIP - why? Milvus does this.
        #
        if self.metric_type == MetricType.COSINE:
            return MetricType.L2
        return self.metric_type

    def index_param(self) -> dict:
        return {
            "metric_type": self.parse_metric(),
            "segments": self.segments,
            "rows": self.rows,
        }

    def search_param(self) -> dict:
        return {
                }


class HNSWConfig(SingleStoreIndexConfig):
    index: IndexType = IndexType.HNSW
    M: int
    efConstruction: int
    ef: int

    def index_param(self) -> dict:
        return super().index_param() | {
            "index_type": "HNSW_FLAT",
            "M": self.M,
            "efConstruction": self.efConstruction
        }

    def search_param(self) -> dict:
        return super().search_param() | {
            "ef": self.ef,
        }


class IVFConfig(SingleStoreIndexConfig):
    nlist: int
    nprobe: int

    def index_param(self) -> dict:
        return super().index_param() | {
            "index_type": "IVF_FLAT",
            "nlist": self.nlist,
        }

    def search_param(self) -> dict:
        return super().search_param() | {
            "nprobe": self.nprobe,
        }


class IVFPQConfig(IVFConfig):
    index: IndexType = IndexType.IVFPQ
    quantizationRatio: int
    reorder_k: int

    def index_param(self) -> dict:
        return super().index_param() | {
            "index_type": "IVF_PQ",
            "m_dim_divisor": self.quantizationRatio,
        }

    def search_param(self) -> dict:
        return super().search_param() | {
            "kk": self.reorder_k,
        }


class IVFPQFSConfig(IVFPQConfig):
    index: IndexType = IndexType.IVFPQFS

    def index_param(self) -> dict:
        return super().index_param() | {
            "index_type": "IVF_PQFS",
            # "with_raw_data" : False,
        }

    def search_param(self) -> dict:
        return super().search_param()

# class HNSWPQConfig(HNSWConfig):
#     nbits: int | None = None

#     def index_param(self) -> dict:
#         return super().index_param() | {
#             "nbits": self.nbits,
#         }

#     def search_param(self) -> dict:
#         return super().search_param()


_singlestore_case_config = {
    IndexType.HNSW: HNSWConfig,
    IndexType.IVFFlat: IVFConfig,
    IndexType.IVFPQ: IVFPQConfig,
    IndexType.IVFPQFS: IVFPQFSConfig,

    IndexType.AUTOINDEX: SingleStoreIndexConfig,
    IndexType.DISKANN: SingleStoreIndexConfig,
    IndexType.Flat: SingleStoreIndexConfig,
}
