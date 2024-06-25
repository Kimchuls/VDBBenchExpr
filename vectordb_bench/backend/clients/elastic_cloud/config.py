from enum import Enum
from pydantic import SecretStr, BaseModel

from ..api import DBConfig, DBCaseConfig, MetricType, IndexType


class ElasticCloudConfig(DBConfig, BaseModel):
    # cloud_id: SecretStr
    password: SecretStr

    def to_dict(self) -> dict:
        return {
            "hosts" : ['https://localhost:9200'],
            "ca_certs" : "/home/ubuntu/other2/elasticsearch/http_ca.crt",

            # "cloud_id": self.cloud_id.get_secret_value(),
            "basic_auth": ("elastic", self.password.get_secret_value()),
        }


class ESElementType(str, Enum):
    float = "float"  # 4 byte
    byte = "byte"  # 1 byte, -128 to 127


class ElasticCloudIndexConfig(BaseModel, DBCaseConfig):
    element_type: ESElementType = ESElementType.float
    index: IndexType = IndexType.ES_HNSW  # ES only support 'hnsw'

    metric_type: MetricType | None = None
    efConstruction: int | None = None
    M: int | None = None
    num_candidates: int | None = None

    def parse_metric(self) -> str:
        if self.metric_type == MetricType.L2:
            return "l2_norm"
        elif self.metric_type == MetricType.IP:
            return "dot_product"
        return "cosine"

    def index_param(self) -> dict:
        params = {
            "type": "dense_vector",
            "index": True,
            "element_type": self.element_type.value,
            "similarity": self.parse_metric(),
            "index_options": {
                "type": "hnsw",
                "m": self.M,
                "ef_construction": self.efConstruction,
            },
        }
        return params

    def search_param(self) -> dict:
        return {
            "num_candidates": self.num_candidates,
        }

class ElasticCloudINT8HNSWIndexConfig(ElasticCloudIndexConfig):
    num_candidates = int

    def index_param(self) -> dict:
        return super().index_param() | {
            "index_options": {
                "type":"int8_hnsw"
            }
        }

_elasticcloud_case_config = {
    # WIP: wut
    IndexType.HNSW: ElasticCloudIndexConfig,

    IndexType.ES_HNSW: ElasticCloudIndexConfig,
    IndexType.HNSWPQ: ElasticCloudINT8HNSWIndexConfig,
}
