from pydantic import BaseModel, SecretStr
from ..api import DBConfig, DBCaseConfig, MetricType, IndexType

POSTGRE_URL_PLACEHOLDER = "postgresql://%s:%s@%s/%s"

class PgVector2Config(DBConfig):
    user_name: SecretStr = "zhan4404"
    password: SecretStr = "zhan4404"
    url: SecretStr = "localhost"
    db_name: str = "pgvtest"

    def to_dict(self) -> dict:
        user_str = self.user_name.get_secret_value()
        pwd_str = self.password.get_secret_value()
        url_str = self.url.get_secret_value()
        return {
            "url" : POSTGRE_URL_PLACEHOLDER%(user_str, pwd_str, url_str, self.db_name)
        }

class PgVector2IndexConfig(BaseModel, DBCaseConfig):
    metric_type: MetricType | None = None
    M: int | None = 16
    efConstruction: int | None = 128
    ef: int | None = 200

    def parse_metric(self) -> str: 
        if self.metric_type == MetricType.L2:
            return "vector_l2_ops"
        elif self.metric_type == MetricType.IP:
            return "vector_ip_ops"
        return "vector_cosine_ops"
    
    def parse_metric_fun_str(self) -> str: 
        if self.metric_type == MetricType.L2:
            return "l2_distance"
        elif self.metric_type == MetricType.IP:
            return "max_inner_product"
        return "cosine_distance"

    def index_param(self) -> dict:
        return {
            "M": self.M,
            "efConstruction": self.efConstruction,
            "metric" : self.parse_metric()
        }
    
    def search_param(self) -> dict:
        return {
            "ef": self.ef,
            "metric_fun" : self.parse_metric_fun_str()
        }
    
_pgvector_case_config = {
    IndexType.IVFPQFS: PgVector2IndexConfig,
}