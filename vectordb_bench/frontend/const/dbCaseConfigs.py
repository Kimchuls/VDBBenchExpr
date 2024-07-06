from enum import IntEnum
import typing
from pydantic import BaseModel
from vectordb_bench.backend.cases import CaseType
from vectordb_bench.backend.clients import DB
from vectordb_bench.backend.clients.api import IndexType
from vectordb_bench.models import CaseConfigParamType

MAX_STREAMLIT_INT = (1 << 53) - 1

DB_LIST = [d for d in DB]

DIVIDER = "DIVIDER"
CASE_LIST_WITH_DIVIDER = [
    CaseType.DataInsertion128D1M,
    CaseType.DataInsertion128D10M,
    DIVIDER,
    CaseType.Performance768D100M,
    CaseType.Performance768D10M,
    CaseType.PerformanceRange768D10M,
    CaseType.Performance768D1M,
    CaseType.Performance128D1M,
    CaseType.PerformanceRange128D1M,
    CaseType.Performance128D10M,
    CaseType.PerformanceRange128D10M,
    CaseType.Performance128D100M,
    CaseType.Performance960D1M,
    CaseType.PerformanceRange960D1M,
    CaseType.Performance96D10M,
    DIVIDER,
    CaseType.Performance1536D5M,
    CaseType.Performance1536D500K,
    DIVIDER,
    CaseType.Performance768D10M1P,
    CaseType.Performance768D1M1P,
    DIVIDER,
    CaseType.Performance1536D5M1P,
    CaseType.Performance1536D500K1P,
    DIVIDER,
    CaseType.Performance768D10M99P,
    CaseType.Performance768D1M99P,
    DIVIDER,
    CaseType.Performance1536D5M99P,
    CaseType.Performance1536D500K99P,
    DIVIDER,
    CaseType.CapacityDim960,
    CaseType.CapacityDim128,
]

CASE_LIST = [item for item in CASE_LIST_WITH_DIVIDER if isinstance(item, CaseType)]


class InputType(IntEnum):
    Text = 20001
    Number = 20002
    Option = 20003


class CaseConfigInput(BaseModel):
    label: CaseConfigParamType
    inputType: InputType = InputType.Text
    inputConfig: dict = {}
    # todo type should be a function
    isDisplayed: typing.Any = lambda x: True


CaseConfigParamInput_IndexType = CaseConfigInput(
    label=CaseConfigParamType.IndexType,
    inputType=InputType.Option,
    inputConfig={
        "options": [
            IndexType.HNSW.value,
            IndexType.IVFFlat.value,
            IndexType.DISKANN.value,
            IndexType.Flat.value,
            IndexType.AUTOINDEX.value,
            IndexType.IVFPQFS.value,
        ],
    },
)

CaseConfigParamInput_M = CaseConfigInput(
    label=CaseConfigParamType.M,
    inputType=InputType.Number,
    inputConfig={
        "min": 4,
        "max": 64,
        "value": 16,
    },
    isDisplayed=lambda config: config.get(CaseConfigParamType.IndexType, None)
    == IndexType.HNSW.value,
)


CaseConfigParamInput_EFConstruction_Milvus = CaseConfigInput(
    label=CaseConfigParamType.EFConstruction,
    inputType=InputType.Number,
    inputConfig={
        "min": 8,
        "max": 512,
        "value": 128,
    },
    isDisplayed=lambda config: config[CaseConfigParamType.IndexType]
    == IndexType.HNSW.value,
)

CaseConfigParamInput_EFConstruction_Weaviate = CaseConfigInput(
    label=CaseConfigParamType.EFConstruction,
    inputType=InputType.Number,
    inputConfig={
        "min": 8,
        "max": 512,
        "value": 128,
    },
)

CaseConfigParamInput_EFConstruction_ES = CaseConfigInput(
    label=CaseConfigParamType.EFConstruction,
    inputType=InputType.Number,
    inputConfig={
        "min": 8,
        "max": 512,
        "value": 128,
    },
)

CaseConfigParamInput_EFConstruction_PgVectoRS = CaseConfigInput(
    label=CaseConfigParamType.EFConstruction,
    inputType=InputType.Number,
    inputConfig={
        "min": 8,
        "max": 512,
        "value": 128,
    },
    isDisplayed=lambda config: config[CaseConfigParamType.IndexType]
    == IndexType.HNSW.value,
)

CaseConfigParamInput_IndexType_ES = CaseConfigInput(
    label=CaseConfigParamType.IndexType,
    inputType=InputType.Option,
    inputConfig={
        "options": [
            IndexType.HNSW.value,
            IndexType.HNSWPQ.value,
        ],
    },
)

CaseConfigParamInput_M_ES = CaseConfigInput(
    label=CaseConfigParamType.M,
    inputType=InputType.Number,
    inputConfig={
        "min": 4,
        "max": 64,
        "value": 16,
    },
)

CaseConfigParamInput_NumCandidates_ES = CaseConfigInput(
    label=CaseConfigParamType.numCandidates,
    inputType=InputType.Number,
    inputConfig={
        "min": 1,
        "max": 10000,
        "value": 100,
    },
)

CaseConfigParamInput_EF_Milvus = CaseConfigInput(
    label=CaseConfigParamType.EF,
    inputType=InputType.Number,
    inputConfig={
        "min": 100,
        "max": MAX_STREAMLIT_INT,
        "value": 100,
    },
    isDisplayed=lambda config: config[CaseConfigParamType.IndexType]
    == IndexType.HNSW.value,
)


CaseConfigParamInput_EF_Weaviate = CaseConfigInput(
    label=CaseConfigParamType.EF,
    inputType=InputType.Number,
    inputConfig={
        "min": -1,
        "max": MAX_STREAMLIT_INT,
        "value": -1,
    },
)

CaseConfigParamInput_MaxConnections = CaseConfigInput(
    label=CaseConfigParamType.MaxConnections,
    inputType=InputType.Number,
    inputConfig={"min": 1, "max": MAX_STREAMLIT_INT, "value": 64},
)

CaseConfigParamInput_SearchList = CaseConfigInput(
    label=CaseConfigParamType.SearchList,
    inputType=InputType.Number,
    inputConfig={
        "min": 100,
        "max": MAX_STREAMLIT_INT,
        "value": 100,
    },
    isDisplayed=lambda config: config.get(CaseConfigParamType.IndexType, None)
    == IndexType.DISKANN.value,
)

CaseConfigParamInput_Nlist = CaseConfigInput(
    label=CaseConfigParamType.Nlist,
    inputType=InputType.Number,
    inputConfig={
        "min": 1,
        "max": 65536,
        "value": 1000,
    },
    isDisplayed=lambda config: config.get(CaseConfigParamType.IndexType, None)
    in (IndexType.IVFFlat.value, IndexType.IVFPQFS.value),
)

CaseConfigParamInput_Nprobe = CaseConfigInput(
    label=CaseConfigParamType.Nprobe,
    inputType=InputType.Number,
    inputConfig={
        "min": 1,
        "max": 65536,
        "value": 20,
    },
    isDisplayed=lambda config: config.get(CaseConfigParamType.IndexType, None)
    in (IndexType.IVFFlat.value, IndexType.IVFPQFS.value),
)

CaseConfigParamInput_Lists = CaseConfigInput(
    label=CaseConfigParamType.lists,
    inputType=InputType.Number,
    inputConfig={
        "min": 1,
        "max": 65536,
        "value": 1000,
    },
)

CaseConfigParamInput_Probes = CaseConfigInput(
    label=CaseConfigParamType.probes,
    inputType=InputType.Number,
    inputConfig={
        "min": 1,
        "max": 65536,
        "value": 20,
    },
)

CaseConfigParamInput_QuantizationType_PgVectoRS = CaseConfigInput(
    label=CaseConfigParamType.quantizationType,
    inputType=InputType.Option,
    inputConfig={
        "options": ["trivial", "scalar", "product"],
    },
)

CaseConfigParamInput_QuantizationRatio_PgVectoRS = CaseConfigInput(
    label=CaseConfigParamType.quantizationRatio,
    inputType=InputType.Number,
    inputConfig={
        "min": 1,
        "max": 256,
        "value": 4,
    },
    isDisplayed=lambda config: config.get(CaseConfigParamType.quantizationType, None)
    == "product"
    or config.get(CaseConfigParamType.IndexType, None) in (IndexType.IVFPQFS.value),
)

CaseConfigParamInput_ReorderK = CaseConfigInput(
    label=CaseConfigParamType.reorderK,
    inputType=InputType.Number,
    inputConfig={
        "min": 1,
        "max": 65536,
        "value": 1000,
    },
    isDisplayed=lambda config: config.get(CaseConfigParamType.IndexType, None)
    in (IndexType.IVFFlat.value, IndexType.IVFPQFS.value),
)

MilvusLoadConfig = [
    CaseConfigParamInput_IndexType,
    CaseConfigParamInput_M,
    CaseConfigParamInput_EFConstruction_Milvus,
    CaseConfigParamInput_Nlist,
]
MilvusPerformanceConfig = [
    CaseConfigParamInput_IndexType,
    CaseConfigParamInput_M,
    CaseConfigParamInput_EFConstruction_Milvus,
    CaseConfigParamInput_EF_Milvus,
    CaseConfigParamInput_SearchList,
    CaseConfigParamInput_Nlist,
    CaseConfigParamInput_Nprobe,
    CaseConfigParamInput_QuantizationRatio_PgVectoRS,
    CaseConfigParamInput_ReorderK,
]

WeaviateLoadConfig = [
    CaseConfigParamInput_MaxConnections,
    CaseConfigParamInput_EFConstruction_Weaviate,
]
WeaviatePerformanceConfig = [
    CaseConfigParamInput_MaxConnections,
    CaseConfigParamInput_EFConstruction_Weaviate,
    CaseConfigParamInput_EF_Weaviate,
]

ESLoadingConfig = [
    CaseConfigParamInput_IndexType_ES,
    CaseConfigParamInput_EFConstruction_ES,
    CaseConfigParamInput_M_ES]
ESPerformanceConfig = [
    CaseConfigParamInput_IndexType_ES,
    CaseConfigParamInput_EFConstruction_ES,
    CaseConfigParamInput_M_ES,
    CaseConfigParamInput_NumCandidates_ES,
]

PgVectorLoadingConfig = [CaseConfigParamInput_Lists]
PgVectorPerformanceConfig = [CaseConfigParamInput_Lists, CaseConfigParamInput_Probes]

PgVector2LoadingConfig = [    
    CaseConfigParamInput_IndexType,
    CaseConfigParamInput_M,
    CaseConfigParamInput_EFConstruction_Milvus,
    ]
PgVector2PerformanceConfig = [
    CaseConfigParamInput_IndexType,
    CaseConfigParamInput_M,
    CaseConfigParamInput_EFConstruction_Milvus,
    CaseConfigParamInput_EF_Milvus,]

PgVectoRSLoadingConfig = [
    CaseConfigParamInput_IndexType,
    CaseConfigParamInput_M,
    CaseConfigParamInput_EFConstruction_PgVectoRS,
    CaseConfigParamInput_Nlist,
    CaseConfigParamInput_QuantizationType_PgVectoRS,
    CaseConfigParamInput_QuantizationRatio_PgVectoRS,
]

PgVectoRSPerformanceConfig = [
    CaseConfigParamInput_IndexType,
    CaseConfigParamInput_M,
    CaseConfigParamInput_EFConstruction_PgVectoRS,
    CaseConfigParamInput_Nlist,
    CaseConfigParamInput_Nprobe,
    CaseConfigParamInput_QuantizationType_PgVectoRS,
    CaseConfigParamInput_QuantizationRatio_PgVectoRS,
]

SingleStoreLoadConfig = [
    CaseConfigParamInput_IndexType,
    CaseConfigParamInput_M,
    CaseConfigParamInput_EFConstruction_Milvus,
    CaseConfigParamInput_Nlist,
]
SingleStorePerformanceConfig = [
    CaseConfigParamInput_IndexType,
    CaseConfigParamInput_M,
    CaseConfigParamInput_EFConstruction_Milvus,
    CaseConfigParamInput_EF_Milvus,
    CaseConfigParamInput_Nlist,
    CaseConfigParamInput_Nprobe,
    CaseConfigParamInput_QuantizationRatio_PgVectoRS,
    CaseConfigParamInput_ReorderK
]

CASE_CONFIG_MAP = {
    DB.Milvus: {
        CaseType.CapacityDim960: MilvusLoadConfig,
        CaseType.CapacityDim128: MilvusLoadConfig,
        CaseType.Performance768D100M: MilvusPerformanceConfig,
        CaseType.Performance768D10M: MilvusPerformanceConfig,
        CaseType.PerformanceRange768D10M: MilvusPerformanceConfig,
        CaseType.Performance768D1M: MilvusPerformanceConfig,
        CaseType.Performance128D1M: MilvusPerformanceConfig,
        CaseType.PerformanceRange128D1M: MilvusPerformanceConfig,
        CaseType.Performance128D10M: MilvusPerformanceConfig,
        CaseType.DataInsertion128D1M: MilvusPerformanceConfig,
        CaseType.DataInsertion128D10M: MilvusPerformanceConfig,
        CaseType.PerformanceRange128D10M: MilvusPerformanceConfig,
        CaseType.Performance128D100M: MilvusPerformanceConfig,
        CaseType.Performance960D1M: MilvusPerformanceConfig,
        CaseType.PerformanceRange960D1M: MilvusPerformanceConfig,
        CaseType.Performance96D10M: MilvusPerformanceConfig,
        CaseType.Performance768D10M1P: MilvusPerformanceConfig,
        CaseType.Performance768D1M1P: MilvusPerformanceConfig,
        CaseType.Performance768D10M99P: MilvusPerformanceConfig,
        CaseType.Performance768D1M99P: MilvusPerformanceConfig,
        CaseType.Performance1536D5M: MilvusPerformanceConfig,
        CaseType.Performance1536D500K: MilvusPerformanceConfig,
        CaseType.Performance1536D5M1P: MilvusPerformanceConfig,
        CaseType.Performance1536D500K1P: MilvusPerformanceConfig,
        CaseType.Performance1536D5M99P: MilvusPerformanceConfig,
        CaseType.Performance1536D500K99P: MilvusPerformanceConfig,
    },
    DB.WeaviateCloud: {
        CaseType.CapacityDim960: WeaviateLoadConfig,
        CaseType.CapacityDim128: WeaviateLoadConfig,
        CaseType.Performance768D100M: WeaviatePerformanceConfig,
        CaseType.Performance768D10M: WeaviatePerformanceConfig,
        CaseType.PerformanceRange768D10M: WeaviatePerformanceConfig,
        CaseType.Performance768D1M: WeaviatePerformanceConfig,
        CaseType.Performance128D1M: WeaviatePerformanceConfig,
        CaseType.PerformanceRange128D1M: WeaviatePerformanceConfig,
        CaseType.Performance128D10M: WeaviatePerformanceConfig,
        CaseType.DataInsertion128D1M: WeaviatePerformanceConfig,
        CaseType.DataInsertion128D10M: WeaviatePerformanceConfig,
        CaseType.PerformanceRange128D10M: WeaviatePerformanceConfig,
        CaseType.Performance128D100M: WeaviatePerformanceConfig,
        CaseType.Performance960D1M: WeaviatePerformanceConfig,
        CaseType.PerformanceRange960D1M: WeaviatePerformanceConfig,
        CaseType.Performance96D10M: WeaviatePerformanceConfig,
        CaseType.Performance768D10M1P: WeaviatePerformanceConfig,
        CaseType.Performance768D1M1P: WeaviatePerformanceConfig,
        CaseType.Performance768D10M99P: WeaviatePerformanceConfig,
        CaseType.Performance768D1M99P: WeaviatePerformanceConfig,
        CaseType.Performance1536D5M: WeaviatePerformanceConfig,
        CaseType.Performance1536D500K: WeaviatePerformanceConfig,
        CaseType.Performance1536D5M1P: WeaviatePerformanceConfig,
        CaseType.Performance1536D500K1P: WeaviatePerformanceConfig,
        CaseType.Performance1536D5M99P: WeaviatePerformanceConfig,
        CaseType.Performance1536D500K99P: WeaviatePerformanceConfig,
    },
    DB.ElasticCloud: {
        CaseType.CapacityDim960: ESLoadingConfig,
        CaseType.CapacityDim128: ESLoadingConfig,
        CaseType.Performance768D100M: ESPerformanceConfig,
        CaseType.Performance768D10M: ESPerformanceConfig,
        CaseType.PerformanceRange768D10M: ESPerformanceConfig,
        CaseType.Performance768D1M: ESPerformanceConfig,
        CaseType.Performance128D1M: ESPerformanceConfig,
        CaseType.PerformanceRange128D1M: ESPerformanceConfig,
        CaseType.Performance128D10M: ESPerformanceConfig,
        CaseType.DataInsertion128D1M: ESPerformanceConfig,
        CaseType.DataInsertion128D10M: ESPerformanceConfig,
        CaseType.PerformanceRange128D10M: ESPerformanceConfig,
        CaseType.Performance128D100M: ESPerformanceConfig,
        CaseType.Performance960D1M: ESPerformanceConfig,
        CaseType.PerformanceRange960D1M: ESPerformanceConfig,
        CaseType.Performance96D10M: ESPerformanceConfig,
        CaseType.Performance768D10M1P: ESPerformanceConfig,
        CaseType.Performance768D1M1P: ESPerformanceConfig,
        CaseType.Performance768D10M99P: ESPerformanceConfig,
        CaseType.Performance768D1M99P: ESPerformanceConfig,
        CaseType.Performance1536D5M: ESPerformanceConfig,
        CaseType.Performance1536D500K: ESPerformanceConfig,
        CaseType.Performance1536D5M1P: ESPerformanceConfig,
        CaseType.Performance1536D500K1P: ESPerformanceConfig,
        CaseType.Performance1536D5M99P: ESPerformanceConfig,
        CaseType.Performance1536D500K99P: ESPerformanceConfig,
    },
    DB.PgVector: {
        CaseType.CapacityDim960: PgVectorLoadingConfig,
        CaseType.CapacityDim128: PgVectorLoadingConfig,
        CaseType.Performance768D100M: PgVectorPerformanceConfig,
        CaseType.Performance768D10M: PgVectorPerformanceConfig,
        CaseType.PerformanceRange768D10M: PgVectorPerformanceConfig,
        CaseType.Performance768D1M: PgVectorPerformanceConfig,
        CaseType.Performance128D1M: PgVectorPerformanceConfig,
        CaseType.PerformanceRange128D1M: PgVectorPerformanceConfig,
        CaseType.Performance128D10M: PgVectorPerformanceConfig,
        CaseType.DataInsertion128D10M: PgVectorPerformanceConfig,
        CaseType.DataInsertion128D1M: PgVectorPerformanceConfig,
        CaseType.PerformanceRange128D10M: PgVectorPerformanceConfig,
        CaseType.Performance128D100M: PgVectorPerformanceConfig,
        CaseType.Performance960D1M: PgVectorPerformanceConfig,
        CaseType.PerformanceRange960D1M: PgVectorPerformanceConfig,
        CaseType.Performance96D10M: PgVectorPerformanceConfig,
        CaseType.Performance768D10M1P: PgVectorPerformanceConfig,
        CaseType.Performance768D1M1P: PgVectorPerformanceConfig,
        CaseType.Performance768D10M99P: PgVectorPerformanceConfig,
        CaseType.Performance768D1M99P: PgVectorPerformanceConfig,
        CaseType.Performance1536D5M: PgVectorPerformanceConfig,
        CaseType.Performance1536D500K: PgVectorPerformanceConfig,
        CaseType.Performance1536D5M1P: PgVectorPerformanceConfig,
        CaseType.Performance1536D500K1P: PgVectorPerformanceConfig,
        CaseType.Performance1536D5M99P: PgVectorPerformanceConfig,
        CaseType.Performance1536D500K99P: PgVectorPerformanceConfig,
    },
    DB.PgVector2: {
        CaseType.CapacityDim960: PgVector2LoadingConfig,
        CaseType.CapacityDim128: PgVector2LoadingConfig,
        CaseType.Performance768D100M: PgVector2PerformanceConfig,
        CaseType.Performance768D10M: PgVector2PerformanceConfig,
        CaseType.PerformanceRange768D10M: PgVector2PerformanceConfig,
        CaseType.Performance768D1M: PgVector2PerformanceConfig,
        CaseType.Performance128D1M: PgVector2PerformanceConfig,
        CaseType.PerformanceRange128D1M: PgVector2PerformanceConfig,
        CaseType.Performance128D10M: PgVector2PerformanceConfig,
        CaseType.DataInsertion128D1M: PgVector2PerformanceConfig,
        CaseType.DataInsertion128D10M: PgVector2PerformanceConfig,
        CaseType.PerformanceRange128D10M: PgVector2PerformanceConfig,
        CaseType.Performance128D100M: PgVector2PerformanceConfig,
        CaseType.Performance960D1M: PgVector2PerformanceConfig,
        CaseType.PerformanceRange960D1M: PgVector2PerformanceConfig,
        CaseType.Performance96D10M: PgVector2PerformanceConfig,
        CaseType.Performance768D10M1P: PgVector2PerformanceConfig,
        CaseType.Performance768D1M1P: PgVector2PerformanceConfig,
        CaseType.Performance768D10M99P: PgVector2PerformanceConfig,
        CaseType.Performance768D1M99P: PgVector2PerformanceConfig,
        CaseType.Performance1536D5M: PgVector2PerformanceConfig,
        CaseType.Performance1536D500K: PgVector2PerformanceConfig,
        CaseType.Performance1536D5M1P: PgVector2PerformanceConfig,
        CaseType.Performance1536D500K1P: PgVector2PerformanceConfig,
        CaseType.Performance1536D5M99P: PgVector2PerformanceConfig,
        CaseType.Performance1536D500K99P: PgVector2PerformanceConfig,
    },
    DB.PgVectoRS: {
        CaseType.CapacityDim960: PgVectoRSLoadingConfig,
        CaseType.CapacityDim128: PgVectoRSLoadingConfig,
        CaseType.Performance768D100M: PgVectoRSPerformanceConfig,
        CaseType.Performance768D10M: PgVectoRSPerformanceConfig,
        CaseType.PerformanceRange768D10M: PgVectoRSPerformanceConfig,
        CaseType.Performance768D1M: PgVectoRSPerformanceConfig,
        CaseType.Performance128D1M: PgVectoRSPerformanceConfig,
        CaseType.PerformanceRange128D1M: PgVectoRSPerformanceConfig,
        CaseType.Performance128D10M: PgVectoRSPerformanceConfig,
        CaseType.DataInsertion128D1M: PgVectoRSPerformanceConfig,
        CaseType.DataInsertion128D10M: PgVectoRSPerformanceConfig,
        CaseType.PerformanceRange128D10M: PgVectoRSPerformanceConfig,
        CaseType.Performance128D100M: PgVectoRSPerformanceConfig,
        CaseType.Performance960D1M: PgVectoRSPerformanceConfig,
        CaseType.PerformanceRange960D1M: PgVectoRSPerformanceConfig,
        CaseType.Performance96D10M: PgVectoRSPerformanceConfig,
        CaseType.Performance768D10M1P: PgVectoRSPerformanceConfig,
        CaseType.Performance768D1M1P: PgVectoRSPerformanceConfig,
        CaseType.Performance768D10M99P: PgVectoRSPerformanceConfig,
        CaseType.Performance768D1M99P: PgVectoRSPerformanceConfig,
        CaseType.Performance1536D5M: PgVectoRSPerformanceConfig,
        CaseType.Performance1536D500K: PgVectoRSPerformanceConfig,
        CaseType.Performance1536D5M1P: PgVectoRSPerformanceConfig,
        CaseType.Performance1536D500K1P: PgVectoRSPerformanceConfig,
        CaseType.Performance1536D5M99P: PgVectorPerformanceConfig,
        CaseType.Performance1536D500K99P: PgVectoRSPerformanceConfig,
    },
    DB.SingleStore: {
        CaseType.CapacityDim960: SingleStoreLoadConfig,
        CaseType.CapacityDim128: SingleStoreLoadConfig,
        CaseType.Performance768D100M: SingleStorePerformanceConfig,
        CaseType.Performance768D10M: SingleStorePerformanceConfig,
        CaseType.PerformanceRange768D10M: SingleStorePerformanceConfig,
        CaseType.Performance768D1M: SingleStorePerformanceConfig,
        CaseType.Performance128D1M: SingleStorePerformanceConfig,
        CaseType.PerformanceRange128D1M: SingleStorePerformanceConfig,
        CaseType.Performance128D10M: SingleStorePerformanceConfig,
        CaseType.DataInsertion128D1M: SingleStorePerformanceConfig,
        CaseType.DataInsertion128D10M: SingleStorePerformanceConfig,
        CaseType.PerformanceRange128D10M: SingleStorePerformanceConfig,
        CaseType.Performance128D100M: SingleStorePerformanceConfig,
        CaseType.Performance960D1M: SingleStorePerformanceConfig,
        CaseType.PerformanceRange960D1M: SingleStorePerformanceConfig,
        CaseType.Performance96D10M: SingleStorePerformanceConfig,
        CaseType.Performance768D10M1P: SingleStorePerformanceConfig,
        CaseType.Performance768D1M1P: SingleStorePerformanceConfig,
        CaseType.Performance768D10M99P: SingleStorePerformanceConfig,
        CaseType.Performance768D1M99P: SingleStorePerformanceConfig,
        CaseType.Performance1536D5M: SingleStorePerformanceConfig,
        CaseType.Performance1536D500K: SingleStorePerformanceConfig,
        CaseType.Performance1536D5M1P: SingleStorePerformanceConfig,
        CaseType.Performance1536D500K1P: SingleStorePerformanceConfig,
        CaseType.Performance1536D5M99P: SingleStorePerformanceConfig,
        CaseType.Performance1536D500K99P: SingleStorePerformanceConfig,
    },
}
