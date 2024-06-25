"""Wrapper around the Pgvector vector database over VectorDB"""

import logging
import tempfile
import json
import os
import traceback
import psutil
import multiprocessing as mp
import threading
from contextlib import contextmanager
from typing import Any

from ..api import VectorDB, DBCaseConfig, MetricType
import singlestoredb as s2
import numpy as np

log = logging.getLogger(__name__)

@contextmanager
def TemporaryPipe(mode="r"):
    """ Context manager for creating and automatically destroying a
        named pipe.

        It returns a file name rather than opening the pipe because
        the open() is expected to block until there's a
        reader. Instead it returns the name and expects you to launch
        a reader and *then* open it.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, 'pipe')
        os.mkfifo(path)
        try:
            yield path
        finally:
            os.unlink(path)

def vector_to_hex(v):
    return np.array(v, np.float32).tobytes(order="C").hex()

# def LoadFromPipe(cursor, path):
#     cursor.execute("load data local infile \"%s\" into table points(id, @v) "
#                    "format csv "
#                    "fields terminated by ',' enclosed by '' escaped by '' "
#                    "lines terminated by '\n' "
#                    f"set embedding = unhex(@v):>vector({self.dim}, F32);"
#                    % (path))


class SingleStore(VectorDB):
    def __init__(
        self,
        dim: int,
        db_config: dict,
        db_case_config: DBCaseConfig,
        collection_name: str = "points",
        drop_old: bool = False,
        **kwargs,
    ):
        self.db_config = db_config
        self.case_config = db_case_config
        self.table_name = collection_name
        self.dim = dim
        self.drop_old = drop_old

        # WIP - work around a bug with spurious plan reuse across different params.
        #
        # import uuid
        # self._uuid = str(uuid.uuid1(node=0))

        self._cur = None
        self._ann_query = None

        # We support deferring index build work until all compaction work for newly inserted data is
        # complete. This is expected to result in significantly faster load times, but is only really an
        # option for an initial load scenario. In other words, "True" reflects build times for a one-time-load
        # case - and is overall going to be faster, if you're not measuring build time - while "False"
        # reflects build times for an ongoing ingest case. Other drivers are generally written around a
        # streaming case.
        #
        # self._compact_before_indexing = False
        self._compact_before_indexing = True


        # Hardcoded DROP_OLD=false semantics for testing without restarting streamlit.
        # self.drop_old=False
        self._keep_old_override = False
        # assert(self.drop_old==False)

        # print(json.dumps(self.db_config))
        log.info(json.dumps(self.db_config))

        ## here is a fix
        conn = s2.connect(**self.db_config)
        # conn=s2.connect(host='svc-5ee568b2-281c-4097-ba5b-18dfd1fb0680-dml.aws-ohio-1.svc.singlestore.com', port=3306, user='admin', password='Purdueb3nchmark?')
        self._cur = conn.cursor()

        # WIP
        if self._drop_old():
            self._cur.execute("drop database if exists db")
            # self.pg_table.drop(pg_engine, checkfirst=True)

        params = self.case_config.index_param()
        if params["segments"] >= psutil.cpu_count(logical=False):
            self.cpus= psutil.cpu_count(logical=False)
        else:
            self.cpus=params["segments"]
        self.rows_per_segment=(int)(params["rows"]*1.1//params["segments"])

        # log.info(params)
        # log.info(self.cpus)
        # log.info(self.rows_per_segment)
        # exit(0)
        self._cur.execute(f"create database if not exists db partitions {self.cpus}")
        # self._cur.execute(f"create database if not exists db partitions {psutil.cpu_count(logical=False)}")
        # self._cur.execute(f"create database if not exists db partitions 128")
        # self._cur.execute(f"create database if not exists db partitions 12")
        # self._cur.execute(f"create database if not exists db partitions 9")
        self._cur.execute("use db")

        # self._cur.execute("set global internal_columnstore_max_uncompressed_blob_size = 1073741824")
        # self._cur.execute("set global columnstore_disk_insert_threshold=1.0")
        # self._cur.execute(f"create table if not exists {self.table_name}( "
        #                   "id int, "
        #                   f"embedding vector({self.dim}, F32) not null, "
        #                   "key () using clustered columnstore "
        #                   "with(columnstore_flush_bytes=1073741824, "
        #                   "     columnstore_segment_rows=185190)) ")
        #                   # "     columnstore_segment_rows=1000000)) ")

        # self._cur.execute("set global internal_columnstore_max_uncompressed_blob_size = 536870912")
        # self._cur.execute("set global columnstore_disk_insert_threshold=1.0")
        # self._cur.execute(f"create table if not exists {self.table_name}( "
        #                   "id int, "
        #                   f"embedding vector({self.dim}, F32) not null, "
        #                   "key () using clustered columnstore "
        #                   "with(columnstore_flush_bytes=536870912)) ")


        # # self._cur.execute("set global internal_columnstore_max_uncompressed_blob_size = 1073741824")
        # self._cur.execute(f"create table if not exists {self.table_name}( "
        #                   "id int option 'Integer', "
        #                   f"embedding vector({self.dim}, F32) not null option 'SeekableString', "
        #                   "key () using clustered columnstore "
        #                 #   "with(columnstore_segment_rows=150000))")
        #                   "with(columnstore_segment_rows=1800000))")


        # self._cur.execute("set global sub_to_physical_partition_ratio = 0")
        # self._cur.execute("set global query_parallelism_per_leaf_core = 0")
        # self._cur.execute("set session query_parallelism_per_leaf_core = 0")
        self._cur.execute("set global internal_columnstore_max_uncompressed_blob_size = 1073741824")
        self._cur.execute(f"create table if not exists {self.table_name}( "
                          "id int option 'Integer', "
                          f"embedding vector({self.dim}, F32) not null option 'SeekableString', "
                          "key () using clustered columnstore "
                          f"with(columnstore_segment_rows={self.rows_per_segment}))")

        
        # exit(0)

        # self._cur.execute(f"create table if not exists {self.table_name}( "
        #                   "id int, "
        #                   f"embedding vector({self.dim}, F32) not null, "
        #                   "key () using clustered columnstore "
        #                   "with(columnstore_segment_rows=150000))")

        if not self._compact_before_indexing:
            self._create_index()

        self._cur = None

    def _create_index(self):
        # assert(self.drop_old==False)
        if self._drop_old():
            params = self.case_config.index_param()
            params.pop("rows")
            params.pop("segments")
            if "m_dim_divisor" in params:
                params["m"] = self.dim // params["m_dim_divisor"]
                del params["m_dim_divisor"]
            self._cur.execute("alter table points "
                              "add vector index(embedding) index_options '%s'"
                              % (json.dumps(params)))

    @contextmanager
    def init(self) -> None:
        """
        Examples:
            >>> with self.init():
            >>>     self.insert_embeddings()
            >>>     self.search_embedding()
        """
        # assert(self.drop_old==False)
        conn = s2.connect(**self.db_config)
        self._cur = conn.cursor()
        self._cur.execute("use db")
        self._warm_query()
        yield
        self._cur = None

    def ready_to_load(self):
        pass

    # This serves two purposes:
    # - Triggers query compilation and waits for it to complete.
    # - Loads the relevant index into memory.
    #
    def _warm_query(self):
        # assert(self.drop_old==False)
        self._cur.execute("set interpreter_mode=llvm")
        query = self.build_query(np.array([0] * self.dim,
                                          dtype=np.dtype('float32')),
                                 10)
        self._cur.execute(query)

        self._cur.execute("select count(*) from information_schema.columnar_segments where column_name=\"embedding\";")
        log.info(f"the segment numbers in singlestore is: {self._cur.fetchall()}")

    def _drop_old(self):
        # return False
        # assert(self.drop_old==False)
        return self.drop_old and not self._keep_old_override

    def optimize(self):
        # assert(self.drop_old==False)
        self._cur.execute("optimize table points flush")
        self._cur.execute("optimize table points full")

        assert self._compact_before_indexing == True

        if self._compact_before_indexing:
            self._create_index()

        self._warm_query()

    def ready_to_search(self):
        # assert(self.drop_old==False)
        pass

    def insert_embeddings(
        self,
        embeddings: list[list[float]],
        metadata: list[int],
        **kwargs: Any,
    ) -> (int, Exception):
        # WIP
        # return len(metadata), None
        try:
            if self._keep_old_override:
                return len(metadata), None

            def LoadFromPipe(path):
                # assert(self.drop_old==False)
                self._cur.execute("load data local infile \"%s\" into table points(id, @v) "
                                  "format csv "
                                  "fields terminated by ',' enclosed by '' escaped by '' "
                                  "lines terminated by '\n' "
                                  f"set embedding = unhex(@v):>vector({self.dim}, F32);"
                                  % (path))

            # We're going to stream data over the socket via LOAD DATA
            # LOCAL INFILE "/path/to/named/pipe". The pipe lets us use
            # LOAD DATA rather than slower manual INSERTs, but also avoids
            # having to actually stage the data as a file on disk.
            #
            with TemporaryPipe() as pipe_name:
                t = threading.Thread(target=LoadFromPipe, args=(pipe_name,))
                t.start()
                with open(pipe_name, "w") as f:
                    for i, embedding in zip(metadata, embeddings):
                        f.write("%d,%s\n" % (i, vector_to_hex(embedding)))
                t.join()

            return len(metadata), None
        except Exception as e:
            log.warning(f"Failed to insert data into table ({self.table_name}), error: {e}")
            return 0, e


    def need_normalize_cosine(self) -> bool:
        """Whether this database need to normalize dataset to support COSINE"""
        # assert(self.drop_old==False)
        return True

    def build_query(self, v, k):
        # assert(self.drop_old==False)
        if not self._ann_query:
            metric = self.case_config.effective_metric()
            assert metric in (MetricType.L2, MetricType.IP, MetricType.COSINE)

            param_str = json.dumps(self.case_config.search_param())

            if metric == MetricType.L2:
                self._ann_query = ("select id FROM points order by embedding <-> X'%s' "
                                   f"search_options = '{param_str}' "
                                   "limit %d")
            else:
                self._ann_query = ("select id FROM points order by embedding <*> X'%s' "
                                   f"search_options = '{param_str}' "
                                   "desc limit %d")
            print(self._ann_query)

            self._warm_query()

        return self._ann_query % (vector_to_hex(v), k)

    def search_embedding(        
        self,
        query: list[float],
        k: int = 100,
        filters: dict | None = None,
        timeout: int | None = None,
    ) -> list[int]:

        # assert(self.drop_old==False)
        assert filters is None
        # print (self.build_query(query, k))
        self._cur.execute(self.build_query(query, k))
        # ret = self._cur.fetchall()
        # for id, partition_id in ret:
        #     print(partition_id)
        # return [id for id, parititon_id in ret]
        return [id for id, in self._cur.fetchall()]
