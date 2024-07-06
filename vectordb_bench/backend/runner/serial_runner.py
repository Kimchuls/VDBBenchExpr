import time
import logging
import traceback
import concurrent
import multiprocessing as mp
import math
import psutil

import numpy as np
import pandas as pd

from ..clients import api
from ...metric import calc_recall
from ...models import LoadTimeoutError, PerformanceTimeoutError
from .. import utils
from ... import config
from vectordb_bench.backend.dataset import DatasetManager

NUM_PER_BATCH = config.NUM_PER_BATCH
LOAD_MAX_TRY_COUNT = 10
WAITTING_TIME = 60

log = logging.getLogger(__name__)

def find_and_replace(all_embeddings, all_metadatas, replace_data):
    metadata_to_embedding = {metadata: embedding for metadata, embedding in zip(all_metadatas, all_embeddings)}
    
    new_embeddings = []
    new_metadatas = []
    
    for data in replace_data:
        if data in metadata_to_embedding:
            new_metadatas.append(data)
            new_embeddings.append(metadata_to_embedding[data])
    
    return new_embeddings, new_metadatas

class SerialInsertRunner:
    def __init__(self, db: api.VectorDB, dataset: DatasetManager, normalize: bool, timeout: float | None = None, replace_data: list[int] | None = None):
        self.timeout = timeout if isinstance(timeout, (int, float)) else None
        self.dataset = dataset
        self.db = db
        self.normalize = normalize
        self.replace_data = replace_data

    def task(self) -> int:
        count = 0
        with self.db.init():
            log.info(f"({mp.current_process().name:16}) Start inserting embeddings in batch {config.NUM_PER_BATCH}")
            start = time.perf_counter()
            for data_df in self.dataset:
                all_metadata = data_df['id'].tolist()

                emb_np = np.stack(data_df['emb'])
                # print("checkpoint", emb_np.shape, self.normalize)
                if self.normalize:
                    log.debug("normalize the 100k train data")
                    all_embeddings = emb_np / np.linalg.norm(emb_np, axis=1)[:, np.newaxis].tolist()
                else:
                    all_embeddings = emb_np.tolist()
                del(emb_np)
                log.debug(f"batch dataset size: {len(all_embeddings)}, {len(all_metadata)}")

                last_batch = self.dataset.data.size - count == len(all_metadata)
                insert_count, error = self.db.insert_embeddings(
                    embeddings=all_embeddings,
                    metadata=all_metadata,
                    last_batch=last_batch,
                )
                if error is not None:
                    raise error

                assert insert_count == len(all_metadata)
                count += insert_count
                if count % 100_000 == 0:
                    log.info(f"({mp.current_process().name:16}) Loaded {count} embeddings into VectorDB")
                    # self.db.rows_check()

            log.info(f"({mp.current_process().name:16}) Finish loading all dataset into VectorDB, dur={time.perf_counter()-start}")
            return count
        
        
    def replace_task(self) -> int:
        count = 0
        del_count = 0
        with self.db.init():
            log.info(f"({mp.current_process().name:16}) Start inserting embeddings in batch {config.NUM_PER_BATCH}")
            start = time.perf_counter()
                
            for data_df in self.dataset:
                all_metadata = data_df['id'].tolist()
                emb_np = np.stack(data_df['emb'])
                if self.normalize:
                    log.debug("normalize the 100k train data")
                    all_embeddings = emb_np / np.linalg.norm(emb_np, axis=1)[:, np.newaxis].tolist()
                else:
                    all_embeddings = emb_np.tolist()
                del(emb_np)
                log.debug(f"batch dataset size: {len(all_embeddings)}, {len(all_metadata)}")
                
                new_embeddings, new_metadatas = find_and_replace(all_embeddings, all_metadata, self.replace_data)

                last_batch = self.dataset.data.size - count == len(all_metadata)
                
                delete_count, error = self.db.delete_embeddings(
                    metadata=new_metadatas,
                    last_batch=last_batch,
                )
                if error is not None:
                    raise error
                assert delete_count == len(new_metadatas)
                
                insert_count, error = self.db.insert_embeddings(
                    embeddings=new_embeddings,
                    metadata=new_metadatas,
                    last_batch=last_batch,
                )
                if error is not None:
                    raise error

                assert insert_count == len(new_metadatas)
                del_count += delete_count
                count += insert_count
            log.info(f"({mp.current_process().name:16}) Deleted {del_count} embeddings from VectorDB")
            log.info(f"({mp.current_process().name:16}) Loaded {count} embeddings into VectorDB")
            log.info(f"({mp.current_process().name:16}) Finish loading all dataset into VectorDB, dur={time.perf_counter()-start}")
            # self.db.rows_check()
            return count

    def endless_insert_data(self, all_embeddings, all_metadata, left_id: int = 0) -> int:
        with self.db.init():
            # unique id for endlessness insertion
            all_metadata = [i+left_id for i in all_metadata]

            NUM_BATCHES = math.ceil(len(all_embeddings)/NUM_PER_BATCH)
            log.info(f"({mp.current_process().name:16}) Start inserting {len(all_embeddings)} embeddings in batch {NUM_PER_BATCH}")
            count = 0
            for batch_id in range(NUM_BATCHES):
                retry_count = 0
                already_insert_count = 0
                metadata = all_metadata[batch_id*NUM_PER_BATCH : (batch_id+1)*NUM_PER_BATCH]
                embeddings = all_embeddings[batch_id*NUM_PER_BATCH : (batch_id+1)*NUM_PER_BATCH]

                log.debug(f"({mp.current_process().name:16}) batch [{batch_id:3}/{NUM_BATCHES}], Start inserting {len(metadata)} embeddings")
                while retry_count < LOAD_MAX_TRY_COUNT:
                    insert_count, error = self.db.insert_embeddings(
                        embeddings=embeddings[already_insert_count :],
                        metadata=metadata[already_insert_count :],
                    )
                    already_insert_count += insert_count
                    if error is not None:
                        retry_count += 1
                        time.sleep(WAITTING_TIME)

                        log.info(f"Failed to insert data, try {retry_count} time")
                        if retry_count >= LOAD_MAX_TRY_COUNT:
                            raise error
                    else:
                        break
                log.debug(f"({mp.current_process().name:16}) batch [{batch_id:3}/{NUM_BATCHES}], Finish inserting {len(metadata)} embeddings")

                assert already_insert_count == len(metadata)
                count += already_insert_count
            log.info(f"({mp.current_process().name:16}) Finish inserting {len(all_embeddings)} embeddings in batch {NUM_PER_BATCH}")
        return count

    @utils.time_it
    def _insert_all_batches(self) -> int:
        """Performance case only"""
        with concurrent.futures.ProcessPoolExecutor(mp_context=mp.get_context('spawn'), max_workers=1) as executor:
            future = executor.submit(self.task)
            try:
                count = future.result(timeout=self.timeout)
            except TimeoutError as e:
                msg = f"VectorDB load dataset timeout in {self.timeout}"
                log.warning(msg)
                for pid, _ in executor._processes.items():
                    psutil.Process(pid).kill()
                raise PerformanceTimeoutError(msg) from e
            except Exception as e:
                log.warning(f"VectorDB load dataset error: {e}")
                log.warning(traceback.print_exc())
                raise e from e
            else:
                return count
            
    @utils.time_it
    def _replace_data(self) -> int:
        """Performance case only"""
        with concurrent.futures.ProcessPoolExecutor(mp_context=mp.get_context('spawn'), max_workers=1) as executor:
            future = executor.submit(self.replace_task)
            try:
                count = future.result(timeout=self.timeout)
            except TimeoutError as e:
                msg = f"VectorDB load dataset timeout in {self.timeout}"
                log.warning(msg)
                for pid, _ in executor._processes.items():
                    psutil.Process(pid).kill()
                raise PerformanceTimeoutError(msg) from e
            except Exception as e:
                log.warning(f"VectorDB load dataset error: {e}")
                log.warning(traceback.print_exc())
                raise e from e
            else:
                return count

    def run_endlessness(self) -> int:
        """run forever util DB raises exception or crash"""
        # datasets for load tests are quite small, can fit into memory
        # only 1 file
        data_df = [data_df for data_df in self.dataset][0]
        all_embeddings, all_metadata = np.stack(data_df["emb"]).tolist(), data_df['id'].tolist()

        start_time = time.perf_counter()
        max_load_count, times = 0, 0
        try:
            with self.db.init():
                self.db.ready_to_load()
            while time.perf_counter() - start_time < self.timeout:
                count = self.endless_insert_data(all_embeddings, all_metadata, left_id=max_load_count)
                max_load_count += count
                times += 1
                log.info(f"Loaded {times} entire dataset, current max load counts={utils.numerize(max_load_count)}, {max_load_count}")
        except Exception as e:
            log.info(f"Capacity case load reach limit, insertion counts={utils.numerize(max_load_count)}, {max_load_count}, err={e}")
            traceback.print_exc()
            return max_load_count
        else:
            msg = f"capacity case load timeout in {self.timeout}s"
            log.info(msg)
            raise LoadTimeoutError(msg)

    def run(self) -> int:
        count, dur = self._insert_all_batches()
        return count
    
    def replace(self) -> int:
        count, dur = self._replace_data()
        return count


class SerialSearchRunner:
    def __init__(
        self,
        db: api.VectorDB,
        test_data: list[list[float]],
        ground_truth: pd.DataFrame,
        k: int = 100,
        filters: dict | None = None,
        distance_data: list[float] |  None = None,
    ):
        self.db = db
        self.k = k
        self.filters = filters

        if isinstance(test_data[0], np.ndarray):
            self.test_data = [query.tolist() for query in test_data]
        else:
            self.test_data = test_data
        self.distance_data = distance_data
        self.ground_truth = ground_truth

    def search(self, args: tuple[list, pd.DataFrame]):
        log.info(f"{mp.current_process().name:14} start search the entire test_data to get recall and latency")
        with self.db.init():
            test_data, ground_truth = args

            log.debug(f"test dataset size: {len(test_data)}")
            log.debug(f"ground truth size: {ground_truth.columns}, shape: {ground_truth.shape}")

            latencies, recalls = [], []
            times = []
            for idx, emb in enumerate(test_data):
                s = time.perf_counter()
                distance  = [] if self.distance_data == [] else [self.distance_data[idx]]
                try:
                    results = self.db.search_embedding(
                        emb,
                        self.k,
                        self.filters,
                        distance,
                    )

                except Exception as e:
                    log.warning(f"VectorDB search_embedding error: {e}")
                    traceback.print_exc(chain=True)
                    raise e from None

                latencies.append(time.perf_counter() - s)

                gt = ground_truth['neighbors_id'][idx].to_list()
                recalls.append(calc_recall(self.k, gt[:self.k], results))   
                times.append(1.0*len(results)/self.k)
                # if len(results)>100:
                #     print(idx, len(results))   
                # print(results)
                # print(self.k)   
                # print(gt[:self.k])  
                # print(emb)  
                # exit(0)   


                if len(latencies) % 1000 == 0:
                    # self.db.rows_check()
                    # self.db._cur.execute("select count(*) as cnt, sum(rows_count) as total from information_schema.COLUMNAR_SEGMENTS where table_name = \"points\" and column_name = \"id\";")
                    # ans=[(cnt, total) for cnt, total, in self.db._cur.fetchall()]
                    # log.info(f"During Searching, {str(ans)} rows.")
                    log.debug(f"({mp.current_process().name:14}) search_count={len(latencies):3}, latest_latency={latencies[-1]}, latest recall={recalls[-1]}")
                # break

        avg_latency = round(np.mean(latencies), 4)
        avg_recall = round(np.mean(recalls), 4)
        avg_scale = round(np.mean(times), 4)
        cost = round(np.sum(latencies), 4)
        p99 = round(np.percentile(latencies, 99), 4)
        log.info(
            f"{mp.current_process().name:14} search entire test_data: "
            f"cost={cost}s, "
            f"queries={len(latencies)}, "
            f"avg_recall={avg_recall}, "
            f"avg_latency={avg_latency}, "
            f"p99={p99}, "
            f"scale={avg_scale}"
         )
        return (avg_recall, p99)


    def _run_in_subprocess(self) -> tuple[float, float]:
        with concurrent.futures.ProcessPoolExecutor(max_workers=1) as executor:
            future = executor.submit(self.search, (self.test_data, self.ground_truth))
            result = future.result()
            return result

    def run(self) -> tuple[float, float]:
        return self._run_in_subprocess()
