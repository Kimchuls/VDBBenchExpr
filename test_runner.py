import singlestoredb as s2
import json
import pandas as pd
import os
import numpy as np

x={"user":'root', "password":"", "local_infile":True}
rows=10_000_000
segments=12
dim=128
name="points"
conn = s2.connect(**x)
cur = conn.cursor()
cur.execute("drop database if exists db")
# cur.execute(f"create database if not exists db partitions {segments}")
cur.execute(f"create database if not exists db")
cur.execute("use db")
cur.execute("set global enable_background_merger = off")
cur.execute("set global enable_background_flusher = off")
cur.execute("set global internal_columnstore_max_uncompressed_blob_size = 1073741824")

rows_per_segment=(int)(rows*1.1//segments)
cur.execute(f"create table if not exists {name}( "
                          "id int option 'Integer', "
                          f"embedding vector({dim}, F32) not null option 'SeekableString', "
                          "key () using clustered columnstore "
                          f"with(columnstore_segment_rows={rows_per_segment}))")
params = {
  "index_type":"IVF_PQFS", 
  "nlist":3162,
  "metric_type":"EUCLIDEAN_DISTANCE"
}
cur.execute("alter table points "
                              "add vector index(embedding) index_options '%s' ; "
                              % (json.dumps(params)))

print("start loading data")
dataset = "/home/ubuntu/memsql/notes/datasets/sift1m/sift1m_large_10m"
datafile = os.path.join(dataset, "shuffle_train.parquet")
data_df = pd.read_parquet(datafile)
print("finish loading data")

def vector_to_hex(v):
    return np.array(v, np.float32).tobytes(order="C").hex()
times = 100
batch = rows//times
for i in range(times):
  print(f"now is batch {i}")
  all_metadata = data_df['id'][i*batch:(i+1)*batch].tolist()
  all_embeddings = np.stack(data_df['emb'][i*batch:(i+1)*batch]).tolist()
  print(f"batch dataset size: {len(all_embeddings)}, {len(all_metadata)}")
  
  # with open("tempfile", "w") as f:
  #   for i, embedding in zip(all_metadata, all_embeddings):
  #     f.write("%d,%s\n" % (i, vector_to_hex(embedding)))
  #   cur.execute("load data local infile \"%s\" into table points(id, @v) "
  #                                 "format csv "
  #                                 "fields terminated by ',' enclosed by '' escaped by '' "
  #                                 "lines terminated by '\n' "
  #                                 f"set embedding = unhex(@v):>vector({dim}, F32);"
  #                                 % ("tempfile"))
  # os.system("rm -rf tempfile")
  
  # with open("tempfile", "w") as f:
  #   f.write("insert into points values ")
  #   for i, embedding in zip(all_metadata, all_embeddings):
  #     f.write("(%d,unhex(\"%s\"):>vector(%d, F32))" % (i, vector_to_hex(embedding),dim))
  #     if i!=len(all_metadata)-1:
  #       f.write(", ")
  #     else:
  #       f.write("; ")
  # # cur.execute("source /home/ubuntu/memsql/notes/VectorDBBench/tempfile;")
  
  # text=open("/home/ubuntu/memsql/notes/VectorDBBench/tempfile","r").readlines()[0]
  # print(text)
  # cur.execute(f)
  # os.system("rm -rf /home/ubuntu/memsql/notes/VectorDBBench/tempfile")
  f=""
  f+="insert into points values "
  for i, embedding in zip(all_metadata, all_embeddings):
    f+="(%d,unhex(\"%s\"):>vector(%d, F32))" % (i, vector_to_hex(embedding),dim)
    if i!=len(all_metadata)-1:
      f+=",\n"
    else:
      f+=";\n"
  # print(f)
  cur.execute(f)
  cur.execute("select count(*), sum(rows_count) from information_schema.columnar_segments where column_name=\"embedding\";")
  print(f"the segment numbers in singlestore is: {cur.fetchall()}")
  break