import laddu as ld
import polars as pl
import time

with ld.mpi.MPI():
    print('reading parquet')
    df = pl.read_parquet('analysis/datasets/data_s20.parquet')
    print('done')
    time.sleep(10)
    print('reading dataset')
    ld.io.from_polars(df)
    print('done')
    time.sleep(10)
