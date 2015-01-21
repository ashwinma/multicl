export TOTAL_FLOAT_ELEMENTS=8388608
echo $TOTAL_FLOAT_ELEMENTS
SNUCL_DEV_0=0 SNUCL_DEV_1=0 numactl --physcpubind=0 ./sample > res.iter1024.float8M.d0d0 2>&1
SNUCL_DEV_0=1 SNUCL_DEV_1=0 numactl --physcpubind=0 ./sample > res.iter1024.float8M.d1d0 2>&1
SNUCL_DEV_0=2 SNUCL_DEV_1=0 numactl --physcpubind=0 ./sample > res.iter1024.float8M.d2d0 2>&1
SNUCL_DEV_0=0 SNUCL_DEV_1=1 numactl --physcpubind=0 ./sample > res.iter1024.float8M.d0d1 2>&1
SNUCL_DEV_0=1 SNUCL_DEV_1=1 numactl --physcpubind=0 ./sample > res.iter1024.float8M.d1d1 2>&1
SNUCL_DEV_0=2 SNUCL_DEV_1=1 numactl --physcpubind=0 ./sample > res.iter1024.float8M.d2d1 2>&1
SNUCL_DEV_0=0 SNUCL_DEV_1=2 numactl --physcpubind=0 ./sample > res.iter1024.float8M.d0d2 2>&1
SNUCL_DEV_0=1 SNUCL_DEV_1=2 numactl --physcpubind=0 ./sample > res.iter1024.float8M.d1d2 2>&1
SNUCL_DEV_0=2 SNUCL_DEV_1=2 numactl --physcpubind=0 ./sample > res.iter1024.float8M.d2d2 2>&1


