#SNUCL_DEV_0=0 SNUCL_DEV_1=0 numactl --physcpubind=0 ./disfd > res.1x1.s1.d0d0 2>&1
#SNUCL_DEV_0=0 SNUCL_DEV_1=1 numactl --physcpubind=0 ./disfd > res.1x1.s1.d0d1 2>&1
#SNUCL_DEV_0=0 SNUCL_DEV_1=2 numactl --physcpubind=0 ./disfd > res.1x1.s1.d0d2 2>&1
#SNUCL_DEV_0=1 SNUCL_DEV_1=0 numactl --physcpubind=0 ./disfd > res.1x1.s1.d1d0 2>&1
SNUCL_DEV_0=1 SNUCL_DEV_1=1 numactl --physcpubind=0 ./disfd > res.1x1.s1.d1d1 2>&1
SNUCL_DEV_0=1 SNUCL_DEV_1=2 numactl --physcpubind=0 ./disfd > res.1x1.s1.d1d2 2>&1
#SNUCL_DEV_0=2 SNUCL_DEV_1=0 numactl --physcpubind=0 ./disfd > res.1x1.s1.d2d0 2>&1
SNUCL_DEV_0=2 SNUCL_DEV_1=1 numactl --physcpubind=0 ./disfd > res.1x1.s1.d2d1 2>&1
SNUCL_DEV_0=2 SNUCL_DEV_1=2 numactl --physcpubind=0 ./disfd > res.1x1.s1.d2d2 2>&1

#SNUCL_DEV_0=1 SNUCL_DEV_1=1 numactl --physcpubind=0 ./disfd > res.1x1.s1.auto 2>&1
