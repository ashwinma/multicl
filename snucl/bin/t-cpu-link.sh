#!/bin/bash

if [ ! -n "$1" ]; then
  echo "usage: $0 <output file> <# of files> [input files] [options]"
  exit;
fi

x86_PATH="$SNUCLROOT/inc/snuclc/kernel/x86"
x86_builtin_PATH="$SNUCLROOT/inc/snuclc/builtins/x86"
SRC_MERGER=$SNUCLROOT/bin/snuclc-merger

CL_BUILTIN_IMPORT=-lsnucl-builtins-lnx64

KERNEL_DIR="`pwd`/snucl_kernels/cpu/`hostname`"

CL_TEMP_SHC_NAME="$KERNEL_DIR/__cl_kernel_info_$1.c"
CL_TEMP_SHO_NAME="$KERNEL_DIR/__cl_kernel_info_$1.so"
CL_TEMP_LAU_NAME="$KERNEL_DIR/__cl_kernel_$1.cpp"
CL_TEMP_OBJ_NAME="$KERNEL_DIR/__cl_kernel_$1.o"
CL_TEMP_BIN_NAME="$KERNEL_DIR/__cl_kernel_$1.so"

GCC_FLAGS='-O3'

NUM_CORES=$2
NUM_INPUTS=$3
INPUT_SRCS=""
INPUT_OBJS=""
for I in `seq $NUM_INPUTS`; do
  CL_TEMP_INP_SRC="$KERNEL_DIR/__cl_kernel_$4.cpp"
  CL_TEMP_INP_OBJ="$KERNEL_DIR/__cl_kernel_$4.o"
  INPUT_SRCS=$INPUT_SRCS" "$CL_TEMP_INP_SRC
  INPUT_OBJS=$INPUT_OBJS" "$CL_TEMP_INP_OBJ
  shift
done

OPTIONS=""
while [ $# -gt 3 ]
do
  OPTIONS=$OPTIONS" "$4
  shift
done

$SRC_MERGER $CL_TEMP_LAU_NAME $CL_TEMP_SHC_NAME $INPUT_SRCS

gcc -shared -g -fPIC -o $CL_TEMP_SHO_NAME $CL_TEMP_SHC_NAME

g++ -DDEF_INCLUDE_X86 -g -c -fPIC $GCC_FLAGS -o $CL_TEMP_OBJ_NAME -I. -I$x86_PATH -I$x86_builtin_PATH $CL_TEMP_LAU_NAME -L$SNUCLROOT/lib -DMAX_CONTEXT_COUNT=$NUM_CORES

g++ -DDEF_INCLUDE_X86 -shared -fPIC $GCC_FLAGS -o $CL_TEMP_BIN_NAME -I. -I$x86_PATH $INPUT_OBJS $CL_TEMP_OBJ_NAME -L$SNUCLROOT/lib $CL_BUILTIN_IMPORT -lpthread -lm $OPTIONS


exit 0

