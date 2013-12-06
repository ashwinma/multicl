#!/bin/sh

if [ ! -n "$1" ]; then
  echo "usage:$0 kernel"
  exit;
fi

TRANSLATOR=$SNUCLROOT/bin/clang

KERNEL_DIR="`pwd`/snucl_kernels/host"

CL_TEMP_SRC_NAME="$KERNEL_DIR/__cl_kernel_$1.cl"
CL_TEMP_TRN_NAME="$KERNEL_DIR/__cl_kernel_$1.cpp"
CL_TEMP_BIN_NAME="$KERNEL_DIR/__cl_kernel_$1.so"
CL_TEMP_SHC_NAME="$KERNEL_DIR/__cl_kernel_info_$1.c"
CL_TEMP_SHO_NAME="$KERNEL_DIR/__cl_kernel_info_$1.so"
CL_TEMP_LOG_NAME="$KERNEL_DIR/__cl_kernel_$1.log"

CL_BIN_HEADER_PREFIX="SNUCL_KERNEL_COMPILE_INFO"

shift
CL_OPTIONS=""
OPTIONS=""
while [ $# -gt 0 ]; do
  case "$1" in
    -cl*)
      CL_OPTIONS="$CL_OPTIONS -plugin-arg-snuclc $1"
      ;;
    *)
      OPTIONS="$OPTIONS $1"
      ;;
  esac
  shift
done

$TRANSLATOR -cc1 -load $SNUCLROOT/lib/libSnuCLTranslator.so -plugin snuclc $CL_OPTIONS -I$SNUCLROOT/src/include/compiler -include $SNUCLROOT/src/include/compiler/cl_kernel.h -I. $OPTIONS $CL_TEMP_SRC_NAME -o $CL_TEMP_TRN_NAME >>$CL_TEMP_LOG_NAME 2>&1

grep --text $CL_BIN_HEADER_PREFIX $CL_TEMP_TRN_NAME > $CL_TEMP_SHC_NAME

gcc -shared -fPIC -o $CL_TEMP_SHO_NAME $CL_TEMP_SHC_NAME

exit 0

