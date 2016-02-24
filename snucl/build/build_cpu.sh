#!/bin/sh

if [ -z $SNUCLROOT ]; then
  echo "[ERROR] Set \"SNUCLROOT\" environment."
fi

echo "********** Build the SnuCL runtime **********"
cd $SNUCLROOT/src/runtime
make -f Makefile.cpu

echo "********** Build the SnuCL CPU compiler **********"
CBT=compiler-build-temp
cd $SNUCLROOT/build/
mkdir -p $CBT
cd $CBT
$SNUCLROOT/src/compiler/configure
make BUILD_EXAMPLES=1 ENABLE_OPTIMIZED=1
cp Release/bin/clang $SNUCLROOT/bin
cp Release/bin/snuclc-merger $SNUCLROOT/bin
cp Release/lib/libSnuCL* $SNUCLROOT/lib

echo "********* Build the built-in functions **********"
cd $SNUCLROOT/src/builtins/x86
make
make install

echo "********** Build complete **********"
