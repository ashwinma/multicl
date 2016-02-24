#!/bin/sh

if [ -z $SNUCLROOT ]; then
  echo "[ERROR] Set \"SNUCLROOT\" environment."
fi

cd $SNUCLROOT/src/runtime
make -f Makefile.legacy
