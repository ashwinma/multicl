#!/bin/sh
#-*-sh-*-

#
# Copyright © 2012-2013 Inria.  All rights reserved.
# See COPYING in top-level directory.
#

HWLOC_top_builddir="/home/aaji/mpich-gpu-merge.git/src/hwloc/build-cuda"

HWLOC_PLUGINS_PATH=${HWLOC_top_builddir}/src
export HWLOC_PLUGINS_PATH

"$@"
