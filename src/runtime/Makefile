include Makefile.common

TARGET := $(SNUCLROOT)/lib/libsnucl_cluster.so
CXX    := mpic++

CXX_FLAGS += -DCLUSTER_PLATFORM -DEXPORT_APIS -DSNUCL_DEBUG

SOURCES += $(wildcard legacy/*.cpp cluster/*.cpp)

OBJ_DIR=objs/cluster

SUBDIRS=legacy cluster
OBJ_SUBDIRS=$(patsubst %,$(OBJ_DIR)/%,$(SUBDIRS))
DEPS_DIR=$(patsubst %,$(OBJ_DIR)/.deps/%,$(SUBDIRS))

include Makefile.rules
