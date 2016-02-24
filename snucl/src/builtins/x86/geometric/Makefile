ifndef TOPDIR
TOPDIR := ..
endif

############################################################
include $(TOPDIR)/config.mk

SUBDIRS    := 
OBJS       := $(patsubst %.c, %.o, $(wildcard *.c))

all: compile $(OBJS)
	@echo "...done"

include $(TOPDIR)/rules.mk
############################################################
