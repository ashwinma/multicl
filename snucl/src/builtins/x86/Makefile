.EXPORT_ALL_VARIABLES:

TOPDIR        := $(shell /bin/pwd)
SUBDIRS       := image miscvec math common async conversion geometric integer reinterpreting vector relational printf

############################################################
include $(TOPDIR)/config.mk

all : compile $(OBJS)
# Output is static library
	mkdir -p $(TARDIR)
	rm -f $(TARDIR)/$(TARGET)
	$(AR) rcs $(TARDIR)/$(TARGET) $(addsuffix /*.o,$(SUBDIRS))

include $(TOPDIR)/rules.mk
############################################################
