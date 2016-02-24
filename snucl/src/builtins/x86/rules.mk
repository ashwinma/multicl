compile:
	@for dir in $(SUBDIRS); do \
	$(MAKE) -C $$dir || exit $?; \
	done

install:
ifndef SNUCLROOT
$(error Missing environment variable SNUCLROOT.)
endif
	cp -f $(TARDIR)/$(TARGET) ${SNUCLROOT}/lib

clean:
	@for dir in $(SUBDIRS); do \
	make -C $$dir clean; \
	done
	rm -rf *.o
