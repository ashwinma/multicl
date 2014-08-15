# Multiple-machine Makefile

SHELL = /bin/sh

# Files

SRC =	ljs.cpp input.cpp integrate.cpp atom.cpp force.cpp neighbor.cpp \
	thermo.cpp comm.cpp timer.cpp output.cpp setup.cpp opencl_wrapper.cpp
INC =	ljs.h atom.h force.h neighbor.h thermo.h timer.h comm.h integrate.h \
	threadData.h opencl_wrapper.h opencl_data.h precision.h variant.h 

# Definitions

ROOT =	miniMD
EXE =	$(ROOT)_$@
OBJ =	$(SRC:.cpp=.o)

# Help

help:
	@echo 'Type "make target" where target is one of:'
	@echo '      nvidia       (using mpicxx with specific nvidia paths for OpenCL)'
	@echo '      opencl       '

# Targets

nvidia:
	@if [ ! -d Obj_$@ ]; then mkdir Obj_$@; fi
	@cp -p $(SRC) $(INC) Obj_$@
	@cp Makefile.$@ Obj_$@/Makefile
	@cd Obj_$@; \
	$(MAKE)  "OBJ = $(OBJ)" "INC = $(INC)" "EXE = ../$(EXE)" ../$(EXE)
#       @if [ -d Obj_$@ ]; then cd Obj_$@; rm $(SRC) $(INC) Makefile*; fi

opencl:
	@if [ ! -d Obj_$@ ]; then mkdir Obj_$@; fi
	@cp -p $(SRC) $(INC) Obj_$@
	@cp Makefile.$@ Obj_$@/Makefile
	@cd Obj_$@; \
	$(MAKE)  "OBJ = $(OBJ)" "INC = $(INC)" "EXE = ../$(EXE)" ../$(EXE)
#       @if [ -d Obj_$@ ]; then cd Obj_$@; rm $(SRC) $(INC) Makefile*; fi

       
# Clean

clean:
	rm -r Obj_*
	
clean_nvidia:
	rm -r Obj_nvidia

clean_opencl:
	rm -r Obj_opencl

# Test

scope=0
force=lj
halfneigh=0
path=""
test:
	bash run_tests ${scope} ${force} ${halfneigh} ${path}   
