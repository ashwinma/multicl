INCDIR=$(SNUCLROOT)/inc/snuclc/builtins/x86/

#### Detect debug parameter
ifdef debug
SNUCL_DEBUG   := yes
TARDIR        := $(shell /bin/pwd)/out/Debug
else
SNUCL_DEBUG   := no
TARDIR        := $(shell /bin/pwd)/out/Release
endif


#### Detect platform
UNAME := $(shell uname -a)
ifeq ($(findstring Linux, $(UNAME)), Linux)
OS := lnx
else 
OS := undifined
endif


ifeq ($(findstring x86_64, $(UNAME)), x86_64)
SNUCL_MACH    := $(OS)64
TARGET        := libsnucl-builtins-lnx64.a
else
SNUCL_MACH    := $(OS)32
TARGET        := libsnucl-builtins-lnx32.a
endif


ifeq ($(SNUCL_DEBUG), yes)
CC_OPT_LEVEL  := -O0 -g
else
CC_OPT_LEVEL  := -O3
endif



#### TARGET DEPENDENT MACRO
ifeq ($(SNUCL_MACH), lnx32)
CC            := g++
LD            := ld
AR            := ar
INCLUDES      := -I. -I$(INCDIR) -I$(SNUCLROOT)/inc
INCLUDES      += -I$(INCDIR)/type -I$(INCDIR)/common
LDFLAGS       := -lm
CFLAGS        := $(CC_OPT_LEVEL) $(INCLUDES)
CFLAGS				+= -fPIC -m32
else 
ifeq ($(SNUCL_MACH), lnx64)
CC            := g++
LD            := ld
AR            := ar
INCLUDES      := -I. -I$(INCDIR) -I$(SNUCLROOT)/inc
INCLUDES      += -I$(INCDIR)/type -I$(INCDIR)/common
LDFLAGS       := -lm
CFLAGS        := $(CC_OPT_LEVEL) $(INCLUDES)
CFLAGS				+= -fPIC
endif
endif

############################################################
.SUFFIXES : .o .c

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@
############################################################
