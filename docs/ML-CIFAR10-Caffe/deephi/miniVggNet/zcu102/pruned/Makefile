PROJECT   =   miniVggNet
PRJCONV   =   miniVggNet_conv

CXX       :=   g++
CC        :=   gcc
OBJ       :=   main.o
OBJCONV	  :=   main.conv.o

# linking libraries of OpenCV
LDFLAGS   =   $(shell pkg-config --libs opencv)
# linking libraries of DNNDK 
LDFLAGS   +=  -lhineon -ln2cube -ldputils -lpthread

CUR_DIR =   $(shell pwd)
SRC     =   $(CUR_DIR)/src
BUILD   =   $(CUR_DIR)/build
MODEL   =   $(CUR_DIR)/model
VPATH   =   $(SRC)
ARCH	= 	$(shell uname -m | sed -e s/arm.*/armv71/ \
				  -e s/aarch64.*/aarch64/ )

CFLAGS    :=  -O2 -Wall -Wpointer-arith -std=c++11 -ffast-math
ifeq ($(ARCH),armv71)
u    CFLAGS +=  -mcpu=cortex-a9 -mfloat-abi=hard -mfpu=neon
	MODEL = $(CUR_DIR)/model/arm32_1152/dpu_miniVggNet_0.elf
endif 
ifeq ($(ARCH),aarch64)
    CFLAGS += -mcpu=cortex-a53
    MODEL = $(CUR_DIR)/model/arm64_4096/dpu_miniVggNet_0.elf 
endif

.PHONY: all clean 

all: $(BUILD) $(PROJECT) $(PRJCONV) 

$(PROJECT) : $(OBJ) 
	$(CXX) $(CFLAGS) $(addprefix $(BUILD)/, $^) $(MODEL) -o $@ $(LDFLAGS)

$(PRJCONV) : $(OBJCONV)
	$(CXX) $(CFLAGS) $(addprefix $(BUILD)/, $^) $(MODEL) -o $@ $(LDFLAGS)


%.o : %.cc
	$(CXX) -c $(CFLAGS) $< -o $(BUILD)/$@

clean:
	$(RM) -rf $(BUILD)
	$(RM) $(PROJECT) 

$(BUILD) : 
	-mkdir -p $@ 
