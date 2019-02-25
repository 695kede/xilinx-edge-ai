PROJECT   =    detection
OBJ       :=   main.o dpu_ssd.elf ssd_detector.o prior_boxes.o


CXX       :=   g++
CC        :=   gcc

# linking libraries of OpenCV
LDFLAGS   = $(shell pkg-config --libs opencv)

# linking libraries of DNNDK 
LDFLAGS   +=  -lhineon -ln2cube -lpthread -ldputils -lSDL

CUR_DIR   =   $(shell pwd)
SRC       =   $(CUR_DIR)/src
BUILD     =   $(CUR_DIR)/build
MODEL	  =   $(CUR_DIR)/model
VPATH     =   $(SRC)

CFLAGS    :=  -g -O0 -mcpu=cortex-a53 -Wpointer-arith -std=c++11 -ffast-math

 
all: $(BUILD) $(PROJECT)
 
$(PROJECT): $(OBJ)
	$(CXX) $(CFLAGS) $(addprefix $(BUILD)/, $^) -o $@ $(LDFLAGS)
 
%.o : %.cc
	$(CXX) -c $(CFLAGS) $< -o $(BUILD)/$@

%.o : %.cpp
	$(CXX) -c $(CFLAGS) $< -o $(BUILD)/$@

%.elf : 
	cp $(MODEL)/$@ $(BUILD)/$@ 

clean:
	$(RM) -r $(BUILD)
	$(RM) $(PROJECT)

$(BUILD) : 
	-mkdir -p $@
