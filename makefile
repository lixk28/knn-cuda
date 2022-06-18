CXX      := nvcc
CXXFLAGS :=
LDFLAGS  := -lm -lcublas -Xcompiler -fopenmp
BUILD    ?=

ifeq ($(BUILD), DEBUG)
    CXXFLAGS += -DDEBUG
endif

.PHONY: all
all:
	$(CXX) $(CXXFLAGS) knn.cu knn_omp.cc main.cc -o main $(LDFLAGS)

.PHONY: debug
debug:
	$(MAKE) BUILD=DEBUG

.PHONY: clean
clean:
	$(RM) main
