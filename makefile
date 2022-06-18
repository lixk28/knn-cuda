CXX      := nvcc
CXXFLAGS :=
LDFLAGS  := -lcublas
BUILD    ?=

ifeq ($(BUILD), DEBUG)
    CXXFLAGS += -DDEBUG
endif

.PHONY: all
all:
	$(CXX) $(CXXFLAGS) knn.cu main.cc -o main $(LDFLAGS)

.PHONY: debug
debug:
	$(MAKE) BUILD=DEBUG

.PHONY: clean
clean:
	$(RM) main
