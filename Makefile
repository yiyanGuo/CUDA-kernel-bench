# CUDA benchmark build file.
#
# How to add a new operator type:
# 1. Add the new kernel/launch source under kernel/<op_name>/.
# 2. Add the matching benchmark entry under benchmark/bench_<op_name>.cu.
# 3. Register the new benchmark entry in main.cpp.
#
# How to add a new implementation for an existing operator type:
# 1. Add the new kernel/launch source under the matching kernel/<op_name>/ folder.
# 2. Register that implementation inside benchmark/bench_<op_name>.cu.
# Source files are collected automatically via wildcard rules below.

NVCC ?= nvcc
TARGET ?= cuda-kernel-bench
CUDA_ARCH ?= sm_89

# Add shared compile-time parameter overrides here, for example:
#   make CUDAFLAGS='-D VECTOR_ADD_N=16777216 -D TRANSPOSE_ROWS=1024'
#   make CUDA_ARCH=sm_86
# Default to Ada Lovelace (RTX 40 series). The include path points at
# ./include so shared headers live in one place.
CUDAFLAGS ?= -std=c++17 -Iinclude
NVCCFLAGS := $(CUDAFLAGS) -arch=$(CUDA_ARCH)

OBJ_DIR := build

# Kernel and benchmark sources are collected automatically.
# New files under kernel/<op_name>/ and benchmark/ are picked up without
# editing this Makefile, but they still need to be registered in the matching
# benchmark file and in main.cpp when applicable.
KERNEL_SRCS := $(sort $(wildcard kernel/*/*.cu))
BENCH_SRCS := $(sort $(wildcard benchmark/*.cu))

# Source list used to generate object list. Headers are intentionally not
# included here because they aren't compiled directly.
SRCS := main.cpp $(KERNEL_SRCS) $(BENCH_SRCS)

# Map source files to object files under $(OBJ_DIR), preserving paths.
# First replace .cu -> $(OBJ_DIR)/%.o, then .cpp -> $(OBJ_DIR)/%.o
OBJS := $(patsubst %.cpp,$(OBJ_DIR)/%.o,$(patsubst %.cu,$(OBJ_DIR)/%.o,$(SRCS)))

.PHONY: all run clean

all: $(TARGET)

$(TARGET): $(OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $(OBJS)

# Pattern rules: compile .cu and .cpp sources into object files in build/
$(OBJ_DIR)/%.o: %.cu
	@mkdir -p $(dir $@)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

$(OBJ_DIR)/%.o: %.cpp
	@mkdir -p $(dir $@)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

# Example:
#   make run OP=vector_add
run: $(TARGET)
	@if [ -z "$(OP)" ]; then \
		echo "Usage: make run OP=<operator_name>"; \
		exit 1; \
	fi
	./$(TARGET) $(OP)

clean:
	rm -rf $(TARGET) $(OBJ_DIR)
