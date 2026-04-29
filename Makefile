# CUDA benchmark build file.
#
# How to add a new operator type:
# 1. Add the new kernel/launch source under kernel/<op_name>/.
# 2. Add the matching benchmark entry under benchmark/bench_<op_name>.cu.
# 3. Append both source files to the lists below.
# 4. Register the new benchmark entry in main.cpp.
#
# How to add a new implementation for an existing operator type:
# 1. Add the new kernel/launch source under the matching kernel/<op_name>/ folder.
# 2. Register that implementation inside benchmark/bench_<op_name>.cu.
# 3. Append the new kernel source to KERNEL_SRCS below.

NVCC ?= nvcc
TARGET ?= cuda-kernel-bench

# Add shared compile-time parameter overrides here, for example:
#   make CUDAFLAGS='-D VECTOR_ADD_N=16777216 -D TRANSPOSE_ROWS=1024'
# The include path points at ./include so shared headers live in one place.
CUDAFLAGS ?= -std=c++17 -Iinclude

OBJ_DIR := build

# Keep the source list explicit so it is obvious where to register new operators.
KERNEL_SRCS := \
	kernel/vector_add/vector_add_naive.cu \
	kernel/vector_add/vector_add_float4.cu \
	kernel/transpose/transpose_naive.cu \
	kernel/transpose/transpose_tile.cu \
	kernel/reduction/reduction_naive.cu \
	kernel/reduction/reduction_presum.cu \
	kernel/reduction/reduction_presum_float4.cu \
	kernel/reduction/reduction_shuffle.cu \
	kernel/reduction/reduction_grid_stride.cu \
	kernel/reduction/reduction_integrate.cu \
	kernel/scan/scan_naive.cu \
	kernel/scan/scan_one_block.cu \
	kernel/scan/scan_multi_block.cu \
	kernel/scan/scan_warp.cu \
	kernel/scan/thrust_exclusive_scan.cu

# Benchmark files are flattened as benchmark/bench_<op_name>.cu.
BENCH_SRCS := \
	benchmark/bench_reduction.cu \
	benchmark/bench_vector_add.cu \
	benchmark/bench_transpose.cu \
	benchmark/bench_scan.cu

# Source list used to generate object list. Headers are intentionally not
# included here because they aren't compiled directly.
SRCS := main.cpp $(KERNEL_SRCS) $(BENCH_SRCS)

# Map source files to object files under $(OBJ_DIR), preserving paths.
# First replace .cu -> $(OBJ_DIR)/%.o, then .cpp -> $(OBJ_DIR)/%.o
OBJS := $(patsubst %.cpp,$(OBJ_DIR)/%.o,$(patsubst %.cu,$(OBJ_DIR)/%.o,$(SRCS)))

.PHONY: all run clean

all: $(TARGET)

$(TARGET): $(OBJS)
	$(NVCC) $(CUDAFLAGS) -o $@ $(OBJS)

# Pattern rules: compile .cu and .cpp sources into object files in build/
$(OBJ_DIR)/%.o: %.cu
	@mkdir -p $(dir $@)
	$(NVCC) $(CUDAFLAGS) -c $< -o $@

$(OBJ_DIR)/%.o: %.cpp
	@mkdir -p $(dir $@)
	$(NVCC) $(CUDAFLAGS) -c $< -o $@

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
