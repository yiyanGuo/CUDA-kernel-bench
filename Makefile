PYTHON ?= python
NVCC ?= nvcc
TORCH_CUDA_ARCH_LIST ?= 8.9
GPU_DETECT_BIN ?= build/gpu_detect

OP ?=
DIMS ?=
WARMUP ?= 2
REPEAT ?= 5

.PHONY: all help run run-all gpu-detect-build gpu-detect clean

all: help

help:
	@echo "Python benchmark harness for CUDA and Triton kernels."
	@echo "Usage:"
	@echo "  make run OP=vector_add DIMS='16777216'"
	@echo "  make run OP=transpose DIMS='2048 4096'"
	@echo "  make run OP=reduction DIMS='16777216'"
	@echo "  make run OP=scan DIMS='16777216'"
	@echo "  make run OP=scan DIMS='16777216' WARMUP=3 REPEAT=10"
	@echo "  make run-all"
	@echo "  make gpu-detect"
	@echo ""
	@echo "Defaults: WARMUP=$(WARMUP), REPEAT=$(REPEAT), TORCH_CUDA_ARCH_LIST=$(TORCH_CUDA_ARCH_LIST)"

run:
	@if [ -z "$(OP)" ]; then \
		echo "Usage: make run OP=<operator_name> [DIMS='<dim0 dim1 ...>'] [WARMUP=$(WARMUP)] [REPEAT=$(REPEAT)]"; \
		exit 1; \
	fi
	TORCH_CUDA_ARCH_LIST=$(TORCH_CUDA_ARCH_LIST) $(PYTHON) main.py $(OP) $(DIMS) --warmup $(WARMUP) --repeat $(REPEAT)

run-all:
	TORCH_CUDA_ARCH_LIST=$(TORCH_CUDA_ARCH_LIST) $(PYTHON) main.py all --warmup $(WARMUP) --repeat $(REPEAT)

gpu-detect-build: $(GPU_DETECT_BIN)

$(GPU_DETECT_BIN): gpu_detect.cpp
	mkdir -p $(@D)
	$(NVCC) -O3 $< -o $@

gpu-detect: gpu-detect-build
	./$(GPU_DETECT_BIN)

clean:
	rm -rf build
	find . -type d -name '__pycache__' -prune -exec rm -rf {} +
