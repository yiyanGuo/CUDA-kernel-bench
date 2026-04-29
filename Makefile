PYTHON ?= python
TORCH_CUDA_ARCH_LIST ?= 8.9

OP ?=
DIMS ?=
WARMUP ?= 2
REPEAT ?= 5

.PHONY: all help run run-all clean

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

clean:
	rm -rf build
	find . -type d -name '__pycache__' -prune -exec rm -rf {} +
