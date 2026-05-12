# CUDA-kernel-bench

这个仓库用于学习 CUDA、Triton 算子开发与优化。框架保持最小化：每个算子目录自己负责 benchmark。

## 结构

```text
kernels/<op>/
  benchmark.py
  *.cu
  *_single_binding.cpp
```

约定：

- `benchmark.py` 是唯一 Python 文件。
- 问题规模、warmup/repeat、是否校验都直接写在 `benchmark.py` 的常量里。
- CUDA 编译结果保留在对应目录的 `.torch_extensions/` 下，后续运行会复用。
- 没有总入口，没有命令行参数，没有共享 Python 框架。

公共头文件保留在 `include/`。

## 环境

```bash
conda activate kernel-bench
```

可选：

```bash
export TORCH_CUDA_ARCH_LIST=8.9
```

## 运行

直接运行算子目录下的 benchmark：

```bash
python kernels/vector_add/benchmark.py
python kernels/reduction/benchmark.py
python kernels/scan/benchmark.py
python kernels/transpose/benchmark.py
python kernels/softmax/benchmark.py
python kernels/rmsnorm/benchmark.py
python kernels/rope/benchmark.py
python kernels/flashattention/benchmark.py
python kernels/hgemm/benchmark.py
```

## 编译缓存

每个 CUDA extension 都在当前算子目录内编译，例如：

```text
kernels/vector_add/.torch_extensions/
kernels/hgemm/.torch_extensions/
```

清理缓存：

```bash
find kernels -type d -name '.torch_extensions' -prune -exec rm -rf {} +
find . -type d -name '__pycache__' -prune -exec rm -rf {} +
```
