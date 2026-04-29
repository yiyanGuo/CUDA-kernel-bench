# CUDA-kernel-bench

这是一个以 Python 为 benchmark 入口的 GPU kernel 实验仓库。

当前结构的目标是把两层职责分开：

- `kernel/` 继续作为主目录，按算子类型分子文件夹。
- 每个算子子目录里既可以放 `.cu`，也可以放 `.py`，不再按后端单独拆目录。
- `benchmark/` 里的 `bench_*.py` 只负责数据准备、参考结果、计时和校验，并把不同后端统一成 Python 函数调用。
- `main.py` 只做算子分发和命令行参数解析。

## 当前后端模型

- CUDA backend：通过 PyTorch 的 `torch.utils.cpp_extension.load(...)` 在运行时编译现有 `.cu` launch 函数，并暴露成 Python callable。
- Triton backend：直接放在 `kernel/<op_name>/` 下的 Python 文件里，由对应 benchmark 按实现注册。

这意味着第一次跑某个 CUDA benchmark 时会有一次 extension 编译开销；之后会复用 `build/torch_extensions/` 下的产物。

## 使用方式

直接运行 Python 入口：

```bash
python main.py vector_add
python main.py vector_add 16777216
python main.py transpose 2048 4096
python main.py reduction 16777216
python main.py scan 16777216
python main.py all
```

默认值：

- `warmup = 2`
- `repeat = 5`

也可以显式覆盖：

```bash
python main.py vector_add 16777216 --warmup 2 --repeat 5
python main.py scan 16777216 --warmup 3 --repeat 10
```

如果你更习惯 `make`，也可以包一层：

```bash
make run OP=vector_add DIMS='16777216'
make run OP=transpose DIMS='2048 4096'
make run OP=scan DIMS='16777216' WARMUP=3 REPEAT=10
make run-all
```

## 目录约定

### `kernel/`

按算子类型分目录，例如：

- `kernel/vector_add/`
- `kernel/reduction/`
- `kernel/scan/`
- `kernel/transpose/`

同一个目录中可以同时存在：

- 原始 CUDA kernel / launch 文件，例如 `.cu`
- Python backend 封装，例如 `*_cuda.py`
- Triton backend，例如 `*_triton.py`
- Python binding 所需的 `*.cpp`

### `benchmark/`

每个算子有一个对应的 Python bench 文件：

- `benchmark/bench_vector_add.py`
- `benchmark/bench_reduction.py`
- `benchmark/bench_scan.py`
- `benchmark/bench_transpose.py`

它们负责：

- 准备输入数据
- 构造 CPU 参考结果
- 拉起不同后端实现
- 统一计时与校验输出

## 如何新增实现

1. 在 `kernel/<op_name>/` 下新增实现文件。
2. 如果是 CUDA 实现，为该算子的 binding / backend Python 文件把新 launch 函数暴露出来。
3. 如果是 Triton 实现，直接在同目录下新增一个 Python backend 文件并返回实现注册表。
4. 在对应 `benchmark/bench_<op_name>.py` 中把该 backend 模块纳入实现列表。

## 当前状态

- `vector_add` 已经同时接通了 CUDA 和 Triton backend。
- `reduction / scan / transpose` 已经接通了 CUDA backend。
- 旧的 C++ benchmark 入口和测试框架文件已经移除，仓库现在只保留 Python benchmark 流程。
