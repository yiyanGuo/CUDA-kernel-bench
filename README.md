# CUDA-kernel-bench

这是一个最小 CUDA 算子 benchmark 框架。

现在已经把 benchmark 测试从 kernel 目录剥离出来：`kernel/` 只保留 kernel 和 launch 函数，`benchmark/` 只保留扁平的 `bench_*.cu` 文件和公共工具文件，公共头文件放在 `include/` 下。每种算子类型在自己的 benchmark 文件里维护一个实现注册表，main 只负责按类型分发一次。

## 使用方式

当前只注册了 `transpose` 和 `vector_add` 两种算子。程序从命令行读取算子名，例如：

```bash
make
./cuda-kernel-bench vector_add
./cuda-kernel-bench transpose
./cuda-kernel-bench all
```

## 参数控制

算子参数暂时通过宏定义控制：

```bash
make CUDAFLAGS='-std=c++17 -Iinclude -D VECTOR_ADD_N=16777216 -D TRANSPOSE_ROWS=2048 -D TRANSPOSE_COLS=4096'
```

默认会执行 2 次预热和 5 次重复计时，输出最短耗时、计算吞吐量和带宽吞吐量，并做 CPU 正确性检查。对于同一种算子的新增优化版本，只需要在对应 benchmark 文件里的注册表中新增一项即可。

## 如何新增算子

1. 在 `kernel/<op_name>/` 下新增一个 `.cu` 文件，只实现该算子的 kernel 和 launch 函数。
2. 在 `benchmark/` 下新增对应的 `bench_<op_name>.cu` 文件，里面维护该类型自己的实现注册表，并负责数据准备、CPU 校验、预热、5 次计时和结果输出。
3. 如果只是给已有类型添加新的优化算子，只需要在对应 `bench_<op_name>.cu` 文件的注册表里新增一项，不需要改 `main.cpp`。
4. 如果要新增一个全新的算子类型，则需要再新增一个 `bench_<op_name>.cu` 文件，并在 `main.cpp` 里注册一个新的类型入口。
