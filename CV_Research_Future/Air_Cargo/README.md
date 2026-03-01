# cpp_matching_engine

一个**最小但工程化**的 C++ 撮合引擎 demo，用来展示：

- L2 limit order book（单品种）
- Price-time priority（同价位 FIFO）
- 支持 `Add` / `Cancel`（可扩展 Modify）
- 输出 trades（成交）与 top-of-book
- 自带 **unit tests** 与 **benchmark 脚手架**（不依赖第三方库，方便面试环境）

> 目标不是“写一个交易所”，而是提供一个能讲清楚设计取舍、能跑、能测、能 benchmark 的作品。

## Build & Run

在仓库根目录：

```zsh
cd cpp_matching_engine
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
./build/me_demo
./build/me_tests
./build/me_benchmark
```

## Verified on my machine (reproducible notes)

下述结果来自本仓库当前版本在本机一次真实构建/运行（用于展示“能跑 + 能测 + 有性能数据”）：

- OS: macOS (arm64)
- CMake: 4.2.3
- Compiler: Apple clang version 17.0.0 (clang-1700.6.3.2)
- Build type: Release

### Unit tests

- `ctest --test-dir build`: PASS（1/1）

### Throughput benchmark

运行：`./build/me_benchmark 2000000`

输出（本次测量）：

- `messages=2000000 seconds=0.37916 msg_per_sec=5.27482e+06`
- `latency_us p50=0 p99=0 max=2706`

说明：这是一个非常简化的 micro-benchmark（随机 add/cancel 流），主要用于演示 pipeline 与性能测量方式，而非宣称 production/HFT 性能。延迟测量使用 `steady_clock`，且在同一线程内计时；当操作耗时小于计时分辨率时，p50/p99 可能显示为 0 微秒，这是计时粒度与极短临界区共同导致的。

## 工程结构

- `include/matching_engine/`：公共头文件
- `src/`：实现与 demo main
- `tests/`：极简自测框架（无第三方依赖）
- `benchmarks/`：吞吐/延迟的 micro-benchmark 脚手架

## 设计说明（面试可讲）

1. **数据结构**：按 `Side` 维护 price levels；每个 price level 内维护 FIFO 队列。
2. **撮合规则**：
   - 买单与最优卖价交叉则成交；卖单与最优买价交叉则成交。
   - 成交价使用被动方价格（maker price）。
3. **取消（O(1) cancel）**：
   - 价位内用 `std::list` 保存 FIFO 队列；
   - 用 `order_id -> (side, price, iterator)` 保存到 `list` 节点的迭代器；
   - cancel 时直接 `erase(iterator)`，因此是均摊 O(1)。
   - 取舍点：`std::list` 指针追踪一般不如连续内存结构 cache-friendly，但能换来稳定 cancel；这是低延迟系统里常见的“复杂度 vs cache”权衡点。
4. **benchmark**：用固定随机种子生成订单流，测吞吐（messages/sec）与延迟分位数（p50/p99/max，单位微秒）。

## 下一步可扩展（可选）

- `Modify`（改价/改量）
- partial cancel
- IOC/FOK
- 多品种、分片（sharding）
- lock-free ring buffer 输入、NUMA-aware
