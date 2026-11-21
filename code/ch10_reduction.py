"""讲义：第 10 章 · 归约（notes/lessons/10-reduction.md）

运行：uv run python ch10_reduction.py
把 block 内的两种树形归约一轮一轮模拟出来，再演示 grid-stride loop 和浮点加法不严格结合。
"""

import numpy as np

from ch07_bank_conflict import conflict_ways


def tree_halves(v: list[float]) -> list[list[float]]:
    """前一半加后一半：for d = n/2 … 1：前 d 个线程 temp[t] += temp[t + d]。返回每一轮之后的 temp。"""
    temp = list(v)
    rows = [temp[:]]
    d = len(temp) // 2
    while d > 0:
        for t in range(d):           # 同一轮里这些线程是同时做的（之前有一道 __syncthreads）
            temp[t] += temp[t + d]
        rows.append(temp[:])
        d //= 2
    return rows


def tree_interleaved(v: list[float]) -> list[list[float]]:
    """相邻两两相加：for s = 1, 2, 4…：tid % (2s) == 0 的线程 temp[t] += temp[t + s]。"""
    temp = list(v)
    rows = [temp[:]]
    s = 1
    while s < len(temp):
        for t in range(0, len(temp), 2 * s):
            temp[t] += temp[t + s]
        rows.append(temp[:])
        s *= 2
    return rows


def active_threads_halves(n: int) -> list[list[int]]:
    """前一半加后一半：每一轮干活的线程号。"""
    out, d = [], n // 2
    while d > 0:
        out.append(list(range(d)))
        d //= 2
    return out


def active_threads_interleaved(n: int) -> list[list[int]]:
    """相邻两两相加：每一轮干活的线程号。"""
    out, s = [], 1
    while s < n:
        out.append([t for t in range(n) if t % (2 * s) == 0])
        s *= 2
    return out


def grid_stride_partials(data: np.ndarray, grid_dim: int, block_dim: int) -> np.ndarray:
    """每个线程从 i = 全局编号 开始，每次跳过整个 grid 的线程数，在"寄存器"里累加。返回每个线程的 acc。"""
    total = grid_dim * block_dim
    acc = np.zeros(total, dtype=data.dtype)
    for tid in range(total):
        acc[tid] = data[tid::total].sum()   # i = tid, tid + total, tid + 2·total, …
    return acc


def main():
    v = [3, 1, 4, 1, 5, 9, 2, 6]
    print("\n第一步 浮点加法不严格结合")
    print(f"  (0.1 + 0.2) + 0.3 = {(0.1 + 0.2) + 0.3!r}，0.1 + (0.2 + 0.3) = {0.1 + (0.2 + 0.3)!r}")
    x = np.random.default_rng(0).random(1_000_000).astype(np.float32)
    seq = np.cumsum(x, dtype=np.float32)[-1]   # 一个一个顺序加（CPU 循环的顺序）
    tree = x.sum(dtype=np.float32)              # NumPy 内部分组两两加（和树形归约一样换了顺序）
    print(f"  100 万个 float32：顺序加 = {seq:.2f}，分组加 = {tree:.2f}，差 {abs(seq - tree):.2f}")

    print("\n第二步 前一半加后一半，blockDim = 8")
    for k, row in enumerate(tree_halves(v)):
        print(f"  第 {k} 轮后：{row}")
    print("  轮数 = log2(8) =", len(tree_halves(v)) - 1, "；加法次数 =", sum(len(a) for a in active_threads_halves(8)))

    print("\n第三步 相邻两两相加：结果一样，干活的线程不一样")
    print("  结果：", tree_interleaved(v)[-1][0])
    print("  前一半加后一半，每轮干活的线程：", active_threads_halves(8))
    print("  相邻两两相加，每轮干活的线程：  ", active_threads_interleaved(8))
    print("  相邻两两相加第 2 轮：线程 t 访问 temp[4k]，跨步 4 →", conflict_ways(np.arange(32) * 4), "路冲突")
    print("  前一半加后一半：连续线程读连续地址 →", conflict_ways(np.arange(32)), "路")

    print("\n第五步 grid-stride loop：2 个 block × 4 个线程，处理 20 个数")
    data = np.arange(1, 21, dtype=np.float64)
    acc = grid_stride_partials(data, 2, 4)
    print("  线程 0 依次读的下标：", list(range(0, 20, 8)))
    print("  8 个线程各自的 acc：", acc)
    print("  再做树形归约：", tree_halves(list(acc))[-1][0], "= sum(1..20) =", data.sum())


if __name__ == "__main__":
    main()
